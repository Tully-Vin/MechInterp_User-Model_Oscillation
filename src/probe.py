import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from io_utils import read_jsonl, latest_file, ensure_dir
from prompt_utils import load_yaml, get_system_prompt


def get_last_hidden(model, tokenizer, messages, layer_idx):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        out = model(inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states[layer_idx]
    vec = hs[0, -1].float().cpu().numpy()
    return vec


def build_probe_dataset(model, tokenizer, prompts, layer_idx, max_per_class):
    system_prompt = get_system_prompt(prompts)
    rows = []
    for domain, groups in prompts["domains"].items():
        for label_name, label_value in [("expert", 1), ("novice", 0)]:
            samples = groups[label_name][:max_per_class]
            for text in samples:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": text})
                vec = get_last_hidden(model, tokenizer, messages, layer_idx)
                rows.append({
                    "domain": domain,
                    "label": label_value,
                    "vector": vec,
                })
    X = np.stack([r["vector"] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y


def apply_probe(model, tokenizer, probe, records, layer_idx):
    rows = []
    for rec in records:
        domain = rec["domain"]
        condition = rec["condition"]
        convo_id = rec["conversation_id"]
        messages = []
        for msg in rec["messages"]:
            messages.append(msg)
            if msg["role"] != "user":
                continue
            vec = get_last_hidden(model, tokenizer, messages, layer_idx)
            prob = probe.predict_proba(vec.reshape(1, -1))[0, 1]
            rows.append({
                "domain": domain,
                "condition": condition,
                "conversation_id": convo_id,
                "user_turn_index": sum(1 for m in messages if m["role"] == "user") - 1,
                "expert_probability": float(prob),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--prompts", default="prompts/probe_prompts.yaml")
    parser.add_argument("--data", default="")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    prompts = load_yaml(Path(args.prompts))

    model_name = cfg["model"]["name"]
    torch_dtype = getattr(torch, cfg["model"]["torch_dtype"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=cfg["model"].get("use_fast_tokenizer", True))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=cfg["model"]["device_map"],
        torch_dtype=torch_dtype,
    )

    layer_idx = cfg["probe"]["layer_idx"]
    max_per_class = cfg["probe"]["max_samples_per_class"]

    X, y = build_probe_dataset(model, tokenizer, prompts, layer_idx, max_per_class)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    probe.fit(X, y)

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = latest_file(Path("data/raw"))

    records = read_jsonl(data_path)
    df = apply_probe(model, tokenizer, probe, records, layer_idx)

    out_dir = Path(cfg["probe"]["output_dir"])
    ensure_dir(out_dir)
    out_path = out_dir / f"probe_{data_path.stem}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
