import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from io_utils import write_jsonl
from prompt_utils import load_yaml, get_system_prompt, build_base_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_response(model, tokenizer, messages, gen_cfg):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            do_sample=gen_cfg["do_sample"],
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--prompts", default="prompts/prompts.yaml")
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

    gen_cfg = cfg["generation"]
    exp_cfg = cfg["experiment"]

    system_prompt = get_system_prompt(prompts)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(exp_cfg["output_dir"])
    output_path = output_dir / f"run_{run_id}.jsonl"

    records = []
    total_generations = (
        len(exp_cfg["domains"])
        * len(exp_cfg["conditions"])
        * exp_cfg["num_conversations"]
        * (1 + exp_cfg["max_feedback_turns"])
    )
    pbar = tqdm(total=total_generations, desc="Generating", unit="gen")
    for domain in exp_cfg["domains"]:
        for condition in exp_cfg["conditions"]:
            feedbacks = prompts["conditions"][condition]["feedbacks"][: exp_cfg["max_feedback_turns"]]
            for i in range(exp_cfg["num_conversations"]):
                seed = gen_cfg["seed"] + i
                set_seed(seed)

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                base_prompt = build_base_prompt(prompts, domain, condition)
                messages.append({"role": "user", "content": base_prompt})

                assistant = generate_response(model, tokenizer, messages, gen_cfg)
                messages.append({"role": "assistant", "content": assistant})
                pbar.update(1)

                for fb in feedbacks:
                    messages.append({"role": "user", "content": fb})
                    assistant = generate_response(model, tokenizer, messages, gen_cfg)
                    messages.append({"role": "assistant", "content": assistant})
                    pbar.update(1)

                convo_id = f"{domain}_{condition}_{i:03d}"
                records.append({
                    "run_id": run_id,
                    "domain": domain,
                    "condition": condition,
                    "conversation_id": convo_id,
                    "seed": seed,
                    "messages": messages,
                    "gen_config": gen_cfg,
                })

    pbar.close()
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} conversations to {output_path}")
    print("Next: run scripts\\analyze.ps1 (or scripts\\analyze.cmd)")


if __name__ == "__main__":
    main()
