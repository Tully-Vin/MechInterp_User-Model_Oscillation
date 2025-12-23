import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io_utils import read_jsonl, latest_file, ensure_dir
from metrics import technicality_components
from prompt_utils import load_yaml


def extract_assistant_rows(records, jargon_map):
    rows = []
    for rec in records:
        domain = rec["domain"]
        condition = rec["condition"]
        convo_id = rec["conversation_id"]
        turn_idx = 0
        for msg in rec["messages"]:
            if msg["role"] != "assistant":
                continue
            comps = technicality_components(msg["content"], jargon_map[domain])
            row = {
                "domain": domain,
                "condition": condition,
                "conversation_id": convo_id,
                "turn_index": turn_idx,
                **comps,
            }
            rows.append(row)
            turn_idx += 1
    return pd.DataFrame(rows)


def zscore(series):
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


def compute_oscillation(df):
    out = []
    for (domain, condition, convo_id), grp in df.groupby(["domain", "condition", "conversation_id"]):
        tech = grp.sort_values("turn_index")["technicality"].to_numpy()
        if len(tech) < 2:
            continue
        deltas = np.diff(tech)
        mean_abs = float(np.mean(np.abs(deltas)))
        flips = np.sum(np.sign(deltas[1:]) != np.sign(deltas[:-1]))
        flip_rate = float(flips / max(1, len(deltas) - 1))
        oi = mean_abs * (1.0 + flip_rate)
        out.append({
            "domain": domain,
            "condition": condition,
            "conversation_id": convo_id,
            "mean_abs_delta": mean_abs,
            "flip_rate": flip_rate,
            "oscillation_index": oi,
        })
    return pd.DataFrame(out)


def plot_technicality(df, out_dir):
    for domain, grp in df.groupby("domain"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for condition, cgrp in grp.groupby("condition"):
            means = cgrp.groupby("turn_index")["technicality"].mean()
            ax.plot(means.index, means.values, marker="o", label=condition)
        ax.set_title(f"Technicality by turn ({domain})")
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Technicality (z-score sum)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"technicality_{domain}.png", dpi=150)
        plt.close(fig)


def plot_oscillation(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    summary = df.groupby(["domain", "condition"])["oscillation_index"].mean().reset_index()
    for domain in summary["domain"].unique():
        sub = summary[summary["domain"] == domain]
        ax.plot(sub["condition"], sub["oscillation_index"], marker="o", label=domain)
    ax.set_title("Oscillation index by condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Oscillation index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "oscillation_index.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--jargon", default="prompts/domain_jargon.yaml")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    jargon = load_yaml(Path(args.jargon))
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = latest_file(Path("data/raw"))

    records = read_jsonl(input_path)
    df = extract_assistant_rows(records, jargon)

    # Compute z-scores per domain
    for col in ["readability_grade", "avg_sentence_len", "long_word_ratio", "jargon_rate", "code_ratio"]:
        df[col + "_z"] = df.groupby("domain")[col].transform(zscore)

    z_cols = [c for c in df.columns if c.endswith("_z")]
    df["technicality"] = df[z_cols].sum(axis=1)

    tech_path = out_dir / f"technicality_{input_path.stem}.csv"
    df.to_csv(tech_path, index=False)

    osc_df = compute_oscillation(df)
    osc_path = out_dir / f"oscillation_{input_path.stem}.csv"
    osc_df.to_csv(osc_path, index=False)

    plot_technicality(df, out_dir)
    plot_oscillation(osc_df, out_dir)

    print(f"Wrote {tech_path}")
    print(f"Wrote {osc_path}")


if __name__ == "__main__":
    main()
