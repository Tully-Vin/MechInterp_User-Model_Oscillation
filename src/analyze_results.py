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

def feedback_directions(condition, n_steps):
    if condition == "consistent_basic":
        return np.full(n_steps, -1.0)
    if condition == "consistent_technical":
        return np.full(n_steps, 1.0)
    return np.array([-1.0 if i % 2 == 0 else 1.0 for i in range(n_steps)])

def compute_step_deltas(df):
    rows = []
    for (domain, condition, convo_id), grp in df.groupby(["domain", "condition", "conversation_id"]):
        tech = grp.sort_values("turn_index")["technicality"].to_numpy()
        if len(tech) < 2:
            continue
        deltas = np.diff(tech)
        dirs = feedback_directions(condition, len(deltas))
        for i, (delta, direction) in enumerate(zip(deltas, dirs)):
            rows.append({
                "domain": domain,
                "condition": condition,
                "conversation_id": convo_id,
                "step_index": i + 1,
                "delta": float(delta),
                "direction": float(direction),
                "directional_gain": float(delta * direction),
            })
    return pd.DataFrame(rows)

def compute_conversation_stats(df, step_df):
    rows = []
    for (domain, condition, convo_id), grp in df.groupby(["domain", "condition", "conversation_id"]):
        tech = grp.sort_values("turn_index")["technicality"].to_numpy()
        if len(tech) < 2:
            continue
        tech_mean = float(np.mean(tech))
        tech_std = float(np.std(tech, ddof=0))
        tech_range = float(np.max(tech) - np.min(tech))
        deltas = np.diff(tech)
        mean_abs = float(np.mean(np.abs(deltas)))
        flips = np.sum(np.sign(deltas[1:]) != np.sign(deltas[:-1]))
        flip_rate = float(flips / max(1, len(deltas) - 1))
        oi = mean_abs * (1.0 + flip_rate)
        step_sub = step_df[(step_df["domain"] == domain) & (step_df["condition"] == condition) & (step_df["conversation_id"] == convo_id)]
        gain_mean = float(step_sub["directional_gain"].mean()) if len(step_sub) else 0.0
        gain_pos_rate = float((step_sub["directional_gain"] > 0).mean()) if len(step_sub) else 0.0
        rows.append({
            "domain": domain,
            "condition": condition,
            "conversation_id": convo_id,
            "technicality_mean": tech_mean,
            "technicality_std": tech_std,
            "technicality_range": tech_range,
            "mean_abs_delta": mean_abs,
            "flip_rate": flip_rate,
            "oscillation_index": oi,
            "gain_mean": gain_mean,
            "gain_pos_rate": gain_pos_rate,
        })
    return pd.DataFrame(rows)


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

def plot_trajectories(df, out_dir):
    for (domain, condition), grp in df.groupby(["domain", "condition"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        for convo_id, cgrp in grp.groupby("conversation_id"):
            cgrp = cgrp.sort_values("turn_index")
            ax.plot(cgrp["turn_index"], cgrp["technicality"], color="gray", alpha=0.25, linewidth=1)
        med = grp.groupby("turn_index")["technicality"].median()
        q25 = grp.groupby("turn_index")["technicality"].quantile(0.25)
        q75 = grp.groupby("turn_index")["technicality"].quantile(0.75)
        ax.plot(med.index, med.values, color="black", linewidth=2, label="median")
        ax.fill_between(med.index, q25.values, q75.values, color="black", alpha=0.1, label="IQR")
        ax.set_title(f"Trajectories ({domain}, {condition})")
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Technicality")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectories_{domain}_{condition}.png", dpi=150)
        plt.close(fig)

def plot_expressivity(stats_df, out_dir):
    for domain, grp in stats_df.groupby("domain"):
        fig, ax = plt.subplots(figsize=(7, 4))
        conditions = list(grp["condition"].unique())
        data = [grp[grp["condition"] == c]["technicality_range"].values for c in conditions]
        ax.boxplot(data, labels=conditions, showfliers=False)
        ax.set_title(f"Expressivity range by condition ({domain})")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Technicality range (max-min)")
        fig.tight_layout()
        fig.savefig(out_dir / f"expressivity_range_{domain}.png", dpi=150)
        plt.close(fig)

def plot_directional_gain(step_df, out_dir):
    for domain, grp in step_df.groupby("domain"):
        fig, ax = plt.subplots(figsize=(7, 4))
        conditions = list(grp["condition"].unique())
        data = [grp[grp["condition"] == c]["directional_gain"].values for c in conditions]
        ax.boxplot(data, labels=conditions, showfliers=False)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Directional gain by condition ({domain})")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Directional gain (positive = follows feedback)")
        fig.tight_layout()
        fig.savefig(out_dir / f"directional_gain_{domain}.png", dpi=150)
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

    step_df = compute_step_deltas(df)
    step_path = out_dir / f"step_deltas_{input_path.stem}.csv"
    step_df.to_csv(step_path, index=False)

    stats_df = compute_conversation_stats(df, step_df)
    stats_path = out_dir / f"conversation_stats_{input_path.stem}.csv"
    stats_df.to_csv(stats_path, index=False)

    plot_technicality(df, out_dir)
    plot_trajectories(df, out_dir)
    plot_expressivity(stats_df, out_dir)
    plot_directional_gain(step_df, out_dir)
    plot_oscillation(stats_df, out_dir)

    print(f"Wrote {tech_path}")
    print(f"Wrote {step_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
