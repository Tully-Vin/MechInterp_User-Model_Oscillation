import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from metrics import technicality_components
from prompt_utils import load_yaml


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def zscore(series):
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


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
            rows.append({
                "domain": domain,
                "condition": condition,
                "conversation_id": convo_id,
                "turn_index": turn_idx,
                **comps,
            })
            turn_idx += 1
    df = pd.DataFrame(rows)
    for col in ["readability_grade", "avg_sentence_len", "long_word_ratio", "jargon_rate", "code_ratio"]:
        df[col + "_z"] = df.groupby("domain")[col].transform(zscore)
    z_cols = [c for c in df.columns if c.endswith("_z")]
    df["technicality"] = df[z_cols].sum(axis=1)
    return df


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

def component_stats(tech_df):
    stats = {}
    for domain, grp in tech_df.groupby("domain"):
        stats[domain] = {}
        for col in ["readability_grade", "avg_sentence_len", "long_word_ratio", "jargon_rate", "code_ratio"]:
            stats[domain][col] = {
                "mean": float(grp[col].mean()),
                "std": float(grp[col].std(ddof=0)),
            }
    return stats

def compute_message_metrics(messages, domain, comp_stats, jargon_map):
    rows = []
    for idx, msg in enumerate(messages):
        comps = technicality_components(msg["content"], jargon_map[domain])
        z_vals = {}
        for col, val in comps.items():
            mean = comp_stats[domain][col]["mean"]
            std = comp_stats[domain][col]["std"]
            z_vals[col + "_z"] = 0.0 if std == 0 else (val - mean) / std
        tech = float(sum(z_vals.values()))
        rows.append({
            "idx": idx,
            "role": msg["role"],
            "technicality": tech,
            "readability_grade": comps["readability_grade"],
            "jargon_rate": comps["jargon_rate"],
            "code_ratio": comps["code_ratio"],
        })
    df = pd.DataFrame(rows)
    df["delta"] = df["technicality"].diff().fillna(0.0)
    return df


def load_run_data(run_file):
    records = read_jsonl(run_file)
    jargon = load_yaml(ROOT / "prompts" / "domain_jargon.yaml")
    tech_file = ROOT / "results" / f"technicality_{run_file.stem}.csv"
    if tech_file.exists():
        tech_df = pd.read_csv(tech_file)
    else:
        tech_df = extract_assistant_rows(records, jargon)
    step_df = compute_step_deltas(tech_df)
    stats_df = compute_conversation_stats(tech_df, step_df)
    comp_stats = component_stats(tech_df)
    return records, tech_df, step_df, stats_df, comp_stats, jargon


def build_trajectory_plot(tech_df, selected_convo, step_idx):
    fig = go.Figure()
    convo_ids = list(tech_df["conversation_id"].unique())
    for convo_id, grp in tech_df.groupby("conversation_id"):
        grp = grp.sort_values("turn_index")
        color = "rgba(164,210,255,0.45)"
        width = 1
        if convo_id == selected_convo:
            color = "rgba(255,182,193,0.95)"
            width = 2
        fig.add_trace(go.Scatter(
            x=grp["turn_index"],
            y=grp["technicality"],
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False,
        ))
        if step_idx in grp["turn_index"].values:
            y = float(grp.loc[grp["turn_index"] == step_idx, "technicality"].iloc[0])
            fig.add_trace(go.Scatter(
                x=[step_idx],
                y=[y],
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False,
            ))
    med = tech_df.groupby("turn_index")["technicality"].median()
    q25 = tech_df.groupby("turn_index")["technicality"].quantile(0.25)
    q75 = tech_df.groupby("turn_index")["technicality"].quantile(0.75)
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="rgba(164,210,255,0.45)", width=2),
        name="Conversations",
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="rgba(255,182,193,0.95)", width=2),
        name="Selected",
    ))
    fig.add_trace(go.Scatter(
        x=med.index,
        y=med.values,
        mode="lines",
        line=dict(color="rgba(255,236,179,0.95)", width=2),
        name="Median",
    ))
    fig.add_trace(go.Scatter(
        x=list(q25.index) + list(q75.index[::-1]),
        y=list(q25.values) + list(q75.values[::-1]),
        fill="toself",
        fillcolor="rgba(255,236,179,0.25)",
        line=dict(color="rgba(255,221,153,0.0)"),
        name="IQR",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Turn index",
        yaxis_title="Technicality",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_chat(messages):
    for msg in messages:
        role = msg["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown("**SYSTEM**")
                st.markdown(msg["content"])


st.set_page_config(page_title="Conversation Explorer", layout="wide")
st.title("Conversation Explorer")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,500,0,0');

.metric-stack {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.metric-mini {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  padding: 8px 10px;
}
.metric-stack.selected .metric-mini {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.14);
}
.metric-mini-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(255, 255, 255, 0.6);
}
.metric-mini-value {
  font-size: 0.95rem;
  font-weight: 600;
  margin-top: 4px;
  word-break: break-word;
}
.msg-anchor {
  display: block;
  height: 0;
  width: 0;
  overflow: hidden;
}
.divider {
  height: 100%;
  min-height: 100%;
  width: 1px;
  background: rgba(255, 255, 255, 0.08);
  margin: 0 auto;
}
div[data-testid="stChatMessageContent"] {
  transition: background 0.2s ease-in-out;
}
div[data-testid="stChatMessage"][data-message-role="user"],
div[data-testid="stChatMessage"]:has(.user-marker),
div[data-testid="stChatMessage"]:has(img[alt*="user" i]) {
  flex-direction: row-reverse;
  justify-content: flex-end;
  align-items: flex-end;
  width: fit-content !important;
  max-width: 75% !important;
  margin-left: auto !important;
  margin-right: 0 !important;
  display: flex !important;
  align-self: flex-end !important;
}
div[data-testid="stChatMessage"][data-message-role="user"] div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has(.user-marker) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has(img[alt*="user" i]) div[data-testid="stChatMessageContent"] {
  text-align: right;
  max-width: 100% !important;
  margin-left: auto !important;
  margin-right: 0 !important;
  flex: 0 1 auto;
  width: fit-content !important;
  display: inline-block !important;
  align-self: flex-end;
}
@keyframes highlight-bounce {
  0% { background: rgba(255, 255, 255, 0.08); }
  35% { background: rgba(255, 255, 255, 0.19); }
  65% { background: rgba(255, 255, 255, 0.12); }
  85% { background: rgba(255, 255, 255, 0.1); }
  100% { background: rgba(255, 255, 255, 0.08); }
}
.highlight-pulse {
  animation: highlight-bounce 0.71s cubic-bezier(0.2, 0.9, 0.2, 1.1);
  will-change: background-color;
}
.kpi-card {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
  text-align: center;
}
.kpi-label {
  font-size: 1.1rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: rgba(255, 255, 255, 0.6);
}
.kpi-value {
  font-size: 2.05rem;
  font-weight: 600;
  margin-top: 6px;
}
.kpi-delta {
  font-size: 1.05rem;
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
  justify-content: center;
}
.kpi-delta.pos { color: #7ee59b; }
.kpi-delta.neg { color: #ff7a7a; }
.kpi-delta.neu { color: #f4c38a; }
.kpi-icon {
  font-family: "Material Symbols Rounded";
  font-size: 1.2rem;
  line-height: 1;
  display: inline-flex;
  vertical-align: middle;
}
.kpi-spacer { height: 20px; }
.st-key-conversation_wrap {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 14px;
  padding: 16px 18px;
  margin-top: 12px;
}
.go-button-wrap {
  display: flex;
  height: 100%;
  align-items: stretch;
  justify-content: flex-end;
}
div[data-testid="stColumn"]:has(.go-button-marker) {
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: stretch;
}
div[data-testid="stColumn"]:has(.go-button-marker) button {
  height: 100%;
  min-height: 76px;
  width: 100%;
  margin-top: 0;
}
</style>
    """,
    unsafe_allow_html=True,
)

run_files = sorted((ROOT / "data" / "raw").glob("run_*.jsonl"))
if not run_files:
    st.error("No run files found in data/raw.")
    st.stop()

run_files_sorted = sorted(run_files, key=lambda p: p.name, reverse=True)
run_file = st.sidebar.selectbox("Run file", run_files_sorted, format_func=lambda p: p.name, index=0)

records, tech_df, step_df, stats_df, comp_stats, jargon_map = load_run_data(run_file)

domains = sorted(stats_df["domain"].unique())
conditions = sorted(stats_df["condition"].unique())

st.sidebar.markdown("### Domains")
domain_checks = {}
for d in domains:
    domain_checks[d] = st.sidebar.checkbox(d, value=True, key=f"domain_{d}")
sel_domains = [d for d, v in domain_checks.items() if v]
if not sel_domains:
    st.session_state[f"domain_{domains[0]}"] = True
    sel_domains = [domains[0]]

st.sidebar.markdown("### Conditions")
cond_checks = {}
for c in conditions:
    cond_checks[c] = st.sidebar.checkbox(c, value=True, key=f"condition_{c}")
sel_conditions = [c for c, v in cond_checks.items() if v]
if not sel_conditions:
    st.session_state[f"condition_{conditions[0]}"] = True
    sel_conditions = [conditions[0]]

metric_options = [
    "technicality_mean",
    "technicality_std",
    "technicality_range",
    "oscillation_index",
    "gain_mean",
    "gain_pos_rate",
]
sort_metric = st.sidebar.selectbox("Sort conversations by", metric_options, index=0)
sort_desc = st.sidebar.checkbox("Sort descending", value=True)
top_n = st.sidebar.slider("Max conversations to show", 5, 200, 60)

st.sidebar.markdown("### Metric filters")
metric_ranges = {}
for metric in metric_options:
    series = stats_df[metric]
    min_val = float(series.min())
    max_val = float(series.max())
    slider_key = f"{metric}_slider"
    min_key = f"{metric}_min"
    max_key = f"{metric}_max"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = (min_val, max_val)
    if min_key not in st.session_state:
        st.session_state[min_key] = min_val
    if max_key not in st.session_state:
        st.session_state[max_key] = max_val
    slider_val = st.sidebar.slider(
        metric,
        min_value=min_val,
        max_value=max_val,
        value=st.session_state[slider_key],
        key=slider_key,
    )
    col1, col2 = st.sidebar.columns(2)
    min_input = col1.number_input(
        f"{metric} min",
        min_value=min_val,
        max_value=max_val,
        value=float(slider_val[0]),
        key=min_key,
        on_change=lambda m=metric: st.session_state.__setitem__(f"{m}_slider", (st.session_state[f"{m}_min"], st.session_state[f"{m}_slider"][1])),
    )
    max_input = col2.number_input(
        f"{metric} max",
        min_value=min_val,
        max_value=max_val,
        value=float(slider_val[1]),
        key=max_key,
        on_change=lambda m=metric: st.session_state.__setitem__(f"{m}_slider", (st.session_state[f"{m}_slider"][0], st.session_state[f"{m}_max"])),
    )
    metric_ranges[metric] = tuple(st.session_state[slider_key])

filt_stats = stats_df[
    stats_df["domain"].isin(sel_domains) & stats_df["condition"].isin(sel_conditions)
].copy()
for metric, (min_v, max_v) in metric_ranges.items():
    filt_stats = filt_stats[(filt_stats[metric] >= min_v) & (filt_stats[metric] <= max_v)]
filt_stats = filt_stats.sort_values(sort_metric, ascending=not sort_desc).head(top_n)

st.subheader("Filtered conversations")
table_df = filt_stats.copy()
table_df["id"] = table_df["conversation_id"].astype(str).str[-3:]
table_df = table_df.set_index("conversation_id")
table_df.insert(0, "select", False)
column_order = ["select", "domain", "condition", "id"] + [
    c for c in table_df.columns if c not in {"select", "domain", "condition", "id"}
]
edited = st.data_editor(
    table_df,
    width="stretch",
    hide_index=True,
    column_order=column_order,
    column_config={
        "select": st.column_config.CheckboxColumn(required=False),
        "id": st.column_config.TextColumn("id"),
    },
    key="filtered_table",
)
selected_ids = edited[edited["select"]].index.tolist()
if selected_ids:
    filt_stats = filt_stats[filt_stats["conversation_id"].isin(selected_ids)]

filt_tech = tech_df[tech_df["conversation_id"].isin(filt_stats["conversation_id"])]

max_turn = int(filt_tech["turn_index"].max()) if len(filt_tech) else 0

st.subheader("Trajectories")
if filt_stats.empty:
    st.warning("No conversations match the current filters.")
    st.stop()

chart_slot = st.empty()
selected_convo = st.selectbox("Conversation", filt_stats["conversation_id"].tolist())
if "turn_index" not in st.session_state:
    st.session_state["turn_index"] = 0
step_idx = int(st.session_state["turn_index"])
if step_idx > max_turn:
    step_idx = max_turn
    st.session_state["turn_index"] = step_idx

def agg_for_turn(df, turn_idx):
    if df.empty:
        return None
    subset = df[df["turn_index"] == turn_idx]
    if subset.empty:
        return None
    return {
        "technicality": float(subset["technicality"].mean()),
        "readability_grade": float(subset["readability_grade"].mean()),
        "jargon_rate": float(subset["jargon_rate"].mean()),
        "code_ratio": float(subset["code_ratio"].mean()),
    }

def delta_badge(delta):
    if delta > 0.0005:
        cls = "pos"
        icon = "arrow_upward"
    elif delta < -0.0005:
        cls = "neg"
        icon = "arrow_downward"
    else:
        cls = "neu"
        icon = "arrow_right_alt"
    return (
        f'<div class="kpi-delta {cls}">'
        f'<span class="material-symbols-rounded kpi-icon">{icon}</span>'
        f'{delta:+.3f}</div>'
    )

agg = agg_for_turn(filt_tech, step_idx)
prev_agg = agg_for_turn(filt_tech, step_idx - 1) if step_idx > 0 else None
if agg:
    deltas = {
        key: (agg[key] - prev_agg[key]) if prev_agg else 0.0
        for key in agg
    }
    kpi_cols = st.columns(4, gap="small")
    kpi_defs = [
        ("Technicality", agg["technicality"], deltas["technicality"]),
        ("Readability", agg["readability_grade"], deltas["readability_grade"]),
        ("Jargon rate", agg["jargon_rate"], deltas["jargon_rate"]),
        ("Code ratio", agg["code_ratio"], deltas["code_ratio"]),
    ]
    for col, (label, value, delta) in zip(kpi_cols, kpi_defs):
        with col:
            st.markdown(
                f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value:.3f}</div>
  {delta_badge(delta)}
</div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

slider_col, button_col = st.columns([4, 1], gap="small")
with slider_col:
    step_idx = st.slider("Turn index", 0, max_turn, step_idx, key="turn_index")
with button_col:
    st.markdown('<div class="go-button-marker go-button-wrap"></div>', unsafe_allow_html=True)
    go_to_message = st.button("Go to message")
fig = build_trajectory_plot(filt_tech, selected_convo, step_idx=step_idx)
chart_slot.plotly_chart(fig, width="stretch")

conversation_box = st.container(key="conversation_wrap")
with conversation_box:
    st.subheader("Conversation content")
    rec_map = {r["conversation_id"]: r for r in records}
    if selected_convo in rec_map:
        convo = rec_map[selected_convo]
        msg_df = compute_message_metrics(convo["messages"], convo["domain"], comp_stats, jargon_map)
        if "scroll_nonce" not in st.session_state:
            st.session_state["scroll_nonce"] = 0
        assistant_turn = -1
        highlight_index = None
        for idx, msg in enumerate(convo["messages"]):
            if msg["role"] == "assistant":
                assistant_turn += 1
                if assistant_turn == step_idx:
                    highlight_index = idx
                    break
        if highlight_index is not None:
            st.markdown(
                f"""
<style>
div[data-testid="stChatMessage"]:has(#msg-{highlight_index}) {{
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  padding: 8px 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  align-self: stretch;
}}
div#metrics-{highlight_index}.metric-stack .metric-mini {{
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.14);
}}
</style>
                """,
                unsafe_allow_html=True,
            )
        if go_to_message:
            st.session_state["scroll_nonce"] += 1
        scroll_target = highlight_index if go_to_message else None
        scroll_nonce = st.session_state["scroll_nonce"]
        for idx, msg in enumerate(convo["messages"]):
            role = msg["role"]
            row = msg_df[msg_df["idx"] == idx].iloc[0]
            col_chat, col_div, col_metrics = st.columns([4, 0.05, 0.5], gap="medium")
            with col_chat:
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(
                            f'<span id="msg-{idx}" class="msg-anchor user-marker"></span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(msg["content"])
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(f'<span id="msg-{idx}" class="msg-anchor"></span>', unsafe_allow_html=True)
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar=":material/terminal:"):
                        st.markdown(f'<span id="msg-{idx}" class="msg-anchor"></span>', unsafe_allow_html=True)
                        st.markdown("**SYSTEM**")
                        st.markdown(msg["content"])
            with col_div:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            with col_metrics:
                if role == "assistant":
                    metrics = {
                        "Technicality": round(row["technicality"], 2),
                        "Delta": round(row["delta"], 2),
                        "Readability": round(row["readability_grade"], 2),
                        "Jargon rate": round(row["jargon_rate"], 2),
                        "Code ratio": round(row["code_ratio"], 2),
                    }
                    cards = []
                    for label, value in metrics.items():
                        cards.append(
                            f'<div class="metric-mini"><div class="metric-mini-label">{label}</div>'
                            f'<div class="metric-mini-value">{value:.2f}</div></div>'
                        )
                    st.markdown(
                        f'<div id="metrics-{idx}" class="metric-stack">{"".join(cards)}</div>',
                        unsafe_allow_html=True,
                    )
        if scroll_target is not None:
            components.html(
                f"""
<script>
(() => {{
  const doc = window.parent.document;
  let attempts = 0;
  const targetId = "msg-{scroll_target}";
  const nonce = {scroll_nonce};
  const tryScroll = () => {{
    const target = doc.getElementById(targetId);
    if (!target) {{
      attempts += 1;
      if (attempts < 6) {{
        setTimeout(tryScroll, 200);
      }}
      return;
    }}
    target.scrollIntoView({{ behavior: "smooth", block: "center" }});
    const container = target.closest('[data-testid="stChatMessage"]');
    if (container) {{
      setTimeout(() => {{
        container.classList.remove("highlight-pulse");
        void container.offsetWidth;
        container.classList.add("highlight-pulse");
        setTimeout(() => container.classList.remove("highlight-pulse"), 1200);
      }}, 650);
    }}
  }};
  setTimeout(tryScroll, 50);
}})();
</script>
                """,
                height=0,
            )
    else:
        st.info("Conversation not found in run file.")
