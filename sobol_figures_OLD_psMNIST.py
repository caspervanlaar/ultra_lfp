import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D
import os
import warnings

warnings.filterwarnings("ignore")

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

# ── PATHS ────────────────────────────────────────────────────────────────────
# Ensure these match your local environment
OUT = "//home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/PMNIST/SOBOL_SWEEPS/H16_SOBOL/"
FINRES = OUT + "SOBOL_FINAL_RESULTS.json" # Or SOBOL_RECOVERED_S2.json
FULHIS = OUT + "SOBOL_FULL_HISTORY.json"

if not os.path.exists(OUT):
    os.makedirs(OUT)

# ── 1. DATA LOADING ───────────────────────────────────────────────────────────
def load_smnist_data(results_path, history_path):
    with open(results_path) as f:
        si_raw = json.load(f)
    with open(history_path) as f:
        history_data = json.load(f)
    
    # Extract the runs dictionary values
    runs_dict = history_data.get("runs", {})
    runs_list = list(runs_dict.values())
    
    return si_raw, runs_list

# Mapping for plotting labels
PARAM_LABELS  = ["λ_slow", "Strength", "Jitter", "Period", "H_inertia"]
SI_METRICS    = ["acc", "rank", "intf", "entr"]
SI_LABELS     = {"acc": "Accuracy", "rank": "Eff. Rank", 
                 "intf": "Interference", "entr": "Entropy",
                 "sync": "Synchrony", "acorr": "Autocorr"}

PALETTE = {"acc": "#2ecc71", "rank": "#3498db", "intf": "#e67e22",
           "entr": "#9b59b6", "acorr": "#e74c3c", "sync": "#f39c12"}

# ── 2. CORRECTED BUILD DATAFRAME ──────────────────────────────────────────────
def build_df(runs_list):
    rows = []
    for r in runs_list:
        cfg = r.get("config", {})
        hist = r.get("history", {})
        
        # Get final values from the history/hidden_metrics lists
        # hidden_metrics is a list of dicts, one per epoch
        hidden = hist.get("hidden_metrics", [])
        last_hidden = hidden[-1] if hidden else {}
        
        acc_list = hist.get("acc", [])
        last_acc = acc_list[-1] if acc_list else np.nan
        
        rows.append({
            # Parameters from config
            "LAMBDA":   cfg.get("tau"), 
            "STRENGTH": cfg.get("strength"),
            "JITTER":   cfg.get("jitter"),
            "PERIOD":   cfg.get("period"),
            "INERTIA":  cfg.get("h_inertia"),
            
            # Metrics from history (Final Epoch)
            "acc":      last_acc,
            "rank":     last_hidden.get("effective_rank"),
            "intf":     last_hidden.get("interference"),
            "entr":     last_hidden.get("entropy"),
            "acorr":    last_hidden.get("a_corr"),
            "sync":     last_hidden.get("synchrony")
        })
        
    df = pd.DataFrame(rows)
    return df.dropna(subset=["acc", "rank"])

# ── PLOTTING FUNCTIONS ────────────────────────────────────────────────────────

def fig_s1_st(si_raw):
    s1_vals = np.array([si_raw[m]["S1"] for m in SI_METRICS])
    st_vals = np.array([si_raw[m]["ST"] for m in SI_METRICS])
    s1_df = pd.DataFrame(s1_vals, index=[SI_LABELS[m] for m in SI_METRICS], columns=PARAM_LABELS)
    st_df = pd.DataFrame(st_vals, index=[SI_LABELS[m] for m in SI_METRICS], columns=PARAM_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    kw = dict(annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5)
    sns.heatmap(s1_df, ax=axes[0], **kw)
    axes[0].set_title("First-Order ($S_1$): Direct Effects")
    sns.heatmap(st_df, ax=axes[1], **kw)
    axes[1].set_title("Total-Order ($S_T$): Total contribution")
    plt.tight_layout()
    plt.savefig(f"{OUT}fig1_s1_st.png")

def fig_s2_matrices(si_raw):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for ax, m in zip(axes, SI_METRICS):
        s2_raw = si_raw[m].get("S2")
        if not s2_raw: continue
        s2_data = np.array([[np.nan if v is None else v for v in row] for row in s2_raw])
        s2_df = pd.DataFrame(s2_data, index=PARAM_LABELS, columns=PARAM_LABELS)
        mask = np.tril(np.ones_like(s2_df, dtype=bool))
        sns.heatmap(s2_df, mask=mask, annot=True, fmt=".3f", cmap="vlag", center=0, ax=ax)
        ax.set_title(f"$S_2$ Interactions: {SI_LABELS[m]}")
    plt.tight_layout()
    plt.savefig(f"{OUT}fig2_s2_matrices.png")

def fig_network(si_raw, metric="acc"):
    raw = si_raw[metric].get("S2")
    if not raw: return
    s2  = np.array([[np.nan if v is None else v for v in row] for row in raw])
    fig, ax = plt.subplots(figsize=(8, 8))
    G = nx.Graph()
    for i in range(len(PARAM_LABELS)):
        for j in range(i+1, len(PARAM_LABELS)):
            v = s2[i, j]
            if not np.isnan(v) and abs(v) > 0.01:
                col = "#c0392b" if v > 0 else "#2980b9"
                G.add_edge(PARAM_LABELS[i], PARAM_LABELS[j], weight=abs(v), color=col, val=v)
    
    if not G.edges(): return
    pos = nx.circular_layout(G)
    colors = [G[u][v]["color"] for u, v in G.edges()]
    widths = [G[u][v]["weight"] * 40 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="#f8f9fa", edgecolors="#333", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, alpha=0.6)
    
    ax.axis("off")
    ax.set_title(f"Interaction Network: {SI_LABELS[metric]} (|S2| > 0.01)")
    plt.savefig(f"{OUT}fig3_network_{metric}.png")

def fig_parallel(df, outcome="acc", label="Accuracy", top_q=0.90):
    cols = ["LAMBDA", "STRENGTH", "JITTER", "PERIOD", "INERTIA", outcome]
    df_p = df[cols].copy()
    for c in cols:
        df_p[c] = (df_p[c] - df_p[c].min()) / (df_p[c].max() - df_p[c].min() + 1e-12)

    q_val = df[outcome].quantile(top_q)
    top_idx = df[df[outcome] >= q_val].index
    base_idx = df[df[outcome] < q_val].index

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cols))
    for idx in base_idx:
        ax.plot(x, df_p.loc[idx], color="#7f0000", alpha=0.05, lw=0.5)
    for idx in top_idx:
        ax.plot(x, df_p.loc[idx], color="#2ecc71", alpha=0.5, lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_title(f"Parallel Coordinates: Top 10% {label} Pathways")
    plt.savefig(f"{OUT}fig7_parallel_{outcome}.png")

def fig_bifurcation(df, param="INERTIA", label="H_Inertia"):
    df_s = df.sort_values(param)
    W = max(5, len(df_s) // 15)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df_s[param], df_s["acc"], color=PALETTE["acc"], alpha=0.1, s=10)
    ax1.plot(df_s[param], df_s["acc"].rolling(W, min_periods=1).mean(), color=PALETTE["acc"], lw=3)
    ax1.set_xlabel(label)
    ax1.set_ylabel("Accuracy", color=PALETTE["acc"])
    ax2 = ax1.twinx()
    ax2.plot(df_s[param], df_s["intf"].rolling(W, min_periods=1).mean(), color=PALETTE["intf"], lw=2, linestyle="--")
    ax2.set_ylabel("Interference", color=PALETTE["intf"])
    plt.title(f"Bifurcation: Effect of {label}")
    plt.savefig(f"{OUT}fig_bifurcation_{param}.png")

# ── EXECUTION ────────────────────────────────────────────────────────────────
try:
    si_raw, runs_list = load_smnist_data(FINRES, FULHIS)
    df = build_df(runs_list)
    
    print(f"Loaded {len(df)} valid runs. Generating figures...")
    
    fig_s1_st(si_raw)
    fig_s2_matrices(si_raw)
    fig_network(si_raw, metric="acc")
    fig_parallel(df, outcome="acc", label="Accuracy")
    fig_bifurcation(df, "INERTIA", "H_Inertia")
    fig_bifurcation(df, "STRENGTH", "Connection Strength")
    
    print(f"\n=== Success: Figures saved to {OUT}")
except Exception as e:
    print(f"\n[ERROR] Failed to process data: {e}")
    import traceback
    traceback.print_exc()