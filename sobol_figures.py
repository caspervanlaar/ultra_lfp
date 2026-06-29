"""
Sobol Sensitivity Analysis - Figures
Adapted from Pmnist_Ultra_lfp_oscillation.ipynb (cells 29, 32-42)
Data: SOBOL_FINAL_RESULTS.json + SOBOL_FULL_HISTORY.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D
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
OUT = "/home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/CIFAR_P_RESULTS/H64_CIFAR_SOBOL/BACK_UP"

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
FINRES= OUT+ "/SOBOL_FINAL_RESULTS.json"
FULHIS= OUT+ "/SOBOL_FULL_HISTORY.json"
with open(FINRES) as f:
    si_raw = json.load(f)

with open(FULHIS) as f:
    runs_list = json.load(f)

# Problem definition (matches Adding_sobol_worker.py)
PARAM_NAMES   = ["LAMBDA_SLOW", "H_INERTIA", "BASE_STRENGTH", "PERIOD", "JITTER_SCALE"]
PARAM_LABELS  = ["λ_slow", "H_inertia", "Strength", "Period", "Jitter"]

# Metrics available in SOBOL_FINAL_RESULTS (drop sync - all NaN there)
SI_METRICS    = ["acc", "rank", "intf", "entr"]
SI_LABELS     = {"acc": "Accuracy", "rank": "Eff. Rank",
                 "intf": "Interference", "entr": "Entropy"}

# All metrics available in the full history
ALL_METRICS   = ["acc", "rank", "sync", "entr", "acorr", "intf"]
ALL_LABELS    = {"acc": "Accuracy", "rank": "Eff. Rank", "sync": "Synchrony",
                 "entr": "Entropy", "acorr": "Autocorr", "intf": "Interference"}

# Palette
PALETTE = {"acc": "#2ecc71", "rank": "#3498db", "intf": "#e67e22",
           "entr": "#9b59b6", "acorr": "#e74c3c", "sync": "#f39c12"}

# ── 2. BUILD DATAFRAME FROM FULL HISTORY ──────────────────────────────────────
rows = []
for r in runs_list:
    p    = r["parameters"]
    last = r["epochs"][-1]
    rows.append({
        "LAMBDA_SLOW":   p["lambda"],
        "H_INERTIA":     p["inertia"],
        "BASE_STRENGTH": p["strength"],
        "PERIOD":        p["period"],
        "JITTER_SCALE":  p["jitter"],
        "acc":   last["acc"],
        "rank":  last["rank"],
        "sync":  last.get("sync", np.nan),
        "entr":  last["entr"],
        "acorr": last["acorr"],
        "intf":  last["intf"],
    })
df = pd.DataFrame(rows).dropna(subset=["rank"])

print(f"Loaded {len(df)} runs | Metrics: {list(df.columns)}")

# ── FIGURE 1 ─ S1 / ST SIDE-BY-SIDE HEATMAPS ─────────────────────────────────
def fig_s1_st():
    s1_vals = np.array([[si_raw[m]["S1"][i]  for i in range(5)] for m in SI_METRICS])
    st_vals = np.array([[si_raw[m]["ST"][i]  for i in range(5)] for m in SI_METRICS])
    s1_conf = np.array([[si_raw[m]["S1_conf"][i] for i in range(5)] for m in SI_METRICS])
    st_conf = np.array([[si_raw[m]["ST_conf"][i] for i in range(5)] for m in SI_METRICS])

    s1_df = pd.DataFrame(s1_vals, index=[SI_LABELS[m] for m in SI_METRICS], columns=PARAM_LABELS)
    st_df = pd.DataFrame(st_vals, index=[SI_LABELS[m] for m in SI_METRICS], columns=PARAM_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)
    kw = dict(annot=True, fmt=".2f", linewidths=0.4,
              cbar_kws={"label": "Index Value", "shrink": 0.8})

    sns.heatmap(s1_df, cmap="RdBu_r", center=0, ax=axes[0], **kw)
    axes[0].set_title("First-Order Sensitivity  $S_1$\n(Direct contribution)", fontsize=13)

    sns.heatmap(st_df, cmap="RdBu_r", center=0, ax=axes[1], **kw)
    axes[1].set_title("Total-Order Sensitivity  $S_T$\n(Including interactions)", fontsize=13)

    plt.suptitle("Sobol Sensitivity Indices — sCIFAR-10 Task", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_sobol_S1_ST.png", bbox_inches="tight")
    plt.close()
    print("Saved fig1_sobol_S1_ST.png")

fig_s1_st()

# ── FIGURE 2 ─ S2 INTERACTION MATRICES (2x2 grid per metric) ─────────────────
def fig_s2():
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax, m in zip(axes, SI_METRICS):
        raw = si_raw[m]["S2"]
        s2  = pd.DataFrame([[np.nan if v is None else v for v in row]
                             for row in raw],
                            index=PARAM_LABELS, columns=PARAM_LABELS).astype(float)
        # Show only upper triangle
        mask = np.tril(np.ones_like(s2, dtype=bool))
        sns.heatmap(s2, mask=mask, annot=True, fmt=".3f",
                    cmap="RdBu_r", center=0,
                    cbar_kws={"label": "S2", "shrink": 0.7},
                    ax=ax, linewidths=0.4)
        ax.set_title(f"$S_2$ Interaction — {SI_LABELS[m]}\n"
                     f"Positive = synergy | Negative = competition",
                     fontsize=11)

    plt.suptitle("Second-Order Sobol Indices ($S_2$) — sCIFAR-10 Task", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_sobol_S2_matrices.png", bbox_inches="tight")
    plt.close()
    print("Saved fig2_sobol_S2_matrices.png")

fig_s2()

# ── FIGURE 3 ─ INTERACTION NETWORK (acc) ─────────────────────────────────────
def fig_network(metric="acc"):
    raw = si_raw[metric]["S2"]
    s2  = np.array([[np.nan if v is None else v for v in row] for row in raw])

    fig, ax = plt.subplots(figsize=(8, 8))
    G = nx.Graph()
    for i in range(5):
        for j in range(i+1, 5):
            v = s2[i, j]
            if not np.isnan(v) and abs(v) > 0.02:
                col = "#c0392b" if v > 0 else "#2980b9"
                G.add_edge(PARAM_LABELS[i], PARAM_LABELS[j],
                           weight=abs(v), color=col, val=v)

    if len(G.edges()) == 0:
        ax.text(0.5, 0.5, "No significant interactions\n(|S2| > 0.02)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
    else:
        pos    = nx.spring_layout(G, k=2.5, seed=42)
        edges  = G.edges()
        colors = [G[u][v]["color"] for u, v in edges]
        widths = [G[u][v]["weight"] * 50 for u, v in edges]
        labels = {(u,v): f"{G[u][v]['val']:.2f}" for u, v in edges}

        nx.draw_networkx_nodes(G, pos, node_size=3500,
                               node_color="#f8f9fa", edgecolors="#333", linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=colors,
                               width=widths, alpha=0.75, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, labels, font_size=8, ax=ax)

    legend = [Line2D([0],[0], color="#c0392b", lw=3, label="Synergy (+)"),
               Line2D([0],[0], color="#2980b9", lw=3, label="Competition (−)")]
    ax.legend(handles=legend, loc="upper right", fontsize=11)
    ax.axis("off")
    ax.set_title(f"Parameter Interaction Network — {SI_LABELS.get(metric, metric)}\n"
                 f"Edge width ∝ |S₂|, threshold |S₂| > 0.02", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_sobol_network_{metric}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig3_sobol_network_{metric}.png")

for m in SI_METRICS:
    fig_network(m)

# ── FIGURE 4 ─ RADAR / SPIDER CHART ─────────────────────────────────────────
def fig_radar():
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for m in SI_METRICS:
        vals  = [si_raw[m]["ST"][i] for i in range(5)]
        # clip to [0,1] for radar readability
        vals  = [max(0, v) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.5, label=SI_LABELS[m],
                color=PALETTE.get(m, None))
        ax.fill(angles, vals, alpha=0.08, color=PALETTE.get(m, None))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PARAM_LABELS, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Total Sensitivity $S_T$ across Metrics", pad=25, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_sobol_radar.png", bbox_inches="tight")
    plt.close()
    print("Saved fig4_sobol_radar.png")

fig_radar()

# ── FIGURE 5 ─ EPOCH EVOLUTION TRAJECTORIES ───────────────────────────────────
def fig_epoch_evolution():
    plot_metrics = [("acc","Accuracy"), ("rank","Eff. Rank"),
                    ("entr","Entropy"), ("acorr","Autocorr"), ("intf","Interference")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Epoch-by-Epoch Evolution Across All Sobol Runs", fontsize=16)
    axes = axes.flatten()

    for ax, (key, label) in zip(axes[:5], plot_metrics):
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        col = PALETTE.get(key, "#555")
        for r in runs_list:
            try:
                y = [ep[key] for ep in r["epochs"]]
                ax.plot(range(1, len(y)+1), y, color=col, alpha=0.12, linewidth=0.8)
            except KeyError:
                continue
        # Overlay median trajectory (pad ragged sequences with NaN)
        seqs     = [[ep.get(key, np.nan) for ep in r["epochs"]] for r in runs_list]
        max_len  = max(len(s) for s in seqs)
        padded   = np.full((len(seqs), max_len), np.nan)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        med = np.nanmedian(padded, axis=0)
        ax.plot(range(1, len(med)+1), med, color=col, linewidth=2.5,
                linestyle="--", label="Median", zorder=5)
        ax.legend(fontsize=9)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_sobol_epoch_evolution.png", bbox_inches="tight")
    plt.close()
    print("Saved fig5_sobol_epoch_evolution.png")

fig_epoch_evolution()

# ── FIGURE 6 ─ PHASE PORTRAITS ────────────────────────────────────────────────
def fig_phase_portraits():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase Portraits — Metric Interactions Coloured by H_INERTIA",
                 fontsize=14)
    pairs = [("intf", "acc"), ("rank", "acc"), ("entr", "acc")]
    for ax, (x_key, y_key) in zip(axes, pairs):
        sc = ax.scatter(df[x_key], df[y_key],
                        c=df["H_INERTIA"], cmap="viridis", alpha=0.5, s=15)
        plt.colorbar(sc, ax=ax, label="H_INERTIA")
        ax.set_xlabel(ALL_LABELS.get(x_key, x_key))
        ax.set_ylabel(ALL_LABELS.get(y_key, y_key))
        ax.set_title(f"{ALL_LABELS.get(x_key, x_key)} vs {ALL_LABELS.get(y_key, y_key)}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_sobol_phase_portraits.png", bbox_inches="tight")
    plt.close()
    print("Saved fig6_sobol_phase_portraits.png")

fig_phase_portraits()

# ── FIGURE 7 ─ PARALLEL COORDINATES — WINNING PATHWAYS ───────────────────────
def fig_parallel(outcome="acc", label="Accuracy", top_q=0.90, cmap_lo="#7f0000", cmap_hi="#2ecc71"):
    cols   = ["LAMBDA_SLOW", "BASE_STRENGTH", "JITTER_SCALE", "PERIOD", "H_INERTIA", outcome]
    df_p   = df[cols].copy()
    # Normalise
    for c in cols:
        mn, mx = df_p[c].min(), df_p[c].max()
        df_p[c] = (df_p[c] - mn) / (mx - mn + 1e-12)

    q_val    = df[outcome].quantile(top_q)
    top_idx  = df[df[outcome] >= q_val].index
    base_idx = df[df[outcome] <  q_val].index

    x = np.arange(len(cols))
    nice_cols = ["λ_slow", "Strength", "Jitter", "Period", "H_inertia", label]

    fig, ax = plt.subplots(figsize=(15, 7))
    for idx in base_idx:
        ax.plot(x, df_p.loc[idx], color=cmap_lo, alpha=0.10, linewidth=0.6)
    for idx in top_idx:
        ax.plot(x, df_p.loc[idx], color=cmap_hi, alpha=0.65, linewidth=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(nice_cols, rotation=15, fontsize=12)
    ax.set_ylabel("Normalised value [0–1]", fontsize=12)
    ax.set_title(f"Parallel Coordinates — Top {int((1-top_q)*100)}% {label} (green) vs Baseline (red)",
                 fontsize=14, pad=15)
    legend_els = [Line2D([0],[0], color=cmap_hi, lw=2.5, label=f"Top {int((1-top_q)*100)}%"),
                  Line2D([0],[0], color=cmap_lo, lw=1.0, alpha=0.3, label="Baseline")]
    ax.legend(handles=legend_els, loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_parallel_{outcome}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig7_parallel_{outcome}.png")

fig_parallel("acc",  "Accuracy",   top_q=0.90)
fig_parallel("rank", "Eff. Rank",  top_q=0.90, cmap_hi="#3498db")

# ── FIGURE 8 ─ BIFURCATION SWEEP — H_INERTIA ─────────────────────────────────
def fig_bifurcation():
    df_s = df.sort_values("H_INERTIA")
    W    = min(20, len(df_s) // 10)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    c_acc  = "#2ecc71"
    c_rank = "#3498db"

    ax1.set_xlabel("H_INERTIA  (persistence parameter)", fontsize=12)
    ax1.set_ylabel("Accuracy", color=c_acc, fontsize=12)
    ax1.plot(df_s["H_INERTIA"], df_s["acc"].rolling(W, min_periods=1).mean(),
             color=c_acc, linewidth=3, label="Accuracy (rolling mean)")
    ax1.scatter(df_s["H_INERTIA"], df_s["acc"], color=c_acc, alpha=0.08, s=8)
    ax1.tick_params(axis="y", labelcolor=c_acc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank", color=c_rank, fontsize=12)
    ax2.plot(df_s["H_INERTIA"], df_s["rank"].rolling(W, min_periods=1).mean(),
             color=c_rank, linewidth=3, linestyle="--", label="Eff. Rank")
    ax2.tick_params(axis="y", labelcolor=c_rank)

    optimal_h = df.loc[df["acc"].idxmax(), "H_INERTIA"]
    ax1.axvline(x=optimal_h, color="red", linestyle=":", alpha=0.6,
                label=f"Optimal H = {optimal_h:.3f}")

    lines  = [Line2D([0],[0], color=c_acc,  lw=3, label="Accuracy"),
               Line2D([0],[0], color=c_rank, lw=3, linestyle="--", label="Eff. Rank"),
               Line2D([0],[0], color="red",  lw=1, linestyle=":", label=f"Optimal H={optimal_h:.3f}")]
    ax1.legend(handles=lines, loc="upper left", fontsize=11)
    ax1.set_title("Bifurcation Analysis: Dominant Effect of H_INERTIA", fontsize=14, pad=15)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig8_bifurcation.png", bbox_inches="tight")
    plt.close()
    print("Saved fig8_bifurcation.png")

fig_bifurcation()

# ── FIGURE 9 ─ INTERFERENCE / SYNCHRONY CRASH ────────────────────────────────
def fig_criticality():
    df_s = df.sort_values("H_INERTIA")
    W    = min(20, len(df_s) // 10)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    c_acc  = "#2ecc71"
    c_intf = "#e67e22"

    ax1.set_xlabel("H_INERTIA", fontsize=12)
    ax1.set_ylabel("Accuracy", color=c_acc, fontsize=12)
    ax1.plot(df_s["H_INERTIA"], df_s["acc"].rolling(W, min_periods=1).mean(),
             color=c_acc, linewidth=3)
    ax1.scatter(df_s["H_INERTIA"], df_s["acc"], color=c_acc, alpha=0.06, s=8)
    ax1.tick_params(axis="y", labelcolor=c_acc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Interference", color=c_intf, fontsize=12)
    ax2.plot(df_s["H_INERTIA"], df_s["intf"].rolling(W, min_periods=1).mean(),
             color=c_intf, linewidth=3, linestyle="-.")
    ax2.tick_params(axis="y", labelcolor=c_intf)

    ax1.axvspan(df["H_INERTIA"].quantile(0.20),
                df["H_INERTIA"].quantile(0.40),
                color="gold", alpha=0.08, label="Functional regime")
    ax1.set_title("Criticality Threshold: Accuracy vs. Interference across H_INERTIA",
                  fontsize=14, pad=15)
    ax1.grid(True, alpha=0.2)
    lines = [Line2D([0],[0], color=c_acc,  lw=3, label="Accuracy"),
              Line2D([0],[0], color=c_intf, lw=3, linestyle="-.", label="Interference"),
              Line2D([0],[0], color="gold", lw=8, alpha=0.3, label="Functional regime")]
    ax1.legend(handles=lines, loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig9_criticality.png", bbox_inches="tight")
    plt.close()
    print("Saved fig9_criticality.png")

fig_criticality()

# ── FIGURE 10 ─ RANK–AUTOCORR TRADE-OFF ──────────────────────────────────────
def fig_rank_autocorr():
    df_s = df.sort_values("H_INERTIA")
    W    = min(20, len(df_s) // 10)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    c_rank  = "#3498db"
    c_acorr = "#e74c3c"

    ax1.set_xlabel("H_INERTIA", fontsize=12)
    ax1.set_ylabel("Effective Rank (Dimensionality)", color=c_rank, fontsize=12)
    ax1.plot(df_s["H_INERTIA"], df_s["rank"].rolling(W, min_periods=1).mean(),
             color=c_rank, linewidth=3)
    ax1.scatter(df_s["H_INERTIA"], df_s["rank"], color=c_rank, alpha=0.07, s=8)
    ax1.tick_params(axis="y", labelcolor=c_rank)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Temporal Autocorrelation", color=c_acorr, fontsize=12)
    ax2.plot(df_s["H_INERTIA"], df_s["acorr"].rolling(W, min_periods=1).mean(),
             color=c_acorr, linewidth=3, linestyle="-.")
    ax2.tick_params(axis="y", labelcolor=c_acorr)

    ax1.set_title("Complexity–Persistence Trade-off: Rank vs. Autocorrelation",
                  fontsize=14, pad=15)
    ax1.grid(True, alpha=0.2)
    lines = [Line2D([0],[0], color=c_rank,  lw=3, label="Eff. Rank"),
              Line2D([0],[0], color=c_acorr, lw=3, linestyle="-.", label="Autocorr")]
    ax1.legend(handles=lines, loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig10_rank_autocorr.png", bbox_inches="tight")
    plt.close()
    print("Saved fig10_rank_autocorr.png")

fig_rank_autocorr()

# ── FIGURE 11 ─ DUAL OBJECTIVE PATHWAY (acc + rank) ───────────────────────────
def fig_dual_pathway():
    cols   = ["LAMBDA_SLOW", "BASE_STRENGTH", "JITTER_SCALE", "PERIOD", "H_INERTIA", "acc", "rank"]
    df_p   = df[cols].copy()
    for c in cols:
        mn, mx = df_p[c].min(), df_p[c].max()
        df_p[c] = (df_p[c] - mn) / (mx - mn + 1e-12)

    q95_acc  = df["acc"].quantile(0.95)
    q95_rank = df["rank"].quantile(0.95)
    acc_top  = df[df["acc"]  >= q95_acc].index
    rank_top = df[df["rank"] >= q95_rank].index
    other    = df.index.difference(acc_top.union(rank_top))

    x      = np.arange(len(cols))
    labels = ["λ_slow", "Strength", "Jitter", "Period", "H_inertia", "Accuracy", "Eff. Rank"]

    fig, ax = plt.subplots(figsize=(16, 8))
    for idx in other:
        ax.plot(x, df_p.loc[idx], color="#555", alpha=0.08, linewidth=0.6)
    for idx in rank_top:
        ax.plot(x, df_p.loc[idx], color="#3498db", alpha=0.60, linewidth=2.0)
    for idx in acc_top:
        ax.plot(x, df_p.loc[idx], color="#2ecc71", alpha=0.65, linewidth=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=12)
    ax.set_ylabel("Normalised range [0–1]", fontsize=12)
    ax.set_title("Dual-Objective Pathway: Top 5% Accuracy (green) vs Top 5% Eff. Rank (blue)",
                 fontsize=14, pad=15)
    legend_els = [Line2D([0],[0], color="#2ecc71", lw=3, label="Top 5% Accuracy"),
                  Line2D([0],[0], color="#3498db", lw=3, label="Top 5% Eff. Rank"),
                  Line2D([0],[0], color="#555",    lw=1, alpha=0.3, label="Baseline")]
    ax.legend(handles=legend_els, fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig11_dual_pathway.png", bbox_inches="tight")
    plt.close()
    print("Saved fig11_dual_pathway.png")

fig_dual_pathway()

# ── FIGURE 12 ─ KDE LANDSCAPE PLOTS ──────────────────────────────────────────
def fig_kde_landscapes():
    df_sync = df.dropna(subset=["sync"])
    configs = [
        ("H_INERTIA", "acc",  "magma",  "Accuracy Landscape",   "#2ecc71", "Acc. cliff"),
        ("H_INERTIA", "rank", "plasma", "Rank Landscape",        "#3498db", "Complexity collapse"),
        ("H_INERTIA", "intf", "YlOrRd", "Interference Landscape","#e67e22", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("KDE State Transition Maps — Global Field Dynamics", fontsize=15)

    for ax, (xk, yk, cmap, title, col, vline_label) in zip(axes, configs):
        try:
            sns.kdeplot(data=df, x=xk, y=yk,
                        cmap=cmap, fill=True, thresh=0, levels=25, ax=ax)
        except Exception:
            ax.hexbin(df[xk], df[yk], gridsize=25, cmap=cmap)
        ax.scatter(df[xk], df[yk], color="white", s=1.5, alpha=0.15)
        if vline_label:
            vx = df.loc[df[yk].idxmax(), xk]
            ax.axvline(x=vx, color=col, linestyle="--", alpha=0.7, label=vline_label)
            ax.legend(fontsize=9)
        ax.set_xlabel("H_INERTIA", fontsize=11)
        ax.set_ylabel(ALL_LABELS.get(yk, yk), fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig12_kde_landscapes.png", bbox_inches="tight")
    plt.close()
    print("Saved fig12_kde_landscapes.png")

fig_kde_landscapes()

# ── FIGURE 13 ─ ST BAR CHART per METRIC ─────────────────────────────────────
def fig_st_bars():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Total-Order Sensitivity  $S_T$ per Metric", fontsize=15)
    axes = axes.flatten()

    for ax, m in zip(axes, SI_METRICS):
        st   = np.array(si_raw[m]["ST"])
        conf = np.array(si_raw[m]["ST_conf"])
        cols = ["#c0392b" if v == max(st) else "#3498db" for v in st]
        bars = ax.bar(PARAM_LABELS, st, color=cols, alpha=0.8,
                      yerr=conf, capsize=5, error_kw={"elinewidth":1.5})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(SI_LABELS[m], fontsize=12)
        ax.set_ylabel("$S_T$")
        ax.set_ylim(min(0, (st - conf).min() - 0.05),
                    max(1.05, (st + conf).max() + 0.05))
        # Label top bar
        top_i = np.argmax(st)
        ax.text(top_i, st[top_i] + conf[top_i] + 0.02,
                f"{st[top_i]:.2f}", ha="center", fontsize=9, color="#c0392b")

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig13_ST_bars.png", bbox_inches="tight")
    plt.close()
    print("Saved fig13_ST_bars.png")

fig_st_bars()
# ── FIGURE 14 ─ BIFURCATION SWEEP — BASE_STRENGTH ────────────────────────────
def fig_bifurcation_strength():
    df_s = df.sort_values("BASE_STRENGTH")
    W    = min(20, len(df_s) // 10)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    c_acc  = PALETTE["acc"]
    c_rank = PALETTE["rank"]

    ax1.set_xlabel("Base Strength", fontsize=12)
    ax1.set_ylabel("acc Loss", color=c_acc, fontsize=12)
    ax1.plot(df_s["BASE_STRENGTH"], df_s["acc"].rolling(W, min_periods=1).mean(),
             color=c_acc, linewidth=3, label="acc (rolling mean)")
    ax1.scatter(df_s["BASE_STRENGTH"], df_s["acc"], color=c_acc, alpha=0.08, s=8)
    ax1.tick_params(axis="y", labelcolor=c_acc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank", color=c_rank, fontsize=12)
    ax2.plot(df_s["BASE_STRENGTH"], df_s["rank"].rolling(W, min_periods=1).mean(),
             color=c_rank, linewidth=3, linestyle="--", label="Eff. Rank")
    ax2.tick_params(axis="y", labelcolor=c_rank)

    # Annotate optimal strength (minimum acc)
    optimal_s = df.loc[df["acc"].idxmin(), "BASE_STRENGTH"]
    ax1.axvline(x=optimal_s, color="black", linestyle=":", alpha=0.6,
                label=f"Min acc Strength = {optimal_s:.3f}")

    lines  = [Line2D([0],[0], color=c_acc,  lw=3, label="acc Loss"),
               Line2D([0],[0], color=c_rank, lw=3, linestyle="--", label="Eff. Rank"),
               Line2D([0],[0], color="black",  lw=1, linestyle=":", label="Min acc Point")]
    ax1.legend(handles=lines, loc="upper right", fontsize=11)
    ax1.set_title("Bifurcation Analysis: Effect of Connection Strength on Performance", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig14_strength_bifurcation.png", bbox_inches="tight")
    plt.close()
    print("Saved fig14_strength_bifurcation.png")

# ── FIGURE 15 ─ INTERFERENCE COLLAPSE — STRENGTH ─────────────────────────────
def fig_strength_criticality():
    df_s = df.sort_values("BASE_STRENGTH")
    W    = min(20, len(df_s) // 10)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    c_acc  = PALETTE["acc"]
    c_intf = PALETTE["intf"]

    ax1.set_xlabel("Base Strength", fontsize=12)
    ax1.set_ylabel("acc Loss", color=c_acc, fontsize=12)
    ax1.plot(df_s["BASE_STRENGTH"], df_s["acc"].rolling(W, min_periods=1).mean(),
             color=c_acc, linewidth=3)
    ax1.tick_params(axis="y", labelcolor=c_acc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Interference", color=c_intf, fontsize=12)
    ax2.plot(df_s["BASE_STRENGTH"], df_s["intf"].rolling(W, min_periods=1).mean(),
             color=c_intf, linewidth=3, linestyle="-.")
    ax2.tick_params(axis="y", labelcolor=c_intf)

    # Highlight the regime where interference explodes
    intf_threshold = df["intf"].quantile(0.75)
    ax2.axhline(y=intf_threshold, color="gray", alpha=0.3, linestyle="--")
    
    ax1.set_title("Criticality: Strength vs. Interference explosion", fontsize=14, pad=15)
    lines = [Line2D([0],[0], color=c_acc,  lw=3, label="acc Loss"),
              Line2D([0],[0], color=c_intf, lw=3, linestyle="-.", label="Interference")]
    ax1.legend(handles=lines, loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig15_strength_criticality.png", bbox_inches="tight")
    plt.close()
    print("Saved fig15_strength_criticality.png")

# ── FIGURE 16 ─ KDE LANDSCAPES — STRENGTH ────────────────────────────────────
def fig_kde_strength():
    configs = [
        ("BASE_STRENGTH", "acc",  "magma",  "Loss Landscape"),
        ("BASE_STRENGTH", "rank", "plasma", "Complexity Landscape"),
        ("BASE_STRENGTH", "intf", "YlOrRd", "Interference Landscape"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("KDE State Transition Maps — Sensitivity to Strength", fontsize=15)

    for ax, (xk, yk, cmap, title) in zip(axes, configs):
        sns.kdeplot(data=df, x=xk, y=yk, cmap=cmap, fill=True, thresh=0, levels=25, ax=ax)
        ax.scatter(df[xk], df[yk], color="white", s=2, alpha=0.1)
        ax.set_xlabel("Base Strength", fontsize=11)
        ax.set_ylabel(ALL_LABELS.get(yk, yk), fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig16_kde_strength.png", bbox_inches="tight")
    plt.close()
    print("Saved fig16_kde_strength.png")

# ── FIGURE 17 ─ COMPLEXITY–PERSISTENCE TRADE-OFF — STRENGTH ─────────────────
def fig_strength_tradeoff():
    df_s = df.sort_values("BASE_STRENGTH")
    W    = min(20, len(df_s) // 10)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    c_rank  = PALETTE["rank"]
    c_acorr = PALETTE["acorr"]

    ax1.set_xlabel("Base Strength", fontsize=12)
    ax1.set_ylabel("Effective Rank", color=c_rank, fontsize=12)
    ax1.plot(df_s["BASE_STRENGTH"], df_s["rank"].rolling(W, min_periods=1).mean(),
             color=c_rank, linewidth=3)
    ax1.tick_params(axis="y", labelcolor=c_rank)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Autocorrelation", color=c_acorr, fontsize=12)
    ax2.plot(df_s["BASE_STRENGTH"], df_s["acorr"].rolling(W, min_periods=1).mean(),
             color=c_acorr, linewidth=3, linestyle="-.")
    ax2.tick_params(axis="y", labelcolor=c_acorr)

    ax1.set_title("Information Flow: Rank vs. Persistence across Strength", fontsize=14, pad=15)
    lines = [Line2D([0],[0], color=c_rank,  lw=3, label="Eff. Rank"),
              Line2D([0],[0], color=c_acorr, lw=3, linestyle="-.", label="Autocorr")]
    ax1.legend(handles=lines, loc="center right", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig17_strength_tradeoff.png", bbox_inches="tight")
    plt.close()
    print("Saved fig17_strength_tradeoff.png")

# ── NEW CALLS ────────────────────────────────────────────────────────────────
fig_bifurcation_strength()
fig_strength_criticality()
fig_kde_strength()
fig_strength_tradeoff()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def fig_bifurcation_inertia_triple():
    # Sort and windowing for smooth curves
    df_s = df.sort_values("H_INERTIA")
    W = min(20, len(df_s) // 10)
    
    fig, ax_acc = plt.subplots(figsize=(14, 7))
    
    # Colors as requested
    c_sync = "#f39c12"  # Orange
    c_acc  = "#2ecc71"  # Green
    c_rank = "#3498db"  # Blue
    
    # 1. Accuracy (Left - Green)
    ax_acc.set_xlabel("H_INERTIA", fontsize=12, fontweight='bold')
    ax_acc.set_ylabel("Accuracy", color=c_acc, fontsize=12, fontweight='bold')
    ax_acc.plot(df_s["H_INERTIA"], df_s["acc"].rolling(W, min_periods=1).mean(),
                color=c_acc, linewidth=4, label="Accuracy")
    ax_acc.scatter(df_s["H_INERTIA"], df_s["acc"], color=c_acc, alpha=0.05, s=10)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)

    # 2. Synchrony (Right 1 - Orange)
    ax_sync = ax_acc.twinx()
    ax_sync.set_ylabel("Synchrony", color=c_sync, fontsize=12, fontweight='bold')
    ax_sync.plot(df_s["H_INERTIA"], df_s["sync"].rolling(W, min_periods=1).mean(),
                 color=c_sync, linewidth=3, label="Synchrony")
    ax_sync.tick_params(axis="y", labelcolor=c_sync)

    # 3. Effective Rank (Right 2 - Blue)
    ax_rank = ax_acc.twinx()
    # Offset the right spine to prevent overlap
    ax_rank.spines["right"].set_position(("axes", 1.12))
    ax_rank.set_ylabel("Effective Rank", color=c_rank, fontsize=12, fontweight='bold')
    ax_rank.plot(df_s["H_INERTIA"], df_s["rank"].rolling(W, min_periods=1).mean(),
                 color=c_rank, linewidth=3, linestyle="--", label="Eff. Rank")
    ax_rank.tick_params(axis="y", labelcolor=c_rank)

    plt.title("Bifurcation Analysis: Metric Sensitivity to H_INERTIA", fontsize=15, pad=20)
    
    # Custom Legend
    custom_lines = [Line2D([0], [0], color=c_acc, lw=4, label='Accuracy (L)'),
                    Line2D([0], [0], color=c_sync, lw=3, label='Synchrony (R1)'),
                    Line2D([0], [0], color=c_rank, lw=3, ls='--', label='Eff. Rank (R2)')]
    ax_acc.legend(handles=custom_lines, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_extra_inertia_bifurcation.png", bbox_inches="tight")
    plt.show()

def fig_bifurcation_strength_triple():
    # Sort and windowing
    df_s = df.sort_values("BASE_STRENGTH")
    W = min(20, len(df_s) // 10)
    
    fig, ax_acc = plt.subplots(figsize=(14, 7))
    
    c_sync = "#f39c12"  # Orange
    c_acc  = "#2ecc71"  # Green
    c_rank = "#3498db"  # Blue
    
    # 1. Accuracy (Left - Green)
    ax_acc.set_xlabel("Base Strength", fontsize=12, fontweight='bold')
    ax_acc.set_ylabel("Accuracy", color=c_acc, fontsize=12, fontweight='bold')
    ax_acc.plot(df_s["BASE_STRENGTH"], df_s["acc"].rolling(W, min_periods=1).mean(),
                color=c_acc, linewidth=4)
    ax_acc.scatter(df_s["BASE_STRENGTH"], df_s["acc"], color=c_acc, alpha=0.05, s=10)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)

    # 2. Synchrony (Right 1 - Orange)
    ax_sync = ax_acc.twinx()
    ax_sync.set_ylabel("Synchrony", color=c_sync, fontsize=12, fontweight='bold')
    ax_sync.plot(df_s["BASE_STRENGTH"], df_s["sync"].rolling(W, min_periods=1).mean(),
                 color=c_sync, linewidth=3)
    ax_sync.tick_params(axis="y", labelcolor=c_sync)

    # 3. Effective Rank (Right 2 - Blue)
    ax_rank = ax_acc.twinx()
    ax_rank.spines["right"].set_position(("axes", 1.12))
    ax_rank.set_ylabel("Effective Rank", color=c_rank, fontsize=12, fontweight='bold')
    ax_rank.plot(df_s["BASE_STRENGTH"], df_s["rank"].rolling(W, min_periods=1).mean(),
                 color=c_rank, linewidth=3, linestyle="--")
    ax_rank.tick_params(axis="y", labelcolor=c_rank)

    plt.title("Bifurcation Analysis: Metric Sensitivity to BASE_STRENGTH", fontsize=15, pad=20)
    
    custom_lines = [Line2D([0], [0], color=c_acc, lw=4, label='Accuracy (L)'),
                    Line2D([0], [0], color=c_sync, lw=3, label='Synchrony (R1)'),
                    Line2D([0], [0], color=c_rank, lw=3, ls='--', label='Eff. Rank (R2)')]
    ax_acc.legend(handles=custom_lines, loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_extra_strength_bifurcation.png", bbox_inches="tight")
    plt.show()

# Execute new plots
fig_bifurcation_inertia_triple()
fig_bifurcation_strength_triple()
print("\n=== All figures saved")