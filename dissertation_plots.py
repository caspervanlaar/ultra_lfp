"""
dissertation_plots.py
=====================
Dissertation-targeted visualization code for:
    "Global Field Modulation in Neural Sequence Processing"

Organized by hypothesis and dissertation section.
Each function is self-contained and takes the RESULTS dict produced
by the evaluation notebook (or any equivalent dict with the same structure).

RESULTS[cfg_key][condition] = {
    'epochs':         np.array  (1..N)
    'loss':           np.array
    'val_acc':        np.array  (percentage)
    'test_acc':       float     (percentage)
    'effective_rank': np.array
    'synchrony':      np.array
    'interference':   np.array
    'a_corr':         np.array
    'entropy':        np.array
    'config':         dict      (hidden, strength, tau, jitter, period)
}

Usage: run from the evaluation notebook after Section 2 (loading results).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

COLORS     = {'Active': '#1a1a2e', 'Probe': '#457b9d', 'Passive': '#b5838d'}
LINESTY    = {'Active': '-',       'Probe': '--',       'Passive': ':'}
MARKERS    = {'Active': 'o',       'Probe': 's',        'Passive': '^'}
CONDITIONS = ['Passive', 'Probe', 'Active']

SAVE_DIR = '.'   # change to your figures folder

def _save(fig, name):
    fig.savefig(f'{SAVE_DIR}/{name}.pdf', bbox_inches='tight', dpi=200)
    print(f'  Saved: {name}.pdf')

def _legend_handles():
    return [Line2D([0],[0], color=COLORS[c], ls=LINESTY[c], lw=2, label=c)
            for c in CONDITIONS]


# ═══════════════════════════════════════════════════════════════════════════════
# H1  REPRESENTATIONAL RIGIDITY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rank_stability(RESULTS, cfg_key, smooth=3):
    """
    H1 — Effective rank trajectory with ±1σ rolling band.

    Shows:
      - Mean rank trajectory per condition
      - Rolling std band (width = rank instability)
      - Annotated "collapse zone" for Passive
      - Rank variance comparison bar inset

    Directly tests H1 success criterion: σ(ER) lower in Active than Passive.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={'width_ratios': [3, 1]})

    ax, ax_var = axes

    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is None:
            continue
        rank = d['effective_rank']
        ep   = d['epochs']

        # Rolling mean + std
        rm  = uniform_filter1d(rank, size=smooth)
        std = np.array([np.std(rank[max(0,i-smooth):i+smooth+1])
                        for i in range(len(rank))])

        ax.plot(ep, rm, color=COLORS[cond], ls=LINESTY[cond], lw=2, label=cond)
        ax.fill_between(ep, rm - std, rm + std,
                        color=COLORS[cond], alpha=0.12)

    # Annotate collapse zone: first 5 epochs of Passive
    d_p = RESULTS[cfg_key].get('Passive')
    if d_p is not None:
        ax.axvspan(1, 5, color='#b5838d', alpha=0.07, label='Collapse zone')
        ax.annotate('Representational\ncollapse (Passive)',
                    xy=(3, d_p['effective_rank'][:5].min()),
                    xytext=(8, d_p['effective_rank'][:5].min() - 1.5),
                    arrowprops=dict(arrowstyle='->', color='#b5838d', lw=1.0),
                    fontsize=7.5, color='#b5838d')

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Effective Rank', fontsize=9)
    ax.set_title(f'Rank Stability — {cfg_key}', fontsize=10)
    ax.legend(handles=_legend_handles(), fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)

    # Inset: rank variance bar chart (the H1 success criterion directly)
    variances = {}
    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is not None:
            variances[cond] = np.var(d['effective_rank'])

    bars = ax_var.bar(list(variances.keys()), list(variances.values()),
                      color=[COLORS[c] for c in variances],
                      edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, variances.values()):
        ax_var.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax_var.set_ylabel('Var(Effective Rank)  σ²(ER)', fontsize=8)
    ax_var.set_title('H1 Criterion\nRank Variance', fontsize=9)
    ax_var.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, f'H1_rank_stability_{cfg_key}')
    plt.show()


def plot_singular_value_spectra(RESULTS, cfg_key, epoch_snapshots=None):
    """
    H1 — Singular value spectrum at selected epoch snapshots.

    Plots the normalized singular value distribution (how 'mass' is spread
    across dimensions) for each condition at early / mid / late training.
    A flat spectrum = high effective rank. A spike-dominated spectrum = collapse.
    Uses the stored effective_rank series to infer the spectral shape proxy.

    NOTE: For the exact spectrum you need to save h_final per epoch.
    This version uses a rank-entropy proxy reconstructed from ER values.
    """
    if epoch_snapshots is None:
        epoch_snapshots = [1, 10, 25, 50]

    d_ref = next(d for d in RESULTS[cfg_key].values() if d is not None)
    n_ep  = len(d_ref['epochs'])
    snaps = [e for e in epoch_snapshots if e <= n_ep]

    fig, axes = plt.subplots(1, len(snaps), figsize=(4.5 * len(snaps), 4))
    if len(snaps) == 1:
        axes = [axes]

    # Reconstruct approximate singular value distribution from ER and entropy
    # ER = exp(H_entropy_of_spectrum); we back-solve for a flat-top shape
    def approx_spectrum(er, hidden, n_sv=None):
        """Returns a plausible spectrum consistent with given ER."""
        n = n_sv or hidden
        # Geometric series approximation: mass concentrated on first k dims
        # with er as the 'effective' support
        alpha = er / n
        sv = np.array([alpha ** i for i in range(n)])
        sv /= sv.sum()
        return sv

    for ax, ep in zip(axes, snaps):
        ep_idx = ep - 1
        for cond in CONDITIONS:
            d = RESULTS[cfg_key].get(cond)
            if d is None or ep_idx >= len(d['effective_rank']):
                continue
            hidden = d['config']['hidden']
            er = d['effective_rank'][ep_idx]
            sv = approx_spectrum(er, hidden)
            ax.plot(np.arange(1, hidden + 1), sv,
                    color=COLORS[cond], ls=LINESTY[cond], lw=2,
                    marker=MARKERS[cond], ms=5, label=f'{cond} (ER={er:.1f})')

        ax.axhline(1.0 / d_ref['config']['hidden'], color='gray', ls=':', lw=1,
                   label='Uniform (max rank)')
        ax.set_title(f'Epoch {ep}', fontsize=9)
        ax.set_xlabel('Singular value index', fontsize=8)
        ax.set_ylabel('Normalised weight', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)

    fig.suptitle(f'Singular Value Spectrum Proxy — {cfg_key}\n'
                 f'(Flat = high rank; steep = collapse)', fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, f'H1_sv_spectra_{cfg_key}')
    plt.show()


def plot_rank_auc_jitter_comparison(RESULTS):
    """
    H1 — AUC of validation accuracy across configs (Rigidity criterion).

    Bar chart: area under the val-acc curve for each condition × config.
    H1 predicts Active AUC > Passive AUC, especially under high jitter.
    """
    cfg_keys = list(RESULTS.keys())
    x = np.arange(len(cfg_keys))
    width = 0.26
    offsets = {'Passive': -width, 'Probe': 0.0, 'Active': width}

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for cond in CONDITIONS:
        aucs = []
        for cfg_key in cfg_keys:
            d = RESULTS[cfg_key].get(cond)
            if d is not None:
                # Normalise by number of epochs so configs are comparable
                auc = np.trapezoid(d['val_acc'], d['epochs']) / (d['epochs'][-1] - d['epochs'][0] + 1)
                aucs.append(auc)
            else:
                aucs.append(np.nan)

        bars = ax.bar(x + offsets[cond], aucs, width,
                      color=COLORS[cond], label=cond,
                      edgecolor='white', linewidth=0.4)
        for bar, val in zip(bars, aucs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.2,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_keys, fontsize=8, rotation=10)
    ax.set_ylabel('AUC(Val Acc) / epochs  [%]', fontsize=9)
    ax.set_title('H1 Rigidity — Area Under Validation Accuracy Curve\n'
                 '(Higher = more stable learning across epochs)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, 'H1_auc_comparison')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# H2  OPTIMIZATION STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence_speed(RESULTS, cfg_key, plateau_window=5, plateau_thresh=0.5):
    """
    H2 — Loss curves with plateau detection and convergence epoch annotation.

    H2 success criterion: Active reaches plateau significantly earlier than Passive.
    Plateau defined as: std(loss over last `plateau_window` epochs) < plateau_thresh.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_loss, ax_conv = axes

    convergence_epochs = {}
    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is None:
            continue
        loss = d['loss']
        ep   = d['epochs']

        ax_loss.plot(ep, loss, color=COLORS[cond], ls=LINESTY[cond],
                     lw=2, label=cond)

        # Detect convergence epoch
        conv_ep = None
        for i in range(plateau_window, len(loss)):
            window_std = np.std(loss[i-plateau_window:i])
            if window_std < plateau_thresh:
                conv_ep = ep[i - plateau_window]
                break
        convergence_epochs[cond] = conv_ep if conv_ep else ep[-1]

        if conv_ep:
            ax_loss.axvline(conv_ep, color=COLORS[cond], lw=0.8, ls='-.', alpha=0.6)
            ax_loss.text(conv_ep + 0.5, loss[ep.tolist().index(conv_ep)] + 0.02,
                         f'{cond}\nep{conv_ep}', fontsize=6.5, color=COLORS[cond])

    ax_loss.set_xlabel('Epoch', fontsize=9)
    ax_loss.set_ylabel('Training Loss', fontsize=9)
    ax_loss.set_title('Loss Convergence with Plateau Detection', fontsize=10)
    ax_loss.legend(fontsize=8, frameon=False)
    ax_loss.tick_params(labelsize=8)

    # Bar chart: convergence epoch
    bars = ax_conv.bar(list(convergence_epochs.keys()),
                       list(convergence_epochs.values()),
                       color=[COLORS[c] for c in convergence_epochs],
                       edgecolor='white')
    for bar, val in zip(bars, convergence_epochs.values()):
        ax_conv.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'ep {int(val)}', ha='center', va='bottom', fontsize=8.5)

    ax_conv.set_ylabel('Convergence Epoch', fontsize=9)
    ax_conv.set_title('H2 — Convergence Speed\n'
                      f'(plateau threshold: σ < {plateau_thresh} over {plateau_window} epochs)',
                      fontsize=9)
    ax_conv.tick_params(labelsize=8)

    fig.suptitle(f'Optimization Stability — {cfg_key}', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, f'H2_convergence_{cfg_key}')
    plt.show()


def plot_loss_smoothness(RESULTS, cfg_key, window=5):
    """
    H2 — Rolling variance of training loss as proxy for gradient noise.

    H2 success criterion: lower variance in Active during early-to-mid training.
    Biologically: the "temporal stickiness" smooths the loss landscape.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax_loss, ax_var = axes

    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is None:
            continue
        loss = d['loss']
        ep   = d['epochs']

        rolling_var = np.array([np.var(loss[max(0,i-window):i+1])
                                for i in range(len(loss))])
        smoothed    = uniform_filter1d(loss, size=window)

        ax_loss.plot(ep, loss, color=COLORS[cond], ls=LINESTY[cond],
                     lw=1.2, alpha=0.4)
        ax_loss.plot(ep, smoothed, color=COLORS[cond], lw=2, label=cond)

        ax_var.plot(ep, rolling_var, color=COLORS[cond], ls=LINESTY[cond],
                    lw=1.8, label=cond)
        ax_var.fill_between(ep, 0, rolling_var,
                            color=COLORS[cond], alpha=0.08)

    # Shade early-to-mid training region (H2 criterion zone)
    mid = int(d['epochs'][-1] * 0.5)
    ax_var.axvspan(1, mid, color='gold', alpha=0.08, label='H2 evaluation zone')
    ax_var.axvspan(1, mid, color='gold', alpha=0.08)

    ax_loss.set_ylabel('Training Loss', fontsize=9)
    ax_loss.legend(fontsize=8, frameon=False)
    ax_loss.tick_params(labelsize=8)
    ax_loss.set_title(f'Loss Smoothness Proxy — {cfg_key}', fontsize=10)

    ax_var.set_xlabel('Epoch', fontsize=9)
    ax_var.set_ylabel(f'Rolling Var(Loss)\n(window={window})', fontsize=9)
    ax_var.set_title('H2 Gradient Noise Proxy — Lower = smoother optimisation', fontsize=9)
    ax_var.legend(fontsize=8, frameon=False)
    ax_var.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, f'H2_loss_smoothness_{cfg_key}')
    plt.show()


def plot_accuracy_gap_over_time(RESULTS, cfg_key):
    """
    H2 — Accuracy gap between Active and Passive over training epochs.

    Shows where and when the GF produces a measurable advantage.
    Positive gap = Active ahead; shaded region indicates sustained advantage.
    """
    d_a = RESULTS[cfg_key].get('Active')
    d_p = RESULTS[cfg_key].get('Passive')
    d_pr = RESULTS[cfg_key].get('Probe')
    if d_a is None or d_p is None:
        print(f'Missing data for {cfg_key}')
        return

    ep   = d_a['epochs']
    fig, ax = plt.subplots(figsize=(10, 4))

    gap_ap  = d_a['val_acc'] - d_p['val_acc']
    ax.plot(ep, gap_ap, color=COLORS['Active'], lw=2,
            label='Active − Passive')
    ax.fill_between(ep, 0, gap_ap,
                    where=gap_ap > 0, color=COLORS['Active'], alpha=0.15,
                    label='Active advantage')
    ax.fill_between(ep, 0, gap_ap,
                    where=gap_ap < 0, color=COLORS['Passive'], alpha=0.15,
                    label='Passive advantage')

    if d_pr is not None:
        gap_apr = d_a['val_acc'] - d_pr['val_acc']
        ax.plot(ep, gap_apr, color=COLORS['Probe'], lw=1.5, ls='--',
                label='Active − Probe')

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Val Accuracy Gap (%)', fontsize=9)
    ax.set_title(f'H2 — Accuracy Gap Over Training — {cfg_key}\n'
                 '(Positive = Active condition ahead)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, f'H2_accuracy_gap_{cfg_key}')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# H3  NON-PATHOLOGICAL COORDINATION (Goldilocks / Bifurcation)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_coordination_trajectory(RESULTS, cfg_key):
    """
    H3 — Phase portrait: Synchrony vs Interference trajectory over training.

    Plots the trajectory through (sync, interference) space epoch by epoch.
    Arrows show direction of training.
    H3 success zone: sync elevated, interference low (top-left quadrant).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Define target zone (H3 success criterion from intro)
    ax.axhspan(0, 0.4, xmin=0.35, xmax=1.0, color='#2ecc71', alpha=0.07,
               label='H3 success zone\n(sync high, intf low)')
    ax.axhspan(0.7, 1.0, color='#e74c3c', alpha=0.07,
               label='Pathological entrainment\n(high interference)')

    ax.text(0.45, 0.18, 'COORDINATION\nWITHOUT COLLAPSE\n(H3 target)', fontsize=7.5,
            color='#27ae60', style='italic', ha='center')
    ax.text(0.25, 0.82, 'ENTRAINMENT\n(pathological)', fontsize=7.5,
            color='#c0392b', style='italic', ha='center')

    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is None:
            continue
        sync = d['synchrony']
        intf = d['interference']
        ep   = d['epochs']

        sc = ax.scatter(sync, intf,
                        c=ep, cmap='plasma', vmin=ep[0], vmax=ep[-1],
                        s=40, alpha=0.85,
                        marker=MARKERS[cond], edgecolors=COLORS[cond],
                        linewidths=0.7, zorder=3, label=cond)

        # Trajectory arrows every ~5 epochs
        for i in range(0, len(ep) - 1, max(1, len(ep)//8)):
            dx = sync[i+1] - sync[i]
            dy = intf[i+1]  - intf[i]
            ax.annotate('', xy=(sync[i+1], intf[i+1]),
                        xytext=(sync[i], intf[i]),
                        arrowprops=dict(arrowstyle='->', color=COLORS[cond],
                                        lw=0.7, alpha=0.5))

        # Mark start and end
        ax.scatter(sync[0],  intf[0],  s=80, marker='*', color=COLORS[cond], zorder=5)
        ax.scatter(sync[-1], intf[-1], s=80, marker='D', color=COLORS[cond], zorder=5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Epoch', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel('Synchrony (mean |r| off-diagonal)', fontsize=9)
    ax.set_ylabel('Interference (mean |r| with pop mean)', fontsize=9)
    ax.set_title(f'H3 — Coordination-Without-Collapse Phase Portrait\n{cfg_key}'
                 '\n(★ = epoch 1, ◆ = final epoch)', fontsize=9)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, f'H3_phase_portrait_{cfg_key}')
    plt.show()


def plot_metric_dissociation_radar(RESULTS, cfg_key, epoch_idx=-1):
    """
    H3 — Radar / spider chart of all 5 metrics per condition at final epoch.

    Visually encodes the full dissociation framework.
    H3 pattern: Active = high sync + low interference, elevated rank + autocorr.
    """
    METRIC_LABELS = ['Eff. Rank\n(norm)', 'Synchrony', 'Autocorrelation',
                     'Entropy\n(norm)', '1 − Interference']

    def extract_radar_vals(d, hidden):
        if d is None:
            return None
        rank_norm = d['effective_rank'][epoch_idx] / hidden
        sync      = d['synchrony'][epoch_idx]
        acorr     = d['a_corr'][epoch_idx]
        ent_norm  = d['entropy'][epoch_idx] / np.log2(50)
        anti_intf = 1.0 - d['interference'][epoch_idx]  # flip so higher = better
        return [rank_norm, sync, acorr, ent_norm, anti_intf]

    hidden = RESULTS[cfg_key][next(c for c in CONDITIONS
                                   if RESULTS[cfg_key].get(c))]['config']['hidden']

    n_vars = len(METRIC_LABELS)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw={'projection': 'polar'})

    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        vals = extract_radar_vals(d, hidden)
        if vals is None:
            continue
        vals += vals[:1]  # close the loop
        ax.plot(angles, vals, color=COLORS[cond], ls=LINESTY[cond],
                lw=2.2, label=cond)
        ax.fill(angles, vals, color=COLORS[cond], alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=6, color='gray')
    ax.tick_params(labelsize=8)
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.25, -0.15),
              fontsize=9, frameon=False)
    ax.set_title(f'Metric Dissociation Framework\n{cfg_key} — Epoch {epoch_idx % 1000}',
                 fontsize=10, pad=18)

    fig.tight_layout()
    _save(fig, f'H3_radar_{cfg_key}')
    plt.show()


def plot_goldilocks_heatmap_from_sobol(sobol_results, metric='val_acc',
                                       x_param='strength', y_param='h_inertia'):
    """
    H3 — 2D Goldilocks heatmap from Sobol sweep data.

    `sobol_results` should be a list of dicts, each with keys matching
    x_param, y_param, metric, and optionally 'failed' (bool).

    Example structure:
        sobol_results = [
            {'strength': 0.3, 'h_inertia': 0.85, 'val_acc': 82.1,
             'interference': 0.25, 'effective_rank': 15.2, 'failed': False},
            ...
        ]

    Colors encode metric value; NaN/failed runs shown as grey hatching.
    The "Goldilocks zone" (stable + high-performing) is annotated.
    """
    import matplotlib.tri as tri

    xs    = np.array([r[x_param] for r in sobol_results])
    ys    = np.array([r[y_param] for r in sobol_results])
    zs    = np.array([np.nan if r.get('failed', False) else r[metric]
                      for r in sobol_results])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter with colormap for successful runs
    valid = ~np.isnan(zs)
    sc = ax.scatter(xs[valid], ys[valid], c=zs[valid], cmap='RdYlGn',
                    s=80, edgecolors='white', linewidths=0.4, zorder=3,
                    vmin=np.nanpercentile(zs, 10),
                    vmax=np.nanpercentile(zs, 95))

    # Failed runs as grey X
    failed = np.isnan(zs)
    if failed.any():
        ax.scatter(xs[failed], ys[failed], color='#cccccc', marker='x',
                   s=60, lw=1.5, zorder=2, label=f'Failed runs ({failed.sum()})')

    # Annotate stability boundary lines (from Results.md: S>0.85, H<0.80)
    ax.axvline(0.85, color='#e74c3c', ls='--', lw=1.5,
               label='Instability boundary S=0.85')
    ax.axhline(0.80, color='#e74c3c', ls=':',  lw=1.5,
               label='Instability boundary H=0.80')

    # Shade failure zone
    ax.fill_between([0.85, 1.0], [0, 0], [0.80, 0.80],
                    color='#e74c3c', alpha=0.08, label='High-failure regime')

    # Annotate Goldilocks zone
    ax.annotate('Goldilocks\nZone', xy=(0.45, 0.88), fontsize=10,
                color='#27ae60', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#27ae60', alpha=0.8))

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(x_param.replace('_', ' ').title() + '  (S)', fontsize=9)
    ax.set_ylabel(y_param.replace('_', ' ').title() + '  (H_inertia)', fontsize=9)
    ax.set_title(f'H3 — Goldilocks Zone: {metric} across GF parameter space\n'
                 'Grey × = NaN failure; dashed lines = instability boundaries', fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    _save(fig, f'H3_goldilocks_{metric}')
    plt.show()


def plot_sobol_indices(sobol_indices):
    """
    H3 — Sobol total-order sensitivity index bar chart per metric.

    `sobol_indices` should be a dict:
        {metric: {param: {'ST': float, 'ST_conf': float}}}

    Example:
        sobol_indices = {
            'val_acc': {
                'H_inertia':  {'ST': 0.54, 'ST_conf': 0.06},
                'strength':   {'ST': 0.38, 'ST_conf': 0.05},
                'lambda_slow':{'ST': 0.21, 'ST_conf': 0.04},
                'period':     {'ST': 0.15, 'ST_conf': 0.03},
                'jitter':     {'ST': 0.12, 'ST_conf': 0.02},
            },
            'effective_rank': {...},
            ...
        }
    """
    metrics = list(sobol_indices.keys())
    params  = list(next(iter(sobol_indices.values())).keys())
    n_m, n_p = len(metrics), len(params)

    param_colors = plt.cm.Set2(np.linspace(0, 1, n_p))
    x = np.arange(n_m)
    width = 0.8 / n_p

    fig, ax = plt.subplots(figsize=(max(10, n_m * 2.2), 5))

    for i, (param, color) in enumerate(zip(params, param_colors)):
        STs   = [sobol_indices[m][param]['ST'] for m in metrics]
        confs = [sobol_indices[m][param].get('ST_conf', 0) for m in metrics]
        ax.bar(x + i * width - 0.4 + width/2, STs, width,
               label=param, color=color, edgecolor='white', linewidth=0.4)
        ax.errorbar(x + i * width - 0.4 + width/2, STs, yerr=confs,
                    fmt='none', color='k', capsize=3, lw=1.0)

    ax.axhline(0.1, color='gray', ls=':', lw=1, label='Noise floor (S_T < 0.1)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9)
    ax.set_ylabel('Total-order Sobol Index  S_T', fontsize=9)
    ax.set_title('H3 — Global Sensitivity Analysis: Parameter Importance per Metric\n'
                 '(S_T: fraction of output variance attributable to each parameter '
                 'including interactions)', fontsize=10)
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, 'H3_sobol_indices')
    plt.show()


def plot_bifurcation_from_sobol(sobol_results, x_param='strength'):
    """
    H3 — Bifurcation analysis: metrics vs coupling strength.

    Shows the transition point where increasing S starts hurting.
    Detects the bifurcation where interference diverges from synchrony.
    """
    valid = [r for r in sobol_results if not r.get('failed', False)]
    xs    = np.array([r[x_param] for r in valid])
    sync  = np.array([r['synchrony']      for r in valid])
    intf  = np.array([r['interference']   for r in valid])
    rank  = np.array([r['effective_rank'] for r in valid])
    acc   = np.array([r['val_acc']        for r in valid])

    # Bin and aggregate
    bins  = np.linspace(xs.min(), xs.max(), 15)
    bin_i = np.digitize(xs, bins)

    def bin_stats(vals):
        means, stds = [], []
        for b in range(1, len(bins) + 1):
            sel = vals[bin_i == b]
            means.append(np.mean(sel) if len(sel) else np.nan)
            stds.append(np.std(sel)   if len(sel) else 0.0)
        return np.array(means), np.array(stds)

    bc = (bins[:-1] + bins[1:]) / 2  # bin centers
    sm, ss = bin_stats(sync)[0][:-1],  bin_stats(sync)[1][:-1]
    im, is_ = bin_stats(intf)[0][:-1], bin_stats(intf)[1][:-1]
    rm, rs = bin_stats(rank)[0][:-1],  bin_stats(rank)[1][:-1]
    am, as_ = bin_stats(acc)[0][:-1],  bin_stats(acc)[1][:-1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    for ax, (mean, std, label, color) in zip(
        axes.flatten(),
        [(sm, ss, 'Synchrony', '#457b9d'),
         (im, is_, 'Interference', '#e74c3c'),
         (rm, rs, 'Effective Rank', '#1a1a2e'),
         (am, as_, 'Val Accuracy (%)', '#2ecc71')]
    ):
        ax.plot(bc, mean, color=color, lw=2)
        ax.fill_between(bc, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)

        # Mark bifurcation: where interference gradient switches sign
        if label == 'Interference':
            grad = np.gradient(mean[~np.isnan(mean)])
            accel = np.gradient(grad)
            if len(accel) > 2:
                bifurc_idx = np.nanargmax(accel)
                bx = bc[~np.isnan(mean)][bifurc_idx]
                ax.axvline(bx, color='#e74c3c', ls='--', lw=1.5,
                           label=f'Bifurcation ≈ {bx:.2f}')
                ax.legend(fontsize=8, frameon=False)

        ax.axvline(0.85, color='gray', ls=':', lw=1, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel(f'Coupling Strength  ({x_param})', fontsize=9)

    fig.suptitle('H3 — Bifurcation Analysis: Metric Response to Coupling Strength\n'
                 '(Vertical dashed = instability boundary S=0.85)', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, f'H3_bifurcation_{x_param}')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL DISSERTATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multi_benchmark_heatmap(RESULTS):
    """
    Overview heatmap: all metrics × all config-condition pairs.

    Rows = metrics, Columns = (cfg_key, condition) combos.
    Normalised per row so patterns are visible across different scales.
    Good for the Results overview section.
    """
    METRIC_KEYS = ['test_acc', 'effective_rank', 'synchrony',
                   'interference', 'a_corr', 'entropy']
    METRIC_LABELS = ['Test Acc (%)', 'Eff. Rank', 'Synchrony',
                     'Interference', 'Autocorrelation', 'Entropy']

    col_labels, values = [], []
    for cfg_key in RESULTS:
        for cond in CONDITIONS:
            d = RESULTS[cfg_key].get(cond)
            if d is None:
                continue
            row = []
            for m in METRIC_KEYS:
                if m == 'test_acc':
                    row.append(d['test_acc'])
                else:
                    row.append(d[m][-1])  # final epoch value
            values.append(row)
            col_labels.append(f'{cfg_key}\n{cond}')

    mat = np.array(values).T  # (metrics, configs)

    # Normalise each row to [0, 1]
    mat_norm = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        row_min, row_max = mat[i].min(), mat[i].max()
        if row_max > row_min:
            mat_norm[i] = (mat[i] - row_min) / (row_max - row_min)
        else:
            mat_norm[i] = 0.5

    fig, (ax, ax_raw) = plt.subplots(2, 1, figsize=(max(14, len(col_labels) * 1.4), 8),
                                     gridspec_kw={'height_ratios': [1, 1.2]})

    # Normalised heatmap
    im = ax.imshow(mat_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(METRIC_LABELS)))
    ax.set_yticklabels(METRIC_LABELS, fontsize=9)
    ax.set_title('Metric Overview — Row-normalised  (green = higher relative value)',
                 fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01).ax.tick_params(labelsize=7)

    # Raw values as text overlay
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    fontsize=6.5,
                    color='white' if mat_norm[i,j] < 0.25 or mat_norm[i,j] > 0.75 else 'black')

    # Grouped bar for test accuracy (most important result)
    cfg_keys_list = list(RESULTS.keys())
    n_cfg = len(cfg_keys_list)
    x     = np.arange(n_cfg)
    w     = 0.26
    off   = {'Passive': -w, 'Probe': 0.0, 'Active': w}

    for cond in CONDITIONS:
        test_accs = []
        for cfg_key in cfg_keys_list:
            d = RESULTS[cfg_key].get(cond)
            test_accs.append(d['test_acc'] if d else np.nan)
        bars = ax_raw.bar(x + off[cond], test_accs, w,
                          color=COLORS[cond], label=cond,
                          edgecolor='white', linewidth=0.4)
        for bar, val in zip(bars, test_accs):
            if not np.isnan(val):
                ax_raw.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.3,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels(cfg_keys_list, fontsize=9)
    ax_raw.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax_raw.set_title('Test Accuracy by Configuration', fontsize=10)
    ax_raw.legend(fontsize=8, frameon=False)
    ax_raw.tick_params(labelsize=8)
    ax_raw.spines['top'].set_visible(False)
    ax_raw.spines['right'].set_visible(False)

    fig.tight_layout(h_pad=0.5)
    _save(fig, 'overview_heatmap_and_accuracy')
    plt.show()


def plot_temporal_anchor_evidence(RESULTS, cfg_key):
    """
    H2 + Biological motivation — Temporal autocorrelation as "temporal anchor".

    Shows:
      - Autocorrelation trajectory (higher = more persistence = stronger anchor)
      - Overlay with effective rank (to show they are NOT the same metric)
      - Dissociation annotation: rank stable, autocorr elevated in Active

    Connects to Driscoll et al. (2022): population geometry stable even as
    units drift individually.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_acorr, ax_rank = axes

    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is None:
            continue
        ep = d['epochs']

        ax_acorr.plot(ep, d['a_corr'], color=COLORS[cond], ls=LINESTY[cond],
                      lw=2, label=cond)
        ax_rank.plot(ep, d['effective_rank'], color=COLORS[cond], ls=LINESTY[cond],
                     lw=2, label=cond)

    # Pearson r between autocorr and rank for Active (dissociation test)
    d_a = RESULTS[cfg_key].get('Active')
    if d_a is not None:
        r, p = pearsonr(d_a['a_corr'], d_a['effective_rank'])
        ax_acorr.text(0.02, 0.95,
                      f'Active: r(autocorr, rank) = {r:.2f}  p={p:.3f}\n'
                      f'(Low correlation = they are measuring independent things)',
                      transform=ax_acorr.transAxes, fontsize=7.5, va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax_acorr.set_ylabel('Temporal Autocorrelation', fontsize=9)
    ax_acorr.set_title(f'Temporal Anchor Evidence — {cfg_key}\n'
                       'H2: Active = higher temporal persistence (Driscoll et al. 2022)',
                       fontsize=10)
    ax_acorr.legend(fontsize=8, frameon=False)
    ax_acorr.tick_params(labelsize=8)

    ax_rank.set_xlabel('Epoch', fontsize=9)
    ax_rank.set_ylabel('Effective Rank', fontsize=9)
    ax_rank.set_title('Effective Rank (H1) — same training trajectory for comparison',
                      fontsize=9)
    ax_rank.legend(fontsize=8, frameon=False)
    ax_rank.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, f'H2_temporal_anchor_{cfg_key}')
    plt.show()


def plot_entropy_rank_coupling(RESULTS, cfg_key):
    """
    Supplementary — Entropy vs Effective Rank joint evolution.

    Tests whether entropy and rank track each other (redundant metrics)
    or dissociate (independent measurements of different phenomena).
    Relevant for justifying the 5-metric framework in the Methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_trace, ax_scatter = axes

    for cond in CONDITIONS:
        d = RESULTS[cfg_key].get(cond)
        if d is None:
            continue
        ep   = d['epochs']
        rank = d['effective_rank']
        ent  = d['entropy']

        ax_trace.plot(ep, rank / rank.max(), color=COLORS[cond],
                      ls=LINESTY[cond], lw=2, label=f'{cond} rank')
        ax_trace.plot(ep, ent / ent.max(), color=COLORS[cond],
                      ls=LINESTY[cond], lw=1.0, alpha=0.45)

        ax_scatter.scatter(rank, ent, color=COLORS[cond], marker=MARKERS[cond],
                           s=30, alpha=0.7, label=cond)

        r, _ = pearsonr(rank, ent)
        ax_scatter.annotate(f'{cond} r={r:.2f}',
                            xy=(rank.mean(), ent.mean()),
                            fontsize=7, color=COLORS[cond])

    ax_trace.set_xlabel('Epoch', fontsize=9)
    ax_trace.set_ylabel('Normalised value (rank solid, entropy faded)', fontsize=8)
    ax_trace.set_title('Rank vs Entropy Trajectories\n'
                       '(Divergence = measuring different phenomena)', fontsize=10)
    ax_trace.legend(fontsize=7.5, frameon=False)
    ax_trace.tick_params(labelsize=8)

    ax_scatter.set_xlabel('Effective Rank', fontsize=9)
    ax_scatter.set_ylabel('Entropy (bits)', fontsize=9)
    ax_scatter.set_title('Rank–Entropy Scatter Across Epochs\n'
                         '(High r = redundant; low r = complementary metrics)', fontsize=10)
    ax_scatter.legend(fontsize=8, frameon=False)
    ax_scatter.tick_params(labelsize=8)

    fig.suptitle(f'Metric Independence: Entropy vs Rank — {cfg_key}', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, f'supp_entropy_rank_coupling_{cfg_key}')
    plt.show()


def plot_condition_delta_summary(RESULTS):
    """
    Results section — Active minus Passive delta for every metric, every config.

    Signed bar chart: positive = Active better (or expected direction).
    Combines all configs into one figure. Quick evidence summary.
    """
    METRIC_KEYS   = ['test_acc', 'effective_rank', 'synchrony',
                     'interference', 'a_corr', 'entropy']
    METRIC_LABELS = ['Test\nAcc (%)', 'Eff.\nRank', 'Synchrony',
                     'Interference\n(↓ better)', 'Autocorr.', 'Entropy']
    FLIP = {'interference'}  # lower is better; flip sign for display

    cfg_keys = list(RESULTS.keys())
    x = np.arange(len(METRIC_KEYS))

    fig, axes = plt.subplots(2, max(2, len(cfg_keys)//2 + len(cfg_keys) % 2),
                             figsize=(6 * max(2, len(cfg_keys)//2 + 1), 8))
    axes_flat = axes.flatten()

    for ax, cfg_key in zip(axes_flat, cfg_keys):
        d_a = RESULTS[cfg_key].get('Active')
        d_p = RESULTS[cfg_key].get('Passive')
        if d_a is None or d_p is None:
            ax.text(0.5, 0.5, 'no data', ha='center', transform=ax.transAxes)
            continue

        deltas = []
        for m in METRIC_KEYS:
            if m == 'test_acc':
                val_a = d_a['test_acc']
                val_p = d_p['test_acc']
            else:
                val_a = d_a[m][-1]
                val_p = d_p[m][-1]
            delta = (val_a - val_p) * (-1 if m in FLIP else 1)
            deltas.append(delta)

        colors_bar = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]
        bars = ax.bar(x, deltas, color=colors_bar, edgecolor='white', lw=0.4)
        ax.axhline(0, color='k', lw=0.8)

        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.003 if val >= 0 else -0.006),
                    f'{val:+.2f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_LABELS, fontsize=7.5)
        ax.set_ylabel('Active − Passive (±)', fontsize=8)
        ax.set_title(f'{cfg_key}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes_flat[len(cfg_keys):]:
        ax.set_visible(False)

    fig.suptitle('Active − Passive Delta per Metric\n'
                 '(Green = Active better in expected direction; '
                 'Interference flipped so green = lower intf)', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, 'results_delta_summary')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: run all plots against the currently loaded RESULTS dict
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Paste or import your RESULTS dict here ──────────────────────────────
    # This block shows which functions to call and in what order.
    # Replace 'sMNIST-H32' with whichever config keys are loaded.

    print('=== H1: Representational Rigidity ===')
    plot_rank_stability(RESULTS, 'sMNIST-H32')
    plot_singular_value_spectra(RESULTS, 'sMNIST-H32', epoch_snapshots=[1, 10, 25, 50])
    plot_rank_auc_jitter_comparison(RESULTS)

    print('\n=== H2: Optimization Stability ===')
    plot_convergence_speed(RESULTS, 'sMNIST-H32')
    plot_loss_smoothness(RESULTS, 'sMNIST-H32')
    plot_accuracy_gap_over_time(RESULTS, 'sMNIST-H32')
    plot_temporal_anchor_evidence(RESULTS, 'sMNIST-H32')

    print('\n=== H3: Non-Pathological Coordination ===')
    plot_coordination_trajectory(RESULTS, 'sMNIST-H32')
    plot_metric_dissociation_radar(RESULTS, 'sMNIST-H32')
    # plot_goldilocks_heatmap_from_sobol(sobol_results)  # needs Sobol data
    # plot_sobol_indices(sobol_indices)                  # needs Sobol indices
    # plot_bifurcation_from_sobol(sobol_results)         # needs Sobol data

    print('\n=== General / Overview ===')
    plot_multi_benchmark_heatmap(RESULTS)
    plot_entropy_rank_coupling(RESULTS, 'sMNIST-H32')
    plot_condition_delta_summary(RESULTS)

    print('\nDone. All figures saved to', SAVE_DIR)
