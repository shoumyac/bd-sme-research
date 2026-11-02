#!/usr/bin/env python3
"""Figures for the manuscript, drawn from results_drift/*.csv.  Run from this folder: python make_figures.py
Writes figures/fig1_design.pdf, fig2_main.pdf, fig3_stress.pdf, fig4_sources.pdf, fig5_external.pdf (and .png copies)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import pearsonr, spearmanr

R = "results_drift"; OUT = "figures"; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY, INK, MUTE = "#2a78d6", "#eb6834", "#c9c8c3", "#0b0b0b", "#52514e"
SEQ = {"low": "#a9c9f0", "medium": "#5b9be0", "high": "#1f5fae"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})
s1 = pd.read_csv(f"{R}/summary_e1.csv"); s1 = s1[s1.baseline == "shared"]
fl = pd.read_csv(f"{R}/summary_noise_floor.csv"); fl = fl[fl.baseline == "shared"]
s3 = pd.read_csv(f"{R}/summary_e3.csv"); s2 = pd.read_csv(f"{R}/summary_e2.csv")
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "median": "Median", "isolated": "Isolated"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1: design
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f2ee", ec=MUTE, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.0, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTE, shrinkA=0, shrinkB=0))

    box(1, 18, 15, 12, "Training data\n960 loans", fc="#e4eefb", ec=BLUE, bold=True)
    box(23, 36, 28, 9, "Reference model\ntrained on the pooled data", fc="#e4eefb", ec=BLUE)
    box(23, 18, 28, 12, "Stylised clients\n(3 institution archetypes\nor label-skew splits)")
    box(23, 1, 28, 12, "50 bootstrap retrainings\nof the pooled model\n(routine retraining)")
    box(58, 18, 20, 12, "Federated models\nFedAvg, FedProx,\nSCAFFOLD, median,\nisolated")
    box(58, 1, 20, 12, "Noise floor:\n5th percentile of the\nbootstrap agreement", fc="#fbe9e0", ec=ORANGE)
    box(64, 36, 35, 9, "Declined applicants\n(reference score $\\geq$ 0.30)", fc="#e4eefb", ec=BLUE, bold=True)
    box(84, 12, 15, 18, "Agreement of\ntop-5 reasons:\noverlap,\nsame top\nreason,\ncosine", fc="#fff6e0", ec="#eda100")
    arrow(16, 27, 23, 40); arrow(16, 24, 23, 24); arrow(16, 21, 23, 7)
    arrow(51, 40.5, 64, 40.5); arrow(51, 24, 58, 24); arrow(51, 7, 58, 7)
    arrow(78, 24, 84, 24); arrow(78, 7, 84, 13); arrow(91.5, 36, 91.5, 30)
    save(fig, "fig1_design")


# ------------------------------------------------------------------ Fig 2: main result
def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    methods = ["fedavg", "fedprox", "scaffold", "median", "isolated"]
    for ax, m in zip(axs, ["LR", "MLP"]):
        f = fl[(fl.model == m) & (fl.floor == "boot")].iloc[0]
        ax.axhspan(f.jac_p05, f.jac_mean, color=GREY, alpha=0.75, lw=0, zorder=0)
        ax.axhline(f.jac_p05, color=MUTE, lw=0.8, ls="--", zorder=1)
        if m == "MLP":
            g = fl[(fl.model == m) & (fl.floor == "boot_init")].iloc[0]
            ax.axhline(g.jac_p05, color=MUTE, lw=0.8, ls=":", zorder=1)
        for j, reg in enumerate(["low", "medium", "high"]):
            for i, meth in enumerate(methods):
                r = s1[(s1.model == m) & (s1.regime == reg) & (s1.method == meth)].iloc[0]
                x = i + (j - 1) * 0.26
                ax.bar(x, r.jac, 0.24, color=SEQ[reg], zorder=2, label=reg if i == 0 else None)
                ax.errorbar(x, r.jac, yerr=r.jac_sd, color=INK, lw=0.7, capsize=1.5, zorder=3)
        ax.set_xticks(range(5)); ax.set_xticklabels([NAMES[k] for k in methods], rotation=0, fontsize=7.6)
        ax.set_ylim(0.5, 1.0); ax.set_title({"LR": "Logistic regression", "MLP": "Neural network (32 hidden units)"}[m], fontsize=9, loc="left")
    axs[0].set_ylabel("Top-5 reason overlap with pooled model")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=SEQ[k], label=k) for k in ["low", "medium", "high"]]
    axs[0].legend(handles=h, title="Heterogeneity", fontsize=7, title_fontsize=7, frameon=False, loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.0)
    h2 = [Patch(color=GREY, label="Retraining noise floor: mean to 5th percentile"),
          Line2D([0], [0], color=MUTE, ls=":", lw=1, label="Floor when the random initialisation also changes")]
    axs[1].legend(handles=h2, fontsize=6.8, frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "fig2_main")


# ------------------------------------------------------------------ Fig 3: stress tests
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    spec = [("dirichlet_alpha", "Label-skew concentration $\\alpha$", [0.1, 0.5, 5.0], True),
            ("clients", "Number of clients", [3.0, 5.0, 10.0], False),
            ("dp_sigma", "Noise $\\sigma$ added to the aggregate", [0.0, 0.01, 0.05, 0.1, 0.2], False)]
    for ax, (fac, lab, levels, logx) in zip(axs, spec):
        for m, col, mk in [("LR", BLUE, "o"), ("MLP", ORANGE, "s")]:
            d = s3[(s3.model == m) & (s3.factor == fac)].sort_values("level")
            if d.empty:
                continue
            x = np.arange(len(d))
            ax.errorbar(x, d.jac, yerr=d.jac_sd, color=col, marker=mk, ms=4.5, lw=1.4, capsize=2, label={"LR": "Logistic regression", "MLP": "Neural network"}[m], zorder=3)
            f = d.floor_jac_p05.mean()
            if m == "LR":
                ax.axhline(f, color=MUTE, ls="--", lw=0.8, zorder=1)
            else:
                ax.axhline(f, color=MUTE, ls=":", lw=0.8, zorder=1)
        d0 = s3[(s3.model == "LR") & (s3.factor == fac)].sort_values("level")
        if fac == "clients":
            ax.set_xticks(range(len(d0))); ax.set_xticklabels([f"{int(k)}" for k in d0.K])
        else:
            ax.set_xticks(range(len(d0))); ax.set_xticklabels([f"{l:g}" for l in d0.level])
        ax.set_xlabel(lab)
    axs[0].set_ylabel("Top-5 reason overlap with pooled model")
    axs[0].set_ylim(0.5, 1.0)
    from matplotlib.lines import Line2D
    hs, ls = axs[0].get_legend_handles_labels()
    hs += [Line2D([0], [0], color=MUTE, ls="--", lw=1), Line2D([0], [0], color=MUTE, ls=":", lw=1)]
    ls += ["Noise floor, logistic regression", "Noise floor, neural network"]
    axs[2].legend(hs, ls, fontsize=6.4, frameon=False, loc="lower left")
    fig.tight_layout(); save(fig, "fig3_stress")


# ------------------------------------------------------------------ Fig 4: sources of cross-client disagreement
def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    cols = [("centralised_client_baseline_mean", "Pooled model, own baselines", "#a9c9f0", None),
            ("fedavg_client_baseline_mean", "FedAvg, own baselines", BLUE, None),
            ("centralised_shared_baseline_mean", "Pooled model, shared baseline", "#f6c1ab", None),
            ("fedavg_shared_baseline_mean", "FedAvg, shared baseline", ORANGE, None)]
    groups = [(m, r) for m in ["LR", "MLP"] for r in ["low", "medium", "high"]]
    for k, (c, lab, col, _) in enumerate(cols):
        vals = [s2[(s2.model == m) & (s2.regime == r)][c].iloc[0] for m, r in groups]
        ax.bar(np.arange(len(groups)) + (k - 1.5) * 0.2, vals, 0.19, color=col, label=lab, zorder=2)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{m}\n{r}" for m, r in groups], fontsize=7.4)
    ax.set_ylim(0.75, 1.0); ax.set_ylabel("Cross-client rank agreement (Spearman)")
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2, columnspacing=0.9, handlelength=1.1)
    ax.set_title("(a)", fontsize=9, loc="left", y=1.18)
    ax = axs[1]
    p = pd.read_csv(f"{R}/e2_proposition.csv")
    ax.scatter(p.predicted, p.measured_A, s=16, color=BLUE, zorder=3)
    for _, r in p.nlargest(3, "predicted").iterrows():
        ax.annotate(r.feature.replace("_bdt", "").replace("_", " "), (r.predicted, r.measured_A), xytext=(-4, 5), textcoords="offset points", fontsize=6.6, color=INK, ha="right")
    rp = pearsonr(p.predicted, p.measured_A)[0]; rs = spearmanr(p.predicted, p.measured_A)[0]
    ax.text(0.97, 0.05, f"Pearson {rp:.2f}\nSpearman {rs:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, color=INK)
    ax.set_xlabel("Predicted dispersion $|w_j|\\,\\mathrm{std}_k(\\mu_j^k)$"); ax.set_ylabel("Measured cross-client spread")
    ax.set_title("(b)", fontsize=9, loc="left", y=1.18)
    fig.tight_layout(); save(fig, "fig4_sources")

# ------------------------------------------------------------------ Fig 5: three public credit datasets
def fig5():
    ER = pd.read_csv(f"{R}/ext_fl_vs_reference.csv"); EF = pd.read_csv(f"{R}/ext_noise_floor.csv")
    DN = [("german", "German credit (1,000 loans)"), ("taiwan_default", "Taiwan card default (5,000 sampled)"), ("taiwan_bankruptcy", "Taiwan bankruptcy (6,819 firms)")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True)
    cols = {"fedavg": BLUE, "fedprox": "#5b9be0", "scaffold": "#a9c9f0", "median": ORANGE, "isolated": "#8f8e89"}
    for r, (m, mt) in enumerate([("LR", "Logistic regression"), ("MLP", "Neural network")]):
        for c, (dn, title) in enumerate(DN):
            ax = axs[r, c]
            g = ER[(ER.dataset == dn) & (ER.model == m) & (ER.regime == "high")]
            f = EF[(EF.dataset == dn) & (EF.model == m) & (EF.regime == "high")]
            lo, hi = f.jac_p05.mean(), f.jac_mean.mean()
            ax.axhspan(lo, hi, color=GREY, alpha=0.8, zorder=1, lw=0)
            ax.axhline(lo, color=MUTE, lw=0.8, ls="--", zorder=2)
            for k, a in enumerate(cols):
                v = g[g.method == a].jac.mean(); sd = g[g.method == a].jac.std()
                ax.bar(k, v, 0.7, color=cols[a], yerr=sd if sd == sd else None, capsize=2, error_kw=dict(lw=0.7), zorder=3)
            ax.set_xticks(range(5)); ax.set_xticklabels([NAMES[a] if a != "scaffold" else "SCAFF." for a in cols], rotation=45, ha="right", fontsize=6.8)
            ax.set_ylim(0.2, 1.0)
            if r == 0: ax.set_title(title, fontsize=7.4)
            if c == 0: ax.set_ylabel(f"{mt}\nTop-5 overlap with pooled model", fontsize=7.4)
    fig.tight_layout(h_pad=1.0); save(fig, "fig5_external")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); print("figures written to", OUT)
