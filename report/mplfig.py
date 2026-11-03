import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)


TEXTWIDTH = 6.5

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.15,
    "lines.markersize": 3.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GRAY = "#8C8C8C"
BAND = "#ECECEC"


COLORS = {
    "naive": "#D55E00",
    "known_T": "#E69F00",
    "uniform_shift": "#0072B2",
    "sharp_suspect": "#009E73",
    "sharp_oracle_flags": "#CC79A7",
    "oracle_clean": "#444444",
}
MARKERS = {
    "naive": "o", "known_T": "s", "uniform_shift": "^",
    "sharp_suspect": "D", "sharp_oracle_flags": "v", "oracle_clean": "P",
}
DASHES = {
    "naive": "-", "known_T": (0, (4, 1.6)), "uniform_shift": (0, (1, 1.2)),
    "sharp_suspect": "-", "sharp_oracle_flags": (0, (3, 1.2, 1, 1.2)),
    "oracle_clean": (0, (5, 1.2)),
}
LABELS = {
    "naive": "Naive CP",
    "known_T": "Known-$T$ correction",
    "uniform_shift": "Rank shift",
    "sharp_suspect": "Exact: confidence flags",
    "sharp_oracle_flags": "Exact: true-error flags",
    "oracle_clean": "Clean-label CP",
}
METHODS = ["naive", "known_T", "uniform_shift", "sharp_suspect",
           "sharp_oracle_flags", "oracle_clean"]


def style(m):
    return dict(color=COLORS[m], marker=MARKERS[m], ls=DASHES[m],
                label=LABELS[m], lw=1.5, ms=4.2, capsize=2.2)


def agg(rows, key, group_fields):

    buckets = {}
    for r in rows:
        if key not in r:
            continue
        g = tuple(r[f] for f in group_fields)
        buckets.setdefault(g, []).append(r[key])
    return {g: (float(np.mean(v)), float(np.std(v)), len(v)) for g, v in buckets.items()}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGS, f"{name}.png"))
    plt.close(fig)


def fig1_reliability(res):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.25), sharey=True)
    panels = [
        ("exp1_controlled", lambda r: r["dataset"] == "mnist" and r["eta_bar"] == 0.2
         and r["n_cal"] == 2000, r"(a) MNIST"),
        ("exp1_controlled", lambda r: r["dataset"] == "fmnist" and r["eta_bar"] == 0.2
         and r["n_cal"] == 2000, r"(b) Fashion-MNIST"),
        ("exp2_cifar10n", lambda r: r["noisy"] == "aggre_label" and r["n_cal"] == 2000
         and r["budget_frac"] == 0.2, r"(c) CIFAR-10N (aggre)"),
    ]
    for ax, (exp, pred, title) in zip(axes, panels):
        rows = [r for r in res.get(exp, []) if pred(r)]
        nominal = sorted({1.0 - r["alpha"] for r in rows})
        for m in METHODS:
            ys, es = [], []
            for n in nominal:
                vals = [r[f"{m}_coverage"] for r in rows
                        if 1.0 - r["alpha"] == n and f"{m}_coverage" in r]
                ys.append(np.mean(vals) if vals else np.nan)
                es.append(np.std(vals) if vals else 0.0)
            ax.errorbar(nominal, ys, yerr=es, **style(m))
        ax.plot([0.77, 0.98], [0.77, 0.98], ls="--", lw=0.8, color=GRAY,
                label="Target", zorder=1)
        ax.set_xlim(0.775, 0.975)
        ax.set_ylim(0.60, 1.025)
        ax.set_yticks([.6, .7, .8, .9, 1.0])
        ax.set_xticks(nominal)
        ax.set_xlabel("Requested coverage $1-\\alpha$")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35, lw=0.5)
    axes[0].set_ylabel("Clean-label coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               frameon=False, columnspacing=1.4, handlelength=1.9)
    save(fig, "fig1_reliability")


def fig2_efficiency(res):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.3), sharey=True)
    panels = [
        ("exp1_controlled", lambda r: r["dataset"] == "mnist" and r["alpha"] == 0.1
         and r["n_cal"] == 2000, r"(a) MNIST", "eta_bar", [0.1, 0.2, 0.4],
         [r"$\bar\eta=0.1$", r"$\bar\eta=0.2$", r"$\bar\eta=0.4$"]),
        ("exp1_controlled", lambda r: r["dataset"] == "fmnist" and r["alpha"] == 0.1
         and r["n_cal"] == 2000, r"(b) Fashion-MNIST", "eta_bar", [0.1, 0.2, 0.4],
         [r"$\bar\eta=0.1$", r"$\bar\eta=0.2$", r"$\bar\eta=0.4$"]),
        ("exp2_cifar10n", lambda r: r["alpha"] == 0.1 and r["n_cal"] == 2000
         and r["budget_frac"] == 0.2, r"(c) CIFAR-10N", "noisy",
         ["aggre_label", "random_label1", "worse_label"],
         ["Agg.\n(9%)", "Ann. 1\n(17%)", "Worse\n(40%)"]),
    ]
    for ax, (exp, pred, title, xfield, xs, xlabels) in zip(axes, panels):
        rows = [r for r in res.get(exp, []) if pred(r)]
        xpos = np.arange(len(xs), dtype=float)
        for m in METHODS:
            means, stds = [], []
            for x in xs:
                vals = [r[f"{m}_size"] for r in rows
                        if r[xfield] == x and f"{m}_size" in r]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)
            ax.errorbar(xpos, means, yerr=stds, **style(m))
        ax.axhline(10.0, ls="--", lw=0.8, color=GRAY, zorder=1)
        ax.set_ylim(0, 10.5)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, fontsize=9.5)
        ax.set_xlim(-0.45, len(xs) - 0.55)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35, lw=0.5)
    axes[0].set_ylabel("Mean number of labels")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               frameon=False, columnspacing=1.4, handlelength=1.9)
    save(fig, "fig2_efficiency")


def fig4_noise_sweep(res):


    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.25))
    rows = [r for r in res.get("exp1_controlled", [])
            if r["dataset"] == "mnist" and r["alpha"] == 0.1 and r["n_cal"] == 2000]
    etas = sorted({r["eta_bar"] for r in rows})
    for m in METHODS:
        for ax, suf in zip(axes, ("coverage", "size")):
            ys, es = [], []
            for e in etas:
                vals = [r[f"{m}_{suf}"] for r in rows
                        if r["eta_bar"] == e and f"{m}_{suf}" in r]
                ys.append(np.mean(vals) if vals else np.nan)
                es.append(np.std(vals) if vals else 0.0)
            ax.errorbar(etas, ys, yerr=es, **style(m))
    axes[0].axhline(0.9, ls="--", lw=0.8, color=GRAY, zorder=1)
    axes[0].text(0.405, 0.903, "Target", ha="right", va="bottom",
                 fontsize=9.5, color=GRAY)
    axes[0].set_ylabel("Clean-label coverage")
    axes[0].set_ylim(0.68, 1.03)
    axes[0].set_yticks([.7, .8, .9, 1.0])
    axes[1].axhline(10.0, ls="--", lw=0.8, color=GRAY, zorder=1)
    axes[1].text(0.405, 10.12, "All 10 labels", ha="right", va="bottom",
                 fontsize=9.5, color=GRAY)
    axes[1].set_ylabel("Mean number of labels")
    axes[1].set_ylim(0, 11.6)
    axes[1].set_yticks([0, 2, 4, 6, 8, 10])
    for ax, t in zip(axes, (r"(a) coverage", r"(b) set size")):
        ax.set_xticks(etas)
        ax.set_xlabel(r"corruption rate $\bar\eta$")
        ax.set_xlim(0.065, 0.435)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.35, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2,
               frameon=False, columnspacing=1.4, handlelength=1.9)
    save(fig, "fig4_noise_sweep")


def fig3_limits(res):
    fig = plt.figure(figsize=(TEXTWIDTH, 5.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1])]


    ax = axes[0]
    rows = [r for r in res.get("exp3_adversarial", []) if r.get("alpha") == 0.1 and r.get("eta_bar") == 0.1]
    if rows:
        lo = float(np.mean([r["rank_bound_low_fixed"] for r in rows]))
        hi = float(np.mean([r["rank_bound_high_fixed"] for r in rows]))
        ax.axvspan(lo, hi, color=BAND, zorder=0)
        ax.text((lo + hi) / 2, len(METHODS) - 0.25, "Fixed-budget rank bounds",
                ha="center", va="top", fontsize=9.5, color="#666666")
        ys = np.arange(len(METHODS))[::-1]
        for y, m in zip(ys, METHODS):
            vals = [r[f"{m}_coverage"] for r in rows if f"{m}_coverage" in r]
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            ax.errorbar(mu, y, xerr=sd, fmt=MARKERS[m], ms=4.2, lw=1.1,
                        capsize=2, color=COLORS[m], zorder=3)
        ax.axvline(0.9, ls="--", lw=0.8, color=GRAY, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABELS[m] for m in METHODS], fontsize=9.5)
        ax.set_xlabel("Clean-label coverage")
        ax.set_xlim(0.62, 1.015)
        ax.set_xticks([.65, .7, .75, .8, .85, .9, .95, 1.0])
        ax.set_ylim(-0.7, len(METHODS) + 0.1)
    ax.set_title("(a) Model-agreement errors can reduce coverage")


    ax = axes[1]
    rows = res.get("exp4_flag_recall", [])
    if rows:
        recalls = sorted({r["recall_target"] for r in rows})
        for m in ["uniform_shift", "sharp_suspect", "naive"]:
            ys, es = [], []
            for rc in recalls:
                vals = [r[f"{m}_coverage"] for r in rows
                        if r["recall_target"] == rc and f"{m}_coverage" in r]
                ys.append(np.mean(vals) if vals else np.nan)
                es.append(np.std(vals) if vals else 0.0)
            kwargs = style(m)
            if m == "sharp_suspect":
                kwargs["label"] = "Exact: partial error flags"
            ax.errorbar(recalls, ys, yerr=es, **kwargs)
        ax.axhline(0.8, ls="--", lw=0.8, color=GRAY, zorder=1)
        ax.set_xlabel("Fraction of true errors flagged")
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=9.5)
        ax.set_ylim(0.6, 1.02)
        ax.set_yticks([.6, .7, .8, .9, 1.0])
    ax.set_ylabel("Clean-label coverage")
    ax.set_title("(b) Missed errors")
    ax.grid(axis="y", alpha=0.35, lw=0.5)


    ax = axes[2]
    s = res.get("exp5_sharpness", {})
    if s:
        vals = [s["lower_attained"], s["upper_attained"]]
        ax.bar([0, 1], vals, color=["#009E73", "#CC79A7"], width=0.5,
               edgecolor="white", linewidth=0.3, zorder=2)
        for x, v in zip([0, 1], vals):
            ax.text(x, v + s["trials"] * 0.035, f"{v} / {s['trials']}",
                    ha="center", fontsize=9.5)
        ax.axhline(s["trials"], ls="--", lw=0.8, color=GRAY, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Lower\nbound $L(q)$", "Upper\nbound $U(q)$"], fontsize=9.5)
        ax.set_ylabel("Verified thresholds")
        ax.set_ylim(0, s["trials"] * 1.28)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_xlim(-0.6, 1.6)
    ax.set_title("(c) Exact bounds")
    ax.grid(axis="y", alpha=0.35, lw=0.5)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncols=2,
                   frameon=False, columnspacing=1.4, handlelength=1.9)
    save(fig, "fig3_limits")


if __name__ == "__main__":
    with open(os.path.join(ROOT, "results", "summary.json")) as f:
        res = json.load(f)
    fig1_reliability(res)
    fig2_efficiency(res)
    fig4_noise_sweep(res)
    fig3_limits(res)
    print("figures written:", sorted(os.listdir(FIGS)))
