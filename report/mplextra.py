import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}"
                           r"\usepackage{libertine}\usepackage{libertinust1math}",
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.15,
    "lines.markersize": 3.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ARMS = [

    ("audit_pipeline", "sharp_suspect", "Disagreement pipeline (sharp)", "#0072B2", "o"),
    ("random_audit_cp", "naive", "Random clean audit", "#E69F00", "s"),
    ("random_audit_fixed_cp", "naive", "Random audit, fixed $r{=}1000$", "#CC79A7", "v"),
    ("audited_only_cp", "naive", "CP on reviewed labels only", "#009E73", "D"),
]
ALPHAS = [0.05, 0.1, 0.2]


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGS, f"{name}.png"))
    plt.close(fig)


def main():
    rows = json.load(open(os.path.join(ROOT, "results", "summary.json")))[
        "exp8_annotation_audit"
    ]

    def cell(audit, alpha, field):
        sel = [r for r in rows if r["audit"] == audit and abs(r["alpha"] - alpha) < 1e-9]
        return np.array([r[f"{field}_{k}"] for r in sel for k in ["coverage"]]), \
               np.array([r[f"{field}_{k}"] for r in sel for k in ["size"]])

    def stats(audit, alpha, field, kind):
        vals = [r[f"{field}_{kind}"] for r in rows
                if r["audit"] == audit and abs(r["alpha"] - alpha) < 1e-9]
        return np.mean(vals), np.std(vals)

    for audit, field, label, _, _ in ARMS:
        alpha = 0.1 if audit.startswith("random_audit") else 0.05
        cov, _ = stats(audit, alpha, field, "coverage")
        size, _ = stats(audit, alpha, field, "size")
        if audit == "audit_pipeline":
            print(f"pipeline a=0.05: cov {cov:.3f} size {size:.2f}")
        if audit == "random_audit_cp":
            print(f"random   a=0.1 : cov {cov:.3f} size {size:.2f}")
        if audit == "random_audit_fixed_cp":
            print(f"fixed1k  a=0.1 : cov {cov:.3f} size {size:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    ax = axes[0]
    for audit, field, label, color, marker in ARMS:
        means = [stats(audit, a, field, "coverage")[0] for a in ALPHAS]
        sds = [stats(audit, a, field, "coverage")[1] for a in ALPHAS]
        ax.errorbar(ALPHAS, means, yerr=sds, color=color, marker=marker, capsize=2,
                    label=label)
    tgt = [1 - a for a in ALPHAS]
    ax.plot(ALPHAS, tgt, "k--", lw=0.9, label=r"target $1-\alpha$")
    ax.set_xticks(ALPHAS)
    ax.set_xlabel(r"nominal level $\alpha$")
    ax.set_ylabel("achieved clean coverage")
    ax.set_title(r"(a) Coverage vs.\ nominal level (E8, 3 seeds)")

    ax = axes[1]
    for audit, field, label, color, marker in ARMS:
        covs, sizes, cs, ss = [], [], [], []
        for a in ALPHAS:
            c, sc = stats(audit, a, field, "coverage")
            z, sz = stats(audit, a, field, "size")
            covs.append(c)
            sizes.append(z)
            cs.append(sc)
            ss.append(sz)
        ax.errorbar(covs, sizes, xerr=cs, yerr=ss, color=color, marker=marker,
                    capsize=2, label=label)
    ax.axvline(0.90, color="black", lw=0.8, linestyle="-.")
    ax.text(0.902, 1.9, r"target $0.90$", fontsize=6.5)
    ax.annotate("pipeline\n" + r"$\alpha{=}0.05$", xy=(0.965, 2.04), xytext=(0.938, 2.35),
                fontsize=6.2, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.annotate("random audit\n(matched cost)\n" + r"$\alpha{=}0.1$", xy=(0.897, 1.31), xytext=(0.825, 1.62),
                fontsize=6.2, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.annotate("fixed " + r"$r{=}1000$" + "\n" + r"$\alpha{=}0.1$", xy=(0.891, 1.28), xytext=(0.775, 1.18),
                fontsize=6.2, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("achieved clean coverage")
    ax.set_ylabel("average set size $\\downarrow$")
    ax.set_title(r"(b) Size--coverage operating points")


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4,
               frameon=False, columnspacing=1.4, handlelength=1.8)

    save(fig, "fig5_matched_audit")
    print("wrote fig5_matched_audit")


if __name__ == "__main__":
    main()
