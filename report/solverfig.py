from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'results/validate.json').read_text())


def mean_sd(rows, key):
    values = np.array([r[key] for r in rows])
    return f"${values.mean():.3f} \\pm {values.std(ddof=1):.3f}$"


lines = [r'\begin{tabular}{rrrrr}', r'\toprule',
         r'$|F|/n$ (target) & $m/n$ & $n$ & Sweep (ms) & MILP + search (ms) \\', r'\midrule']
for fraction in [.1, .5]:
    for n in [100, 1000, 10000, 100000]:
        rows = [r for r in data['generated_scaling']
                if r['flag_fraction'] == fraction and r['n'] == n]
        sweep = 1000 * np.median([r['sweep_s'] for r in rows])
        milp = f"{1000*np.median([r['milp_s'] for r in rows]):.2f}" if n <= 1000 else 'Not run'
        lines.append(f"{fraction:.1f} & {fraction/10:.2f} & {n:,} & {sweep:.2f} & {milp}" + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'revision_scaling.tex').write_text('\n'.join(lines) + '\n')

lines = [r'\begin{tabular}{llrr}', r'\toprule',
         r'Dataset & Rule & Coverage & Set size \\', r'\midrule']
for dataset, label in [('mnist', 'MNIST'), ('fmnist', 'F-MNIST')]:
    for method, name in [('sharp', 'Exact (equals naive)'), ('shift', 'Rank shift')]:
        rows = [r[method] for r in data['controlled_flagged_channel'] if r['dataset'] == dataset]
        lines.append(f"{label} & {name} & {mean_sd(rows, 'coverage')} & {mean_sd(rows, 'avg_set_size')}" + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'revision_channel.tex').write_text('\n'.join(lines) + '\n')

configs = [('audit_pipeline', .05, 'Robust pipeline\n951-1047 reviews', '#206493'),
           ('random_audit_fixed_cp', .1, 'Random clean audit\n1000 reviews', '#b75e19')]
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9.5,
                     'ytick.labelsize': 9.5, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), layout='constrained')
for i, (arm, alpha, label, color) in enumerate(configs):
    rows = sorted([r for r in data['audit_30seeds'] if r['audit'] == arm and r['alpha'] == alpha], key=lambda r: r['seed'])
    assert len(rows) == 30
    for ax, key in zip(axes, ['sharp_suspect_coverage', 'sharp_suspect_size']):
        values = np.array([r[key] for r in rows])
        ax.scatter(i + np.linspace(-.085, .085, len(rows)), values, s=18, alpha=.75, color=color,
                   marker='o' if i == 0 else 's')
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt='_',
                    markersize=24, capsize=5, lw=1.5, color='black', zorder=4)
for ax in axes:
    ax.set_xticks([0, 1], [c[2] for c in configs], fontsize=9.5)
    ax.set_xlim(-.5, 1.5)
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].axhline(.9, color='#666666', ls='--', lw=1)
axes[0].set_ylim(.86, 1)
axes[0].set_yticks([.86, .9, .94, .98, 1.0])
axes[1].set_ylim(1.15, 2.65)
axes[1].set_yticks([1.2, 1.6, 2.0, 2.4, 2.6])
axes[0].set_ylabel('Clean-label coverage')
axes[0].set_title('(a) Coverage: target 90%', fontsize=10)
axes[1].set_ylabel('Mean number of labels')
axes[1].set_title('(b) Fewer labels is better', fontsize=10)
fig.savefig(OUT / 'fig_revision_audit.pdf')
fig.savefig(OUT / 'fig_revision_audit.png', dpi=180)
plt.close(fig)
