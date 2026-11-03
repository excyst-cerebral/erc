import csv
import json
from pathlib import Path

import numpy as np
import plots as g

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'figures'
DATA = json.loads((ROOT/'results/allocation.json').read_text())
CONFIG = [('cifar10','residual','(a) CIFAR-10, residual'),
          ('cifar10','aps','(b) CIFAR-10, APS'),
          ('mammal_other','residual','(c) Mammal vs. other, residual'),
          ('mammal_other','aps','(d) Mammal vs. other, APS')]
METHODS = [('budget_gain','exact','sharp_suspect','Budget gain'),
           ('interval','exact','known_T','Label gain'),
           ('random','exact','uniform_shift','Random, fixed bound'),
           ('budget_gain','refreshed_uniform_audit','sharp_oracle_flags','Random, fresh bound'),
           ('budget_gain','partial_matched_target','naive','Partial-label'),
           ('budget_gain','random_clean_cp','oracle_clean','Random clean CP')]


def select(task, score, cost, policy):
    return [r for r in DATA['rows'] if r['task']==task and r['score']==score and r['total_reviews']==cost and r['policy']==policy]


def stats(values):
    values = np.asarray(values)
    return f'${values.mean():.3f}\\pm{values.std(ddof=1):.3f}$'


def main():
    OUT.mkdir(exist_ok=True)
    body = ''
    manifest = []
    flat = []
    for panel, (task, score, title) in enumerate(CONFIG):
        curves = ''
        for policy, method, style, name in METHODS:
            points = []
            for cost in DATA['protocol']['total_review_costs']:
                rr = select(task,score,cost,policy)
                values = [r[method]['avg_set_size'] for r in rr]
                points.append((cost,float(np.mean(values)),float(np.std(values,ddof=1))))
                flat.append(dict(task=task,score=score,cost=cost,policy=policy,method=method,
                                 mean_size=float(np.mean(values)),sd_size=float(np.std(values,ddof=1)),
                                 mean_coverage=float(np.mean([r[method]['coverage'] for r in rr])),
                                 sd_coverage=float(np.std([r[method]['coverage'] for r in rr],ddof=1))))
            curves += g.line(style,points)
            manifest.append(dict(task=task,score=score,policy=policy,method=method,points=points))
        x = [42,282][panel%2]
        y = 265 if panel<2 else 105
        options = 'xmin=120,xmax=2080,xtick={200,800,1200,2000},'
        options += ('ymin=.9,ymax=11.5,ytick={1,4,7,10},' if panel<2 else 'ymin=.85,ymax=2.05,ytick={1,1.5,2},')
        options += 'ylabel={Mean set size},' if panel%2==0 else 'yticklabels={},'
        options += 'xlabel={Total reviewed labels}' if panel>=2 else 'xticklabels={}'
        body += g.axis(x,y,178,110,options,curves)
        body += g.panel(x,y+126,title)
    body += g.legend(y=45, methods=[m[2] for m in METHODS],names=[m[3] for m in METHODS])
    g.build('fig_review_allocation',g.wrap(body,410))
    (OUT/'review_allocation_values.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (OUT/'review_allocation_grid.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    lines = [r'\begin{tabular}{lcccc}',r'\toprule',
             r'Method & CIFAR, residual & CIFAR, APS & Binary, residual & Binary, APS \\',r'\midrule']
    for policy,method,style,name in METHODS:
        cells=[name]
        for task,score,title in CONFIG:
            cells.append(stats([r[method]['avg_set_size'] for r in select(task,score,800,policy)]))
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'review_allocation_primary.tex').write_text('\n'.join(lines)+'\n')
    print('Wrote allocation figure, table, values and full-grid summary.')


if __name__=='__main__':
    main()
