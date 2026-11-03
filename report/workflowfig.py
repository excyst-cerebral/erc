import csv
import json
from pathlib import Path
import numpy as np
import plots as g

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'
D=json.loads((ROOT/'results/targeted.json').read_text())
B=json.loads((ROOT/'results/benchmark.json').read_text())


def stats(values,digits=3):
    v=np.asarray(values)
    return f'${v.mean():.{digits}f}\\pm{v.std(ddof=1):.{digits}f}$'


def table(name,columns,header,rows):
    lines=[r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/(name+'.tex')).write_text('\n'.join(lines)+'\n')


def main():
    rows=[]
    for n in [100,1000,10000,100000,1000000]:
        rr=[v for v in B['paired_scaling'] if v['n']==n]
        rows.append([f'$10^{int(np.log10(n))}$']+[f'{1000*np.median([v[m+"_s"] for v in rr]):.3f}' for m in ['decomposed','greedy_binary','sweep']])
    table('practical_timing','rrrr',r'$n$ & Two quantiles (ms) & Greedy search (ms) & Sweep (ms)',rows)
    rows=[]
    primary=[v for v in D['rows'] if v['task']=='mammal_other' and v['score']=='aps' and v['uncertain_pool']==400 and v['target']==.95]
    for method,label in [('exact','Exact (budget)'),('coarse','$F$-coarse'),('partial_label_same_alpha','Partial-label, same $\\alpha$'),('partial_label_matched_target','Partial-label, target-matched'),('clean_audit','Random clean audit')]:
        rows.append([label,stats([v[method]['coverage'] for v in primary]),stats([v[method]['avg_set_size'] for v in primary])])
    table('practical_primary','lcc',r'Method & Coverage & Mean set size',rows)
    flat=[]
    for source in ['disagreement','targeted']:
        data=json.loads((ROOT/'results'/f'{source}.json').read_text())
        keys=['exact','coarse','global_shift','clean_audit']+(['partial_label_same_alpha','partial_label_matched_target'] if source=='targeted' else [])
        for row in data['rows']:
            for method in keys:
                flat.append(dict(stage=source,task=row['task'],score=row['score'],target=row['target'],seed=row['seed'],reviews=row['review_cost'],uncertain_pool=row.get('uncertain_pool',''),method=method,coverage=row[method]['coverage'],set_size=row[method]['avg_set_size'],budget_success=row['budget_success']))
    with (OUT/'practical_full_grid.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=flat[0].keys());writer.writeheader();writer.writerows(flat)
    body=''
    methods=[('exact','sharp_suspect'),('coarse','naive'),('partial_label_matched_target','sharp_oracle_flags'),('clean_audit','oracle_clean')]
    for i,metric in enumerate(['coverage','avg_set_size']):
        curves=''
        for method,style in methods:
            points=[]
            for target in [.9,.95,.99]:
                rr=[v[method][metric] for v in D['rows'] if v['task']=='mammal_other' and v['score']=='aps' and v['uncertain_pool']==400 and v['target']==target]
                points.append((target,float(np.mean(rr)),float(np.std(rr,ddof=1))))
            curves+=g.line(style,points)
        if i==0:curves+=r'\addplot[reference,dashed] coordinates {(.885,.885) (1,1)};'+'\n'
        opts='xmin=.885,xmax=1,xtick={.9,.95,.99},xlabel={Coverage target},'
        opts+=('ymin=.87,ymax=1.015,ytick={.9,.95,1},ylabel={Measured coverage}' if i==0 else 'ymin=.9,ymax=2.1,ytick={1,1.5,2},ylabel={Mean set size}')
        body+=g.axis([42,282][i],66,178,130,opts,curves)
        body+=g.panel([42,282][i],211,['(a) Coverage','(b) Set size'][i])
    body+=g.legend(methods=[s for m,s in methods],names=['Exact (budget)','$F$-coarse','Partial-label','Clean audit'])
    g.build('fig_targeted_review',g.wrap(body,225))
    print('Wrote direct-computation and targeted-review artifacts.')


if __name__=='__main__':
    main()
