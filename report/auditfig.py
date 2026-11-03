import json
from pathlib import Path
import numpy as np
import plots as g

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'
data=json.loads((ROOT/'results/audit.json').read_text())


def stats(values,digits=3):
    v=np.asarray(values)
    return f'${v.mean():.{digits}f}\\pm{v.std(ddof=1):.{digits}f}$'


def table(name,columns,header,rows):
    text=['\\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']
    text += [' & '.join(map(str,row))+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    (OUT/(name+'.tex')).write_text('\n'.join(text)+'\n')


def main():
    primary=[v for v in data['audit_frontier'] if v['r']==200 and v['gamma']==.01]
    clean=[v for v in data['clean_audit_frontier'] if v['reviews']==1000]
    table('review_primary','lrrr',r'Procedure & Reviewed labels & Coverage & Set size',[
        ['Robust pipeline','951--1047',stats([v['exact']['coverage'] for v in primary]),stats([v['exact']['avg_set_size'] for v in primary])],
        ['Random clean audit',1000,stats([v['coverage'] for v in clean]),stats([v['avg_set_size'] for v in clean])]])
    rows=[]
    for ds,label in [('mnist','MNIST'),('fmnist','F-MNIST')]:
        for eta in [.1,.2,.4]:
            rr=[r for r in data['budget_accounting']['rate_budgets'] if r['dataset']==ds and r['eta_bar']==eta]
            rows.append([label,eta,rr[0]['m'],f"{min(r['n_corrupt'] for r in rr)}--{max(r['n_corrupt'] for r in rr)}",
                         f"{min(r['m_rate'] for r in rr)}--{max(r['m_rate'] for r in rr)}",f"{sum(r['n_corrupt']<=r['m_rate'] for r in rr)}/{len(rr)}"])
    table('review_budget_accounting','lrrrrr',r'Dataset & $\bar\eta$ & Nominal $m$ & Realized $M$ & $m_\delta$ & $M\le m_\delta$',rows)
    rows=[]
    for alpha in [.1,.2]:
        rr=[r for r in data['matched_attack_budgets'] if r['alpha']==alpha]
        row=[alpha,rr[0]['attack_cap'],f"{min(r['n_corrupt'] for r in rr)}--{max(r['n_corrupt'] for r in rr)}"]
        for method in ['global_shift','coarse','exact']:
            row.append(r'\shortstack{'+stats([r[method]['coverage'] for r in rr])+r'\\'+stats([r[method]['avg_set_size'] for r in rr],2)+'}')
        rows.append(row)
    table('review_matched_attack','rrrccc',r'$\alpha$ & $m_{\rm atk}=m_{\rm def}$ & $M$ & Global & $F$-coarse & Exact',rows)
    rows=[]
    for key,label in [('historical_score_comparisons','Main confidence'),('score_comparisons','Small budget')]:
        for ds,dlabel in [('mnist','MNIST'),('fmnist','F-MNIST'),('cifar10n','CIFAR-10N')]:
            if key=='score_comparisons' and ds=='cifar10n':continue
            for score in ['residual','aps']:
                rr=[r for r in data[key] if r['dataset']==ds and r['score']==score]
                if not rr:continue
                rows.append([dlabel,('Confidence' if key=='historical_score_comparisons' else 'Small / true $F$'),score.upper() if score=='aps' else 'Residual',len(rr),
                             sum(r['flagged_covered']!=r['spoilable'] for r in rr),sum(r['predictions_differ'] for r in rr),
                             stats([r['coarse']['avg_set_size']-r['exact']['avg_set_size'] for r in rr])])
    for score in ['residual','aps']:
        rr=[r for r in data['score_comparisons'] if r['dataset']=='cifar10n' and r['score']==score]
        rows.append(['CIFAR-10N','Confidence',score.upper() if score=='aps' else 'Residual',len(rr),sum(r['flagged_covered']!=r['spoilable'] for r in rr),sum(r['predictions_differ'] for r in rr),stats([r['coarse']['avg_set_size']-r['exact']['avg_set_size'] for r in rr])])
    table('review_exact_relief','lllrrrc',r'Dataset & Setting & Score & Runs & Count differs & Sets differ & Size reduction',rows)
    rows=[]
    for ds,label in [('mnist','MNIST'),('fmnist','F-MNIST')]:
        for eta in [.01,.03]:
            for score in ['residual','aps']:
                rr=[r for r in data['score_comparisons'] if r['dataset']==ds and r['eta_bar']==eta and r['score']==score]
                row=[label,eta,'APS' if score=='aps' else 'Residual']
                for method in ['exact','coarse','global_shift']:
                    row.append(r'\shortstack{'+stats([r[method]['coverage'] for r in rr])+r'\\'+stats([r[method]['avg_set_size'] for r in rr],2)+'}')
                rows.append(row)
    table('review_small_budgets','lrlccc',r'Dataset & $\bar\eta$ & Score & Exact & $F$-coarse & Global shift',rows)
    rows=[]
    for r in [100,200,400,600]:
        for gamma in [.05,.01,.005]:
            rr=[v for v in data['audit_frontier'] if v['r']==r and v['gamma']==gamma]
            rows.append([r,gamma,f"{min(v['review_cost'] for v in rr)}--{max(v['review_cost'] for v in rr)}",
                         stats([v['exact']['coverage'] for v in rr]),stats([v['exact']['avg_set_size'] for v in rr])])
    table('review_audit_frontier','rrrcc',r'Consensus reviews $r$ & $\gamma$ & Total reviews & Coverage & Set size',rows)
    rows=[]
    for n in [100,1000,10000,100000]:
        rr=[v for v in data['greedy_scaling'] if v['n']==n]
        rows.append([f'$10^{int(np.log10(n))}$',f"{1000*np.median([v['sweep_s'] for v in rr]):.3f}",f"{1000*np.median([v['greedy_binary_s'] for v in rr]):.3f}",f"{sum(v['agree'] for v in rr)}/{len(rr)}"])
    table('review_greedy_timing','rrrr',r'$n$ & Sweep (ms) & Greedy + binary search (ms) & Agreement',rows)
    body=''
    styles=['naive','sharp_suspect','sharp_oracle_flags','known_T']
    names=['$r=100$','$r=200$','$r=400$','$r=600$']
    for i,metric in enumerate(['coverage','avg_set_size']):
        curves=''
        for r,style in zip([100,200,400,600],styles):
            points=[]
            for gamma in [.001,.005,.01,.02,.05]:
                rr=[v['exact'][metric] for v in data['audit_frontier'] if v['r']==r and v['gamma']==gamma]
                points.append((gamma,float(np.mean(rr)),float(np.std(rr,ddof=1))))
            curves+=g.line(style,points)
        base=[v[metric] for v in data['clean_audit_frontier'] if v['reviews']==1000]
        b=float(np.mean(base))
        curves+=r'\addplot[ink,densely dotted,line width=.9pt] coordinates {'+f'(.0008,{b}) (.062,{b})'+'};\n'
        if i==0:curves+=r'\addplot[reference,dashed] coordinates {(.0008,.9) (.062,.9)};'+'\n'
        opts=r'xmode=log,xmin=.0008,xmax=.062,xtick={.001,.01,.05},xticklabels={0.001,0.01,0.05},xlabel={Budget failure $\gamma$},'
        opts+=('ymin=.87,ymax=1,ytick={.9,.95,1},ylabel={Clean-label coverage}' if i==0 else 'ymin=1.05,ymax=4.1,ytick={1.5,2,3,4},ylabel={Mean set size}')
        body+=g.axis([42,282][i],66,178,130,opts,curves)
        body+=g.panel([42,282][i],211,['(a) Coverage','(b) Set size'][i])
    body+=g.legend(methods=styles+['oracle_clean'],names=names+['Clean audit (1000)'])
    g.build('fig_audit_frontier',g.wrap(body,225))
    print('Wrote review evidence tables and vector frontier figure.')


if __name__=='__main__':
    main()
