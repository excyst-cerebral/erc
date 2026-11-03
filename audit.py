import argparse
import hashlib
import json
import math
import pickle
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from method import (sharp_suspect_quantile, f_aware_coarse_quantile,
                    greedy_binary_quantile, aps_score_matrix, alt_max_scores,
                    naive_quantile, uniform_shift_quantile, evaluate_sets,
                    flagger_low_confidence, poisson_binomial_budget, conformal_k, decomposed_quantile)
from experiments import instance_dependent_corruption
from validate import generated


def metrics(S, y, q):
    return dict(q=None if math.isinf(q) else q, **evaluate_sets(S <= q, y))


def compare(P, observed, flags, m, alpha, Pt, yt, kind, score_seed=20261022):
    rng_score = np.random.default_rng(score_seed)
    S = 1-P if kind == 'residual' else aps_score_matrix(P, rng_score.random(len(P)))
    St = 1-Pt if kind == 'residual' else aps_score_matrix(Pt, rng_score.random(len(Pt)))
    a = S[np.arange(len(observed)), observed]
    exact = sharp_suspect_quantile(S, observed, flags, m, alpha)
    coarse = f_aware_coarse_quantile(S, observed, flags, m, alpha)
    binary = greedy_binary_quantile(S, observed, flags, m, alpha)
    assert exact == binary == decomposed_quantile(S, observed, flags, m, alpha) and exact <= coarse
    flagged_covered = int((flags & (a <= exact)).sum())
    spoil = int((flags & (a <= exact) & (alt_max_scores(S, observed) > exact)).sum())
    result = dict(exact=metrics(St, yt, exact), coarse=metrics(St, yt, coarse),
                  global_shift=metrics(St, yt, uniform_shift_quantile(a, alpha, m)),
                  naive=metrics(St, yt, naive_quantile(a, alpha)),
                  flagged_covered=flagged_covered, spoilable=spoil,
                  effective_count_relief=min(m,flagged_covered)-min(m,spoil),
                  predictions_differ=bool(np.any((St <= exact)!=(St <= coarse))))
    return result


def load_cifar(cache_dir, data_dir):
    labels=[]
    for batch in range(1,6):
        with open(data_dir/f'cifar-10-batches-py/data_batch_{batch}','rb') as f:
            labels.extend(pickle.load(f,encoding='bytes')[b'labels'])
    with open(data_dir/'cifar-10-batches-py/test_batch','rb') as f:
        test=np.array(pickle.load(f,encoding='bytes')[b'labels'])
    base=data_dir/'cifar10n/cifar-10-100n-main'
    human=np.load(base/'data/CIFAR-10_human_ordered.npy',allow_pickle=True).item()
    labels=np.array(labels);clean=np.asarray(human['clean_label']).ravel()
    noisy={k:np.asarray(human[k]).ravel() for k in ['random_label1','random_label2','random_label3']}
    if not np.array_equal(labels,clean):
        perm=np.load(base/'image_order_c10.npy').ravel()
        if np.array_equal(labels[perm],clean):labels=labels[perm]
        elif np.array_equal(labels,clean[perm]):
            clean=clean[perm];noisy={k:v[perm] for k,v in noisy.items()}
        else:raise ValueError('Unverifiable annotation ordering')
    assert np.array_equal(labels,clean)
    z=np.load(cache_dir/'cache_cifar10.npz')
    return labels,test,noisy,z['probs_tr'],z['probs_te']


def comparisons(cache_dir, data_dir):
    rows=[]
    for dataset in ['mnist','fmnist']:
        z=np.load(cache_dir/f'cache_{dataset}.npz')
        for seed in range(30):
            for eta in [.01,.03]:
                rng=np.random.default_rng(6000+seed)
                idx=rng.choice(len(z['y_cal_pool']),2000,replace=False)
                P=z['probs_cal_pool'][idx];y=z['y_cal_pool'][idx]
                observed,errors,rates=instance_dependent_corruption(y,1-P[np.arange(2000),y],eta,1.5,rng)
                m=poisson_binomial_budget(rates,.01)
                flags=errors.copy()
                for kind in ['residual','aps']:
                    row=dict(dataset=dataset,seed=seed,eta_bar=eta,score=kind,n=2000,alpha=.09,
                             m=m,n_corrupt=int(errors.sum()),flag_source='true_errors',
                             premise_hold=bool(errors.sum()<=m))
                    row.update(compare(P,observed,flags,m,.09,z['probs_te'],z['y_te'],kind));rows.append(row)
        print('small budgets',dataset,flush=True)
    labels,test,noisy,Pall,Pt=load_cifar(cache_dir,data_dir)
    pool=np.arange(0,len(labels),5)
    for seed in range(30):
        rng=np.random.default_rng(777+seed);idx=pool[rng.choice(len(pool),2000,replace=False)]
        P=Pall[idx];y=labels[idx];obs=noisy['random_label1'][idx]
        flags=flagger_low_confidence(P,obs,.2)
        for kind in ['residual','aps']:
            row=dict(dataset='cifar10n',seed=seed,score=kind,n=2000,alpha=.1,m=400,
                     flag_source='confidence',n_corrupt=int((obs!=y).sum()),premise_hold=False)
            row.update(compare(P,obs,flags,400,.1,Pt,test,kind));rows.append(row)
    return rows


def historical_comparisons(cache_dir, data_dir):
    rows=[]
    original=json.loads(Path('results/summary.json').read_text())
    for dataset in ['mnist','fmnist']:
        z=np.load(cache_dir/f'cache_{dataset}.npz')
        for old in original['exp1_controlled']:
            if old['dataset']!=dataset or old['n_cal']!=2000 or old['alpha']!=.1:continue
            rng=np.random.default_rng(1000*old['seed']+17)
            idx=rng.choice(len(z['y_cal_pool']),2000,replace=False)
            P=z['probs_cal_pool'][idx];y=z['y_cal_pool'][idx]
            obs,err,rates=instance_dependent_corruption(y,1-P[np.arange(2000),y],old['eta_bar'],1.5,rng)
            assert int(err.sum())==old['n_corrupt']
            flags=flagger_low_confidence(P,obs,old['eta_bar'])
            for kind in ['residual','aps']:
                row=dict(dataset=dataset,seed=old['seed'],eta_bar=old['eta_bar'],score=kind,
                         flag_source='confidence',alpha=.1,m=old['m'],premise_hold=old['premise_hold'])
                row.update(compare(P,obs,flags,old['m'],.1,z['probs_te'],z['y_te'],kind))
                if kind=='residual':
                    assert abs(row['exact']['avg_set_size']-old['sharp_suspect_size'])<1e-10
                rows.append(row)
    z=np.load(cache_dir/'cache_fmnist.npz')
    matched=[]
    for alpha,cap in [(.1,200),(.2,100)]:
        for seed in range(5):
            rng=np.random.default_rng(31+seed);idx=rng.choice(len(z['y_cal_pool']),2000,replace=False)
            P=z['probs_cal_pool'][idx];y=z['y_cal_pool'][idx]
            a=1-P[np.arange(2000),y];k=conformal_k(2000,alpha);q=np.sort(a)[k-1]
            pred=P.argmax(1);defl=1-P[np.arange(2000),pred]
            cross=np.flatnonzero((pred!=y)&(a>q)&(defl<=q))
            attack=sorted(cross,key=lambda i:defl[i])[:cap];obs=y.copy();obs[attack]=pred[attack]
            flags=flagger_low_confidence(P,obs,cap/2000)
            row=dict(alpha=alpha,seed=seed,attack_cap=cap,n_corrupt=len(attack),defense_budget=cap)
            row.update(compare(P,obs,flags,cap,alpha,z['probs_te'],z['y_te'],'residual'));matched.append(row)
    return rows,matched


def budget_accounting():
    original=json.loads(Path('results/summary.json').read_text())
    rows=[]
    for r in original['exp1_controlled']:
        if r['alpha']==.1 and r['n_cal']==2000:
            rows.append({k:r[k] for k in ['dataset','seed','eta_bar','m','m_rate','n_corrupt']})
    attack=[]
    for r in original['exp3_adversarial']:
        attack.append({k:r[k] for k in ['seed','eta_bar','alpha','attack_cap','defense_budget','n_crossings']})
    all_rows=[]
    for key in ['exp1_controlled','exp2_cifar10n','exp3_adversarial','exp4_flag_recall','exp8_annotation_audit']:
        for old in original[key]:
            n=int(old.get('n_cal',2000))
            actual=int(old.get('n_corrupt',old.get('n_crossings',round(n*old['realized_noise']))))
            if key=='exp1_controlled':budget=old['m']
            elif key=='exp2_cifar10n':budget=int(math.ceil(old['budget_frac']*n))
            elif key=='exp3_adversarial':budget=old['defense_budget']
            elif key=='exp4_flag_recall':budget=actual
            else:budget=old['m']
            row=dict(experiment=key,seed=old['seed'],n_cal=n,alpha=old['alpha'],realized_errors=actual,
                     global_shift_budget=budget,confidence_or_protocol_flags_budget=budget,
                     unflagged_exact_budget=budget,
                     true_error_flags_budget=actual if 'sharp_oracle_flags_coverage' in old else None,
                     clean_reference_budget=0 if 'oracle_clean_coverage' in old else None)
            for k in ['dataset','eta_bar','noisy','budget_frac','recall_target','audit','attack_cap','m_rate']:
                if k in old:row[k]=old[k]
            all_rows.append(row)
    return dict(rate_budgets=rows,attack_budgets=attack,all_row_budgets=all_rows)


def audit_frontier(cache_dir,data_dir):
    labels,test,noisy,Pall,Pt=load_cifar(cache_dir,data_dir);pool=np.arange(0,len(labels),5)
    rows=[];clean=[]
    for seed in range(30):
        rng=np.random.default_rng(777+seed);idx=pool[rng.choice(len(pool),2000,replace=False)]
        P=Pall[idx];y=labels[idx];obs=noisy['random_label1'][idx]
        disagree=(noisy['random_label2'][idx]!=obs)|(noisy['random_label3'][idx]!=obs)
        cons=np.flatnonzero(~disagree);N=len(cons)
        for r in [100,200,400,600]:
            rng2=np.random.default_rng(1200+seed);review=rng2.choice(cons,min(r,N),replace=False)
            found=int((obs[review]!=y[review]).sum())
            adj=obs.copy();adj[disagree]=y[disagree];adj[review]=y[review]
            flags=~disagree;flags[review]=False
            for gamma in [.05,.02,.01,.005,.001]:
                Ks=np.arange(found,N+1)
                valid=Ks[hypergeom.cdf(found,N,Ks,len(review))>=gamma]
                up=int(valid[-1]);m=up-found;alpha=.1-gamma
                row=dict(seed=seed,r=r,gamma=gamma,alpha=alpha,N=N,errors_found=found,K_up=up,m=m,
                         n_corrupt=int((adj!=y).sum()),review_cost=int(disagree.sum()+len(review)),
                         premise_hold=bool((adj!=y).sum()<=m))
                row.update(compare(P,adj,flags,m,alpha,Pt,test,'residual'));rows.append(row)
        for count in [800,1000,1200,1400]:
            rng4=np.random.default_rng(4400+seed);ri=np.sort(rng4.choice(2000,count,replace=False))
            q=naive_quantile(1-P[ri,y[ri]],.1)
            clean.append(dict(seed=seed,reviews=count,**metrics(1-Pt,test,q)))
        print('audit frontier',seed,flush=True)
    return rows,clean


def greedy_scaling():
    rng=np.random.default_rng(4421);rows=[]
    for f,budget in [(.1,.01),(.5,.05)]:
        for n in [100,1000,10000,100000]:
            for rep in range(5):
                S,y,flags,m=generated(rng,n,10,f,budget)
                row=dict(n=n,flag_fraction=f,rep=rep)
                qs={}
                for name,fn in [('sweep',sharp_suspect_quantile),('greedy_binary',greedy_binary_quantile)]:
                    start=time.perf_counter();qs[name]=fn(S,y,flags,m,.1)
                    row[name+'_s']=time.perf_counter()-start
                assert qs['sweep']==qs['greedy_binary']
                row['agree']=True;rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/audit.json'))
    args=parser.parse_args()
    out=dict(protocol='Fixed grid; gamma=0.01 and r=200 specified before inspecting test results. Every grid point is evaluated separately; no test-selected certificate.',
             environment=dict(python=platform.python_version(),numpy=np.__version__),
             budget_accounting=budget_accounting(),greedy_scaling=greedy_scaling(),
             score_comparisons=comparisons(args.cache_dir,args.data_dir))
    out['historical_score_comparisons'],out['matched_attack_budgets']=historical_comparisons(args.cache_dir,args.data_dir)
    out['audit_frontier'],out['clean_audit_frontier']=audit_frontier(args.cache_dir,args.data_dir)
    out['input_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.cache_dir.glob('cache_*.npz')}
    args.output.write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
    print('Saved',args.output,flush=True)


if __name__=='__main__':
    main()
