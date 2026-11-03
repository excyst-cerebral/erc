import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from method import aps_score_matrix, conformal_k, decomposed_quantile, naive_quantile
from audit import load_cifar, metrics


def review_gains(S, observed, flags, revealed, q):
    a = S[np.arange(len(S)), observed] <= q
    v = S[np.arange(len(S)), revealed] <= q
    e = observed != revealed
    spoil = np.asarray(flags) & a & (S.max(1) > q)
    g = (v.astype(int)-a.astype(int)+e.astype(int))*flags
    h = (v.astype(int)-a.astype(int)+spoil.astype(int))*flags
    return g, h


def lower_count(S, observed, flags, m, q):
    a = S[np.arange(len(S)), observed] <= q
    spoil = flags & a & (S.max(1) > q)
    return int(a.sum())-min(m, int(spoil.sum()))


def certificate_cost(S, observed, flags, m, revealed, q, k):
    a = S[np.arange(len(S)), observed] <= q
    spoil = flags & a & (S.max(1) > q)
    g, h = review_gains(S, observed, flags, revealed, q)
    costs = []
    for base, gains in [(int(a.sum())-m, g), (int(a.sum())-int(spoil.sum()), h)]:
        required = k-base
        cumulative = np.r_[0, np.cumsum(np.sort(gains[flags])[::-1])]
        options = np.flatnonzero(cumulative >= required)
        costs.append(int(options[0]) if len(options) else None)
    possible = [v for v in costs if v is not None]
    return min(possible) if possible else None


def upper_errors(N, r, found, gamma):
    low, high = found, N-r+found
    while low < high:
        mid = (low+high+1)//2
        if hypergeom.cdf(found, N, mid, r) >= gamma:
            low = mid
        else:
            high = mid-1
    return low-found


def exhaustive_checks():
    rng = np.random.default_rng(202610201)
    checked = 0
    transcripts = 0
    for n in range(1, 6):
        for K in [1, 2, 3]:
            for _ in range(5):
                S = rng.integers(0, 5, (n, K)).astype(float)
                observed = rng.integers(K, size=n)
                flags = rng.random(n) < .8
                m = int(rng.integers(int(flags.sum())+1))
                worlds = [np.array(y) for y in itertools.product(range(K), repeat=n)
                          if np.all(np.array(y)[~flags]==observed[~flags]) and np.sum(np.array(y)!=observed)<=m]
                actual = worlds[int(rng.integers(len(worlds)))]
                subsets = [np.array([i for i in range(n) if mask>>i&1], dtype=int)
                           for mask in range(1<<n) if all(flags[i] for i in range(n) if mask>>i&1)]
                for q in np.unique(S):
                    g, h = review_gains(S, observed, flags, actual, q)
                    a = S[np.arange(n), observed] <= q
                    spoil = flags & a & (S.max(1)>q)
                    bounds = []
                    for Q in subsets:
                        updated = observed.copy()
                        updated[Q] = actual[Q]
                        remaining = flags.copy()
                        remaining[Q] = False
                        budget = m-int(np.sum(actual[Q]!=observed[Q]))
                        direct = lower_count(S, updated, remaining, budget, q)
                        additive = max(int(a.sum())-m+int(g[Q].sum()), int(a.sum())-int(spoil.sum())+int(h[Q].sum()))
                        compatible = [y for y in worlds if np.array_equal(y[Q], actual[Q])]
                        brute = min(int(np.sum(S[np.arange(n), y]<=q)) for y in compatible)
                        assert direct == additive == brute
                        bounds.append(direct)
                        transcripts += 1
                    for k in range(1, n+1):
                        costs = [len(Q) for Q, bound in zip(subsets, bounds) if bound>=k]
                        expected = min(costs) if costs else None
                        assert certificate_cost(S, observed, flags, m, actual, q, k)==expected
                        checked += 1
    for N in [5, 10, 25]:
        for r in range(1, N+1):
            for found in range(r+1):
                feasible = [v for v in range(found, N-r+found+1) if hypergeom.cdf(found, N, v, r)>=.01]
                assert upper_errors(N,r,found,.01)==max(feasible)-found
    return dict(certificate_cases=checked, transcript_bounds=transcripts)


def choose(S, P, observed, flags, m, alpha, policy, order):
    available = np.flatnonzero(flags)
    if policy == 'random':
        return int(next(i for i in order if flags[i]))
    q = decomposed_quantile(S, observed, flags, m, alpha)
    values = np.unique(S)
    below = values[values < q]
    pivot = below[-1] if len(below) else q
    a = S[np.arange(len(S)), observed] <= pivot
    v = (P*(S<=pivot)).sum(1)
    spoil = flags & a & (S.max(1)>pivot)
    if policy == 'interval':
        probability = v-a.astype(int)+spoil.astype(int)
        return int(available[np.argmax(probability[available])])
    g = np.maximum(0, v-a.astype(int)+1-P[np.arange(len(P)), observed])
    h = np.maximum(0, v-a.astype(int)+spoil.astype(int))
    k = conformal_k(len(S), alpha)
    deficits = [k-(int(a.sum())-m), k-(int(a.sum())-int(spoil.sum()))]
    gains = [g, h]
    choices = []
    for deficit, gain in zip(deficits, gains):
        ordered = available[np.argsort(-gain[available], kind='stable')]
        cumulative = np.cumsum(gain[ordered])
        achievable = np.flatnonzero(cumulative>=deficit)
        estimated_cost = int(achievable[0])+1 if len(achievable) else math.inf
        choices.append((estimated_cost, -float(gain[ordered[0]]), int(ordered[0])))
    return min(choices)[2]


def run(cache_dir, data_dir, output):
    protocol = dict(stage='Budget-coupled follow-up after unrestricted exact-recovery study',
        n=2000, seeds=list(range(30)), calibration_seed_offset=70100, audit_seed_offset=71100,
        review_seed_offset=72100, aps_seed_offset=73100, pilot=200,
        total_review_costs=[200,400,800,1200,1700,2000], alpha=.09, gamma=.01,
        tasks=['cifar10','mammal_other'], scores=['residual','aps'],
        policies=['random','interval','budget_gain'],
        policy='One-step surrogate review gains at the next lower score event; true labels enter the policy only after a charged query. Class probabilities guide selection but are not assumed to be calibrated for validity.',
        scope='Real CIFAR-10N annotations; dataset reference labels emulate accurate review. One cached classifier, reused test set, exploratory evaluation. Initial uniform audit supplies a fixed simultaneous residual-error budget; subsequent review decrements it by observed errors. Oracle certificate is an explicitly privileged diagnostic, never a deployable policy.',
        stronger_comparators=['Uniform clean-subset conformal at the same total review cost and target .90', 'Uniform audit with a freshly inverted error budget using all reviewed labels', 'Partial-label calibration on each policy transcript at matched target .90'],
        posthoc_diagnostic='After inspecting the fixed-budget policy results, compare the clairvoyant certificate cost under the initial survey bound with its cost under the exact realized residual error count. Both use true labels and are explicitly privileged diagnostics.',
        not_evaluated=['Published label-efficient stopping algorithms optimize a different stopping-cost objective and are discussed as related work, not claimed to be outperformed'])
    output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    checks = exhaustive_checks()
    y, yt, human, P, Pt = load_cifar(cache_dir, data_dir)
    pool = np.arange(0,len(y),5)
    rows = []
    for task in protocol['tasks']:
        if task=='cifar10':
            pc, pt, labels, test, noisy = P, Pt, y, yt, human['random_label1']
        else:
            pos=[3,4,5,7]
            neg=[v for v in range(10) if v not in pos]
            pc=np.c_[P[:,neg].sum(1), P[:,pos].sum(1)].astype(float)
            pt=np.c_[Pt[:,neg].sum(1), Pt[:,pos].sum(1)].astype(float)
            pc/=pc.sum(1,keepdims=True)
            pt/=pt.sum(1,keepdims=True)
            labels=np.isin(y,pos).astype(int)
            test=np.isin(yt,pos).astype(int)
            noisy=np.isin(human['random_label1'],pos).astype(int)
        for seed in protocol['seeds']:
            idx=pool[np.random.default_rng(70100+seed).choice(len(pool),2000,replace=False)]
            original=noisy[idx].copy()
            truth=labels[idx]
            initial=np.random.default_rng(71100+seed).permutation(2000)
            pilot=initial[:200]
            found=int(np.sum(original[pilot]!=truth[pilot]))
            m0=upper_errors(2000,200,found,.01)
            flags0=np.ones(2000,bool)
            flags0[pilot]=False
            obs0=original.copy()
            obs0[pilot]=truth[pilot]
            budget_success=bool(np.sum(obs0!=truth)<=m0)
            for score in protocol['scores']:
                srng=np.random.default_rng(73100+seed)
                S=1-pc[idx] if score=='residual' else aps_score_matrix(pc[idx],srng.random(2000))
                St=1-pt if score=='residual' else aps_score_matrix(pt,srng.random(len(pt)))
                clean_q=naive_quantile(S[np.arange(2000),truth],.09)
                oracle=certificate_cost(S,obs0,flags0,m0,truth,clean_q,conformal_k(2000,.09)) if budget_success else None
                actual_budget=int(np.sum(obs0!=truth))
                oracle_tight=certificate_cost(S,obs0,flags0,actual_budget,truth,clean_q,conformal_k(2000,.09))
                for policy in protocol['policies']:
                    observed=obs0.copy()
                    flags=flags0.copy()
                    m=m0
                    extra_errors=0
                    contradiction=False
                    order=initial if policy=='random' else np.random.default_rng(72100+seed).permutation(2000)
                    previous=math.inf
                    for total in range(200,2001):
                        if total in protocol['total_review_costs']:
                            q=decomposed_quantile(S,observed,flags,m,.09)
                            assert contradiction or q<=previous
                            previous=q
                            row=dict(task=task,score=score,seed=seed,policy=policy,total_reviews=total,
                                     initial_budget=m0,residual_budget=m,initial_budget_success=budget_success,
                                     true_remaining_errors=int(np.sum(observed!=truth)),revealed_extra_errors=extra_errors,
                                     observed_budget_contradiction=contradiction,
                                     oracle_extra_reviews_to_full_clean_quantile=oracle,
                                     initial_budget_slack=m0-actual_budget,
                                     oracle_extra_reviews_with_exact_budget=oracle_tight,
                                     exact=metrics(St,test,q),full_clean=metrics(St,test,clean_q))
                            if budget_success:
                                assert q>=clean_q
                            worst=S[np.arange(2000),observed].copy()
                            worst[flags]=S[flags].max(1)
                            row['partial_matched_target']=metrics(St,test,naive_quantile(worst,.1))
                            uniform=initial[:total]
                            row['random_clean_cp']=metrics(St,test,naive_quantile(S[uniform,truth[uniform]],.1))
                            uniform_obs=original.copy()
                            uniform_obs[uniform]=truth[uniform]
                            uniform_flags=np.ones(2000,bool)
                            uniform_flags[uniform]=False
                            uniform_found=int(np.sum(original[uniform]!=truth[uniform]))
                            uniform_m=upper_errors(2000,total,uniform_found,.01)
                            row['refreshed_uniform_audit']=metrics(St,test,decomposed_quantile(S,uniform_obs,uniform_flags,uniform_m,.09))
                            rows.append(row)
                        if total==2000:
                            break
                        i=choose(S,pc[idx],observed,flags,m,.09,policy,order)
                        error=int(observed[i]!=truth[i])
                        observed[i]=truth[i]
                        flags[i]=False
                        extra_errors+=error
                        m-=error
                        if m<0:
                            contradiction=True
                            m=int(flags.sum())
            print(task,seed,flush=True)
    output.write_text(json.dumps(dict(protocol=protocol,checks=checks,
        input_sha256=hashlib.sha256((cache_dir/'cache_cifar10.npz').read_bytes()).hexdigest(),rows=rows),indent=1,allow_nan=False)+'\n')
    print('Saved',output,checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/allocation.json'))
    args=parser.parse_args()
    run(args.cache_dir,args.data_dir,args.output)
