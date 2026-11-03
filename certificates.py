import argparse
import json
from pathlib import Path

import numpy as np

from method import aps_score_matrix, naive_quantile
from audit import load_cifar, metrics


def recover(S, query, k, observed, policy, seed=0):
    n = len(S)
    if k > n:
        return float('inf'), 0
    if k < 1:
        raise ValueError('The requested rank must be positive')
    lower = S.min(1).copy()
    upper = S.max(1).copy()
    reviewed = lower == upper
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cost = 0
    while True:
        lo = float(np.partition(lower, k-1)[k-1])
        hi = float(np.partition(upper, k-1)[k-1])
        if lo == hi:
            return hi, cost
        available = np.flatnonzero(~reviewed)
        if policy == 'random':
            i = int(next(i for i in order if not reviewed[i]))
        else:
            possible = available[(upper[available] >= lo) & (lower[available] <= hi)]
            estimate = S[np.arange(n), observed].copy()
            estimate[reviewed] = lower[reviewed]
            pivot = np.clip(np.partition(estimate, k-1)[k-1], lo, hi)
            crosses = possible[(lower[possible] <= pivot) & (upper[possible] >= pivot)]
            if len(crosses):
                possible = crosses
            priority = np.abs(estimate[possible]-pivot)
            i = int(possible[np.argmin(priority)])
        value = float(query(i))
        if not lower[i] <= value <= upper[i]:
            raise ValueError('Query returned a score outside the candidate-label range')
        lower[i] = upper[i] = value
        reviewed[i] = True
        cost += 1


def checks():
    rng = np.random.default_rng(20261020)
    for n in range(1, 16):
        for classes in [1, 2, 5]:
            for _ in range(15):
                S = rng.integers(0, 8, (n, classes)).astype(float)
                y = rng.integers(classes, size=n)
                obs = rng.integers(classes, size=n)
                k = int(rng.integers(1, n+1))
                expected = float(np.partition(S[np.arange(n), y], k-1)[k-1])
                for policy in ['targeted', 'random']:
                    q, cost = recover(S, lambda i: S[i, y[i]], k, obs, policy)
                    assert q == expected and 0 <= cost <= n
    return 1350


def run(cache_dir, data_dir, output):
    protocol = dict(n=2000, seeds=list(range(10)), seed_offset=30100,
                    tasks=['cifar10', 'mammal_other'], scores=['residual', 'aps'],
                    target=.9, objective='Recover exactly the full-pool clean conformal threshold',
                    query='Dataset reference label emulates adjudication; the policy accesses it only through a charged query',
                    limitations='Exploratory cached-model study. Existing test set; no new human adjudication. Not a comparison against label-efficient stopping or prediction-powered calibration.')
    output.with_suffix('.protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    checked = checks()
    y, yt, human, P, Pt = load_cifar(cache_dir, data_dir)
    rows = []
    pool = np.arange(0, len(y), 5)
    for task in protocol['tasks']:
        if task == 'cifar10':
            pc, pt, labels, test, observed = P, Pt, y, yt, human['random_label1']
        else:
            pos = [3, 4, 5, 7]
            neg = [v for v in range(10) if v not in pos]
            pc = np.c_[P[:, neg].sum(1), P[:, pos].sum(1)].astype(float)
            pt = np.c_[Pt[:, neg].sum(1), Pt[:, pos].sum(1)].astype(float)
            pc /= pc.sum(1, keepdims=True)
            pt /= pt.sum(1, keepdims=True)
            labels, test = np.isin(y, pos).astype(int), np.isin(yt, pos).astype(int)
            observed = np.isin(human['random_label1'], pos).astype(int)
        for seed in protocol['seeds']:
            rng = np.random.default_rng(protocol['seed_offset']+seed)
            idx = pool[rng.choice(len(pool), 2000, replace=False)]
            for score in protocol['scores']:
                srng = np.random.default_rng(40100+seed)
                S = 1-pc[idx] if score == 'residual' else aps_score_matrix(pc[idx], srng.random(2000))
                St = 1-pt if score == 'residual' else aps_score_matrix(pt, srng.random(len(pt)))
                k = int(np.ceil(2001*.9))
                truth = S[np.arange(2000), labels[idx]]
                expected = float(np.partition(truth, k-1)[k-1])
                row = dict(task=task, seed=seed, score=score, n=2000, **metrics(St, test, expected))
                for policy in ['targeted', 'random']:
                    q, cost = recover(S, lambda i: S[i, labels[idx][i]], k, observed[idx], policy, 50100+seed)
                    assert q == expected
                    row[policy+'_queries'] = cost
                clean_idx = np.random.default_rng(60100+seed).choice(2000, row['targeted_queries'], replace=False)
                row['same_cost_random_clean_cp'] = metrics(St, test, naive_quantile(truth[clean_idx], .1))
                row['strict_crossing_lower_bound'] = int(((S.min(1)<expected) & (S.max(1)>expected)).sum()) if np.sum(truth==expected)==1 else None
                rows.append(row)
            print(task, seed, flush=True)
    output.write_text(json.dumps(dict(protocol=protocol, generated_cases_checked=checked, rows=rows), indent=2)+'\n')
    for task in protocol['tasks']:
        for score in protocol['scores']:
            selected = [r for r in rows if r['task']==task and r['score']==score]
            print(task, score, {key:float(np.mean([r[key] for r in selected])) for key in ['targeted_queries','random_queries','coverage','avg_set_size']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/certificates.json'))
    args = parser.parse_args()
    run(args.cache_dir, args.data_dir, args.output)
