import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from method import aps_score_matrix, naive_quantile
from audit import load_cifar, compare, metrics


PROTOCOL = {
    'tasks': ['original_10class', 'animal_vehicle'],
    'vehicle_classes': [0, 1, 8, 9],
    'animal_classes': [2, 3, 4, 5, 6, 7],
    'coverage_targets': [.90, .95, .99],
    'total_adjudication_budgets': [500, 1000],
    'scores': ['residual', 'aps'],
    'seeds': list(range(30)),
    'calibration_seed_offset': 8100,
    'review_seed_offset': 9800,
    'clean_audit_seed_offset': 14300,
    'gamma_fraction_of_total_error': .1,
    'aps_seed_offset': 20261022,
    'allocation': 'Correct mapped annotation disagreements, then review a uniform consensus sample to exhaust the fixed total budget. If disagreements exceed the total budget, use a uniform audit of the whole pool instead.',
    'scope': 'Real recorded human annotations and cached image classifiers. Corrections use dataset reference labels to emulate adjudication. Binary labels are a fixed coarsening of the original task, not new human annotations. All grid configurations are reported; no test-selected guarantee or new independent classifier training is claimed.'
}


def task_data(task, y, yt, human, P, Pt):
    if task == 'original_10class':
        return y, yt, human, P, Pt
    vehicles = np.array(PROTOCOL['vehicle_classes'])
    transform = lambda v: np.isin(v, vehicles).astype(int)
    pc = np.c_[P[:, PROTOCOL['animal_classes']].sum(1), P[:, vehicles].sum(1)]
    pt = np.c_[Pt[:, PROTOCOL['animal_classes']].sum(1), Pt[:, vehicles].sum(1)]
    pc = pc.astype(float); pt = pt.astype(float)
    pc /= pc.sum(1, keepdims=True); pt /= pt.sum(1, keepdims=True)
    return transform(y), transform(yt), {k:transform(v) for k,v in human.items()}, pc, pt


def run(cache_dir, data_dir, output):
    y, yt, human, P, Pt = load_cifar(cache_dir, data_dir)
    pool = np.arange(0, len(y), 5)
    rows = []
    for task in PROTOCOL['tasks']:
        yall, test, noisy, Pall, testP = task_data(task, y, yt, human, P, Pt)
        for seed in PROTOCOL['seeds']:
            rng = np.random.default_rng(PROTOCOL['calibration_seed_offset'] + seed)
            idx = pool[rng.choice(len(pool), 2000, replace=False)]
            labels = yall[idx]; pc = Pall[idx]; obs = noisy['random_label1'][idx]
            disagree = (noisy['random_label2'][idx] != obs) | (noisy['random_label3'][idx] != obs)
            for budget in PROTOCOL['total_adjudication_budgets']:
                fallback = int(disagree.sum()) > budget
                consensus = np.arange(2000) if fallback else np.flatnonzero(~disagree)
                r = budget if fallback else budget - int(disagree.sum())
                rng_review = np.random.default_rng(PROTOCOL['review_seed_offset'] + seed)
                reviewed = rng_review.choice(consensus, r, replace=False)
                observed = obs.copy()
                flags = np.ones(2000, bool)
                if not fallback:
                    observed[disagree] = labels[disagree]; flags[disagree] = False
                found = int((observed[reviewed] != labels[reviewed]).sum())
                observed[reviewed] = labels[reviewed]; flags[reviewed] = False
                assert int((~flags).sum()) == budget
                for target in PROTOCOL['coverage_targets']:
                    epsilon = 1-target; gamma = .1*epsilon; alpha = .9*epsilon
                    counts = np.arange(found, len(consensus)+1)
                    valid = counts[hypergeom.cdf(found, len(consensus), counts, r) >= gamma]
                    upper = int(valid[-1]); m = upper-found
                    for score in PROTOCOL['scores']:
                        row = dict(task=task, seed=seed, score=score, n_cal=2000,
                                   target=target, alpha=alpha, gamma=gamma, review_cost=budget,
                                   disagreements=int(disagree.sum()), consensus_size=len(consensus),
                                   consensus_review=r, errors_found=found, fallback_uniform=fallback,
                                   m=m, remaining_errors=int((observed!=labels).sum()),
                                   budget_success=bool((observed!=labels).sum()<=m))
                        row.update(compare(pc, observed, flags, m, alpha, testP, test, score, score_seed=20261022+seed))
                        rng_clean = np.random.default_rng(PROTOCOL['clean_audit_seed_offset']+seed)
                        clean_idx = rng_clean.choice(2000, budget, replace=False)
                        rng_score = np.random.default_rng(20261022+seed)
                        S = 1-pc if score=='residual' else aps_score_matrix(pc, rng_score.random(len(pc)))
                        St = 1-testP if score=='residual' else aps_score_matrix(testP, rng_score.random(len(testP)))
                        q = naive_quantile(S[clean_idx, labels[clean_idx]], epsilon)
                        row['clean_audit'] = metrics(St, test, q)
                        rows.append(row)
            print(task, seed, flush=True)
    output.write_text(json.dumps(dict(protocol=PROTOCOL, rows=rows,
        input_sha256={'cache_cifar10.npz':hashlib.sha256((cache_dir/'cache_cifar10.npz').read_bytes()).hexdigest()}), indent=1, allow_nan=False)+'\n')
    print('Saved', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/disagreement.json'))
    args = parser.parse_args()
    protocol_path = args.output.with_suffix('.protocol.json')
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    run(args.cache_dir, args.data_dir, args.output)
