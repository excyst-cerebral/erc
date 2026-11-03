import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from method import aps_score_matrix, naive_quantile
from audit import load_cifar, compare, metrics


PROTOCOL = {
    'stage': 'Mechanism-motivated follow-up after the fixed-cost disagreement grid showed only small exact/coarse gains.',
    'tasks': {'animal_vehicle': [0, 1, 8, 9], 'mammal_other': [3, 4, 5, 7]},
    'uncertain_pool_and_uniform_review': [[400, 100], [800, 200]],
    'coverage_targets': [.90, .95, .99],
    'scores': ['residual', 'aps'],
    'seeds': list(range(30)),
    'calibration_seed_offset': 18100,
    'review_seed_offset': 19800,
    'clean_audit_seed_offset': 24300,
    'gamma_fraction_of_total_error': .1,
    'aps_seed_offset': 20261022,
    'allocation': 'Select the fixed-size pool with smallest maximum binary probability, using index tie breaking. Correct every label outside that pool and a uniform sample inside it. Leave the unaudited pool as the suspect set and invert a hypergeometric test for its residual corruption budget.',
    'scope': 'Fixed binary coarsenings of real CIFAR-10N human annotations. Dataset reference labels emulate correct adjudication. One cached classifier; no newly conducted human review. All prespecified follow-up configurations are retained, including null and negative comparisons. This is exploratory evidence, not an independently replicated deployment claim.'
}


def run(cache_dir, data_dir, output):
    y, yt, human, P, Pt = load_cifar(cache_dir, data_dir)
    pool = np.arange(0, len(y), 5)
    rows = []
    for task, positive in PROTOCOL['tasks'].items():
        negative = [v for v in range(10) if v not in positive]
        pcal = np.c_[P[:, negative].sum(1), P[:, positive].sum(1)].astype(float)
        ptest = np.c_[Pt[:, negative].sum(1), Pt[:, positive].sum(1)].astype(float)
        pcal /= pcal.sum(1, keepdims=True); ptest /= ptest.sum(1, keepdims=True)
        yall = np.isin(y, positive).astype(int); test = np.isin(yt, positive).astype(int)
        noisy = np.isin(human['random_label1'], positive).astype(int)
        for seed in PROTOCOL['seeds']:
            rng = np.random.default_rng(PROTOCOL['calibration_seed_offset']+seed)
            idx = pool[rng.choice(len(pool), 2000, replace=False)]
            pc = pcal[idx]; labels = yall[idx]; obs = noisy[idx]
            order = np.argsort(pc.max(1), kind='stable')
            for size, r in PROTOCOL['uncertain_pool_and_uniform_review']:
                uncertain = order[:size]
                rng_review = np.random.default_rng(PROTOCOL['review_seed_offset']+seed)
                reviewed = rng_review.choice(uncertain, r, replace=False)
                flags = np.zeros(2000, bool); flags[uncertain] = True; flags[reviewed] = False
                observed = labels.copy(); observed[flags] = obs[flags]
                found = int((obs[reviewed] != labels[reviewed]).sum())
                budget = 2000-size+r
                assert int((~flags).sum()) == budget
                for target in PROTOCOL['coverage_targets']:
                    epsilon = 1-target; gamma = .1*epsilon; alpha = .9*epsilon
                    counts = np.arange(found, size+1)
                    valid = counts[hypergeom.cdf(found, size, counts, r) >= gamma]
                    m = int(valid[-1])-found
                    for score in PROTOCOL['scores']:
                        row = dict(task=task, score=score, seed=seed, n_cal=2000,
                                   uncertain_pool=size, pool_review=r, review_cost=budget,
                                   n_flags=int(flags.sum()), errors_found=found, m=m,
                                   remaining_errors=int((observed!=labels).sum()),
                                   budget_success=bool((observed!=labels).sum()<=m),
                                   target=target, alpha=alpha, gamma=gamma)
                        row.update(compare(pc, observed, flags, m, alpha, ptest, test, score, score_seed=20261022+seed))
                        rng_clean = np.random.default_rng(PROTOCOL['clean_audit_seed_offset']+seed)
                        clean_idx = rng_clean.choice(2000, budget, replace=False)
                        rng_score = np.random.default_rng(20261022+seed)
                        S = 1-pc if score=='residual' else aps_score_matrix(pc, rng_score.random(2000))
                        St = 1-ptest if score=='residual' else aps_score_matrix(ptest, rng_score.random(len(ptest)))
                        worst = S[np.arange(2000), observed].copy()
                        worst[flags] = S[flags].max(axis=1)
                        row['partial_label_same_alpha'] = metrics(St, test, naive_quantile(worst, alpha))
                        row['partial_label_matched_target'] = metrics(St, test, naive_quantile(worst, epsilon))
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
    parser.add_argument('--output', type=Path, default=Path('results/targeted.json'))
    args = parser.parse_args()
    args.output.with_suffix('.protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    run(args.cache_dir, args.data_dir, args.output)
