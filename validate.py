import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
import solvers as reference
from method import (sharp_suspect_quantile, uniform_shift_quantile,
                    naive_quantile, score_matrix, evaluate_sets, predict_set)


def generated(rng, n, classes, flag_fraction, budget_fraction, tied=False):
    logits = rng.normal(size=(n, classes))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    scores = 1 - probs
    if tied:
        scores = np.round(scores, 1)
    labels = probs.argmax(axis=1)
    flags = rng.random(n) < flag_fraction
    budget = min(int(flags.sum()), max(0, int(n * budget_fraction)))
    return scores, labels, flags, budget


def certified_threshold(*args):
    calls = []
    original = reference.milp

    def record(**kwargs):
        result = original(**kwargs)
        calls.append({"status": int(result.status), "success": bool(result.success),
                      "mip_gap": float(result.mip_gap) if
                      getattr(result, "mip_gap", None) is not None else None})
        return result

    with patch.object(reference, "milp", side_effect=record):
        threshold = reference.milp_threshold(*args, time_limit=10)
    return threshold, calls


def exactness():
    rng = np.random.default_rng(20261020)
    brute, milp = [], []
    for trial in range(200):
        n, classes = int(rng.integers(2, 8)), int(rng.integers(2, 5))
        args = generated(rng, n, classes, .5 if trial % 2 else 1, .3,
                         tied=bool(trial % 3 == 0))
        alpha = .2 if trial % 2 else .5
        expected = reference.brute_force_threshold(*args, alpha)
        actual = sharp_suspect_quantile(*args, alpha)
        assert expected == actual
        brute.append({"trial": trial, "n": n, "K": classes,
                      "tied": trial % 3 == 0, "q": actual, "agree": True})
    for trial in range(100):
        n, classes = int(rng.integers(15, 61)), int(rng.integers(2, 7))
        args = generated(rng, n, classes, .5 if trial % 2 else 1, .05,
                         tied=bool(trial % 3 == 0))
        alpha = .1 if trial % 2 else .2
        actual = sharp_suspect_quantile(*args, alpha)
        expected, calls = certified_threshold(*args, alpha)
        assert expected is not None and expected == actual
        assert all(c["success"] and c["status"] == 0 for c in calls)
        milp.append({"trial": trial, "n": n, "K": classes, "q": actual,
                     "agree": True, "solver_calls": calls})
    return {"bruteforce": brute, "milp": milp}


def scaling():
    rng = np.random.default_rng(20261021)
    reference.milp_threshold(*generated(rng, 20, 3, .5, .05), .1)
    rows = []
    for fraction, budget_fraction in [(.1, .01), (.5, .05)]:
        for n in [100, 1000, 10000, 100000]:
            for rep in range(3):
                args = generated(rng, n, 10, fraction, budget_fraction)
                start = time.perf_counter()
                q = sharp_suspect_quantile(*args, .1)
                row = {"n": n, "K": 10, "flag_fraction": fraction,
                       "budget_fraction": budget_fraction, "rep": rep,
                       "sweep_s": time.perf_counter() - start, "q": q,
                       "mean_calibration_set_size": float((args[0] <= q).sum(1).mean())}
                if n <= 1000:
                    start = time.perf_counter()
                    qm, calls = certified_threshold(*args, .1)
                    row.update(milp_s=time.perf_counter() - start,
                               solver_calls=calls, q_milp=qm, agree=qm == q)
                    assert qm is not None and qm == q
                rows.append(row)
                print("scaling", n, fraction, rep, round(row['sweep_s'], 4), flush=True)
    return rows


def cached_experiments(cache_dir, data_dir):


    import pickle
    import experiments as experiments
    with open(data_dir / 'cifar-10-batches-py/test_batch', 'rb') as f:
        test_labels = np.array(pickle.load(f, encoding='bytes')[b'labels'])
    train_labels = []
    for batch in range(1, 6):
        with open(data_dir / f'cifar-10-batches-py/data_batch_{batch}', 'rb') as f:
            train_labels.extend(pickle.load(f, encoding='bytes')[b'labels'])
    train_labels = np.array(train_labels)
    base = data_dir / 'cifar10n/cifar-10-100n-main'
    human = np.load(base / 'data/CIFAR-10_human_ordered.npy', allow_pickle=True).item()
    clean = np.asarray(human['clean_label']).reshape(-1)
    keys = ['random_label1', 'random_label2', 'random_label3']
    noisy = {k: np.asarray(human[k]).reshape(-1) for k in keys}
    if not np.array_equal(clean, train_labels):
        perm = np.load(base / 'image_order_c10.npy').reshape(-1)
        if np.array_equal(clean, train_labels[perm]):
            train_labels = train_labels[perm]
        elif np.array_equal(clean[perm], train_labels):
            noisy = {k: v[perm] for k, v in noisy.items()}
            clean = clean[perm]
        else:
            raise ValueError('CIFAR-10N ordering cannot be verified')
    assert np.array_equal(clean, train_labels)
    z = np.load(cache_dir / 'cache_cifar10.npz')
    audit = experiments.exp8_annotation_audit(
        ((None, train_labels, None, test_labels), noisy), noisy,
        z['probs_tr'], z['probs_te'], seeds=30)


    channel = []
    for dataset in ['mnist', 'fmnist']:
        z = np.load(cache_dir / f'cache_{dataset}.npz')
        for seed in range(30):
            rng = np.random.default_rng(5500 + seed)
            idx = rng.choice(len(z['y_cal_pool']), 2000, replace=False)
            probs = z['probs_cal_pool'][idx]
            labels = z['y_cal_pool'][idx].copy()
            flags = np.zeros(2000, bool)
            flags[rng.choice(2000, 100, replace=False)] = True
            observed = labels.copy()
            observed[flags] = probs[flags].argmin(1)
            scores = score_matrix(probs)
            a = scores[np.arange(2000), observed]
            errors = observed != labels
            assert not np.any(errors & ~flags) and errors.sum() <= 100
            row = {"dataset": dataset, "seed": seed, "n_cal": 2000,
                   "m": 100, "n_flags": 100, "n_corrupt": int(errors.sum()),
                   "premise_hold": True, "alpha": .1}
            thresholds = {'sharp': sharp_suspect_quantile(scores, observed, flags, 100, .1),
                          'shift': uniform_shift_quantile(a, .1, 100),
                          'naive': naive_quantile(a, .1)}
            for name, q in thresholds.items():
                metrics = evaluate_sets(predict_set(z['probs_te'], q), z['y_te'])
                row[name] = dict(q=q, **metrics)
            channel.append(row)
    return audit, channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/validate.json'))
    args = parser.parse_args()
    out = {"protocol": "Fixed seeds and configurations in validate.py",
           "environment": {"python": platform.python_version(),
                           "machine": platform.machine(), "numpy": np.__version__,
                           "scipy": scipy.__version__},
           "generated_exactness": exactness(), "generated_scaling": scaling()}
    if args.cache_dir and args.data_dir:
        out['audit_30seeds'], out['controlled_flagged_channel'] = cached_experiments(
            args.cache_dir, args.data_dir)
        inputs = {name: args.cache_dir / name for name in
                  ['cache_mnist.npz', 'cache_fmnist.npz', 'cache_cifar10.npz']}
        inputs['CIFAR-10_human_ordered.npy'] = (args.data_dir /
            'cifar10n/cifar-10-100n-main/data/CIFAR-10_human_ordered.npy')
        out['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in inputs.items()}
        out['provenance_notes'] = (
            'Cached classifiers are reused; original initialization seeds are not '
            'recorded in cache metadata. This run adds calibration/review '
            'repetitions, not independent classifier fits.')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
