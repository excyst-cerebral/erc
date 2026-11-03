import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from method import (
    naive_quantile,
    oracle_quantile,
    uniform_shift_quantile,
    sharp_suspect_quantile,
    decomposed_quantile,
    known_T_inverted_quantile,
    flagger_low_confidence,
    predict_set,
    evaluate_sets,
    conformal_k,
    score_matrix,
    prob_residual_scores,
    poisson_binomial_budget,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
K_CLASSES = 10


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def train_model(model, X, y, epochs, lr=1e-3, batch=128, wd=0.0, augment=False, seed=0):
    seed_all(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.from_numpy(X).float().to(DEVICE)
    y = torch.from_numpy(y).long().to(DEVICE)
    n = len(X)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], y[idx]
            if augment:
                pad = F.pad(xb, (2, 2, 2, 2), mode="reflect")
                offs = torch.randint(0, 4, (len(xb), 2), device=DEVICE)
                xb = torch.stack([pad[j, :, offs[j, 0]:offs[j, 0] + 32,
                                     offs[j, 1]:offs[j, 1] + 32] for j in range(len(xb))])
                flip = torch.rand(len(xb), device=DEVICE) < 0.5
                xb[flip] = torch.flip(xb[flip], dims=[3])
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return time.time() - t0


@torch.no_grad()
def get_probs(model, X, batch=256):
    model.eval()
    X = torch.from_numpy(X).float().to(DEVICE)
    out = []
    for i in range(0, len(X), batch):
        out.append(F.softmax(model(X[i:i + batch]), dim=-1).cpu().numpy())
    return np.concatenate(out)


def load_mnist_like(kind):
    ds = torchvision.datasets.MNIST if kind == "mnist" else torchvision.datasets.FashionMNIST
    tf = T.ToTensor()
    tr = ds(DATA, train=True, download=True, transform=tf)
    te = ds(DATA, train=False, download=True, transform=tf)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy()
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy()
    return Xtr, ytr, Xte, yte


def load_cifar10n():


    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    Xtr = tr.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    ytr = np.array(tr.targets)
    Xte = te.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    yte = np.array(te.targets)

    lab = np.load(os.path.join(DATA, "cifar10n", "cifar-10-100n-main", "data",
                               "CIFAR-10_human_ordered.npy"), allow_pickle=True).item()
    noisy_keys = ["aggre_label", "worse_label", "random_label1",
                  "random_label2", "random_label3"]
    clean = np.array(lab["clean_label"]).reshape(-1)
    noisy = {k: np.array(lab[k]).reshape(-1) for k in noisy_keys}

    if not np.array_equal(clean, ytr):
        perm = np.asarray(np.load(os.path.join(DATA, "cifar10n", "cifar-10-100n-main",
                                               "image_order_c10.npy"))).reshape(-1)
        if np.array_equal(clean, ytr[perm]):
            Xtr, ytr = Xtr[perm], ytr[perm]
        elif np.array_equal(clean[perm], ytr):
            noisy = {k: v[perm] for k, v in noisy.items()}
            clean = clean[perm]
        else:
            raise AssertionError("CIFAR-10N label order does not match torchvision CIFAR-10")
    assert np.array_equal(clean, ytr), "CIFAR-10N order mismatch after alignment"
    return (Xtr, ytr, Xte, yte), noisy


ATTACK_CAPS = {0.1: 200, 0.2: 100}


def instance_dependent_corruption(y_clean, hardness, eta_bar, gamma, rng):
    h = (hardness - hardness.mean()) / (hardness.std() + 1e-8)
    w = np.clip(1.0 + gamma * h, 0.05, None)
    eta_i = np.clip(eta_bar * w / w.mean(), 0.0, 0.95)
    corrupt = rng.random(len(y_clean)) < eta_i
    y_tilde = y_clean.copy()
    for i in np.where(corrupt)[0]:
        choices = list(range(K_CLASSES))
        choices.remove(int(y_clean[i]))
        y_tilde[i] = rng.choice(choices)
    return y_tilde, corrupt, eta_i


def marginal_T(eta_bar, K=K_CLASSES):
    Tm = np.full((K, K), eta_bar / (K - 1))
    np.fill_diagonal(Tm, 1.0 - eta_bar)
    return Tm


def run_methods(S_cal, y_cal_tilde, probs_cal, S_test_scores_only, probs_test, y_test,
                flags, m, alpha, true_T=None, clean_scores_cal=None, m_rate=None):

    n = len(y_cal_tilde)
    out = {}
    noisy = S_cal[np.arange(n), y_cal_tilde]
    q_naive = naive_quantile(noisy, alpha)
    q_shift = uniform_shift_quantile(noisy, alpha, m)
    q_sharp = decomposed_quantile(S_cal, y_cal_tilde, flags, m, alpha)
    q_nosus = decomposed_quantile(S_cal, y_cal_tilde, np.ones(n, dtype=bool), m, alpha)
    out["naive"] = evaluate_sets(predict_set(probs_test, q_naive), y_test)
    out["uniform_shift"] = evaluate_sets(predict_set(probs_test, q_shift), y_test)
    out["sharp_suspect"] = evaluate_sets(predict_set(probs_test, q_sharp), y_test)
    out["sharp_no_suspect"] = evaluate_sets(predict_set(probs_test, q_nosus), y_test)
    if clean_scores_cal is not None:
        q_or = oracle_quantile(clean_scores_cal, alpha)
        out["oracle_clean"] = evaluate_sets(predict_set(probs_test, q_or), y_test)
    if true_T is not None:
        q_nt = known_T_inverted_quantile(S_cal, y_cal_tilde, true_T, alpha)
        out["known_T"] = evaluate_sets(predict_set(probs_test, q_nt), y_test)
    if m_rate is not None:
        q_shift_r = uniform_shift_quantile(noisy, alpha, m_rate)
        q_sharp_r = decomposed_quantile(S_cal, y_cal_tilde, flags, m_rate, alpha)
        out["uniform_shift_rate"] = evaluate_sets(predict_set(probs_test, q_shift_r), y_test)
        out["sharp_rate"] = evaluate_sets(predict_set(probs_test, q_sharp_r), y_test)
    return out


def exp0_knownT_control(cache, seeds=5):


    rows = []
    for kind in ["mnist", "fmnist"]:
        probs_tr, probs_cal_pool, probs_te, y_tr, y_cal_pool, y_te, X_cal_pool = cache[kind]
        for eta in [0.1, 0.2, 0.4]:
            for alpha in [0.05, 0.1, 0.2]:
                for seed in range(seeds):
                    rng = np.random.default_rng(2024 + seed)
                    n_cal = 2000
                    idx = rng.choice(len(y_cal_pool), n_cal, replace=False)
                    y_cal_true = y_cal_pool[idx]
                    Pc = probs_cal_pool[idx]
                    flip = rng.random(n_cal) < eta
                    y_tilde = y_cal_true.copy()
                    for i in np.where(flip)[0]:
                        choices = [y for y in range(K_CLASSES) if y != y_cal_true[i]]
                        y_tilde[i] = rng.choice(choices)
                    S_cal = score_matrix(Pc)
                    clean_scores = prob_residual_scores(Pc, y_cal_true)
                    res = run_methods(
                        S_cal, y_tilde, Pc, None, probs_te, y_te,
                        flags=np.ones(n_cal, dtype=bool), m=int(flip.sum()),
                        alpha=alpha, true_T=marginal_T(eta),
                        clean_scores_cal=clean_scores)
                    row = {"dataset": kind, "eta": eta, "alpha": alpha, "seed": seed,
                           "realized_noise": float(flip.mean())}
                    row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                    row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                    rows.append(row)
    return rows


def exp1_controlled(cache, seeds=5):
    rows = []
    for kind in ["mnist", "fmnist"]:
        probs_tr, probs_cal_pool, probs_te, y_tr, y_cal_pool, y_te, X_cal_pool = cache[kind]
        n_test = len(y_te)
        for eta_bar in [0.1, 0.2, 0.4]:
            for n_cal in [500, 1000, 2000]:
                for alpha in [0.05, 0.1, 0.2]:
                    for seed in range(seeds):
                        rng = np.random.default_rng(1000 * seed + 17)
                        idx = rng.choice(len(y_cal_pool), n_cal, replace=False)
                        y_cal_true = y_cal_pool[idx]
                        Pc = probs_cal_pool[idx]
                        hardness = 1.0 - Pc[np.arange(n_cal), y_cal_true]
                        y_tilde, corrupt, eta_i = instance_dependent_corruption(
                            y_cal_true, hardness, eta_bar, gamma=1.5, rng=rng)
                        S_cal = score_matrix(Pc)
                        clean_scores = prob_residual_scores(Pc, y_cal_true)
                        flags = flagger_low_confidence(Pc, y_tilde, budget_frac=eta_bar)
                        m = int(math.ceil(eta_bar * n_cal))


                        m_rate = poisson_binomial_budget(eta_i, delta=0.05)
                        Tm = marginal_T(eta_bar)
                        res = run_methods(
                            S_cal, y_tilde, Pc, None, probs_te, y_te,
                            flags=flags, m=m, alpha=alpha, true_T=Tm,
                            clean_scores_cal=clean_scores, m_rate=m_rate)

                        flags_or = corrupt.copy()
                        res["sharp_oracle_flags"] = evaluate_sets(
                            predict_set(probs_te, sharp_suspect_quantile(
                                S_cal, y_tilde, flags_or, int(corrupt.sum()), alpha)), y_te)
                        row = {"dataset": kind, "eta_bar": eta_bar, "n_cal": n_cal,
                               "alpha": alpha, "seed": seed,
                               "realized_noise": float(corrupt.mean()),
                               "flag_recall": float(flags[corrupt].mean()) if corrupt.any() else 1.0,
                               "flag_frac": float(flags.mean()),
                               "m": m, "m_rate": m_rate,
                               "n_corrupt": int(corrupt.sum()),
                               "premise_hold": bool(flags[corrupt].all()
                                                    and corrupt.sum() <= m)}
                        row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                        row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                        rows.append(row)
    return rows


def exp2_cifar10n(cifar_cache, noisy, probs_cal_all, probs_te_all, seeds=3):
    (Xtr, ytr, Xte, yte), _ = cifar_cache
    rows = []
    model_tr = [i for i in range(len(ytr)) if i % 5 != 0]
    cal_pool = [i for i in range(len(ytr)) if i % 5 == 0]
    for noisy_name in ["aggre_label", "random_label1", "worse_label"]:
        y_noisy_full = noisy[noisy_name]
        for n_cal in [1000, 2000, 5000]:
            for budget_frac in [0.1, 0.2, 0.4]:
                for alpha in [0.05, 0.1, 0.2]:
                    for seed in range(seeds):
                        rng = np.random.default_rng(777 + seed)
                        idx = np.array(cal_pool)[rng.choice(len(cal_pool), n_cal, replace=False)]
                        y_cal_true = ytr[idx]
                        y_tilde = y_noisy_full[idx]
                        Pc = probs_cal_all[idx]
                        corrupt = y_tilde != y_cal_true
                        S_cal = score_matrix(Pc)
                        clean_scores = prob_residual_scores(Pc, y_cal_true)
                        flags = flagger_low_confidence(Pc, y_tilde, budget_frac=budget_frac)
                        m = int(math.ceil(budget_frac * n_cal))
                        res = run_methods(
                            S_cal, y_tilde, Pc, None, probs_te_all, yte,
                            flags=flags, m=m, alpha=alpha, true_T=None,
                            clean_scores_cal=clean_scores)
                        flags_or = corrupt.copy()
                        res["sharp_oracle_flags"] = evaluate_sets(
                            predict_set(probs_te_all, sharp_suspect_quantile(
                                S_cal, y_tilde, flags_or, int(corrupt.sum()), alpha)), yte)
                        row = {"dataset": "cifar10n", "noisy": noisy_name, "n_cal": n_cal,
                               "budget_frac": budget_frac, "alpha": alpha, "seed": seed,
                               "realized_noise": float(corrupt.mean()),
                               "flag_recall": float(flags[corrupt].mean()) if corrupt.any() else 1.0,
                               "flag_precision": float(corrupt[flags].mean()) if flags.any() else 0.0,
                               "flag_frac": float(flags.mean()),
                               "n_corrupt": int(corrupt.sum()),
                               "premise_hold": bool(flags[corrupt].all()
                                                    and corrupt.sum() <= m)}
                        row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                        row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                        rows.append(row)
    return rows


def exp3_adversarial(cache, seeds=5):


    rows = []
    kind = "fmnist"
    probs_tr, probs_cal_pool, probs_te, y_tr, y_cal_pool, y_te, X_cal_pool = cache[kind]
    for eta_bar in [0.1, 0.2]:
        for alpha in [0.1, 0.2]:
            for seed in range(seeds):
                rng = np.random.default_rng(31 + seed)
                n_cal = 2000
                idx = rng.choice(len(y_cal_pool), n_cal, replace=False)
                y_cal_true = y_cal_pool[idx]
                Pc = probs_cal_pool[idx]
                S_cal = score_matrix(Pc)
                clean_scores = prob_residual_scores(Pc, y_cal_true)
                m = int(math.ceil(eta_bar * n_cal))
                m_cap = ATTACK_CAPS[alpha]


                kq = conformal_k(n_cal, alpha)
                q_ref = float(np.sort(clean_scores)[kq - 1])
                pred = Pc.argmax(1)
                mis = np.where(pred != y_cal_true)[0]
                defl = 1.0 - Pc[np.arange(n_cal), pred]
                cross = [int(i) for i in mis
                         if clean_scores[i] > q_ref and defl[i] <= q_ref]
                cross.sort(key=lambda i: defl[i])
                atk = cross[:m_cap]
                y_tilde = y_cal_true.copy()
                for i in atk:
                    y_tilde[i] = int(pred[i])
                corrupt = y_tilde != y_cal_true
                assert int(corrupt.sum()) <= m_cap, "attack exceeded predeclared cap"
                flags_pred = flagger_low_confidence(Pc, y_tilde, budget_frac=eta_bar)
                res = run_methods(
                    S_cal, y_tilde, Pc, None, probs_te, y_te,
                    flags=flags_pred, m=m, alpha=alpha, true_T=marginal_T(eta_bar),
                    clean_scores_cal=clean_scores)
                res["sharp_oracle_flags"] = evaluate_sets(
                    predict_set(probs_te, sharp_suspect_quantile(
                        S_cal, y_tilde, corrupt.copy(), int(corrupt.sum()), alpha)), y_te)
                row = {"eta_bar": eta_bar, "alpha": alpha, "seed": seed,
                       "realized_noise": float(corrupt.mean()),
                       "n_crossings": len(atk),
                       "attack_cap": m_cap,
                       "defense_budget": m,

                       "crossing_heuristic_cov": (kq - len(atk)) / (n_cal + 1),


                       "rank_bound_low": (kq - int(corrupt.sum())) / (n_cal + 1),
                       "rank_bound_high": kq / (n_cal + 1),


                       "rank_bound_low_fixed": (kq - m_cap) / (n_cal + 1),
                       "rank_bound_high_fixed": kq / (n_cal + 1),
                       "flag_recall": float(flags_pred[corrupt].mean()) if corrupt.any() else 1.0,
                       "premise_hold": bool(flags_pred[corrupt].all()
                                            and corrupt.sum() <= m)}
                row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                rows.append(row)
    return rows


def exp4_flag_recall(cache, seeds=5):


    rows = []
    kind = "fmnist"
    probs_tr, probs_cal_pool, probs_te, y_tr, y_cal_pool, y_te, X_cal_pool = cache[kind]
    alpha = 0.2
    eta_bar = 0.1
    for recall in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for seed in range(seeds):
            rng = np.random.default_rng(91 + seed)
            n_cal = 2000
            idx = rng.choice(len(y_cal_pool), n_cal, replace=False)
            y_cal_true = y_cal_pool[idx]
            Pc = probs_cal_pool[idx]
            S_cal = score_matrix(Pc)
            clean_scores = prob_residual_scores(Pc, y_cal_true)
            m_cap = ATTACK_CAPS[alpha]
            kq = conformal_k(n_cal, alpha)
            q_ref = float(np.sort(clean_scores)[kq - 1])
            pred = Pc.argmax(1)
            mis = np.where(pred != y_cal_true)[0]
            defl = 1.0 - Pc[np.arange(n_cal), pred]
            cross = [int(i) for i in mis
                     if clean_scores[i] > q_ref and defl[i] <= q_ref]
            cross.sort(key=lambda i: defl[i])
            atk = cross[:m_cap]
            y_tilde = y_cal_true.copy()
            for i in atk:
                y_tilde[i] = int(pred[i])
            corrupt = y_tilde != y_cal_true
            assert int(corrupt.sum()) <= m_cap, "attack exceeded predeclared cap"

            m = int(corrupt.sum())
            corrupted_idx = np.where(corrupt)[0]
            n_keep = int(round(recall * len(corrupted_idx)))
            keep = corrupted_idx[rng.permutation(len(corrupted_idx))[:n_keep]]
            flags = np.zeros(n_cal, dtype=bool)
            flags[keep] = True
            res = run_methods(
                S_cal, y_tilde, Pc, None, probs_te, y_te,
                flags=flags, m=m, alpha=alpha, true_T=None,
                clean_scores_cal=clean_scores)
            row = {"recall_target": recall, "alpha": alpha, "seed": seed,
                   "realized_noise": float(corrupt.mean()),
                   "n_crossings": len(atk),
                   "crossing_heuristic_cov": (kq - len(atk)) / (n_cal + 1),
                   "rank_bound_low": (kq - int(corrupt.sum())) / (n_cal + 1),
                   "rank_bound_high": kq / (n_cal + 1),
                   "flag_recall_realized": float(flags[corrupt].mean()) if corrupt.any() else 1.0,
                   "flag_frac": float(flags.mean())}
            row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
            row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
            rows.append(row)
    return rows


def exp5_sharpness(cache, trials=200):


    kind = "mnist"
    probs_tr, probs_cal_pool, probs_te, y_tr, y_cal_pool, y_te, X_cal_pool = cache[kind]
    rng = np.random.default_rng(5)
    n_cal = 400
    idx = rng.choice(len(y_cal_pool), n_cal, replace=False)
    y_cal_true = y_cal_pool[idx]
    Pc = probs_cal_pool[idx]
    hardness = 1.0 - Pc[np.arange(n_cal), y_cal_true]
    y_tilde, corrupt, _ = instance_dependent_corruption(
        y_cal_true, hardness, eta_bar=0.2, gamma=1.5, rng=rng)
    S_cal = score_matrix(Pc)
    flags = flagger_low_confidence(Pc, y_tilde, budget_frac=0.2)
    m = 80
    n_ok_L, n_ok_U = 0, 0
    for _ in range(trials):
        q = float(rng.uniform(0.0, 0.6))
        good = S_cal[np.arange(n_cal), y_tilde] <= q
        alt_max = np.full(n_cal, -np.inf)
        alt_min = np.full(n_cal, np.inf)
        for y in range(K_CLASSES):
            mask = y_tilde != y
            alt_max[mask] = np.maximum(alt_max[mask], S_cal[mask, y])
            alt_min[mask] = np.minimum(alt_min[mask], S_cal[mask, y])
        flippable_bad = flags & good & (alt_max > q)
        flippable_good = flags & (~good) & (alt_min <= q)
        Gt = int(good.sum())
        L = Gt - min(m, int(flippable_bad.sum()))
        U = Gt + min(m, int(flippable_good.sum()))

        y_adv = y_tilde.copy()
        n_flip = min(m, int(flippable_bad.sum()))
        flip_set = np.where(flippable_bad)[0][:n_flip]
        for i in flip_set:
            cand = [y for y in range(K_CLASSES) if y != y_tilde[i] and S_cal[i, y] > q]
            y_adv[i] = cand[0]
        G_adv = int((S_cal[np.arange(n_cal), y_adv] <= q).sum())
        if G_adv == L:
            n_ok_L += 1

        y_adv2 = y_tilde.copy()
        n_flip2 = min(m, int(flippable_good.sum()))
        flip_set2 = np.where(flippable_good)[0][:n_flip2]
        for i in flip_set2:
            cand = [y for y in range(K_CLASSES) if y != y_tilde[i] and S_cal[i, y] <= q]
            y_adv2[i] = cand[0]
        G_adv2 = int((S_cal[np.arange(n_cal), y_adv2] <= q).sum())
        if G_adv2 == U:
            n_ok_U += 1
    return {"trials": trials, "lower_attained": n_ok_L, "upper_attained": n_ok_U}


def exp8_annotation_audit(cifar_cache, noisy, probs_cal_all, probs_te_all, seeds=3):


    rows = []
    (Xtr, ytr, Xte, yte), _ = cifar_cache
    cal_pool = [i for i in range(len(ytr)) if i % 5 == 0]
    a_full = {k: noisy[k] for k in ["random_label1", "random_label2", "random_label3"]}
    for n_cal in [2000]:


        for alpha in [0.05, 0.1, 0.2]:
            for seed in range(seeds):
                rng = np.random.default_rng(777 + seed)
                idx = np.array(cal_pool)[rng.choice(len(cal_pool), n_cal, replace=False)]
                y_cal_true = ytr[idx]
                obs = a_full["random_label1"][idx]
                a2 = a_full["random_label2"][idx]
                a3 = a_full["random_label3"][idx]
                Pc = probs_cal_all[idx]
                S_cal = score_matrix(Pc)
                clean_scores = prob_residual_scores(Pc, y_cal_true)
                err = obs != y_cal_true
                variants = {
                    "audit_1pass": ((a2 != obs),
                                    int(((a2 != obs) & (a3 == a2)).sum())),
                    "audit_2pass": (((a2 != obs) | (a3 != obs)),
                                    int(((a2 == a3) & (a2 != obs)).sum())),
                }


                from scipy.stats import hypergeom as hypergeom_dist
                F1 = (a2 != obs) | (a3 != obs)
                obs_adj = obs.copy()
                obs_adj[F1] = y_cal_true[F1]
                cons_idx = np.where(~F1)[0]
                rng2 = np.random.default_rng(1200 + seed)
                r_rev = min(200, len(cons_idx))
                review_idx = rng2.choice(cons_idx, r_rev, replace=False)
                k_found = int((obs_adj[review_idx] != y_cal_true[review_idx]).sum())
                obs_adj[review_idx] = y_cal_true[review_idx]
                gamma = 0.05
                N_cons = int(len(cons_idx))
                K_up = int(k_found)
                for K in range(int(k_found), N_cons + 1):
                    if hypergeom_dist.cdf(k_found, N_cons, K, r_rev) >= gamma:
                        K_up = K
                    else:
                        break
                unaudited = np.setdiff1d(cons_idx, review_idx)
                m_cons = int(max(0, K_up - k_found))
                flags_p = np.zeros(n_cal, dtype=bool)
                flags_p[unaudited] = True
                err_adj = obs_adj != y_cal_true
                variants["audit_pipeline"] = (flags_p, m_cons)
                for name, (flags, m) in variants.items():
                    obs_use = obs_adj if name == "audit_pipeline" else obs
                    err_use = err_adj if name == "audit_pipeline" else err
                    res = run_methods(
                        S_cal, obs_use, Pc, None, probs_te_all, yte,
                        flags=flags, m=m, alpha=alpha, true_T=None,
                        clean_scores_cal=clean_scores)
                    row = {"audit": name, "n_cal": n_cal, "alpha": alpha,
                           "seed": seed,
                           "realized_noise": float(err_use.mean()),
                           "n_flags": int(flags.sum()), "m": m,
                           "n_corrupt": int(err_use.sum()),
                           "flag_recall": float(flags[err_use].mean()) if err_use.any() else 1.0,
                           "flag_precision": float(err_use[flags].mean()) if flags.any() else 0.0,
                           "premise_hold": bool(flags[err_use].all() and err_use.sum() <= m)}
                    if name == "audit_pipeline":
                        row.update({"review_size": int(r_rev),
                                    "errors_found_by_review": k_found,
                                    "K_up": K_up,
                                    "residual_errors": int(err_adj.sum())})
                    row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                    row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                    rows.append(row)


                aud_idx = np.union1d(np.where(F1)[0], review_idx)
                res_aud = run_methods(
                    S_cal[aud_idx], obs_adj[aud_idx], Pc[aud_idx], None,
                    probs_te_all, yte,
                    flags=np.zeros(len(aud_idx), dtype=bool), m=0, alpha=alpha,
                    true_T=None, clean_scores_cal=clean_scores[aud_idx])
                row = {"audit": "audited_only_cp", "n_cal": int(len(aud_idx)),
                       "alpha": alpha, "seed": seed,
                       "realized_noise": float(err_adj[aud_idx].mean()),
                       "n_flags": 0, "m": 0,
                       "n_corrupt": int(err_adj[aud_idx].sum()),
                       "flag_recall": 1.0, "flag_precision": 1.0,
                       "premise_hold": True,
                       "review_size": int(r_rev),
                       "errors_found_by_review": k_found}
                row.update({f"{k1}_coverage": v["coverage"] for k1, v in res_aud.items()})
                row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res_aud.items()})
                rows.append(row)


                rng3 = np.random.default_rng(3300 + seed)
                rnd_idx = np.sort(rng3.choice(n_cal, len(aud_idx), replace=False))
                obs_rnd = obs.copy()
                err_before = obs_rnd != y_cal_true
                obs_rnd[rnd_idx] = y_cal_true[rnd_idx]
                res_rnd = run_methods(
                    S_cal[rnd_idx], obs_rnd[rnd_idx], Pc[rnd_idx], None,
                    probs_te_all, yte,
                    flags=np.zeros(len(rnd_idx), dtype=bool), m=0, alpha=alpha,
                    true_T=None, clean_scores_cal=clean_scores[rnd_idx])
                row = {"audit": "random_audit_cp", "n_cal": int(len(rnd_idx)),
                       "alpha": alpha, "seed": seed,
                       "realized_noise": float(err_before[rnd_idx].mean()),
                       "n_flags": 0, "m": 0,
                       "n_corrupt": int((obs_rnd[rnd_idx] != y_cal_true[rnd_idx]).sum()),
                       "flag_recall": 1.0, "flag_precision": 1.0,
                       "premise_hold": True,
                       "review_size": int(len(rnd_idx)),
                       "errors_found_by_review": int(err_before[rnd_idx].sum())}
                row.update({f"{k1}_coverage": v["coverage"] for k1, v in res_rnd.items()})
                row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res_rnd.items()})
                rows.append(row)


                r_fix = 1000
                rng4 = np.random.default_rng(4400 + seed)
                ridx_fix = np.sort(rng4.choice(n_cal, r_fix, replace=False))
                obs_fix = obs.copy()
                err_b2 = obs_fix != y_cal_true
                obs_fix[ridx_fix] = y_cal_true[ridx_fix]
                res_fix = run_methods(
                    S_cal[ridx_fix], obs_fix[ridx_fix], Pc[ridx_fix], None,
                    probs_te_all, yte,
                    flags=np.zeros(len(ridx_fix), dtype=bool), m=0, alpha=alpha,
                    true_T=None, clean_scores_cal=clean_scores[ridx_fix])
                row = {"audit": "random_audit_fixed_cp", "n_cal": int(r_fix),
                       "alpha": alpha, "seed": seed,
                       "realized_noise": float(err_b2[ridx_fix].mean()),
                       "n_flags": 0, "m": 0,
                       "n_corrupt": int((obs_fix[ridx_fix] != y_cal_true[ridx_fix]).sum()),
                       "flag_recall": 1.0, "flag_precision": 1.0,
                       "premise_hold": True,
                       "review_size": int(r_fix),
                       "errors_found_by_review": int(err_b2[ridx_fix].sum()),
                       "review_count_predeclared": True}
                row.update({f"{k1}_coverage": v["coverage"] for k1, v in res_fix.items()})
                row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res_fix.items()})
                rows.append(row)

                for bf in [0.2]:
                    flags_c = flagger_low_confidence(Pc, obs, budget_frac=bf)
                    m_c = int(math.ceil(bf * n_cal))
                    res = run_methods(
                        S_cal, obs, Pc, None, probs_te_all, yte,
                        flags=flags_c, m=m_c, alpha=alpha, true_T=None,
                        clean_scores_cal=clean_scores)
                    row = {"audit": f"confidence_{bf}", "n_cal": n_cal,
                           "alpha": alpha, "seed": seed,
                           "realized_noise": float(err.mean()),
                           "n_flags": int(flags_c.sum()), "m": m_c,
                           "n_corrupt": int(err.sum()),
                           "flag_recall": float(flags_c[err].mean()) if err.any() else 1.0,
                           "flag_precision": float(err[flags_c].mean()) if flags_c.any() else 0.0,
                           "premise_hold": bool(flags_c[err].all() and err.sum() <= m_c)}
                    row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                    row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                    rows.append(row)
                flags_or = err.copy()
                res = run_methods(
                    S_cal, obs, Pc, None, probs_te_all, yte,
                    flags=flags_or, m=int(err.sum()), alpha=alpha, true_T=None,
                    clean_scores_cal=clean_scores)
                row = {"audit": "oracle_flags", "n_cal": n_cal, "alpha": alpha,
                       "seed": seed, "realized_noise": float(err.mean()),
                       "n_flags": int(flags_or.sum()), "m": int(err.sum()),
                       "n_corrupt": int(err.sum()), "flag_recall": 1.0,
                       "flag_precision": 1.0, "premise_hold": True}
                row.update({f"{k1}_coverage": v["coverage"] for k1, v in res.items()})
                row.update({f"{k1}_size": v["avg_set_size"] for k1, v in res.items()})
                rows.append(row)
    return rows


def build_cache():
    cache = {}
    for kind in ["mnist", "fmnist"]:
        f = os.path.join(RESULTS, f"cache_{kind}.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            cache[kind] = (z["probs_tr"], z["probs_cal_pool"], z["probs_te"],
                           z["y_tr"], z["y_cal_pool"], z["y_te"], z["X_cal_pool"])
            print(f"[cache] {kind} loaded; test acc = "
                  f"{(z['probs_te'].argmax(1) == z['y_te']).mean():.4f}")
            continue
        Xtr, ytr, Xte, yte = load_mnist_like(kind)
        seed_all(0)
        model = MLP()


        t = train_model(model, Xtr[:50000], ytr[:50000],
                        epochs=8 if kind == "mnist" else 20,
                        lr=1e-3, seed=0)
        probs_tr = get_probs(model, Xtr)
        probs_te = get_probs(model, Xte)
        acc = float((probs_te.argmax(1) == yte).mean())
        print(f"[{kind}] trained in {t:.0f}s, test acc = {acc:.4f}")

        np.savez_compressed(f, probs_tr=probs_tr[:50000], probs_cal_pool=probs_tr[50000:],
                            probs_te=probs_te, y_tr=ytr[:50000],
                            y_cal_pool=ytr[50000:], y_te=yte,
                            X_cal_pool=Xtr[50000:60000].reshape(-1, 784)[:10000])
        z = np.load(f, allow_pickle=True)
        cache[kind] = (z["probs_tr"], z["probs_cal_pool"], z["probs_te"],
                       z["y_tr"], z["y_cal_pool"], z["y_te"], z["X_cal_pool"])
    return cache


def main(stages):
    t_start = time.time()
    out = {}
    prev = os.path.join(RESULTS, "summary.json")
    if os.path.exists(prev):
        with open(prev) as fjs:
            out = json.load(fjs)

    cache = None
    if stages & {"e0", "e1", "e3", "e4", "e5"}:
        print("== building model caches (real training) ==")
        cache = build_cache()

    if stages & {"e2", "e8"}:
        print("== CIFAR-10N ==")
        cifar_cache = load_cifar10n()
        (Xtr, ytr, Xte, yte), noisy = cifar_cache
        f = os.path.join(RESULTS, "cache_cifar10.npz")
        if os.path.exists(f):
            z = np.load(f)
            probs_cache_cifar, probs_cache_cifar_te = z["probs_tr"], z["probs_te"]
            print(f"[cache] cifar10 loaded; test acc = "
                  f"{(probs_cache_cifar_te.argmax(1) == yte).mean():.4f}")
        else:
            seed_all(0)
            model = SmallCNN()


            tr_idx = np.array([i for i in range(len(ytr)) if i % 5 != 0])
            t = train_model(model, Xtr[tr_idx], ytr[tr_idx], epochs=35, lr=1e-3,
                            wd=5e-4, augment=True, seed=0)
            probs_cache_cifar = get_probs(model, Xtr)
            probs_cache_cifar_te = get_probs(model, Xte)
            acc = float((probs_cache_cifar_te.argmax(1) == yte).mean())
            print(f"[cifar10] trained in {t:.0f}s, test acc = {acc:.4f}")
            np.savez_compressed(f, probs_tr=probs_cache_cifar,
                                probs_te=probs_cache_cifar_te)
        if "e2" in stages:
            print("== E2 CIFAR-10N real human noise ==")
            out["exp2_cifar10n"] = exp2_cifar10n(cifar_cache, noisy,
                                                 probs_cache_cifar, probs_cache_cifar_te)
        if "e8" in stages:
            print("== E8 annotation-audit suspect sets ==")
            out["exp8_annotation_audit"] = exp8_annotation_audit(
                cifar_cache, noisy, probs_cache_cifar, probs_cache_cifar_te)

    if "e0" in stages:
        print("== E0 known-T class-conditional control ==")
        out["exp0_knownT_control"] = exp0_knownT_control(cache)
    if "e1" in stages:
        print("== E1 controlled instance-dependent noise ==")
        out["exp1_controlled"] = exp1_controlled(cache)
    if "e3" in stages:
        print("== E3 adversarial stress ==")
        out["exp3_adversarial"] = exp3_adversarial(cache)
    if "e4" in stages:
        print("== E4 flag recall ablation ==")
        out["exp4_flag_recall"] = exp4_flag_recall(cache)
    if "e5" in stages:
        print("== E5 sharpness ==")
        out["exp5_sharpness"] = exp5_sharpness(cache)

    out["_meta"] = {"device": DEVICE, "torch": torch.__version__,
                    "elapsed_sec": time.time() - t_start,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(RESULTS, "summary.json"), "w") as fjs:
        json.dump(out, fjs, indent=1)
    print(f"stage(s) {sorted(stages)} done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    import sys
    main(set(sys.argv[1:]) or {"e1", "e2", "e3", "e4", "e5"})
