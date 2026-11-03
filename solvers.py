import json
import math
import os
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from method import conformal_k, score_matrix, sharp_suspect_quantile

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
RNG_SEED = 20261016


def brute_force_threshold(S, y_tilde, flags, m, alpha):


    n, K = S.shape
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    F = np.where(flags)[0]
    assign = y_tilde.copy()
    rows = np.arange(n)
    best = -math.inf

    def rec(j, changed):
        nonlocal best
        if j == len(F):
            sc = S[rows, assign]
            t = float(np.sort(sc)[k - 1])
            if t > best:
                best = t
            return
        rec(j + 1, changed)
        if changed < m:
            i = F[j]
            for y in range(K):
                if y != y_tilde[i]:
                    assign[i] = y
                    rec(j + 1, changed + 1)
            assign[i] = y_tilde[i]

    rec(0, 0)
    return best


def milp_min_covered(S, y_tilde, flags, m, q, time_limit):

    n, K = S.shape
    F = np.where(flags)[0]
    f = len(F)
    obs = S[np.arange(n), y_tilde]
    const = int(((obs <= q) & (~flags)).sum())
    if f == 0:
        return const, True
    H = (S[F] <= q).astype(float)


    nvar = f * K
    rr, cc, vv = [], [], []
    lb = np.full(f + 1, -np.inf)
    ub = np.full(f + 1, np.inf)
    for i in range(f):
        for y in range(K):
            rr.append(i); cc.append(i * K + y); vv.append(1.0)
        lb[i] = ub[i] = 1.0
    for i in range(f):
        rr.append(f); cc.append(i * K + y_tilde[F[i]]); vv.append(1.0)
    lb[f] = f - m
    A = coo_matrix((vv, (rr, cc)), shape=(f + 1, nvar)).tocsc()
    res = milp(c=H.reshape(-1), integrality=np.ones(nvar),
               bounds=Bounds(0.0, 1.0),
               constraints=LinearConstraint(A, lb, ub),
               options={"time_limit": time_limit, "presolve": True,
                        "mip_rel_gap": 0.0})
    if not res.success or res.status != 0 or res.fun is None:
        return None, False
    return float(res.fun) + const, True


def milp_threshold(S, y_tilde, flags, m, alpha, time_limit=60.0):


    n, K = S.shape
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    cand = np.unique(S)
    lo, hi = 0, len(cand) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        mg, ok = milp_min_covered(S, y_tilde, flags, m, float(cand[mid]),
                                  time_limit)
        if not ok:
            return None
        if mg >= k:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo])


def random_instance(rng, n, K, m, f_frac, realistic_probs=None):
    if realistic_probs is not None:
        idx = rng.choice(len(realistic_probs), n, replace=False)
        S = score_matrix(realistic_probs[idx])
    else:
        logits = rng.normal(size=(n, K))
        p = np.exp(logits - logits.max(1, keepdims=True))
        S = score_matrix(p / p.sum(1, keepdims=True))
    y_tilde = rng.integers(0, K, n)
    flags = np.zeros(n, dtype=bool)
    if f_frac >= 1.0:
        flags[:] = True
    else:
        flags[rng.choice(n, max(1, int(f_frac * n)), replace=False)] = True
    m_eff = min(m, int(flags.sum()))
    return S, y_tilde, flags, m_eff


def exp_equivalence(realistic_probs, trials=200):
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    mismatch_bf = mismatch_milp = 0

    for t in range(trials):
        n = int(rng.integers(6, 21))
        K = int(rng.integers(2, 7))
        m = int(rng.integers(0, min(4, n) + 1))
        f_frac = float(rng.choice([1.0, 0.5]))
        S, y_tilde, flags, m = random_instance(
            rng, n, K, m, f_frac,
            realistic_probs=realistic_probs if t % 2 else None)
        alpha = float(rng.choice([0.1, 0.2, 0.3]))
        q_bf = brute_force_threshold(S, y_tilde, flags, m, alpha)
        q_sw = sharp_suspect_quantile(S, y_tilde, flags, m, alpha)
        agree = (q_bf == q_sw) or (math.isinf(q_bf) and math.isinf(q_sw))
        mismatch_bf += 0 if agree else 1
        rows.append({"n": n, "K": K, "m": m, "alpha": alpha,
                     "f_frac": f_frac, "q_bruteforce": q_bf, "q_sweep": q_sw,
                     "agree": bool(agree)})

    milp_rows = []
    for t in range(100):
        n = int(rng.integers(15, 61))
        K = int(rng.integers(2, 6))
        m = int(rng.integers(0, min(6, n) + 1))
        f_frac = float(rng.choice([1.0, 0.5]))
        S, y_tilde, flags, m = random_instance(
            rng, n, K, m, f_frac, realistic_probs=realistic_probs)
        alpha = float(rng.choice([0.1, 0.2]))
        q_mp = milp_threshold(S, y_tilde, flags, m, alpha)
        q_sw = sharp_suspect_quantile(S, y_tilde, flags, m, alpha)
        agree = (q_mp is not None) and (q_mp == q_sw)
        mismatch_milp += 0 if agree else 1
        milp_rows.append({"n": n, "K": K, "m": m, "alpha": alpha,
                          "q_milp": q_mp, "q_sweep": q_sw,
                          "agree": bool(agree)})
    return {"trials_bruteforce": trials, "mismatches_bruteforce": mismatch_bf,
            "trials_milp": 100, "mismatches_milp": mismatch_milp,
            "bruteforce_rows": rows, "milp_rows": milp_rows}


def exp_scaling(realistic_probs, time_limit=60.0):
    rng = np.random.default_rng(RNG_SEED + 1)

    S0, y0, f0, m0 = random_instance(rng, 20, 3, 1, 1.0)
    milp_threshold(S0, y0, f0, m0, 0.1, time_limit=time_limit)
    rows = []
    sizes = [100, 300, 1000, 3000, 10000, 30000, 100000]
    K = 10
    for n in sizes:
        alpha = 0.1

        ts, tm, agrees, q_sw = [], [], [], None
        for rep in range(3):
            S, y_tilde, flags, m = random_instance(
                rng, n, K, max(1, n // 100), 1.0,
                realistic_probs=realistic_probs if n <= 10000 else None)
            t0 = time.perf_counter()
            q_sw = sharp_suspect_quantile(S, y_tilde, flags, m, alpha)
            ts.append(time.perf_counter() - t0)
            if n <= 10000:
                t0 = time.perf_counter()
                q_mp = milp_threshold(S, y_tilde, flags, m, alpha,
                                      time_limit=time_limit)
                tm.append(time.perf_counter() - t0 if q_mp is not None
                          else math.inf)
                agrees.append(q_mp == q_sw if q_mp is not None else None)
        sweep_s = float(np.median(ts))
        row = {"n": n, "K": K, "m": max(1, n // 100), "alpha": alpha,
               "sweep_s": sweep_s, "q_sweep": q_sw}
        if n <= 10000:
            settled = [a for a in agrees if a is not None]
            if math.isinf(max(tm)):
                row["milp_s"] = f">{time_limit:.0f} (timeout)"
                row["agree"] = bool(all(settled)) if settled else None
            else:
                dt = float(np.median(tm))
                row["milp_s"] = dt
                row["agree"] = bool(all(settled)) if settled else None
                row["speedup"] = dt / sweep_s
        else:
            row["milp_s"] = "skipped (n too large for the 0-1 program)"
        rows.append(row)
        print(f"  n={n:6d}  sweep {sweep_s*1e3:8.2f} ms   milp {row['milp_s']}",
              flush=True)
    return {"rows": rows, "time_limit_s": time_limit}


def main():
    z = np.load(os.path.join(RESULTS, "cache_fmnist.npz"), allow_pickle=True)
    probs_cal = z["probs_cal_pool"]
    print("== E7a: equivalence (brute force and MILP vs sharp sweep) ==")
    eq = exp_equivalence(probs_cal)
    print(f"brute force: {eq['trials_bruteforce'] - eq['mismatches_bruteforce']}"
          f"/{eq['trials_bruteforce']} agree")
    print(f"MILP:        {eq['trials_milp'] - eq['mismatches_milp']}"
          f"/{eq['trials_milp']} agree")
    print("== E7b: speed and scalability ==")
    sc = exp_scaling(probs_cal)

    path = os.path.join(RESULTS, "summary.json")
    with open(path) as f:
        out = json.load(f)
    out["milp_equivalence"] = eq
    out["milp_scaling"] = sc
    out["_meta"] = {"device": "cpu (scipy/HiGHS MILP; method sweep numpy)",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
