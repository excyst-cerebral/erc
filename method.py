import math
import numpy as np


def prob_residual_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:

    return 1.0 - probs[np.arange(len(labels)), labels]


def score_matrix(probs: np.ndarray) -> np.ndarray:

    return 1.0 - probs


def ceil_int(x: float) -> int:
    return int(math.ceil(x - 1e-12))


def conformal_k(n: int, alpha: float) -> int:

    return ceil_int((n + 1) * (1.0 - alpha))


def naive_quantile(noisy_scores: np.ndarray, alpha: float) -> float:

    n = len(noisy_scores)
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    return float(np.sort(noisy_scores)[k - 1])


def oracle_quantile(clean_scores: np.ndarray, alpha: float) -> float:

    return naive_quantile(clean_scores, alpha)


def uniform_shift_quantile(noisy_scores: np.ndarray, alpha: float, m: int) -> float:


    n = len(noisy_scores)
    k = conformal_k(n, alpha)
    if k + m > n:
        return math.inf
    return float(np.sort(noisy_scores)[k + m - 1])


def alt_max_scores(S: np.ndarray, y_tilde: np.ndarray) -> np.ndarray:

    n = len(y_tilde)
    b = np.full(n, -np.inf)
    for y in range(S.shape[1]):
        mask = y_tilde != y
        b[mask] = np.maximum(b[mask], S[mask, y])
    return b


def sharp_suspect_quantile(
    S: np.ndarray,
    y_tilde: np.ndarray,
    flags: np.ndarray,
    m: int,
    alpha: float,
) -> float:


    n = len(y_tilde)
    k = conformal_k(n, alpha)
    a = S[np.arange(n), y_tilde]
    b = alt_max_scores(S, y_tilde)
    spoilable = np.asarray(flags, dtype=bool) & (b > a)


    ns = int(spoilable.sum())
    vals = np.concatenate([a, a[spoilable], b[spoilable]])
    d_gt = np.concatenate([np.ones(n), np.zeros(2 * ns)]).astype(np.int64)
    d_c = np.concatenate([np.zeros(n), np.ones(ns), -np.ones(ns)]).astype(np.int64)

    order = np.argsort(vals, kind="stable")
    vals, d_gt, d_c = vals[order], d_gt[order], d_c[order]

    gt = 0
    c = 0
    i = 0
    total = len(vals)
    while i < total:
        v = vals[i]
        j = i
        while j < total and vals[j] == v:
            j += 1
        gt += int(d_gt[i:j].sum())
        c += int(d_c[i:j].sum())
        if gt - min(m, c) >= k:
            return float(v)
        i = j
    return math.inf


def predict_set(probs_test: np.ndarray, q: float) -> np.ndarray:

    return (1.0 - probs_test) <= q


def evaluate_sets(sets: np.ndarray, y_true: np.ndarray) -> dict:
    cov = float(np.mean(sets[np.arange(len(y_true)), y_true]))
    size = float(np.mean(sets.sum(axis=1)))
    return {"coverage": cov, "avg_set_size": size}


def known_T_inverted_quantile(
    S: np.ndarray,
    y_tilde: np.ndarray,
    T: np.ndarray,
    alpha: float,
) -> float:


    n, K = S.shape
    k = conformal_k(n, alpha)
    Tinv = np.linalg.inv(T)
    w = Tinv[y_tilde]
    vals = S.reshape(-1)
    ws = w.reshape(-1)
    order = np.argsort(vals, kind="stable")
    vals, ws = vals[order], ws[order]
    cum = np.cumsum(ws)
    j = 0
    total = len(vals)
    while j < total:
        v = vals[j]
        e = j
        while e < total and vals[e] == v:
            e += 1
        if cum[e - 1] >= k - 1e-9:
            return float(v)
        j = e
    return math.inf


def poisson_binomial_budget(eta_i: np.ndarray, delta: float) -> int:


    pmf = np.array([1.0])
    for e in np.asarray(eta_i, dtype=float).ravel():
        pmf = np.convolve(pmf, np.array([1.0 - e, float(e)]))
    cdf = np.cumsum(pmf)
    m = int(np.searchsorted(cdf, 1.0 - delta))
    return min(m, len(eta_i))


def hoeffding_budget(n: int, mu: float, delta: float) -> int:


    return int(math.ceil(mu + math.sqrt(0.5 * n * math.log(1.0 / delta)) - 1e-12))


def bernstein_budget(mu: float, delta: float) -> int:


    L = math.log(1.0 / delta)
    return int(math.ceil(mu + math.sqrt(2.0 * mu * L) + 2.0 * L / 3.0 - 1e-12))


def flagger_low_confidence(probs_cal: np.ndarray, y_tilde: np.ndarray, budget_frac: float) -> np.ndarray:


    conf = probs_cal[np.arange(len(y_tilde)), y_tilde]
    thresh = np.quantile(conf, budget_frac)
    return conf <= thresh


def f_aware_coarse_quantile(S, y_tilde, flags, m, alpha):
    n = len(y_tilde)
    k = conformal_k(n, alpha)
    a = S[np.arange(n), y_tilde]
    order = np.argsort(a, kind='stable')
    values = a[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), n - 1]
    flagged = np.cumsum(np.asarray(flags, dtype=bool)[order])[ends]
    lower = ends + 1 - np.minimum(m, flagged)
    good = np.flatnonzero(lower >= k)
    return float(values[ends[good[0]]]) if len(good) else math.inf


def greedy_binary_quantile(S, y_tilde, flags, m, alpha):
    n = len(y_tilde)
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    a = S[np.arange(n), y_tilde]
    b = alt_max_scores(S, y_tilde)
    active = np.asarray(flags, dtype=bool) & (b > a)
    candidates = np.unique(np.r_[a, b[active]])
    low, high = 0, len(candidates)
    while low < high:
        mid = (low + high) // 2
        q = candidates[mid]
        covered = a <= q
        count = int(covered.sum()) - min(m, int((active & covered & (b > q)).sum()))
        if count >= k:
            high = mid
        else:
            low = mid + 1
    return float(candidates[low]) if low < len(candidates) else math.inf


def aps_score_matrix(probs, uniforms=None):
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=1, kind='stable')
    cumulative = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    cumulative[:, -1] = 1.0
    if uniforms is not None:
        cumulative -= np.asarray(uniforms)[:, None] * np.take_along_axis(probs, order, axis=1)
    scores = np.empty_like(cumulative)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores


def two_budget_quantile(S, y_tilde, flags, m_in, m_out, alpha):
    n = len(y_tilde)
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    a = S[np.arange(n), y_tilde]
    b = alt_max_scores(S, y_tilde)
    active = b > a
    group = np.asarray(flags, dtype=bool)
    candidates = np.unique(np.r_[a, b[active]])
    low, high = 0, len(candidates)
    while low < high:
        mid = (low + high) // 2
        q = candidates[mid]
        covered = a <= q
        spoil = covered & active & (b > q)
        count = int(covered.sum()) - min(m_in, int((spoil & group).sum())) - min(m_out, int((spoil & ~group).sum()))
        if count >= k:
            high = mid
        else:
            low = mid + 1
    return float(candidates[low]) if low < len(candidates) else math.inf


def decomposed_quantile(S, y_tilde, flags, m, alpha):
    n = len(y_tilde)
    k = conformal_k(n, alpha)
    if k > n:
        return math.inf
    a = S[np.arange(n), y_tilde]
    u = a.copy()
    group = np.asarray(flags, dtype=bool)
    u[group] = S[group].max(axis=1)
    partial = float(np.partition(u, k-1)[k-1])
    shifted = float(np.partition(a, k+m-1)[k+m-1]) if k+m <= n else math.inf
    return min(partial, shifted)
