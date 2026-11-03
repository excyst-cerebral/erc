import itertools
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from method import (conformal_k, sharp_suspect_quantile,
                    f_aware_coarse_quantile, greedy_binary_quantile, aps_score_matrix, two_budget_quantile, decomposed_quantile)
from solvers import milp_threshold, milp_min_covered


def admissible(S, y, flags, budget):
    for labels in itertools.product(range(S.shape[1]), repeat=len(y)):
        candidate = np.array(labels)
        changed = candidate != y
        if changed.sum() <= budget and not np.any(changed & ~flags):
            yield S[np.arange(len(y)), candidate]


class ExactCalibrationTests(unittest.TestCase):
    def test_worst_case_quantiles_and_sharp_counts(self):
        rng = np.random.default_rng(20261018)
        for trial in range(100):
            n, classes = int(rng.integers(1, 6)), int(rng.integers(2, 5))
            S = rng.integers(0, 8, (n, classes)) / 7.0
            y = rng.integers(classes, size=n)
            flags = rng.random(n) < 0.6
            budget = int(rng.integers(0, n + 1))
            alpha = float(rng.choice([0.1, 0.3, 0.5, 0.8]))
            k = conformal_k(n, alpha)
            feasible = list(admissible(S, y, flags, budget))
            expected = max(math.inf if k > n else float(np.sort(a)[k - 1]) for a in feasible)
            self.assertEqual(sharp_suspect_quantile(S, y, flags, budget, alpha), expected)
            self.assertEqual(decomposed_quantile(S, y, flags, budget, alpha), expected)
            q = float(rng.choice(S.ravel()))
            observed = S[np.arange(n), y] <= q
            alternatives = S.copy()
            alternatives[np.arange(n), y] = np.nan
            spoil = flags & observed & (np.nanmax(alternatives, axis=1) > q)
            fix = flags & ~observed & (np.nanmin(alternatives, axis=1) <= q)
            lo = int(observed.sum()) - min(budget, int(spoil.sum()))
            hi = int(observed.sum()) + min(budget, int(fix.sum()))
            self.assertEqual({int((a <= q).sum()) for a in feasible}, set(range(lo, hi + 1)))

    def test_coarse_and_binary_with_ties(self):
        rng = np.random.default_rng(992)
        for _ in range(100):
            n = int(rng.integers(1, 12))
            S = rng.integers(0, 6, (n, 4)) / 5
            y = rng.integers(4, size=n)
            flags = rng.random(n) < .5
            m = int(rng.integers(n + 1))
            alpha = float(rng.choice([.1, .3, .8]))
            exact = sharp_suspect_quantile(S, y, flags, m, alpha)
            self.assertEqual(greedy_binary_quantile(S, y, flags, m, alpha), exact)
            self.assertEqual(decomposed_quantile(S, y, flags, m, alpha), exact)
            coarse = f_aware_coarse_quantile(S, y, flags, m, alpha)
            self.assertGreaterEqual(coarse, exact)
            a = S[np.arange(n), y]
            candidates = np.unique(a)
            valid = [q for q in candidates if int((a <= q).sum()) -
                     min(m, int((flags & (a <= q)).sum())) >= conformal_k(n, alpha)]
            self.assertEqual(coarse, valid[0] if valid else math.inf)
        P = np.array([[.2, .6, .2], [.5, .3, .2]])
        np.testing.assert_allclose(aps_score_matrix(P), [[.8, .6, 1], [.5, .8, 1]])

    def test_two_budget_exhaustive(self):
        rng = np.random.default_rng(780)
        for _ in range(50):
            n = int(rng.integers(2, 6))
            S = rng.integers(0, 5, (n, 3)) / 4
            y = rng.integers(3, size=n)
            flags = rng.random(n) < .5
            mi, mo = rng.integers(0, n+1, size=2)
            alpha = .5
            qs = []
            for labels in itertools.product(range(3), repeat=n):
                candidate = np.array(labels)
                diff = candidate != y
                if (diff & flags).sum() <= mi and (diff & ~flags).sum() <= mo:
                    qs.append(float(np.sort(S[np.arange(n), candidate])[conformal_k(n,alpha)-1]))
            self.assertEqual(two_budget_quantile(S,y,flags,mi,mo,alpha), max(qs))

    def test_two_quantile_branches_and_boundaries(self):
        S = np.array([[.1,.9],[.2,.8],[.3,.7]])
        y = np.zeros(3, dtype=int)
        F = np.ones(3, dtype=bool)
        self.assertEqual(decomposed_quantile(S,y,F,0,.5),.2)
        self.assertEqual(decomposed_quantile(S,y,F,3,.5),.8)
        self.assertEqual(decomposed_quantile(S,y,F,1,.01),math.inf)
        self.assertEqual(decomposed_quantile(np.array([[.1],[.2],[.3]]),y,F,3,.5),.2)

    def test_alternative_score_breakpoint(self):
        self.assertEqual(sharp_suspect_quantile(np.array([[0.1, 0.9]]),
                         np.array([0]), np.array([True]), 1, 0.5), 0.9)

    def test_unsettled_milp_incumbent_is_not_an_exact_certificate(self):
        S = np.array([[0.1, 0.9], [0.3, 0.7]])
        with patch("solvers.milp", return_value=SimpleNamespace(
                success=False, status=1, fun=1.0)):
            result, solved = milp_min_covered(S, np.array([0, 0]),
                np.array([True, True]), 1, 0.5, 0.01)
        self.assertIsNone(result)
        self.assertFalse(solved)

    def test_milp_matches_exact_sweep(self):
        rng = np.random.default_rng(8)
        for _ in range(12):
            n = int(rng.integers(3, 10))
            S = rng.integers(0, 10, (n, 3)) / 10.0
            y = rng.integers(3, size=n)
            flags = rng.random(n) < 0.6
            budget = int(rng.integers(0, n + 1))
            q = milp_threshold(S, y, flags, budget, 0.3)
            self.assertIsNotNone(q, "Unsettled solver output is not agreement")
            self.assertEqual(q, sharp_suspect_quantile(S, y, flags, budget, 0.3))


if __name__ == "__main__":
    unittest.main()
