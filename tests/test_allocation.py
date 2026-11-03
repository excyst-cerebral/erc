import unittest

import numpy as np
from scipy.stats import hypergeom

from certificates import recover
from allocation import certificate_cost, exhaustive_checks, upper_errors


class ReviewAllocationTests(unittest.TestCase):
    def test_certificates_against_compatible_label_enumeration(self):
        counts = exhaustive_checks()
        self.assertEqual(counts['certificate_cases'], 826)
        self.assertEqual(counts['transcript_bounds'], 2614)

    def test_one_extra_error_budget_can_require_the_whole_suspect_pool(self):
        for k in [3, 25, 100, 1000]:
            for errors in [1, k//2]:
                S = np.tile([0., 1.], (k+2, 1))
                S[k:] = 1.
                observed = np.zeros(k+2, dtype=int)
                observed[:errors] = 1
                truth = np.zeros(k+2, dtype=int)
                flags = np.arange(k+2)<k
                self.assertEqual(certificate_cost(S, observed, flags, errors, truth, 0., k), errors)
                self.assertEqual(certificate_cost(S, observed, flags, errors+1, truth, 0., k), k)

    def test_hypergeometric_budget_has_finite_population_coverage(self):
        for N in range(2, 13):
            for r in range(1, N+1):
                for actual in range(N+1):
                    for gamma in [.01, .1]:
                        failure = 0.
                        for found in range(max(0,r-N+actual), min(r,actual)+1):
                            residual = upper_errors(N, r, found, gamma)
                            if actual-found > residual:
                                failure += hypergeom.pmf(found, N, actual, r)
                        self.assertLessEqual(failure, gamma+1e-12)

    def test_interval_recovery_queries_only_through_the_callback(self):
        S = np.array([[.1,.9],[.2,.8],[.3,.7],[.4,.6]])
        labels = np.array([0,1,0,1])
        observed = np.zeros(4, dtype=int)
        for policy in ['targeted','random']:
            queried = []
            def query(i):
                self.assertNotIn(i, queried)
                queried.append(i)
                return S[i, labels[i]]
            q, cost = recover(S, query, 3, observed, policy)
            self.assertEqual(q, .6)
            self.assertEqual(cost, len(queried))
        self.assertEqual(recover(S, lambda i: self.fail('No query is needed'), 5, observed, 'targeted'), (float('inf'), 0))


if __name__ == '__main__':
    unittest.main()
