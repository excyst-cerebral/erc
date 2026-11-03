# Aggregate experiment results

summary.json preserves historical aggregate measurements; its local execution timestamp was omitted. Numerical values are unchanged.

validate.json records fresh exactness, certified MILP calls, nontrivial generated timings, a 30-seed audit, and a controlled corruption-channel diagnostic on real image scores. It includes input-file SHA-256 hashes, not paths, images, labels, model weights, or prediction arrays. Seeds 0-2 reproduce the original audit exactly. Calibration/review repetitions reuse fixed cached classifiers.

The hardened MILP implementation requires certified success with zero requested relative gap. Historical MILP rows lack termination certificates; fresh records include status and gap for every solver call. Run python validate.py for dataset-free checks; see the root README for data-backed experiments.

audit.json contains paired sweep/greedy timings, residual and randomized APS score comparisons, matched E3 attack/defense caps, a 630-row method-budget ledger, and a 30-seed audit grid over four review sizes and five failure allocations. Each grid point is a separate fixed configuration; no test-selected certificate is claimed. Budget-accounting entries are drawn from the original recorded experiment rows. Cache files are represented only by hashes.

disagreement.json and targeted.json retain both exploratory grids (720 runs each), including null and unfavorable cells. Their adjacent protocol JSON files describe the pre-execution configurations; the targeted follow-up was motivated by the first grid. Reference labels emulate review. benchmark.json records 2,000 random tied-instance identity checks and 50 paired runtime cases through n=1,000,000. The direct method, greedy search and sweep return identical thresholds.

allocation.json contains the complete fixed-cost adaptive-review study
(2,160 records) and exhaustive certificate checks (2,614 transcript bounds and
826 minimum-cost cases). Its adjacent protocol distinguishes the fixed policy
study from the post hoc exact-budget oracle diagnostic. `initial_budget_slack`
is the initial bound minus the true residual error count. Both oracle review
cost fields require true labels; one additionally assumes the exact error count.
They are not deployed policy costs. Every per-policy result includes a
matched-cost random clean-subset comparator, a freshly inverted uniform-audit
bound and a partial-label comparator on that policy's transcript.

certificates.json contains the earlier 40 paired exact-recovery experiments.
Its diagnostic count of 1,350 means policy calls on 675 generated tied-score
instances. Those generated checks are distinct from the real-data experiments.
Both experiment protocols were saved before their respective runs; all
configurations, including negative comparisons, are retained.
