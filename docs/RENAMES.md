# File renames

Files were renamed to short names that match the paper's terminology. The figure and table generators now live in `report/`. Saved results keep their recorded contents unchanged, so their internal provenance fields may still mention the earlier names.

| Earlier name | Current name |
| --- | --- |
| `compare_robust.py` | `solvers.py` |
| `revision_validation.py` | `validate.py` |
| `review_validation.py` | `audit.py` |
| `review_allocation.py` | `allocation.py` |
| `interval_review.py` | `certificates.py` |
| `practical_advantage.py` | `disagreement.py` |
| `targeted_review.py` | `targeted.py` |
| `decomposition_benchmark.py` | `benchmark.py` |
| `run_experiments.py` | `experiments.py` |
| `make_tables.py` | `report/tables.py` |
| `make_pgf_figures.py` | `report/plots.py` |
| `make_review_artifacts.py` | `report/auditfig.py` |
| `make_allocation_artifacts.py` | `report/allocfig.py` |
| `make_practical_artifacts.py` | `report/workflowfig.py` |
| `make_revision_validation_artifacts.py` | `report/solverfig.py` |
| `make_figures.py` | `report/mplfig.py` |
| `make_figures_extra.py` | `report/mplextra.py` |
| `make_revision_artifacts.py` | `report/legacyfig.py` |
| `check_pgf_figures.py` | `report/checkpgf.py` |
| `check_figure_layout.py` | `report/checklayout.py` |
| `scripts/prepare_cifar10n.py` | `scripts/cifar10n.py` |
| `tests/test_review_allocation.py` | `tests/test_allocation.py` |
| `GRAPH_DESIGN.md` | `docs/figures.md` |
| `results/results_summary.json` | `results/summary.json` |
| `results/revision_validation.json` | `results/validate.json` |
| `results/review_validation.json` | `results/audit.json` |
| `results/review_allocation.json (+ protocol)` | `results/allocation.json (+ protocol)` |
| `results/interval_review.json (+ protocol)` | `results/certificates.json (+ protocol)` |
| `results/practical_advantage.json (+ protocol)` | `results/disagreement.json (+ protocol)` |
| `results/targeted_review.json (+ protocol)` | `results/targeted.json (+ protocol)` |
| `results/decomposition_benchmark.json` | `results/benchmark.json` |
| `results/reviewer_*.json (fresh-run outputs)` | `results/rerun-*.json` |

Generated figure and table names under `figures/` are unchanged, so the manuscript's `\input` and `\includegraphics` paths still resolve.
