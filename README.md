# Exact Robust Conformal Calibration

Code for the paper *Exact Robust Conformal Calibration with Suspect Labels and Corruption Budgets* (Daniel Huang). File names follow the paper's terminology; earlier internal names are listed in [docs/RENAMES.md](docs/RENAMES.md).

Reproducibility code for *Exact Robust Conformal Calibration with Suspect Labels
and Corruption Budgets*. The preferred exact implementation, `decomposed_quantile`, selects two order
statistics in `O(n + |F|K)` time after score evaluation. The event sweep and
greedy binary search remain independent reference implementations. Its coverage guarantee requires
all actual label errors to lie in the suspect set and their number to fit the
budget, or an explicitly accounted failure probability.

## Quick verification

Use Python 3.10 or newer and a compatible PyTorch/torchvision pair:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

These dataset-free tests check exact thresholds, sharp count endpoints, ties,
full-set cases, and small-instance MILP agreement. For the larger exhaustive,
certified MILP, and generated timing checks, write fresh output separately:

```bash
python validate.py --output results/rerun-validate.json
```

This leaves the bundled manuscript measurements intact. MILP comparisons
require successful solver termination and zero requested relative gap.

## Reproduce manuscript tables and figures

The bundled `results/summary.json` contains the original aggregate
measurements. `results/validate.json` contains the certified solver
checks, generated timings, 30-seed audit, and controlled-channel diagnostic.
`results/audit.json` adds a 630-row budget ledger, the suspect-aware
coarse baseline, randomized APS scores, small-budget comparisons, a fixed
audit-allocation grid and paired greedy-search timings.
No datasets, model weights, or prediction caches are distributed.

```bash
python report/tables.py
python report/solverfig.py
python report/plots.py
python report/auditfig.py
python report/workflowfig.py
python report/allocfig.py
python report/checkpgf.py
```

The final figures require pdfLaTeX with TikZ, PGFPlots, standalone and Latin
Modern, plus Poppler tools on PATH. PDFs, PNG previews, editable standalone
TeX sources and plotted-value manifests are written under `figures/`.
Run the three TikZ/PGFPlots generators after the earlier Matplotlib generators, which can overwrite some filenames. See [docs/figures.md](docs/figures.md) for figure conventions.

## Repeat the data experiments

MNIST, Fashion-MNIST and CIFAR-10 images are downloaded through torchvision.
Obtain CIFAR-10N from its [official release](https://github.com/UCSC-REAL/cifar-10-100n),
including `data/CIFAR-10_human_ordered.npy` and `image_order_c10.npy`, then run:

```bash
python scripts/cifar10n.py /path/to/cifar-10-100n
python experiments.py e0 e1 e3 e4 e5
python experiments.py e2 e8
python solvers.py
python validate.py --cache-dir results --data-dir data --output results/rerun-validate.json
```

The preparation helper copies the required files into `data/cifar10n/`.
Use the trusted upstream label file: its dictionary is loaded with pickle.
The experiment loader verifies clean-label ordering against torchvision.
Training uses Apple MPS when available and CPU otherwise; CPU training can
be slow. Stages reuse prediction caches and merge original-stage outputs
into `results/summary.json`. Archive existing caches and outputs
before requesting fresh training.

MNIST-family fitting holds out the final 10,000 training examples for
calibration; CIFAR fitting holds out every fifth training example. Repeated
seeds vary calibration, corruption and review draws around one cached
classifier per dataset. They are not independent classifier fits. Legacy
caches lack initialization-seed metadata, so exact retraining of those
classifiers is not claimed. Future training seeds initialization before
model construction. The recorded environment used Python 3.12, NumPy 2.5.4,
SciPy 1.18.1, PyTorch 2.14.1, torchvision 0.29.1 and Matplotlib 3.11.2;
requirements use lower bounds rather than a locked environment.

To reproduce the additional comparisons from the same cached classifiers:

```bash
python audit.py --cache-dir results --data-dir data --output results/rerun-audit.json
```

The fixed random seed for randomized APS is recorded in this driver. The
small-budget study uses privileged true-error flags. Its exact and suspect-aware
coarse rules coincide; the paired timing study finds greedy binary search
faster than the current sweep implementation. The two-part budget extension
is checked against exhaustive labelings in the dataset-free tests.

The exact threshold also equals the minimum of the global rank-shift and
pessimistic partial-label thresholds. The two-selection implementation returns
identical values to the reference algorithms and avoids event sorting.

Additional paired checks and exploratory review grids can be reproduced with:

```bash
python benchmark.py
python disagreement.py --cache-dir results --data-dir data
python targeted.py --cache-dir results --data-dir data
```

These commands write separate aggregate files under `results/`. The review
protocols are exported before execution. The targeted binary-task follow-up
was motivated by the first grid's weak gains and is explicitly exploratory.
It uses recorded human annotations and reference labels to emulate corrections.
APS uniforms are resampled per seed and shared between paired methods. The
results show gains over the coarse rule; the pessimistic partial-label branch
explains the featured statistical improvement. The fixed-cost clean audit
remains a stronger efficiency comparator in that example.

## Interpret the evidence

Controlled corruption is simulated on real image data; CIFAR-10N supplies
real human annotation noise. E8 uses dataset clean labels to emulate accurate
adjudication, rather than reporting a newly performed human audit. Its
30-seed comparison uses a fixed 1,000-label clean-audit baseline at the same
guaranteed coverage target. The primary robust configuration uses alpha=0.09, gamma=0.01 and 200 consensus
reviews. It returns larger sets than the fixed 1000-label clean audit. The
released frontier evaluates each alpha/gamma and review-size configuration
separately; selecting on test outcomes is not covered by these guarantees. The controlled logged channel has suspect containment by
construction; the exact method equals naive CP there and improves only over
the coarse rank shift.

Generated runtime instances measure computation, not image-data coverage.
The MILP baseline is an implementation of the budgeted formulation in
[Zargarbashi et al. (2024)](https://arxiv.org/abs/2407.09165), not a timing
reproduction of their released software. The transition-correction comparator
uses the setting discussed by
[Penso et al. (2025)](https://proceedings.mlr.press/v266/penso25a.html).

## Code and formal checks

| File | Purpose |
| --- | --- |
| `method.py` | Exact two-selection rule, count bounds, reference quantiles and baselines |
| `experiments.py` | Image classifiers and experiments E0–E5/E8 |
| `solvers.py` | Exhaustive/MILP reference and historical timing checks |
| `validate.py` | Certified checks, timing grid and 30-seed studies |
| `audit.py` | Budget ledger, coarse/APS baselines, audit frontier and direct timings |
| `report/auditfig.py` | Comparison tables and the audit-frontier vector figure |
| `benchmark.py` | Identity checks and paired three-algorithm timings |
| `disagreement.py` | Fixed-cost disagreement-review grid |
| `allocation.py` | Adaptive-review certificates, minimum review cost and budget-slack diagnostics (paper Appendix G) |
| `certificates.py` | Earlier unrestricted-interval exact-recovery prototype |
| `targeted.py` | Model-uncertain-pool follow-up on binary coarsenings |
| `report/workflowfig.py` | Direct timing and targeted-review tables and figures |
| `report/tables.py` | Original experimental tables |
| `report/solverfig.py` | Solver-validation tables and audit summary |
| `report/allocfig.py` | Audit-allocation tables and figures |
| `report/legacyfig.py` | Early Matplotlib audit figure (superseded by the vector figures) |
| `report/plots.py` | The seven base vector figures |
| `report/checkpgf.py` | Non-mutating figure bounds/font checks |
| `tests/test_method.py` | Dataset-free numerical correctness tests |
| `lean/ConformalLabelNoise.lean` | Supporting formal lemmas |

Install Lean through elan and use the included toolchain:

```bash
lean lean/ConformalLabelNoise.lean
```

The Lean source checks count/order-statistic lemmas and a finite
non-identifiability witness. It does not formalize exchangeability, the
entire sharpness theorem, the sweep implementation, or the neural experiments.
Dataset access and licensing remain governed by the providers. The research code
is released under the MIT License (see `LICENSE`).

## Adaptive review certificates

`allocation.py` implements the fixed-threshold additive review gains, the
exact minimum certificate cost given true labels, the budget-slack diagnostic,
and a probability-guided allocation heuristic. The minimum certificate is a
privileged diagnostic, not an achieved review cost. A positive error-budget slack
can rule out the shifted-count certificate at the exact clean quantile, forcing
the partial-label certificate instead. The tests enumerate compatible clean
labelings and review subsets and check the one-unit slack counterexample.

```bash
python allocation.py --cache-dir /path/to/caches --data-dir /path/to/data --output results/rerun-allocation.json
python certificates.py --cache-dir /path/to/caches --data-dir /path/to/data --output results/rerun-certificates.json
python report/allocfig.py
```

The first driver writes its protocol before execution and retains all 2,160
policy/cost records: 30 paired seeds, two tasks, two scores, three policies and
six fixed review costs. An initial uniform 200-label audit supplies a residual
error bound; later reviews decrement it by revealed errors. The full-grid
summary is `figures/review_allocation_grid.csv`. The budget-gain policy improves
on random review with a frozen bound but has larger mean prediction sets than
ordinary random-subset conformal calibration at every intermediate tested cost.
At full review, its fixed alpha of .09 remains conservative; using alpha .1
recovers ordinary clean calibration. A fresh bound from the complete uniform
audit also beats the proposed policy at several costs. This is not evidence of
a generally superior label-efficient method. Published label-efficient stopping
and selection with explorable uncertainty are discussed in the manuscript;
no performance superiority to those algorithms is claimed.

`certificates.py` retains the earlier unrestricted-interval prototype and its
40 paired configurations. Both studies use existing CIFAR-10N human annotations
and dataset reference labels to emulate review, a cached classifier and a reused
test set. No new human adjudication or independent model replication is claimed.
The exact-budget oracle comparison was added after inspecting policy results
and is marked as a post hoc diagnostic in the protocol.
