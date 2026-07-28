# Can You Trust the Slope?

This repository contains the AAAI-27 manuscript, proofs, and reproducible experiments for:

> **Can You Trust the Slope? Identifiability Limits and Certified Scaling-Law Extrapolation**

The project asks when a scaling law fitted on a bounded set of pilot runs can support a scientifically defensible prediction at a larger scale. Its central conclusion is that a small in-range residual is not an extrapolation certificate: unseen spectral mass and floor--exponent confounding can leave the future statistically unidentified.

## Main results

1. **Exact spectral realizability.** Positive mixtures of power-law predictive energies induce exact Hurwitz-zeta risk tails in infinite-dimensional linear regression. Ordinary floor-plus-power curves are exactly realizable by telescoping energies, and the slowest supported positive spectral population determines the asymptotic exponent even without an atom at that exponent.
2. **Identifiability and pilot-design lower bounds.** A component below pilot noise can change the asymptotic exponent and produce unbounded compute-to-target disagreement. Even within the ordinary one-power family, value-and-slope matching yields a quadratic extrapolation barrier controlled by the fourth-moment effective log-span `T_eff=(mean(t_i^4))^(1/4)`, not by run count alone.
3. **Continuous-exponent SCALE-CERT.** The method returns the full target-risk range compatible with simultaneous pilot bands and a declared positive spectral class. A secant-curvature construction converts all exponent mixtures on a continuous interval into two finite linear programs with a rigorous `O(grid_width^2)` outer correction; inversion gives a simultaneous compute-to-target interval.
4. **Exact trained-regression bridge.** For Gaussian design, the expected test risk of `M`-dimensional OLS trained on `n` samples is the oracle spectral risk times `(n-1)/(n-M-1)`. An exact Hotelling/F representation permits inexpensive finite-sample simulation and mean debiasing.
5. **Stress tests.** The evaluation includes exact lower-bound constructions, a controlled pilot-placement study, 90 noisy synthetic pilot problems, 5,000-replicate finite-sample OLS curves, and 27 rolling-origin predictions on public NMT and image-classification learning curves.

## Reproduce everything

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

pytest -q
python experiments/run_all.py
SYNTHETIC_REPETITIONS=30 \
  BOOTSTRAP_REPETITIONS=20 \
  OLS_REPETITIONS=5000 \
  python experiments/run_submission.py --stage all

make -C paper all
make -C paper checklist
```

The submission runner executes its four stages in fresh subprocesses:

- `continuous`: off-grid theorem/refinement check;
- `ols`: exact finite-sample trained-regression bridge;
- `synthetic`: parametric, bootstrap, Laplace, and certificate comparison;
- `public`: rolling-origin structural-mismatch sensitivity on pinned NSL-PFN data.

The public stage downloads two files from a fixed upstream commit and verifies SHA-256 checksums. No GPU or external API is required.

## Current reproduced highlights

- All 18 theorem and implementation tests pass.
- For a matched floor--exponent pair, moving the same nine-run budget from the center to the endpoints raises the exact Gaussian KL divergence from `4.34e-5` to `8.22e-2`, exposing the importance of scale placement.
- Halving exponent-cell width reduces the certified interpolation correction by a factor of four.
- Across the OLS dimensions used in the paper, the maximum relative error of the debiased 5,000-replicate mean is below `6.1e-5`.
- On the three-component synthetic alternative, conventional bootstrap and Laplace intervals have `0%` target coverage and `100%` false-confidence rate; the broad continuous certificate has `100%` coverage and `0%` false-confidence rate.
- The public study is deliberately descriptive because checkpoint standard errors are unavailable. It reports the coverage--width frontier obtained by explicitly inflating the smallest in-range structural residual rather than presenting that residual as a distribution-free guarantee.

Machine-readable per-run and aggregate outputs are committed under `results/`.

## Paper and repository layout

- `paper/main.tex`: anonymous AAAI manuscript.
- `paper/supplement.tex`: full proofs and experimental details.
- `paper/ReproducibilityChecklist.tex`: completed AAAI checklist responses (`paper/checklist.tex` is the build wrapper).
- `src/scaling_extrapolation/`: exact risks, certificates, OLS theory, and baselines.
- `experiments/run_all.py`: exact lower-bound, pilot-design, and finite-dictionary demonstrations.
- `experiments/run_submission.py`: isolated submission benchmark pipeline.
- `results/`: deterministic CSV outputs.
- `tests/`: theorem-level and implementation tests.
- `notes/theorem_audit.md`: assumptions, proof dependencies, and claim audit.

The AAAI manuscript uses the full seven-page technical-content allowance, followed by two reference-only pages. `paper/preflight_pages.py` verifies the rendered page boundary so that technical floats cannot spill into the reference allowance. All embedded fonts are non-Type-3, and CI regenerates the figures and results before rebuilding the manuscript, supplement, and checklist. The repository is released under the MIT License.
