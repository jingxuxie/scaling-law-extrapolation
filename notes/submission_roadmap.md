# Submission roadmap

## Completed in the proof-foundation branch

- Exact spectral and exact-power realizability.
- Hidden-crossover statistical lower bound.
- Compute-to-target divergence corollary.
- One-power floor--exponent lower bound with explicit constants.
- LP partial-identification method and simultaneous coverage theorem.
- Exact-risk figures, calibration experiment, tests, and AAAI paper skeleton.
- Current-literature positioning against broken laws, Bayesian/PFN extrapolation, downstream unreliability, and 2026 tokens-per-parameter identifiability work.

## Priority 1: empirical credibility

1. Add rolling-origin experiments on at least two public scaling-law datasets.
2. Compare point error, interval coverage, interval width, and false-confidence rate against:
   - nonlinear floor-plus-power fitting;
   - broken power-law fitting;
   - parametric bootstrap;
   - a Bayesian floor-plus-power baseline;
   - an oracle dictionary where appropriate.
3. Report results as a function of observed log-span and extrapolation horizon, not only random held-out points.

**Go/no-go:** SCALE-CERT should materially reduce false-confident extrapolations while remaining nonvacuous on genuinely simple curves.

## Priority 2: bridge from exact risk to trained regression

- Generate finite-sample Gaussian linear-regression data with one- and two-component source spectra.
- Train ridge regression and gradient descent/SGD at increasing truncation dimensions.
- Verify that the hidden-regime and floor-confounding phenomena persist under training and finite validation sets.
- Separate training randomness from evaluation noise using repeated seeds.

## Priority 3: strengthen the positive theory

Choose one of the following as the next theorem contribution:

- a continuous-exponent certificate with a provable discretization error;
- a semi-infinite optimization formulation;
- an optimal pilot-design result for fixed run count or cost;
- a matching or near-matching upper bound in the local one-power model;
- correlated Gaussian pilot errors with a known or estimated covariance matrix.

The continuous-dictionary direction is the best match to the current method. The pilot-design direction is the best match to the paper's practical message.

## Priority 4: paper polish

- Compress the main text to the AAAI seven-page technical-content limit after final experiments are inserted.
- Replace preliminary calibration numbers with the full benchmark table.
- Add a single overview figure combining observational equivalence, future divergence, and certificate/abstention.
- Complete the reproducibility checklist.
- Audit every theorem condition and bibliography entry one final time.

## Main risks

- **Prior-work overlap:** 2026 work proves coefficient ill-conditioning for collinear parameter/data designs. The paper must retain its distinct focus on one-dimensional bounded-range extrapolation, hidden spectral alternatives, minimax testing lower bounds, and set-valued certification.
- **Vacuous intervals:** broad dictionaries can yield wide intervals. Report nested assumption classes and a coverage--informativeness frontier rather than hiding this tradeoff.
- **Model mismatch:** positive spectral mixtures exclude nonmonotone curves. Empty confidence sets should be treated as model rejection, not repaired by silent overfitting.
- **Empirical scope:** exact-risk experiments establish mechanism and correctness, but public learning curves are necessary for a competitive archival submission.
