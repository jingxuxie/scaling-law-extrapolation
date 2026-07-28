# Submission readiness and remaining extensions

## Completed

- Exact zeta-tail and exact-power realizability.
- Proof that the slowest supported positive spectral population controls the asymptotic exponent, even without an atom at the boundary.
- Hidden-crossover exponent lower bound and unbounded compute-to-target corollary.
- Floor--exponent lower bound within the ordinary one-power family.
- Design-aware refinement through the fourth-moment effective log-span and a corresponding necessary safe-horizon inequality.
- Finite-dictionary partial identification and simultaneous coverage.
- Continuous-exponent outer certificate with an explicit quadratic grid correction.
- Simultaneous compute-to-target interval obtained by inverting the target-risk envelope.
- Exact finite-sample Gaussian OLS mean, debiasing factor, and distributional sampler.
- Stable one-power, two-power, parametric-bootstrap, and weak-prior Laplace baselines.
- Fixed-budget pilot-placement experiment, ninety-run synthetic coverage/false-confidence benchmark, and twenty-seven rolling-origin public-curve evaluations.
- Expanded related work with 29 verified references and a clearer separation of exponent, target-risk, and resource identification.
- Full seven-page technical AAAI manuscript followed by two reference-only pages, full supplement, completed checklist, deterministic results, CI, and rendered-page/font preflight.

## Submission claims retained

The paper claims:

1. statistical non-identifiability on bounded pilot ranges under the constructed spectral classes;
2. class-conditional simultaneous coverage of SCALE-CERT;
3. rigorous coverage of off-grid positive mixtures on a declared exponent interval;
4. simultaneous target-risk and compute-to-target certification;
5. an exact trained-OLS bridge;
6. a local design-dependent lower bound governed by the fourth moment of pilot log-scales;
7. synthetic evidence that parametric intervals can be false-confident under hidden structure;
8. descriptive public evidence that in-range residual calibration alone is not enough to select a future mismatch radius.

It does **not** claim distribution-free public-curve coverage, universal superiority in point error, globally optimal pilot design, or validity outside the declared positive spectral class.

## Useful post-submission extensions

These are scientifically valuable but not required for the current paper's central claim:

- correlated pilot errors and generalized least-squares bands;
- adaptive or cost-aware sequential pilot design beyond the fixed-design lower bound;
- ridge and early-stopped SGD analogues of the exact OLS bridge;
- nonmonotone or signed spectral classes;
- a held-out public benchmark with independent replicated standard errors;
- a continuous semi-infinite solver that avoids numerical exponent cells entirely.
