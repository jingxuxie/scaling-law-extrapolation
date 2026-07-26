# Literature positioning

## Closest strands

1. **Empirical and broken scaling laws.** These works propose flexible parametric forms and evaluate point extrapolation. The present paper asks what is identifiable before selecting a preferred form.
2. **Bayesian/PFN extrapolation.** These methods return posterior uncertainty conditional on synthetic-function priors. SCALE-CERT instead gives frequentist coverage over an explicit positive spectral class and can abstain when the class does not identify the target.
3. **Spectral scaling theory in linear regression.** Existing results derive model/data/compute exponents under power-law spectra and source conditions. The present paper uses exact spectral tails to construct statistically indistinguishable learning problems.
4. **Tokens-per-parameter coverage (2026).** Kricheli et al. study local ill-conditioning of bivariate parameter--data laws when pilot runs lie along a nearly collinear ray. The present paper is complementary: it proves univariate out-of-range lower bounds for arbitrary designs inside a bounded interval, permits future hidden regimes, and constructs set-valued target certificates.
5. **Downstream scaling reliability.** Recent meta-analysis shows that simple downstream trends are not universal. This motivates, but does not replace, a theorem characterizing when extrapolation cannot be identified.

## Defensible novelty statement

The intended novelty is the combination of:

- exact linear-regression realizability of observationally indistinguishable scaling curves;
- finite-sample minimax lower bounds for asymptotic exponent and target prediction;
- unbounded compute-to-target disagreement from a below-noise spectral component;
- an LP-computable, target-specific partial-identification interval with simultaneous coverage.

Avoid claiming that this is the first work on uncertainty, identifiability, experimental design, broken regimes, or unreliable scaling laws in isolation.
