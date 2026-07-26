# Theorem audit

This note records the exact claim dependencies and the checks performed before the initial proof branch was opened.

## 1. Spectral realizability

For uncorrelated coordinates with predictive energies `q_j = lambda_j theta_j^2`, the population-optimal truncation risk is

`R(M) = E + sum_{j>M} q_j`.

Two exact constructions are used:

- `q_j = sum_l w_l j^(-1-alpha_l)`, giving Hurwitz-zeta tails.
- `q_j = A[j^(-alpha) - (j+1)^(-alpha)]`, giving the exact power `A(M+1)^(-alpha)` by telescoping.

All predictive energies are nonnegative and summable for positive exponents. An explicit underlying regression model is obtained with independent standard-normal coordinates, `lambda_j=1`, `theta_j=sqrt(q_j)`, and independent label noise of variance `E`.

## 2. Hidden-crossover lower bound

The alternatives are

- `R0(M)=E+A phi_beta(M)`;
- `R1(M)=E+A phi_beta(M)+B phi_alpha(M)`, with `0<alpha<beta`.

Choosing `B=eta/phi_alpha(M_min)` makes the pointwise pilot gap at most `eta`. For `m` independent observations with variance `sigma^2`,

`KL(P0||P1) <= m eta^2/(2 sigma^2)`.

With `eta=sigma/(2 sqrt(m))`, Pinsker gives total variation at most `1/4`. The absolute-loss two-point reduction then gives the factor `3/16` for exponent separation. Integral tail bounds give both the future relative-risk divergence and the compute-to-target ratio.

## 3. Matched floor--exponent pair

The pair

- `f0(t)=E+S exp(-alpha t)`;
- `f1(t)=E+S(1-alpha/beta)+(alpha S/beta) exp(-beta t)`

matches value and first derivative at zero. Writing `H(z)=1-(1+z)exp(-z)` gives the exact gap identity

`f1(t)-f0(t)=integral_alpha^beta alpha*S/u^2 H(ut) du`.

The proof uses

- `0 <= H(z) <= exp(|z|) z^2/2` for all real `z`;
- `H(z) >= exp(-z) z^2/2` for `z>=0`.

These inequalities were checked analytically and by numerical tests. The local horizon/log-span corollary chooses the exponent gap at the KL boundary and explicitly requires that this gap remain inside the declared exponent range.

## 4. SCALE-CERT

The confidence set intersects nonnegative spectral coefficients with Bonferroni-corrected Gaussian observation bands. Conditional on the event that all pilot noises lie inside their bands, the true coefficient vector is feasible; the same event implies coverage for every target size simultaneously. The image of the convex feasible set under the scalar target functional is an interval, so the lower and upper LP optima are sharp relative to the bands.

## Automated checks

`pytest -q` currently verifies:

- direct-sum agreement of the Hurwitz-zeta tail;
- exact value and slope matching;
- upper and lower quadratic gap inequalities;
- telescoping exact-power energies;
- saturation of the hidden-component pilot tolerance;
- containment of a correctly specified curve by the LP certificate.
