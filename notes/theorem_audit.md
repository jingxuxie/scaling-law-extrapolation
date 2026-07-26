# Theorem and claim audit

This note records the exact assumptions behind each contribution and the checks used before marking the manuscript ready for review.

## 1. Exact spectral realizability

For uncorrelated coordinates with predictive energies `q_j=lambda_j theta_j^2`, the population-optimal truncation risk is

`R(M)=E+sum_{j>M} q_j`.

Two constructions are used:

- `q_j=sum_l w_l j^(-1-alpha_l)`, giving exact Hurwitz-zeta tails;
- `q_j=A[j^(-alpha)-(j+1)^(-alpha)]`, giving the exact power `A(M+1)^(-alpha)`.

For a finite positive measure over exponents, Tonelli's theorem gives the continuous positive mixture. All energies are nonnegative and summable for positive exponents. An explicit regression model uses independent standard-normal coordinates and `theta_j=sqrt(q_j)`.

## 2. Hidden-crossover lower bound

The alternatives are

- `R0(M)=E+A phi_beta(M)`;
- `R1(M)=E+A phi_beta(M)+B phi_alpha(M)`, with `0<alpha<beta`.

Choosing `B=eta/phi_alpha(M_min)` makes the pilot gap at most `eta` at every design point. With independent Gaussian noise of variance `sigma^2`,

`KL(P0||P1) <= m eta^2/(2 sigma^2)`.

At `eta=sigma/(2 sqrt(m))`, Pinsker gives total variation at most `1/4`; the absolute-loss two-point reduction gives the constant `3/16`. Integral bounds on the zeta tail prove both future relative-risk divergence and unbounded compute-to-target ratio as the target excess risk approaches zero.

## 3. Matched floor--exponent pair

The pair

- `f0(t)=E+S exp(-alpha t)`;
- `f1(t)=E+S(1-alpha/beta)+(alpha S/beta) exp(-beta t)`

matches value and first derivative at zero. With `H(z)=1-(1+z)exp(-z)`,

`f1(t)-f0(t)=integral_alpha^beta alpha*S/u^2 H(ut) du`.

The proof uses

- `0 <= H(z) <= exp(|z|) z^2/2` for real `z`;
- `H(z) >= exp(-z) z^2/2` for `z>=0`.

Selecting the exponent gap at the KL boundary yields the local lower bound proportional to

`(sigma/sqrt(m)) (1+h/T)^2`,

under the stated local exponent-range conditions. This is a lower bound, not a universal error formula for fitted laws.

## 4. Finite-dictionary and continuous SCALE-CERT

For a finite dictionary, simultaneous Gaussian pilot bands contain the true coefficient vector with probability at least `1-delta`; lower and upper target values are the extrema of a linear functional over the surviving nonnegative coefficient set.

For continuous exponents, `phi_alpha(M)` is convex in `alpha` and

`partial_alpha^2 phi_alpha(M)=sum_{j>M}(log j)^2 j^(-1-alpha)`.

On each exponent cell, the secant exceeds the true basis by at most

`cell_width^2 * curvature_upper / 8`.

Barycentric endpoint masses map every positive exponent measure into a finite nonnegative vector satisfying

`D_M theta-G_M theta <= R(M) <= D_M theta`.

The outer pilot set therefore contains the mapped true measure on the simultaneous-band event, and the two target LPs cover every off-grid mixture on the declared exponent interval. For true total spectral mass at most `W`, the true secant representation error is at most `W max_cell epsilon_cell(M)=O(W Delta_max^2)`.

The guarantee is conditional on positivity, exponent support, pilot-band validity, and any declared structural-mismatch radius. Empty sets are model rejection; unbounded upper objectives are abstention.

## 5. Exact finite-sample OLS bridge

The mean identity assumes whitened Gaussian retained covariates and an independent omitted-coordinate residual with variance `R(M)`. Gaussianity of the residual is not needed for

`E R_OLS(M)=R(M)(n-1)/(n-M-1)`.

The exact F-distribution identity additionally assumes that residual is Gaussian. It follows from a normal vector mixed by an inverse-Wishart matrix:

`R_OLS(M)/R(M) =d 1 + M/(n-M+1) F_{M,n-M+1}`.

The implementation is checked both against the theoretical mean and against explicit Gaussian-design least squares.

## 6. Empirical claim boundaries

- Synthetic coverage uses known Gaussian standard errors and is a direct finite-sample check of the stated class-conditional guarantee.
- Public learning curves do not provide independent checkpoint standard errors. Their results are labelled descriptive and report a structural-mismatch sensitivity frontier; they are not used to claim theorem-level coverage.
- The public files are pinned to NSL-PFN commit `10a5a60911a2a22ef453d8b4ddf1408b6a6075d9` and checksum verified.
- The broad synthetic certificate uses exponent support `[0.05,1.5]`. Narrower results are retained in the CSV outputs but are not used for the hidden-component coverage claim.

## Automated and document checks

`pytest -q` checks 16 identities or implementation properties, including direct zeta sums, both lower-bound constructions, secant curvature, off-grid coverage, quadratic refinement, OLS mean/debiasing, exact OLS sampling, direct OLS simulation, baseline recovery, and minimum mismatch radius.

The submission workflow runs all experiments in isolated jobs, compiles the manuscript, supplement, and checklist, and checks page count and embedded font types. The local preflight additionally renders every PDF page for visual inspection.
