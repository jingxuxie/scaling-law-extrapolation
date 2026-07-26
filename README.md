# Scaling-Law Extrapolation: Identifiability Limits and Certificates

This repository develops a theory and lightweight experimental testbed for a basic question:

> When can a scaling law fitted on a finite pilot range be trusted outside that range?

The current draft studies exact population-risk curves in infinite-dimensional linear regression. It proves that small in-range residuals do not, by themselves, identify the asymptotic exponent or the compute required to reach a demanding target. It also introduces **SCALE-CERT**, a linear-programming procedure that returns the range of future losses compatible with simultaneous pilot-error bands and a declared positive spectral dictionary.

## Current results

The proof-complete foundation contains:

1. **Spectral realizability.** Positive mixtures of power-law predictive-energy spectra induce exact Hurwitz-zeta tail risks. Exact floor-plus-power curves are also realizable by telescoping predictive energies.
2. **Hidden-crossover impossibility.** A slower spectral component can remain below the pilot noise on any finite range while changing the asymptotic exponent. The resulting compute-to-target error can be unbounded.
3. **Floor--exponent confounding.** Two ordinary floor-plus-power laws can match both value and slope at the center of the pilot interval. A two-point lower bound yields a local quadratic dependence on the extrapolation-horizon/log-span ratio.
4. **Finite-sample certificates.** SCALE-CERT solves two linear programs and has simultaneous finite-sample coverage when the stated spectral dictionary is correctly specified.
5. **Exact-risk validation.** Reproducible laptop-scale experiments verify the constructions, quantify compute-to-target failure of a nearly perfect single-power fit, and test certificate calibration.

The manuscript is in `paper/main.tex`; complete proofs are in `paper/appendix_body.tex` and `paper/supplement.tex`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

pytest -q
python experiments/run_all.py
make -C paper draft
```

`experiments/run_all.py` regenerates all numerical summaries in `results/` and all paper figures in `paper/figures/` from a fixed random seed.

To compile the AAAI version, retain the repository's AAAI-27 author-kit files and run:

```bash
make -C paper main
make -C paper supplement
```

The Makefile searches the repository for `aaai2027.sty` and uses the corresponding bibliography style directory.

## Repository layout

- `paper/`: AAAI manuscript, generic build fallback, bibliography, and full supplement.
- `src/scaling_extrapolation/`: exact spectral risks and LP certificates.
- `experiments/`: deterministic scripts for every current figure and table.
- `results/`: machine-readable numerical outputs.
- `tests/`: theorem-level and implementation tests.
- `notes/`: theorem roadmap, audit notes, and submission plan.

## Reproducibility status

The current exact-risk experiments take seconds on a laptop and require no GPU. The test suite checks the zeta-tail implementation, exact telescoping construction, matched value/slope pair, theorem inequalities, hidden-component calibration, and LP containment under correct specification.

## Remaining submission milestones

The current branch is a strong proof-complete first paper draft, not yet the final empirical submission. The highest-priority additions are rolling-origin evaluation on public neural scaling curves, finite-sample ridge/SGD validation, comparison against conventional nonlinear and Bayesian intervals, and a continuous-exponent or discretization-robust certificate. See `notes/submission_roadmap.md`.
