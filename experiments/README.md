# Experiments

Run all current experiments from the repository root:

```bash
python experiments/run_all.py
```

The script is deterministic and writes:

- `matched_floor_exponent.csv`: pilot gap, target gap, and Gaussian KL for the value-and-slope-matched pair.
- `hidden_crossover.csv`: visible/hidden tail ratios across scales.
- `single_power_fit.csv`: compute-to-target underestimation from an almost exact pilot fit.
- `certificate_coverage.csv`: marginal and simultaneous coverage plus interval widths.
- paper figures under `paper/figures/`.

All current experiments use exact Hurwitz-zeta tails and finish in seconds on a laptop.
