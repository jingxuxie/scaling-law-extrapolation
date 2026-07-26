# Experiments

All experiments are CPU-only and deterministic.

## Exact constructions

```bash
PYTHONPATH=src python experiments/run_all.py
```

This produces the matched floor--exponent pair, hidden crossover, compute-to-target comparison, and finite-dictionary coverage demonstration.

## Submission benchmark

```bash
PYTHONPATH=src \
  SYNTHETIC_REPETITIONS=30 \
  BOOTSTRAP_REPETITIONS=20 \
  OLS_REPETITIONS=5000 \
  python experiments/run_submission.py --stage all
```

Each stage can also be run independently with `--stage continuous`, `ols`, `synthetic`, or `public`. The `all` mode launches each stage in a fresh subprocess to isolate repeated nonlinear fits.

The public stage downloads two files from the NSL-PFN repository at commit `10a5a60911a2a22ef453d8b4ddf1408b6a6075d9`, verifies their SHA-256 checksums, and evaluates chronological prefixes only. See `data/public/README.md`.

Outputs under `results/` include both per-run predictions and aggregate summaries. PDF figures use embedded TrueType fonts (`matplotlib.pdf.fonttype=42`) to satisfy AAAI's prohibition on Type-3 fonts.
