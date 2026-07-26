# Public learning-curve data

The submission pipeline downloads, but does not vendor, two public CSV files from the official NSL-PFN repository:

- `benchmark.lang.csv`
- `benchmark.vision.csv`

Source repository: `DongWooLee-Eli/nslpfn`
Pinned commit: `10a5a60911a2a22ef453d8b4ddf1408b6a6075d9`
Upstream license: MIT
Associated paper: Lee et al., *Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks*, ICML 2025.

The runner verifies:

```text
benchmark.lang.csv   63c346db00a66b692e1ea9a0f71a2f2b267e55f74be5b863a92e199dffbf0983
benchmark.vision.csv 23b00692d71e58aa5a2a941e831d57688e0977fdcf5f5125ff7b64283b5d9e83
```

The paper uses five NMT curves and four image-classification curves, with three chronological pilot cutoffs per curve. Because the files do not provide independent checkpoint standard errors, these evaluations are explicitly descriptive model-mismatch sensitivity studies rather than direct applications of the known-standard-error coverage theorem.
