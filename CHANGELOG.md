# Changelog

The full, formatted changelog is in the documentation:
https://middoe.readthedocs.io/en/latest/changelog.html

## 0.2.0

The code now matches the methods of the MIDDoE paper (Tabrizi et al., *Digital Chemical
Engineering* 17 (2025) 100276), with many bug fixes and more robust parameter estimation.
**Results can differ from 0.1.x.** Main changes in behaviour:

- `parmest` minimises the paper's objectives (eqs. S5–S7, per-point σ); `uncert` reports the same values.
- `iden_opt['tol']` is a relative tolerance for every optimiser; use a tight value such as `1e-8`.
- Design priors are off by default (`des_opt['priors']`) and correctly scaled when on.
- Sobol first-order indices use the Jansen estimator.
- The `'Chi'` objective divides by `max(|y|, σ)`.
- `uncert` uses the system `'unc'` when the `MES_E` column is missing.
- `var-cov='H'` and `'B'` are correct for every optimiser; bootstrap replicate filtering is reported.
- New optimiser options `pscale`, `grad`, `grad_step`, `fnorm`.
- Paper option names (`'G'`, `'Ls'`, `'Ln'`, `'SQP'`, `'NM'`, `'LBFGSB'`, `'CS'`, `'J'`) are accepted.
- `run_postprocessing(..., show=False)` for scripts.
- Restructured documentation with a getting-started page and an API reference in workflow order.
