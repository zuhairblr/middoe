# Changelog

The full, formatted changelog is in the documentation:
https://middoe.readthedocs.io/en/latest/changelog.html


## 0.2.2

Tutorials and documentation only; the code is unchanged.

- Tutorials page in two parts: tutorials, one per function with notebooks that compare its options
  (estimation and uncertainty; sensitivity of parameters and inputs; estimability; design for
  precision with D, A, E, ME, ORB, SEM and profile/sampling constraints; design for discrimination
  with HR and BFF; cross-validation), and the case studies of the MIDDoE paper.
- Every new tutorial has a `QUICK` switch for a quick check of the set-up.
- Interactive figures for every tutorial.

## 0.2.1

Documentation and package information only; the code is the same as in 0.2.0.

- New documentation front page and a hands-on tutorials page (five notebooks on two real systems, each with an animation; they open in Google Colab).
- Clearer PyPI description, keywords, classifiers and links; the README starts with what MIDDoE does and when to use it.
- `llms.txt`: a plain-text summary of MIDDoE at the root of the documentation site.
- The logo is served from the documentation itself.

## 0.2.0

The code now matches the methods of the MIDDoE paper (Tabrizi et al., *Digital Chemical
Engineering* 17 (2025) 100276), with many bug fixes and more robust parameter estimation.
**Results can differ from 0.1.x.** Main changes in behaviour:

- `parmest` minimises the paper's objectives (eqs. S5–S7, per-point σ); `uncert` reports the same values.
- `iden_opt['tol']` is a relative tolerance for every optimiser; use a tight value such as `1e-8`.
- Design priors are off by default (`des_opt['priors']`) and correctly scaled when on.
- Sobol first-order indices use the Jansen estimator.
- The `'Chi'` objective divides by `max(|y|, σ)`.
- `uncert` uses the system `'unc'` when the `MES_E` column is missing; empty `MES_E` cells are filled the same way.
- A missing `'unc'` gives a clear error in `expera` and a warning in the designs (which assume σ = 1).
- `var-cov='H'` and `'B'` are correct for every optimiser; bootstrap replicate filtering is reported.
- Bootstrap replicates start from the reference estimate plus 4 random starts (`boot_start='mixed'`), about 4x faster than a full multi-start per replicate with the same optima.
- New optimiser options `pscale`, `grad`, `grad_step`, `fnorm`.
- Paper option names (`'G'`, `'Ls'`, `'Ln'`, `'SQP'`, `'NM'`, `'LBFGSB'`, `'CS'`, `'J'`) are accepted.
- `run_postprocessing(..., show=False)` for scripts; optional `true_theta=`.
- `expera` shows a small figure of each new data set and its noise (`insilicos['plt']`, default on).
- gPROMS models (`'gpr'`) work again: they are no longer loaded as a Python file first.
- `des_opt['conv_rep']` now defaults to `False`, and the per-iteration design metrics (FIM metrics and a curvature probe, about as many extra model evaluations as the optimisation itself) are computed only when it is `True`. Designs are unchanged; MBDoE-PP runs about 2x faster by default.
- Tutorial 1 has four short companion notebooks (`tests/tutorials/01_simple/`, variants A–D: relative or absolute noise, Fisher, Hessian or bootstrap uncertainty, fast or multi-start optimiser).
- Plots corrected and extended: 95 % confidence ellipses (were 1σ), physical estimates ± CI per round, error bars, prediction bands, weighted residuals and parity plots, richer Excel reports, design figures displayed from the main process. `iden_opt['plt_s']` controls display for `save_rounds`, `estima` and `validation`; `models['theta_names']` labels parameters.
- Restructured documentation: an onboarding guide, a getting-started page, interactive figures, the full page tree in the side menu and an API reference in workflow order.
- The four case studies of the MIDDoE paper are notebooks in `tests/middoe_paper/` (one folder per case: the notebook and its model files). These and the tutorials open in Google Colab.
