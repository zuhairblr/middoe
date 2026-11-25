# MIDDoE tutorials

Short, heavily explained notebooks. All of them use the same model: the fed-batch yeast fermentation
of Case Study 1 of the MIDDoE paper (Monod growth with maintenance, 4 parameters). Each notebook
covers one group of MIDDoE modules and explains every option it uses.

| Folder | Covers | Modules | Run time |
|---|---|---|---|
| `01_estimation_and_uncertainty` | model file, `system` and `models` dictionaries, in-silico data, `data.xlsx` layout; estimation with every optimiser and objective; uncertainty with every covariance method; error types and noise levels; reporting (rounds, plots, saved results, final report) | `krnl_expera`, `iden_parmest`, `iden_uncert`, `log_utils`, `iden_utils` | about 25 min |

More notebooks (experiment design, sensitivity and estimability analysis, validation and reports) will
follow.

**How to run:** open `tutorial.ipynb` from its own folder in Jupyter or PyCharm and run all cells.
Each notebook deletes and recreates its own outputs (`data.xlsx`, `error_study/`, plot folders,
`post_processing/`, `iden_results.jac`) on every run. Every notebook ends with the reporting step.
