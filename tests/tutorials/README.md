# MIDDoE tutorials

Short, heavily explained notebooks, one MIDDoE function each, on two systems: a fed-batch yeast
fermentation (Monod growth, 4 parameters) and a pharmaceutical batch reaction (Arrhenius kinetics,
6 parameters). The paper case studies are in `tests/middoe_paper`. All notebooks open in Google Colab
from the [tutorials page](https://middoe.readthedocs.io/en/latest/tutorials.html).

| Folder | Question | Functions |
|---|---|---|
| `01_estimation_and_uncertainty` | Fit a Monod model and know how sure you are (every optimiser, objective, covariance method) | `expera`, `parmest`, `uncert`, `save_rounds`, `run_postprocessing` |
| `01_simple/a` ... `d` | Five estimation recipes side by side | `iden_opt` settings |
| `02_sensitivity/a_kinetic_parameters` | Which kinetic constants shape the product and the impurity? | `sensa` (parameters) |
| `02_sensitivity/b_operating_conditions` | Temperature or feed: which operating condition matters? | `sensa` (inputs) |
| `03_estimability` | How many of six Arrhenius parameters can four batches support? | `estima` |
| `04_design_precision/a_criteria` | Which feeding policy pins down the growth kinetics? D, A, E, ME | `mbdoe_pp` |
| `04_design_precision/b_ill_conditioned` | Correlated Arrhenius parameters: ORB and SEM designs | `mbdoe_pp` |
| `04_design_precision/c_profiles_and_sampling` | Steps, ramps, monotone feeds, fewer samples: what each rule costs | `mbdoe_pp`, `system` |
| `05_design_discrimination/a_hunter_reiner` | Monod or Contois? The run that tells them apart | `mbdoe_md` (HR) |
| `05_design_discrimination/b_hr_vs_bff` | Is there a maintenance term? HR versus BFF | `mbdoe_md` (HR, BFF) |
| `06_validation` | Does the model predict batches it has never seen? | `validation` |

Tutorials 2 to 6 have a `QUICK` switch in their first cell: `True` checks the set-up in minutes with tiny
budgets and a single design, `False` gives the real result.

**How to run:** open `tutorial.ipynb` from its own folder in Jupyter or PyCharm and run all cells.
Each notebook deletes and recreates its own outputs (`data.xlsx`, `error_study/`, plot folders,
`post_processing/`, `iden_results.jac`) on every run. Every notebook ends with the reporting step.
