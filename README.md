<p align="center">
  <img src="https://raw.githubusercontent.com/zuhairblr/middoe/main/docs/source/_static/middoe-logo.png" width="420" alt="MIDDoE logo">
</p>

<h1 align="center">MIDDoE: Model Identification, Discrimination and Design of Experiments</h1>

<p align="center">
    <a href="https://pypi.org/project/middoe/"><img src="https://img.shields.io/pypi/v/middoe?color=blue&label=PyPI&logo=pypi&logoColor=white" alt="PyPI"></a>
    <a href="https://middoe.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/middoe?label=docs" alt="Documentation"></a>
    <a href="https://doi.org/10.1016/j.dche.2025.100276"><img src="https://img.shields.io/badge/paper-DCHE%202025-informational" alt="Paper"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

**MIDDoE is an open-source Python package for model-based design of experiments (MBDoE), parameter
estimation and model discrimination for dynamic models**, such as ODE kinetic models of chemical,
biological, pharmaceutical and mineral processes. It fits the unknown parameters of your model to
data, tells you how precise they are, picks the right model among rival ones, and designs the next
experiment (inputs and sampling times) that is most informative.

Use MIDDoE when you want to:

- **estimate the parameters** of a dynamic (ODE) model from experimental data, with confidence
  intervals, t-tests and correlations;
- **design optimal experiments** for parameter precision (D-, A-, E- and modified E-optimal designs
  from the Fisher information matrix);
- **discriminate between rival models** and design the experiment that tells them apart
  (Hunter-Reiner and Buzzi-Ferraris-Forzatti criteria, P-test);
- find **which parameters matter** (global Sobol sensitivity analysis) and **which can be estimated**
  from the data you have (estimability analysis);
- run the whole **identification loop** (design, experiment, fit, check) in silico before going to
  the lab, then with real data.

You describe the study in a few plain Python dictionaries and call one function per step; the model
is an ordinary Python function (or a gPROMS model).

---

## 🚀 Installation

```bash
pip install middoe
```

Or clone the repository and install it with `pip install .` from its folder. Python 3.9 to 3.13.

## ⚡ Quick start

```python
from middoe.krnl_expera import expera
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
from middoe.des_pp import mbdoe_pp

est = parmest(system, models, iden_opt)                # fit the parameters
unc = uncert(est, system, models, iden_opt)            # how sure are they?
design = mbdoe_pp(des_opt, system, models, round=1)    # plan the next experiment
expera(system, models, insilicos, design, expr=3, swps=design['swps'])  # run it (in silico)
```

The [getting-started page](https://middoe.readthedocs.io/en/latest/getting_started.html) builds the
dictionaries for a complete example that runs in a minute.

## 📚 Documentation and tutorials

- **Documentation:** [middoe.readthedocs.io](https://middoe.readthedocs.io/en/latest/): an onboarding
  guide, interactive figures, every configuration key and the full API reference.
- **Hands-on tutorials:** [five notebooks](https://middoe.readthedocs.io/en/latest/tutorials.html) on
  two real systems (a fed-batch yeast fermentation and a pharmaceutical batch reaction), from a first
  fit to designing experiments, including the case studies of the MIDDoE paper. They run in the
  browser with Google Colab. Files: [`tests/tutorials`](https://github.com/zuhairblr/middoe/tree/main/tests/tutorials)
  and [`tests/middoe_paper`](https://github.com/zuhairblr/middoe/tree/main/tests/middoe_paper).

## ⚙️ What is inside

| Step | Function | Method |
|---|---|---|
| Sensitivity analysis | `sensa` | global Sobol indices (scrambled Sobol-sequence sampling, Jansen estimator) |
| In-silico experiments | `expera` | simulation with absolute or relative measurement noise |
| Parameter estimation | `parmest` | weighted least squares, maximum likelihood; local, multi-start or global optimisers |
| Uncertainty analysis | `uncert` | Fisher information, Hessian or bootstrap covariance; confidence intervals, t-tests, P-test |
| Estimability analysis | `estima` | orthogonalisation ranking and the rCC criterion |
| MBDoE for parameter precision | `mbdoe_pp` | D, A, E, modified E; ORB and SEM for ill-conditioned systems |
| MBDoE for model discrimination | `mbdoe_md` | Hunter-Reiner, Buzzi-Ferraris-Forzatti |
| Validation and reporting | `validation`, `save_rounds`, `run_postprocessing` | cross-validation, plots and Excel reports per round |

Applied to pharmaceutical, biological, mineral and chemical processes.

---

## 💬 Getting help

- Questions and bug reports: [GitHub issues](https://github.com/zuhairblr/middoe/issues), or the `#MIDDoE` tag on **StackOverflow**.
- Contact the development team:
    - **Zuhair Tabrizi** — [zuhairtabrizi@gmail.com](mailto:zuhairtabrizi@gmail.com)
    - **Dr. Elena Barbera** — [elena.barbera@unipd.it](mailto:elena.barbera@unipd.it)
    - **Dr. Wilson Ricardo Leal Da Silva** — [wilson.dasilva@flsmidth.com](mailto:wilson.dasilva@flsmidth.com)
    - **Prof. Fabrizio Bezzo** — [fabrizio.bezzo@unipd.it](mailto:fabrizio.bezzo@unipd.it)

## 👨‍💻 Contributing

Contributions are welcome: report issues or suggest features on the
[GitHub repository](https://github.com/zuhairblr/middoe). By contributing you agree that your
contributions are submitted under the **MIT License** and that you have the right to submit them.

## 🛡️ License

MIT License. See the `LICENSE` file.

---

## 📖 Citation

If you use MIDDoE in your research, please cite:

1. Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
   MIDDoE: An MBDoE Python package for model identification, discrimination, and calibration.
   *Digital Chemical Engineering*, 17, 100276.
   [https://doi.org/10.1016/j.dche.2025.100276](https://doi.org/10.1016/j.dche.2025.100276)

2. Tabrizi, S.Z.B., Barbera, E., Da Silva, W.R.L., & Bezzo, F. (2025).
   A Python/Numpy-based package to support model discrimination and identification.
   In J.F.M. Van Impe et al. (Eds.), *Systems and Control Transactions*, 4, 1282-1287.
   [https://doi.org/10.69997/sct.192104](https://doi.org/10.69997/sct.192104)

For the design criteria for ill-conditioned systems (ORB and SEM), please also cite:

3. Tabrizi, Z., Leal da Silva, W.R., & Bezzo, F. (2026).
   Two novel criteria for model-based design of experiments in ill-conditioned systems.
   *Computers & Chemical Engineering*, 214, 109806.
   [https://doi.org/10.1016/j.compchemeng.2026.109806](https://doi.org/10.1016/j.compchemeng.2026.109806)

### Applied in

Tabrizi, Z., Rodriguez, C., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
Wet Carbonation of Industrial Recycled Concrete Fines: Experimental Study and Reaction Kinetic Modeling.
*Ind Eng Chem Res*, 64(45), 21412–21425.
[https://doi.org/10.1021/acs.iecr.5c02835](https://doi.org/10.1021/acs.iecr.5c02835)

---

## 🙏 Acknowledgements

This work is part of the **CO2Valorize** project, funded by the European Union’s Horizon Europe research and innovation programme under the **Marie Skłodowska-Curie Grant Agreement No. 101073547**.

**MIDDoE** is a collaborative effort between:
- 🏫 The **[Computer-Aided Process Engineering (CAPE) Laboratory](https://research.dii.unipd.it/capelab/)** at the University of Padova, Italy
- 🏢 The **[Green Innovation team at FLSmidth Cement](https://www.flsmidth-cement.com/events/green-cement-and-concrete-innovation)**, Denmark

MIDDoE also integrates with gPROMS, a commercial process and model simulator developed by **[Siemens Industry Software Limited](https://www.siemens.com/global/en/products/automation/industry-software/gproms-digital-process-design-and-operations.html)**, UK.

We warmly thank the research community for their valuable feedback and ongoing contributions that have helped shape and improve this project.

---

<p align="center">
💻 Developed with ❤️ by the MIDDoE team
</p>
