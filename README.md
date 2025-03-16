[//]: # (<p align="center">)

[//]: # (  <img src="https://research.dii.unipd.it/capelab/wp-content/uploads/sites/36/2025/03/logo.png?ver=1741697849" width="500">)

[//]: # (</p>)

[//]: # ()
[//]: # ()
[//]: # (# MIDDoE: Model-&#40;based&#41; Identification, Discrimination, and Design of Experiments)

[//]: # ()
[//]: # (MIDDOE is an open-source Python package developed to support model identification for dynamic lumped models. )

[//]: # (It addresses gaps in existing tools by offering a structured framework that integrates key techniques )

[//]: # (for model identification, discrimination, and experimental design. MIDDOE is designed to balance flexibility, )

[//]: # (accessibility, and practical usability.)

[//]: # ()
[//]: # (## Key features:)

[//]: # ()
[//]: # ( -  Comprehensiveness and Consistency: Ensures essential steps of model identification are included within a structured workflow.)

[//]: # ()
[//]: # ( -  Flexibility: Allows integration with external simulators while also providing monolithic built-in options.)

[//]: # ()
[//]: # ( -  Adaptability: Easily accommodates common physical constraints to enhance practical applicability.)

[//]: # ()
[//]: # ( -  Accessibility: Utilises NumPy-based structures to improve generality and ensure minimal dependencies.)

[//]: # ()
[//]: # ( -  Practicality: Offers a user-friendly interface suitable for experiments beyond the process systems engineering field.)

[//]: # ()
[//]: # ()
[//]: # (## Functionalities:)

[//]: # ()
[//]: # (A collection of numerical capabilities is embedded in MIDDOE to facilitate the model identification process. These include:)

[//]: # ()
[//]: # (-  Sensitivity Analysis: Evaluates the influence of parameters on model behaviour.)

[//]: # ()
[//]: # (-  Estimability Analysis: Assesses which parameters can be reliably estimated from available data.)

[//]: # ()
[//]: # (-  Parameter Estimation: Estimates model parameters based on experimental data.)

[//]: # ()
[//]: # (-  Uncertainty Analysis: Quantifies uncertainties in parameter estimates and model predictions.)

[//]: # ()
[//]: # (-  Model-Based Design of Experiments for Model Discrimination &#40;MBDoE-MD&#41;: Designs experiments to distinguish between competing models.)

[//]: # ()
[//]: # (-  Model-Based Design of Experiments for Parameter Precision &#40;MBDoE-PP&#41;: Designs experiments to improve parameter precision and model robustness.)

[//]: # ()
[//]: # (-  Model Validation: Assesses the model's predictive capability using independent data.)

[//]: # ()
[//]: # (Some service functionalities are also provided to support usage, and post-processing of results, including: )

[//]: # (-  Data handling,)

[//]: # (-  Plotting and reporting, )

[//]: # (-  Insilico data generator.)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (## Applications:)

[//]: # ()
[//]: # (MIDDOE has been tested across a variety of domains, including:)

[//]: # ()
[//]: # (-  Pharmaceutical systems)

[//]: # ()
[//]: # (-  Biological processes)

[//]: # ()
[//]: # (-  Mineral systems)

[//]: # ()
[//]: # (-  Chemical processes)

[//]: # ()
[//]: # (### Installation)

[//]: # ()
[//]: # (#### PyPI )

[//]: # ()
[//]: # (    pip install middoe)

[//]: # ()
[//]: # (#### git clone)

[//]: # ()
[//]: # (    git clone https://github.com/zuhairblr/middoe.git)

[//]: # ()
[//]: # ()
[//]: # (### Tutorials and Examples)

[//]: # ()
[//]: # (* A set of MIDDoE case studies to call the identification workflow in cases of Pharmaceutical, Biological ,Mineral and Chemical systems are added to the package.)

[//]: # (* A documentation to guide the user through the package functionalities and how to use them will be available soon.)

[//]: # ()
[//]: # (### Getting Help)

[//]: # ()
[//]: # (For help and community support, you can:)

[//]: # (* use the #MIDDOE tage on StackOverflow)

[//]: # (* contact the developer team &#40;Zuhair Tabrizi: zuhairtabrizi@gmail.com, Prof. Fabrizio Bezzo: fabrizio.bezzo@unipd.it &#41;)

[//]: # ()
[//]: # (### Developers)

[//]: # ()
[//]: # (Contributions are welcome! If you'd like to improve MIDDOE, report issues, or suggest new features, please visit the GitHub repository for guidelines.)

[//]: # (By contributing to this project, you are agreeing to the following terms and conditions:)

[//]: # (1. You agree your contributions are submitted under the MIT License. )

[//]: # (2. You confirm that you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer.)

[//]: # ()
[//]: # (### License)

[//]: # ()
[//]: # (MIDDOE is licensed under the MIT License. See the LICENSE file for details.)

[//]: # ()
[//]: # (### Acknowledgements)

[//]: # ()
[//]: # (This work is part of the CO2Valorize project that has received funding from the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073547.)

[//]: # (MIDDoE is a collaborative effort between the Computer-Aided Process Engineering &#40;CAPE&#41; group at the University of Padova, Italy, and the Green Innovation team at FLSmidth Cement, Denmark)

[//]: # (MIDDOE was developed to address gaps in existing tools and has benefited from insights gained through applications in various disciplines. Special thanks to the research community for their ongoing contributions and feedback.)

[//]: # ()


<p align="center">
  <img src="https://research.dii.unipd.it/capelab/wp-content/uploads/sites/36/2025/03/logo-Page-5.png" width="500">
</p>


<h1 align="center">MIDDoE: Model-(based) Identification, Discrimination, and Design of Experiments </h1>

<p align="center">
    <a href="https://pypi.org/project/middoe/"><img src="https://img.shields.io/pypi/v/middoe?color=blue&label=PyPI&logo=pypi&logoColor=white" alt="PyPI"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
    <a href="https://github.com/zuhairblr/middoe"><img src="https://img.shields.io/github/stars/zuhairblr/middoe.svg?style=social" alt="GitHub Stars"></a>
</p>

---

## 🌍 About MIDDoE
**MIDDoE** is an open-source Python package designed to streamline **model identification** for dynamic lumped models. Developed to address gaps in existing tools, MIDDoE offers a structured framework integrating:

✅ **Model Identification**  
✅ **Model Discrimination**  
✅ **Experimental Design**  

With its flexible and user-friendly design, MIDDoE ensures practical usability across various scientific disciplines.

---

## ✨ Key Features
✅ **Comprehensive Workflow** — A structured framework that covers all essential steps in model identification.  
✅ **Flexible Integration** — Supports external simulators while offering built-in options.  
✅ **Adaptable Design** — Easily accommodates physical constraints for practical applications.  
✅ **Accessible Framework** — Uses NumPy-based structures for improved generality and minimal dependencies.  
✅ **User-Friendly Interface** — Designed for use beyond traditional process systems engineering applications.  

---

## ⚙️ Functionalities
MIDDoE offers a wide range of numerical capabilities to support model identification, including:

🔍 **Sensitivity Analysis** — Identifies key parameters influencing model behaviour.  
📊 **Estimability Analysis** — Determines which parameters can be reliably estimated.  
📈 **Parameter Estimation** — Estimates model parameters based on experimental data.  
📉 **Uncertainty Analysis** — Evaluates confidence in model predictions.  
🧪 **MBDoE for Model Discrimination (MBDoE-MD)** — Optimises experiments to distinguish between competing models.  
🎯 **MBDoE for Parameter Precision (MBDoE-PP)** — Designs experiments to improve parameter precision.  
🧪 **Model Validation** — Assesses predictive accuracy using independent data.  

Additional service functionalities include:  
- 📂 **Data Handling**  
- 📑 **Plotting and Reporting**  
- 🧬 **In-silico Data Generation**  

---

## 🧪 Applications
MIDDoE has been successfully applied across various domains, including:  
- 💊 **Pharmaceutical systems**  
- 🧫 **Biological processes**  
- 🪨 **Mineral systems**  
- ⚗️ **Chemical processes**  

---

## 🚀 Installation
MIDDoE can be installed via **PyPI** or by cloning the repository:

### PyPI Installation
```bash
pip install middoe
```

### Git Clone
```bash
git clone https://github.com/zuhairblr/middoe.git
```

---

## 📚 Tutorials and Examples
MIDDoE provides a comprehensive set of tutorials and case studies demonstrating its application in:

- 📋 **Pharmaceutical Systems**  
- 🧬 **Biological Processes**  
- 🪨 **Mineral Systems**  
- ⚗️ **Chemical Processes**  

📝 **Documentation** will be available soon to guide users through package functionalities.

---

## 💬 Getting Help
For support and community interaction:  
- 🏷️ Use the `#MIDDoE` tag on **StackOverflow**.  
- 📧 Contact the development team:  
    - **Zuhair Tabrizi** — [zuhairtabrizi@gmail.com](mailto:zuhairtabrizi@gmail.com)
    - **Dr. Elena Barbera** — [elena.barbera@unipd.it ](mailto:elena.barbera@unipd.it )
    - **Dr. Wilson Ricardo Leal Da Silva** — [wilson.dasilva@flsmidth.com ](mailto:wilson.dasilva@flsmidth.com )
    - **Prof. Fabrizio Bezzo** — [fabrizio.bezzo@unipd.it](mailto:fabrizio.bezzo@unipd.it)
       

---

## 👨‍💻 Developers
We welcome contributions! If you'd like to improve MIDDoE, report issues, or suggest new features, please visit the [GitHub repository](https://github.com/zuhairblr/middoe) for guidelines.

### Contributing Terms
By contributing to MIDDoE, you agree to the following terms:  
1️⃣ Your contributions are submitted under the **MIT License**.  
2️⃣ You confirm that you have the rights to submit these contributions.  

---

## 🛡️ License
MIDDoE is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgements
This work is part of the **CO2Valorize** project, funded by the European Union’s Horizon Europe research and innovation programme under the **Marie Skłodowska-Curie Grant Agreement No. 101073547**.

**MIDDoE** is a collaborative effort between:  
- 🏫 The **[Computer-Aided Process Engineering (CAPE)](https://research.dii.unipd.it/capelab/)** lab at the University of Padova, Italy  
- 🏢 The **[Green Innovation team at FLSmidth Cement](https://www.flsmidth-cement.com/events/green-cement-and-concrete-innovation)**, Denmark  

Special thanks to the research community for their valuable contributions and feedback.

---

<p align="center">
💻 Developed with ❤️ by the MIDDoE team
</p>