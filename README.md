<p align="center">
  <img src="https://research.dii.unipd.it/capelab/wp-content/uploads/sites/36/2025/03/logo.png" width="500">
</p>


# MIDDoE: Model Identification, Discrimination, and Design of Experiments

MIDDOE is an open-source Python package developed to support model identification for dynamic lumped models. 
It addresses gaps in existing tools by offering a structured framework that integrates key techniques 
for model identification, discrimination, and experimental design. MIDDOE is designed to balance flexibility, 
accessibility, and practical usability.

## Key features:

 -  Comprehensiveness and Consistency: Ensures essential steps of model identification are included within a structured workflow.

 -  Flexibility: Allows integration with external simulators while also providing monolithic built-in options.

 -  Adaptability: Easily accommodates common physical constraints to enhance practical applicability.

 -  Accessibility: Utilises NumPy-based structures to improve generality and ensure minimal dependencies.

 -  Practicality: Offers a user-friendly interface suitable for experiments beyond the process systems engineering field.


## Functionalities:

A collection of numerical capabilities is embedded in MIDDOE to facilitate the model identification process. These include:

-  Sensitivity Analysis: Evaluates the influence of parameters on model behaviour.

-  Estimability Analysis: Assesses which parameters can be reliably estimated from available data.

-  Parameter Estimation: Estimates model parameters based on experimental data.

-  Uncertainty Analysis: Quantifies uncertainties in parameter estimates and model predictions.

-  Model-Based Design of Experiments for Model Discrimination (MBDoE-MD): Designs experiments to distinguish between competing models.

-  Model-Based Design of Experiments for Parameter Precision (MBDoE-PP): Designs experiments to improve parameter precision and model robustness.

-  Model Validation: Assesses the model's predictive capability using independent data.

Some service functionalities are also provided to support usage, and post-processing of results, including: 
-  Data handling,
-  Plotting and reporting, 
-  Insilico data generator.



## Applications:

MIDDOE has been tested across a variety of domains, including:

-  Pharmaceutical systems

-  Biological processes

-  Mineral systems

-  Chemical processes

### Installation

#### PyPI 

    pip install middoe

#### git clone

    git clone https://github.com/zuhairblr/middoe.git


### Tutorials and Examples

* A set of MIDDoE case studies to call the identification workflow in cases of Pharmaceutical, Biological ,Mineral and Chemical systems are added to the package.
* A documentation to guide the user through the package functionalities and how to use them will be available soon.

### Getting Help

For help and community support, you can:
* use the #MIDDOE tage on StackOverflow
* contact the developer (zuhairtabrizi@gmail.com)

### Developers

Contributions are welcome! If you'd like to improve MIDDOE, report issues, or suggest new features, please visit the GitHub repository for guidelines.
By contributing to this project, you are agreeing to the following terms and conditions:
1. You agree your contributions are submitted under the MIT License. 
2. You confirm that you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer.

### License

MIDDOE is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements

This work is part of the CO2Valorize project that has received funding from the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073547.
MIDDOE was developed to address gaps in existing tools and has benefited from insights gained through applications in various disciplines. Special thanks to the research community for their ongoing contributions and feedback.

