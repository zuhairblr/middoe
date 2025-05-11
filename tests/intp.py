import os
from middoe.iden_utils import Plotting_FinalResults, Plot_estimability, save_sobol_results_to_excel
from middoe.log_utils import load_from_jac

# Usage
file_path = 'C:\\datasim\\14\\results.jac'  # Replace with your actual file path
file_path2 = 'C:\\datasim\\14\\results2.jac'  # Replace with your actual file path

# Load the results
results = load_from_jac(file_path)
results2 = load_from_jac(file_path2)

# Extract directory from file_path
directory_path = os.path.dirname(file_path)
base_path = 'C:\\datasim'
modelling_folder = str(14)  # No leading backslash here

# Join the base path and modelling folder
path = os.path.join(base_path, modelling_folder)

save_sobol_results_to_excel(results2, path)

# Call the plotting functions only if results are successfully loaded
if results:
    # Generate aggregated plots for P and T values across selected rounds
    pcomp_plot_instance = Plotting_FinalResults(results, None, [])  # No specific solver for overall plots
    pcomp_plot_instance.pcomp_plot(directory_path)
    pcomp_plot_instance.tcomp_plot(directory_path)

    # Iterate over solvers to create solver-specific plots
    for solver in ['f20']:
        # Create an instance of the Plotting_FinalResults class for the specific solver and rounds
        selected_rounds = [1, 2, 3]  # Define the rounds to include
        plotting = Plotting_FinalResults(results, solver, selected_rounds)

        # Define filenames for each type of plot
        confidence_intervals_filename = f"{directory_path}/{solver}_confint.png"
        ellipsoid_volume_filename = f"{directory_path}/{solver}_ellipsoid_volume.png"
        std_devs_filename = f"{directory_path}/{solver}_parameter_std_devs.png"
        parameter_estimates_filename = f"{directory_path}/{solver}_parameter_estimates.png"
        tval_plot_filename = f"{directory_path}/{solver}_tval.png"
        fit_plot_filename = f"{directory_path}/{solver}_fit_plot"

        # Generate confidence interval and parameter estimation plots
        plotting.conf_plot(
            filename=confidence_intervals_filename,
            ellipsoid_volume_filename=ellipsoid_volume_filename,
            std_devs_filename=std_devs_filename,
            parameter_estimates_filename=parameter_estimates_filename
        )

        # Generate P and T value plots for the solver
        plotting.pt_plot(tval_plot_filename, results)

        # Generate reporter data
        plotting.reporter(fit_plot_filename)

        # Additional solver-specific plotting (if required)
        Plot_estimability(results, directory_path, solver)
