import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, chi2
from scipy.special import gamma
from scipy.interpolate import interp1d
from pathlib import Path

logger = logging.getLogger(__name__)

# # Set Times New Roman font globally for all elements
# rcParams['font.family'] = 'arial'

def Plot_estimability(round_data, path, solver):
    """
    Plot the estimability of parameters across different rounds for a given model. Estimability plotter post analysis.

    Parameters:
    round_data (dict): Data related to each round of experiments.
    path (str): Base path where the plot will be saved.
    model (str): The name of the model for which the estimability is being plotted.

    Returns:
    None
    """

    # Helper function for pseudo-logarithmic transformation
    def pseudo_log_transform(value, offset=1e-6):
        if value > 0:
            return np.log(value + 1)
        elif value < 0:
            return -np.log(abs(value) + 1)
        else:
            return offset

    # Check if any round has valid rCC_values for the specified model
    has_rcc_data = any(
        round_info.get('rCC_values') is not None and solver in round_info['rCC_values'] and
        round_info['rCC_values'][solver] is not None
        for round_info in round_data.values()
    )

    # If no rCC_values are found, print a message and exit the function
    if not has_rcc_data:
        print("No rCC_values found for any round. Plotting is skipped.")
        return

    # Create the folder path once
    base_path = path
    modelling_folder = 'estimability'
    full_path = os.path.join(base_path, modelling_folder)

    # Ensure the 'estimability' directory exists
    os.makedirs(full_path, exist_ok=True)

    # Set up the plot for all rounds
    plt.figure(figsize=(6, 4), dpi=150)  # Larger figure size for better visibility

    # Iterate over the round data and plot each round's data
    for round_key, round_info in round_data.items():
        # Check if 'rCC_values' exists and is a dictionary, and not None
        if round_info.get('rCC_values') is not None and isinstance(round_info['rCC_values'], dict):
            rCC_values = round_info['rCC_values'].get(solver)

            # If rCC_values is None, replace with a list of zeros
            if rCC_values is None:
                print(f"'rCC_values' for model '{solver}' in {round_key} is None. Replacing with zeros.")
                rCC_values = [0] * len(round_info['rCC_values'].get(list(round_info['rCC_values'].keys())[0], []))

            # Replace NaN values with 0
            rCC_values = [0 if np.isnan(value) else value for value in rCC_values]

            # Generate x_values as 1, 2, ..., len(rCC_values)
            x_values = list(range(1, len(rCC_values) + 1))

            # Apply pseudo-logarithmic transformation
            transformed_rCC_values = [pseudo_log_transform(value) for value in rCC_values]

            # Plot the transformed values for this round
            plt.plot(x_values, transformed_rCC_values, marker='o', linestyle='-', linewidth=2,
                     label=f'{round_key}')

    # Set x-axis ticks to integer values
    plt.xticks(x_values)  # Ensure x-axis ticks are integers

    # Configure the plot settings
    # plt.xlabel('number of selected parameters', fontsize=14)
    # plt.ylabel('corrected critical ratio', fontsize=14)
    # plt.title(f'rCC vs k for All Rounds in model_{model}', fontsize=16)
    plt.xlabel('number of selected parameters')
    plt.ylabel('corrected critical ratio')
    plt.title(f'rCC vs k for All Rounds in model_{solver}')

    # Set custom y-ticks based on the transformed values
    all_rCC_values = [val for round_info in round_data.values() if round_info.get('rCC_values') is not None
                      for val in round_info['rCC_values'].get(solver, [])]
    original_values = sorted(set(all_rCC_values))
    transformed_ticks = [pseudo_log_transform(val) for val in original_values]

    # Filter out closely spaced ticks
    min_spacing = 0.5  # Minimum spacing between ticks
    filtered_ticks = []
    filtered_labels = []

    for i in range(len(transformed_ticks)):
        if not filtered_ticks or abs(transformed_ticks[i] - filtered_ticks[-1]) > min_spacing:
            filtered_ticks.append(transformed_ticks[i])
            filtered_labels.append(f"{original_values[i]:.2f}")

    plt.yticks(filtered_ticks, filtered_labels)

    # Add a legend to distinguish different rounds
    plt.legend()

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Define the filename for the plot and save it
    filename = os.path.join(full_path, f'rCC vs parameters in model_{solver}.png')
    plt.savefig(filename, dpi=300)

    # Optionally show the plot
    plt.show()

    print(f"Plot saved in {full_path}")


def plot_rCC_vs_k(x_values, rCC_values, round, solver):
    """
    Plot rCC values against k for a specific round and model. Estimability plotter while analysis.

    Parameters:
    x_values (list): List of x values (k values).
    rCC_values (list): List of rCC values corresponding to x values.
    round (int): The round number.
    framework_settings (dict): Settings related to the framework, including paths and case information.
    model (str): The name of the model for which the plot is being generated.

    Returns:
    None
    """
    # # Create the folder path once
    # base_path = framework_settings['path']
    # modelling_folder = str(framework_settings['case'])  # Convert case to string
    # full_path = os.path.join(base_path, modelling_folder, 'estimability')
    #
    #
    # # Ensure the 'estimability' directory exists
    # os.makedirs(full_path, exist_ok=True)
    # filename = os.path.join(full_path, f'rCC vs k round_{str(round)} in model_{model}.png')

    # Create path: ./estimability/
    estimability_dir = Path.cwd() / "estimability"
    estimability_dir.mkdir(parents=True, exist_ok=True)
    # Define the output file path
    filename = estimability_dir / f"rCC vs k round_{str(round)} in model_{solver}.png"

    # Improve plot quality and save
    plt.figure(figsize=(8, 5), dpi=150)  # Higher DPI for better quality

    plt.plot(x_values, rCC_values, marker='o', linestyle='-', color='b', markersize=8, linewidth=2)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('rCC', fontsize=14)
    plt.title(f'rCC vs k round_{str(round)} in model_{solver}', fontsize=16)
    plt.grid(True)

    # Set the x-axis ticks to be integers only (1, 2, 3, ..., len(x_values))
    plt.xticks(ticks=x_values)

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Define the filename for the plot and save it
    plt.savefig(filename, dpi=300)

    # Optionally show the plot
    plt.show()


def plot_sobol_results(time_samples, sobol_analysis_results, sobol_problem, solver, response_key):
    """
    Plot Sobol sensitivity analysis results for a given model

    Parameters:
    time_samples (list): List of time for samples (time span)
    sobol_analysis_results (dict): Results of the Sobol sensitivity analysis.
    sobol_problem (dict): Problem definition for the Sobol analysis.
    model (str): The name of the model for which the plot is being generated.
    response_key (str): The response key for which the sensitivities are plotted.
    framework_settings (dict): User provided - Settings related to the framework, including paths and case information.

    Returns:
    None
    """
    num_samples = len(time_samples)
    num_keys = sobol_problem['num_vars']
    names = sobol_problem['names']
    path = Path.cwd()
    filename = os.path.join(path, f'Sobol_SIs_response_{response_key} for model_{solver}.png')

    # Initialize arrays for first-order and total-order sensitivities
    first_order_sensitivities = np.zeros((num_samples, num_keys))
    total_order_sensitivities = np.zeros((num_samples, num_keys))

    # Optionally handle confidence intervals if available
    first_order_conf = np.zeros((num_samples, num_keys))
    total_order_conf = np.zeros((num_samples, num_keys))

    # Extract sensitivities from the Sobol analysis results
    for j in range(num_samples):
        first_order_sensitivities[j, :] = sobol_analysis_results[j]["S1"]
        total_order_sensitivities[j, :] = sobol_analysis_results[j]["ST"]

        # Check if confidence intervals exist
        if "S1_conf" in sobol_analysis_results[j]:
            first_order_conf[j, :] = sobol_analysis_results[j]["S1_conf"]
        if "ST_conf" in sobol_analysis_results[j]:
            total_order_conf[j, :] = sobol_analysis_results[j]["ST_conf"]

    # Plotting first-order and total-order sensitivities
    fig, axes = plt.subplots(1, 2, figsize=(7, 5), dpi=300, constrained_layout=True)

    # First-order sensitivity plot
    for idx, label in enumerate(names):
        axes[0].plot(time_samples, first_order_sensitivities[:, idx], label=f"{label} S1", linestyle='--', linewidth=2)
        # Plot shaded confidence intervals if available
        if np.any(first_order_conf):
            axes[0].fill_between(time_samples, first_order_sensitivities[:, idx] - first_order_conf[:, idx],
                                 first_order_sensitivities[:, idx] + first_order_conf[:, idx], alpha=0.2)

    # Total-order sensitivity plot
    for idx, label in enumerate(names):
        axes[1].plot(time_samples, total_order_sensitivities[:, idx], label=f"{label} Total Sobol Index", linewidth=2)
        # Plot shaded confidence intervals if available
        if np.any(total_order_conf):
            axes[1].fill_between(time_samples, total_order_sensitivities[:, idx] - total_order_conf[:, idx],
                                 total_order_sensitivities[:, idx] + total_order_conf[:, idx], alpha=0.2)

    # Customize the plot - titles, gridlines, labels
    axes[0].set_title(f'S1 : Response:{response_key} - Model:{solver}', fontsize=10)
    axes[1].set_title(f'ST : Response:{response_key} - Model:{solver}', fontsize=10)

    for ax in axes:
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Sensitivity Index', fontsize=12)
        # ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim([time_samples[0], time_samples[-1]])

    # Save the figure to the constructed filename
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()



class Plotting_Results:
    """
    A class to handle plotting of results while analysis.

    Attributes:
    path (str): Base path for saving plots.
    case (str): Case identifier for the experiment.
    mutation_settings (dict): Settings related to mutation in the modelling process.
    modelling_folder (str): Folder name for storing modelling results.
    """
    def __init__(self, models,  pltshow, round=None):
        """
        Initialize the Plotting_Results class with modelling settings.

        Parameters:
        models (dict): Settings related to the modelling process.
        round (int, optional): Round number, if you want to tag subfolders.
        """
        self.base_path = Path.cwd()
        self.round = str(round)
        self.mutation_settings = models['mutation']
        self.modelling_folder = self.base_path / 'modelling'
        self.confidence_folder = self.base_path / 'confidence'
        self.modelling_folder.mkdir(parents=True, exist_ok=True)
        self.confidence_folder.mkdir(parents=True, exist_ok=True)
        self.pltshowing= pltshow


    def fit_plot(self, data, result, system):
        """
        Generate and save the model fitting to experimental data plot for the given data and results.

        Parameters:
        data (dict): Experimental data.
        result (dict): Modelling results.
        round (int): The round number.
        case (str): The case type (e.g., 'doe' or 'classic').
        """
        color_map = ['k', 'b', 'r', 'c', 'm', 'y', 'g']
        line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (5, 5)), (0, (1, 10))]  # Different line styles for solvers
        symbols = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', '|', '_', '+', 'x', 'X', 'P', '8']  # Different markers for experimental data

        solver_colors = {}  # To store color mapping for each variable
        variable_styles = {}  # To store line style for each model
        filtered_data = {}
        # Loop through the data sheets
        for sheet_index, (sheet_name, sheet_data) in enumerate(data.items()):
            # t_exp = np.array(sheet_data['t'], dtype=float)  # Experimental time points
            filtered_data[sheet_name] = {}

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
            fig.suptitle(f'Experiment = {sheet_name}', fontsize=12)

            ax_list_1 = [ax1]  # List to handle multiple y-axes in the first subplot

            # First subplot: Only time-variant output variables
            for var_index, var_name in enumerate(result[next(iter(result))]['tv_output_m'][sheet_name].keys()):
                if f'MES_Y:{var_name}' in sheet_data and f'MES_X:{var_name}' in sheet_data:
                    y_exp = np.array([float(val) if val != 'none' else np.nan for val in sheet_data[f'MES_Y:{var_name}']],
                                     dtype=float)
                    t_exp = np.array([float(val) if val != 'none' else np.nan for val in sheet_data[f'MES_X:{var_name}']],
                                     dtype=float)
                    valid_indices = ~np.isnan(y_exp)  # Filter out NaN values

                    # Save filtered data
                    filtered_data[sheet_name][var_name] = {'time': t_exp[valid_indices], 'values': y_exp[valid_indices]}

                    # Create a new y-axis for each variable (except the first one)
                    if var_index > 0:
                        ax_new = ax_list_1[0].twinx()  # Create a new axis sharing the same x-axis
                        ax_new.spines['right'].set_position(('outward', 60 * (var_index - 1)))  # Adjust positioning
                        ax_list_1.append(ax_new)
                    else:
                        ax_new = ax_list_1[0]

                    # Scatter plot for experimental data (time-variant outputs only)
                    ax_new.scatter(filtered_data[sheet_name][var_name]['time'],
                                   filtered_data[sheet_name][var_name]['values'],
                                   color=color_map[var_index % len(color_map)],
                                   marker=symbols[var_index % len(symbols)],
                                   label=f'Experiment {var_name}', alpha=0.6)

                    # Set the y-axis label and tick colors for the variable
                    ax_new.set_ylabel(f'{var_name} (output)', color=color_map[var_index % len(color_map)])
                    ax_new.tick_params(axis='y', labelcolor=color_map[var_index % len(color_map)])

            # Plot model data for different solvers and variables
            for solver_index, solver_name in enumerate(result.keys()):
                # Assign line style for the model (consistent across all variables)
                if solver_name not in variable_styles:
                    variable_styles[solver_name] = line_styles[solver_index % len(line_styles)]

                t_model = np.array(result[solver_name]['t_m'][sheet_name], dtype=float)  # Model time points

                # Plot model data for time-variant outputs
                for var_index, var_name in enumerate(result[solver_name]['tv_output_m'][sheet_name].keys()):
                    model_output = np.array(result[solver_name]['tv_output_m'][sheet_name][var_name], dtype=float)

                    ax_new = ax_list_1[var_index] if var_index < len(ax_list_1) else ax_list_1[0]

                    # Plot model data for this time-variant output
                    ax_new.plot(t_model, model_output, color=color_map[var_index % len(color_map)],
                                linestyle=variable_styles[solver_name], label=f'{solver_name} {var_name}', alpha=0.8)

            # Add legend for the first subplot
            handles_1, labels_1 = ax1.get_legend_handles_labels()
            if handles_1:
                ax1.legend(ncol=2)

            ax1.set_xlabel('Time (t)')
            # ax1.grid(True)

            # Second subplot: Handle DOE or Classic case
            ax_list_2 = [ax2]

            # Only take one model for subplot 2
            single_solver = next(iter(result.keys()))

            # if sheet_data['piecewise_func'].iloc[0] != 'none':
            #     # DOE case: use @ data for this subplot
            #     if '@t' in sheet_data:
            #         doe_t = np.array(sheet_data['@t'], dtype=float)  # Use time data from `data`
            #     else:
            #         raise KeyError(f"Missing '@t' key in data for sheet '{sheet_name}'.")

            # controls
            doe_vars = {key: np.array(sheet_data[key], dtype=float)
                        for key in sheet_data.keys() if key in system['tvi']}

            doe_t = np.array(sheet_data['X:all'], dtype=float)

            for var_index, (var_name, doe_values) in enumerate(doe_vars.items()):
                if var_name not in variable_styles:
                    variable_styles[var_name] = line_styles[var_index % len(line_styles)]

                # Create a new y-axis for each variable (except the first one)
                if var_index > 0:
                    ax_new = ax_list_2[0].twinx()  # Create a new axis sharing the same x-axis
                    ax_new.spines['right'].set_position(('outward', 60 * (var_index - 1)))  # Closer positioning
                    ax_list_2.append(ax_new)
                else:
                    ax_new = ax_list_2[0]

                # Plot the DOE time-variant input data
                ax_new.plot(doe_t, doe_values, color='black',  # All lines in black
                            linestyle=variable_styles[var_name],
                            label=f'{var_name} (input)', alpha=0.8)

                # Slightly adjust the y-axis limits to avoid overlapping straight lines
                y_min, y_max = ax_new.get_ylim()
                ax_new.set_ylim([y_min - 0.05 * y_max, y_max + 0.05 * y_max])  # Adding small margins

                # Set y-axis label and tick colors for the variable
                ax_new.set_ylabel(f'{var_name} (Input)', color='black')
                ax_new.tick_params(axis='y', labelcolor='black')

            # else:
            #     # Classic case or sheets where `piecewise_func` is 'none': use `tv_input_m` data
            #     t_model = np.array(result[single_solver]['t_m'][sheet_name], dtype=float)
            #
            #     # Plot time-invariant inputs in the second subplot
            #     for var_index, var_name in enumerate(result[single_solver]['tv_input_m'][sheet_name].keys()):
            #         input_data = np.array(result[single_solver]['tv_input_m'][sheet_name][var_name], dtype=float)
            #         min_len = min(len(t_model), len(input_data))
            #         t_model_trimmed = t_model[:min_len]
            #         input_trimmed = input_data[:min_len]
            #
            #         if var_name not in variable_styles:
            #             variable_styles[var_name] = line_styles[var_index % len(line_styles)]
            #
            #         # Create a new y-axis for each variable (except the first one)
            #         if var_index > 0:
            #             ax_new = ax_list_2[0].twinx()  # Create a new axis sharing the same x-axis
            #             ax_new.spines['right'].set_position(('outward', 60 * (var_index - 1)))  # Closer positioning
            #             ax_list_2.append(ax_new)
            #         else:
            #             ax_new = ax_list_2[0]
            #
            #         # Plot input data (time-invariant inputs only)
            #         ax_new.plot(t_model_trimmed, input_trimmed, color='black',  # All lines in black
            #                     linestyle=variable_styles[var_name],
            #                     label=f'{var_name} (input)', alpha=0.8)
            #
            #         # Slightly adjust the y-axis limits to avoid overlapping straight lines
            #         y_min, y_max = ax_new.get_ylim()
            #         ax_new.set_ylim([y_min - 0.05 * y_max, y_max + 0.05 * y_max])  # Adding small margins
            #
            #         # Set y-axis label and tick colors for the variable
            #         ax_new.set_ylabel(f'{var_name} (Input)', color='black')
            #         ax_new.tick_params(axis='y', labelcolor='black')

            # **Add legend for the second subplot**
            # Collect handles and labels from all the y-axes for ax2 and its twinned axes
            handles_2, labels_2 = ax2.get_legend_handles_labels()
            for extra_ax in ax_list_2[1:]:
                extra_handles, extra_labels = extra_ax.get_legend_handles_labels()
                handles_2 += extra_handles
                labels_2 += extra_labels

            # Create the combined legend
            if handles_2:
                ax2.legend(handles_2, labels_2, ncol=2)

            ax2.set_xlabel('Time (t)')
            # ax2.grid(True)

            filename = self.modelling_folder / f"{self.round}_round_{sheet_name}.png"

            # Now save the plot
            plt.tight_layout()

            # Save the figure to the constructed filename
            plt.savefig(filename, dpi=300)
            if self.pltshowing == True:
                plt.show()

            # Close the plot
            plt.close()

    def conf_plot(self, parameters, cov_matrices, confidence_intervals):
        """
        Generate and save a confidence ellipsoid plot for estimated parameters, across different rounds of identification for different models.

        Parameters:
        parameters (dict): Model parameters.
        cov_matrices (dict): Covariance matrices for the estimated parameters
        confidence_intervals (dict): Confidence intervals for the parameters.
        round (int): The round number.
        """
        for solver, theta in parameters.items():
            cov_matrix = cov_matrices[solver]
            param_indices = np.where(self.mutation_settings[solver])[0]  # Get indices where mutation is True
            m = len(param_indices)
            fig, axs = plt.subplots(m, m, figsize=(12, 12))
            fig.suptitle(f'Confidence for {solver} in round {self.round}', fontsize=16)

            for i in range(m):
                for j in range(m):
                    axs[i, j].grid(True)

                    # Plot diagonal (PDFs)
                    if i == j:
                        idx = param_indices[i]
                        std_dev = np.sqrt(cov_matrix[idx, idx])
                        ci_low = theta[idx] - confidence_intervals[solver][i]
                        ci_high = theta[idx] + confidence_intervals[solver][i]
                        x = np.linspace(theta[idx] - 3 * std_dev, theta[idx] + 3 * std_dev, 100)
                        y = norm.pdf(x, loc=theta[idx], scale=std_dev)

                        axs[i, j].plot(x, y, 'b-', label=f'PDF of $\\theta_{idx + 1}$')
                        axs[i, j].axvline(ci_low, color='red', linestyle='--', label='95% CI')
                        axs[i, j].axvline(ci_high, color='red', linestyle='--')
                        axs[i, j].set_xlim(x[0], x[-1])
                        axs[i, j].set_xlabel(f'$\\theta_{idx + 1}$')
                        axs[i, j].set_ylabel('Density')
                        axs[i, j].legend()

                    # Plot off-diagonal (Ellipses)
                    elif i < j:
                        idx_i = param_indices[i]
                        idx_j = param_indices[j]
                        mean = [theta[idx_i], theta[idx_j]]
                        cov = [[cov_matrix[idx_i, idx_i], cov_matrix[idx_i, idx_j]],
                               [cov_matrix[idx_j, idx_j], cov_matrix[idx_j, idx_j]]]

                        vals, vecs = np.linalg.eigh(cov)
                        order = vals.argsort()[::-1]
                        vals, vecs = vals[order], vecs[:, order]
                        theta_ = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                        width, height = 2 * np.sqrt(vals)
                        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta_, edgecolor='red',
                                          fc='None', lw=2)

                        axs[i, j].add_patch(ellipse)
                        max_radius = max(width, height) / 2
                        axs[i, j].set_xlim(mean[0] - max_radius, mean[0] + max_radius)
                        axs[i, j].set_ylim(mean[1] - max_radius, mean[1] + max_radius)
                        axs[i, j].scatter(*mean, color='red')
                        axs[i, j].set_xlabel(f'$\\theta_{idx_i + 1}$')
                        axs[i, j].set_ylabel(f'$\\theta_{idx_j + 1}$')
                        axs[i, j].set_title(f'Ellipse between $\\theta_{idx_i + 1}$ and $\\theta_{idx_j + 1}$')

                    # Remove lower triangular plots
                    else:
                        axs[i, j].axis('off')

            plt.tight_layout()

            filename = self.confidence_folder / f"{self.round}_th_round_{solver}.png"

            # Save the figure to the constructed filename
            plt.savefig(filename, dpi=300)
            if self.pltshowing == True:
                plt.show()

            # Close the plot
            plt.close()


class Plotting_FinalResults:
    """
    A class to handle final plotting and reporting of results, post-identification.

    Attributes
    ----------
    round_data : dict
        Data related to each round of identification (e.g. 'Round 1', 'Round 2', ...).
    winner_solver : str
        The name of the winning model/model (most probable).
    selected_rounds : list of int
        Round numbers the user wants to include. If empty, all rounds are used.
    """

    def __init__(self, round_data, winner_solver, selected_rounds):
        """
        Initialize Plotting_FinalResults with round data, winner model, and selected rounds.

        Parameters
        ----------
        round_data : dict
            Data keyed by 'Round X' strings.
        winner_solver : str
            The name of the winning model/model.
        selected_rounds : list of int
            The rounds to include. If empty, all rounds are considered.
        """
        self.round_data = round_data
        self.winner_solver = winner_solver
        self.selected_rounds = selected_rounds  # if empty => all rounds

    # ---------------------------------------------------------------------
    #                           UTILITY METHODS
    # ---------------------------------------------------------------------
    def _get_round_labels(self):
        """
        Return a sorted list of all round labels that exist in self.round_data.

        Returns
        -------
        list of str
            Labels like ['Round 1', 'Round 2', ...], sorted numerically.
        """
        round_labels = [lbl for lbl in self.round_data if lbl.startswith('Round ')]
        round_labels.sort(key=lambda x: int(x.replace('Round ', '')))
        return round_labels

    def _get_round_number(self, round_label):
        """
        Given a label like 'Round 3', return the integer 3.

        Parameters
        ----------
        round_label : str
            A label such as 'Round 3'.

        Returns
        -------
        int
            The numeric portion of the round label.
        """
        return int(round_label.split(' ')[-1])

    def _rounds_to_process(self, exclude_broad_cov=False):
        """
        Determine which rounds to process based on self.selected_rounds
        and optionally skip rounds whose covariance matrix is not positive definite.

        Parameters
        ----------
        exclude_broad_cov : bool, optional
            If True, skip any round where the model's covariance matrix is not positive-definite.

        Returns
        -------
        list of (int, str)
            A sorted list of tuples (round_number, round_label) for the rounds to process.
        """
        all_round_labels = self._get_round_labels()
        rounds_out = []

        for lbl in all_round_labels:
            rnd_num = self._get_round_number(lbl)

            # If user provided a list of rounds, skip if not in it
            if self.selected_rounds and (rnd_num not in self.selected_rounds):
                continue

            # If we must exclude broad covariances, check positive-definiteness
            if exclude_broad_cov and (self.winner_solver in self.round_data[lbl]['result']):
                solver_data = self.round_data[lbl]['result'][self.winner_solver]
                if 'V_matrix' in solver_data:
                    cov_mat = solver_data['V_matrix']
                    if not np.all(np.linalg.eigvals(cov_mat) > 0):
                        continue  # skip non-positive-def

            rounds_out.append((rnd_num, lbl))

        return sorted(rounds_out, key=lambda x: x[0])

    # ---------------------------------------------------------------------
    #                             MAIN METHODS
    # ---------------------------------------------------------------------
    def conf_plot(self, filename,
                  ellipsoid_volume_filename='ellipsoid_volume_across_rounds.png',
                  std_devs_filename='parameter_std_devs_across_rounds.png',
                  parameter_estimates_filename='parameter_estimates_across_rounds.png'):
        """
        Plot parameter PDFs (on the diagonal) and confidence ellipses (off-diagonal) for
        the winning model across selected rounds. Then call `conf_plot_metrics` and `accvsprec_plot`.

        Parameters
        ----------
        filename : str
            Filename for the main confidence plot figure.
        ellipsoid_volume_filename : str, optional
            Filename for the ellipsoid volume plot.
        std_devs_filename : str, optional
            Filename for the standard deviations plot.
        parameter_estimates_filename : str, optional
            Filename for the parameter estimates plot.
        """
        # Determine which rounds to plot
        all_round_labels = self._get_round_labels()
        if self.selected_rounds:
            round_labels_to_plot = [f"Round {r}" for r in self.selected_rounds
                                    if f"Round {r}" in all_round_labels]
            # Warn if any user-selected rounds don't exist
            missing = [f"Round {r}" for r in self.selected_rounds
                       if f"Round {r}" not in all_round_labels]
            for m in missing:
                logger.warning(f"Selected round {m} is missing from round_data.")
        else:
            round_labels_to_plot = all_round_labels

        if not round_labels_to_plot:
            logger.warning("No valid rounds to plot. Aborting conf_plot.")
            return

        # Identify parameter dimension from the earliest possible round
        dimension_round_label = "Round 1" if "Round 1" in round_labels_to_plot else round_labels_to_plot[0]
        if (dimension_round_label not in self.round_data or
                self.winner_solver not in self.round_data[dimension_round_label]['result']):
            # Fallback to the very first in the list if needed
            dimension_round_label = round_labels_to_plot[0]

        if (dimension_round_label not in self.round_data or
                self.winner_solver not in self.round_data[dimension_round_label]['result']):
            logger.error("Cannot determine parameter dimension; no model data found.")
            logger.warning("Aborting conf_plot.")
            return

        solver_data_dim = self.round_data[dimension_round_label]['result'][self.winner_solver]
        if 'optimization_result' not in solver_data_dim:
            logger.error(f"No 'optimization_result' in {dimension_round_label} for model {self.winner_solver}.")
            return

        # Extract dimension and parameter names
        theta_dim = solver_data_dim['optimization_result'].x
        max_n = len(theta_dim)
        parameter_names = [f'θ_{i + 1}' for i in range(max_n)]

        # Prepare figure
        num_rounds = len(round_labels_to_plot)
        colors = cm.get_cmap('Dark2', num_rounds)
        line_styles = ['-', '--', '-.', ':']

        fig, axs = plt.subplots(nrows=max_n, ncols=max_n,
                                figsize=(max(5, max_n * 3), max(5, max_n * 3)), dpi=300)
        fig.suptitle('Comparison of Confidence Spaces Across Rounds')

        if max_n == 1:
            axs = np.array([[axs]])

        handles, labels = [], []
        plots_count = 0

        def get_active_data(round_label):
            """
            Retrieve data needed for plotting:
            (active_indices, active_theta, covariance_matrix, confidence_intervals).
            Returns None if any data is missing or inconsistent.
            """
            rd = self.round_data.get(round_label, {})
            solver_result = rd.get('result', {}).get(self.winner_solver, {})

            for req_key in ['optimization_result', 'V_matrix', 'CI']:
                if req_key not in solver_result:
                    logger.info(f"Skipping {round_label}: missing key '{req_key}' in solver_data.")
                    return None

            theta_vals = solver_result['optimization_result'].x
            cov_matrix = solver_result['V_matrix']
            ci_vals = solver_result['CI']

            # Retrieve 'mutation' from round-level data
            mutation_dict = rd.get('mutation', {})
            if self.winner_solver not in mutation_dict:
                logger.info(f"Skipping {round_label}: no mutation info for model {self.winner_solver}.")
                return None

            mut_settings = mutation_dict[self.winner_solver]
            if len(mut_settings) != len(theta_vals):
                logger.warning(f"Skipping {round_label}: mismatch in 'mutation' length vs. parameters.")
                return None

            active_indices = [i for i, active in enumerate(mut_settings) if active]
            if not active_indices:
                logger.info(f"Skipping {round_label}: no active parameters in mutation.")
                return None

            active_theta = theta_vals[active_indices]
            return active_indices, active_theta, cov_matrix, ci_vals

        # Plot each round
        for i_r, round_label in enumerate(round_labels_to_plot):
            active_data = get_active_data(round_label)
            if not active_data:
                continue

            active_indices, active_theta, active_cov, active_ci = active_data
            color = colors(i_r)
            style = line_styles[i_r % len(line_styles)]

            # If original positions exist, use them; otherwise default to range(max_n)
            round_info = self.round_data[round_label]
            orig_pos = round_info.get('original_positions', {}).get(self.winner_solver, [])
            if not orig_pos:
                orig_pos = range(max_n)

            # Diagonal => PDF plots, Off-diagonal => ellipse
            for i_row in orig_pos:
                for j_col in orig_pos:
                    ax_ij = axs[i_row, j_col]
                    if i_row == j_col and i_row in active_indices:
                        # Single-parameter PDF
                        param_name = parameter_names[i_row]
                        local_i = active_indices.index(i_row)
                        try:
                            std_dev = np.sqrt(active_cov[local_i, local_i])
                            ci_low = active_theta[local_i] - active_ci[local_i]
                            ci_high = active_theta[local_i] + active_ci[local_i]

                            xvals = np.linspace(active_theta[local_i] - 3 * std_dev,
                                                active_theta[local_i] + 3 * std_dev, 100)
                            pdf_vals = norm.pdf(xvals, loc=active_theta[local_i], scale=std_dev)

                            ax_ij.plot(xvals, pdf_vals, color=color, linestyle=style,
                                       label=f'{round_label} PDF')
                            ax_ij.plot([ci_low, ci_low],
                                       [0, norm.pdf(ci_low, active_theta[local_i], std_dev)],
                                       marker='*', color=color)
                            ax_ij.plot([ci_high, ci_high],
                                       [0, norm.pdf(ci_high, active_theta[local_i], std_dev)],
                                       marker='*', color=color)
                            ax_ij.set_title(f'PDF of {param_name}')
                            plots_count += 1

                            # Manage legend handles
                            if round_label not in labels:
                                handles.append(ax_ij.lines[-1])
                                labels.append(round_label)

                        except Exception as e:
                            logger.warning(f"Error plotting PDF for {round_label}, param {param_name}: {e}")
                            ax_ij.axis('off')

                    elif i_row < j_col and i_row in active_indices and j_col in active_indices:
                        # Two-parameter confidence ellipse
                        param_i = parameter_names[i_row]
                        param_j = parameter_names[j_col]
                        li = active_indices.index(i_row)
                        lj = active_indices.index(j_col)
                        try:
                            mean_ij = [active_theta[li], active_theta[lj]]
                            cov_ij = [
                                [active_cov[li, li], active_cov[li, lj]],
                                [active_cov[lj, li], active_cov[lj, lj]]
                            ]
                            vals, vecs = np.linalg.eigh(cov_ij)
                            order = vals.argsort()[::-1]
                            vals, vecs = vals[order], vecs[:, order]
                            angle_deg = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                            w, h = 2 * np.sqrt(vals)

                            ellipse = Ellipse(xy=mean_ij, width=w, height=h,
                                              angle=angle_deg, edgecolor=color,
                                              fc='None', lw=2, linestyle=style)
                            ax_ij.add_patch(ellipse)
                            ax_ij.scatter(*mean_ij, color=color)
                            ax_ij.set_title(f'{param_i} vs {param_j}')
                            plots_count += 1

                            if round_label not in labels:
                                handles.append(ellipse)
                                labels.append(round_label)

                        except Exception as e:
                            logger.warning(f"Error plotting ellipse for {round_label}, "
                                           f"{param_i} vs {param_j}: {e}")
                            ax_ij.axis('off')

        # Hide empty subplots
        for i_row in range(max_n):
            for j_col in range(max_n):
                ax_ij = axs[i_row, j_col]
                if not (ax_ij.has_data() or ax_ij.patches):
                    ax_ij.axis('off')

        if plots_count == 0:
            logger.warning("No data was plotted. Possibly no active parameters or missing keys.")
            return

        # Create a single legend on the top-left subplot
        if handles and labels:
            if len(labels) > 1:
                axs[0, 0].legend(handles, labels, loc='upper right')
            else:
                axs[0, 0].legend(handles, labels)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        # Generate auxiliary plots
        self.conf_plot_metrics(ellipsoid_volume_filename, std_devs_filename)
        self.accvsprec_plot(parameter_estimates_filename)

    def conf_plot_metrics(self, ellipsoid_volume_filename, std_devs_filename):
        """
        Compute and plot:
          1) The volume of the 95% confidence ellipsoid.
          2) The parameter standard deviations across selected rounds,
        excluding rounds whose covariance matrix is broad/non-positive-definite.

        Parameters
        ----------
        ellipsoid_volume_filename : str
            Filename to save the confidence ellipsoid volume plot.
        std_devs_filename : str
            Filename to save the standard deviations plot.
        """
        round_list = self._rounds_to_process(exclude_broad_cov=True)
        if not round_list:
            logger.info("No valid rounds for conf_plot_metrics (possibly all broad).")
            return

        ref_label = round_list[0][1]
        solver_data = self.round_data[ref_label]['result'].get(self.winner_solver, {})
        if 'optimization_result' not in solver_data:
            logger.warning(f"Missing model optimization_result in {ref_label}. Cannot plot metrics.")
            return

        theta_ref = solver_data['optimization_result'].x
        max_n = len(theta_ref)

        rounds = []
        volumes = []
        param_stds = {i: [] for i in range(max_n)}

        # Collect volumes and std devs
        for (rnd_num, lbl) in round_list:
            sol_data = self.round_data[lbl]['result'].get(self.winner_solver, {})
            if 'V_matrix' not in sol_data:
                continue
            cov_mat = sol_data['V_matrix']

            # Must be positive-definite
            if not np.all(np.linalg.eigvals(cov_mat) > 0):
                continue

            n_params = cov_mat.shape[0]
            chi2_val = chi2.ppf(0.95, n_params)
            det_cov = np.linalg.det(cov_mat)
            vol = ((np.pi ** (n_params / 2)) / gamma(n_params / 2 + 1)) \
                  * (chi2_val ** (n_params / 2)) * np.sqrt(det_cov)

            volumes.append(vol)
            rounds.append(rnd_num)

            stds = np.sqrt(np.diag(cov_mat))
            for i in range(n_params):
                param_stds[i].append(stds[i])

        if not rounds:
            logger.info("No positive-definite covariance among chosen rounds.")
            return

        # Sort data by round
        rounds = np.array(rounds)
        sort_idx = np.argsort(rounds)
        volumes = np.array(volumes)[sort_idx]
        rounds = rounds[sort_idx]
        for i in param_stds:
            param_stds[i] = np.array(param_stds[i])[sort_idx]

        # Plot volume
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, volumes, marker='o')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Volume of 95% Confidence Ellipsoid')
        plt.xlabel('Round')
        plt.ylabel('Ellipsoid Volume')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.savefig(ellipsoid_volume_filename, dpi=300)
        plt.show()

        # Plot standard deviations
        plt.figure(figsize=(10, 6))
        for i in range(max_n):
            plt.plot(rounds, param_stds[i], marker='o', label=f'θ_{i + 1} std')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Parameter Standard Deviations vs. Round')
        plt.xlabel('Round')
        plt.ylabel('Std Dev')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.savefig(std_devs_filename, dpi=300)
        plt.show()

    def accvsprec_plot(self, filename):
        """
        Plot parameter estimates and confidence intervals for the selected rounds,
        skipping any with broad covariance.

        Parameters
        ----------
        filename : str
            Filename to save the accuracy vs. precision plot.
        """
        round_list = self._rounds_to_process(exclude_broad_cov=True)
        if not round_list:
            logger.info("No valid rounds for accvsprec_plot (all broad or none selected).")
            return

        ref_label = round_list[0][1]
        solver_data_ref = self.round_data[ref_label]['result'].get(self.winner_solver, {})
        if 'optimization_result' not in solver_data_ref or 'CI' not in solver_data_ref:
            logger.warning(f"Missing optimization_result or CI in {ref_label}. Cannot plot acc vs. prec.")
            return

        theta_ref = solver_data_ref['optimization_result'].x
        ci_ref = solver_data_ref['CI']
        max_n = len(theta_ref)
        param_names = [f'θ_{i + 1}' for i in range(max_n)]

        # Gather data
        rounds = []
        param_est = {i: [] for i in range(max_n)}
        param_ci = {i: [] for i in range(max_n)}

        for (rnd_num, lbl) in round_list:
            sol_data = self.round_data[lbl]['result'].get(self.winner_solver, {})
            if 'optimization_result' not in sol_data or 'CI' not in sol_data:
                continue

            theta_vals = sol_data['optimization_result'].x
            ci_vals = sol_data['CI']
            cov_mat = sol_data.get('V_matrix', None)
            if cov_mat is None or not np.all(np.linalg.eigvals(cov_mat) > 0):
                continue

            rounds.append(rnd_num)
            for i in range(max_n):
                param_est[i].append(theta_vals[i])
                val_ci = ci_vals[i]
                if np.isnan(val_ci):
                    val_ci = np.inf
                param_ci[i].append(val_ci)

        if not rounds:
            logger.info("All chosen rounds had broad covariance or missing data.")
            return

        # Sort by round
        rounds = np.array(rounds)
        sidx = np.argsort(rounds)
        rounds = rounds[sidx]
        for i in range(max_n):
            param_est[i] = np.array(param_est[i])[sidx]
            param_ci[i] = np.array(param_ci[i])[sidx]

        # Plot
        fig, axs = plt.subplots(nrows=max_n, ncols=1, figsize=(10, 4 * max_n), sharex=True)
        if max_n == 1:
            axs = [axs]

        for i in range(max_n):
            est_arr = param_est[i]
            ci_arr = param_ci[i]
            lb = est_arr - ci_arr
            ub = est_arr + ci_arr

            # Replace infinite CI bounds
            finite_lb = lb[np.isfinite(lb)]
            finite_ub = ub[np.isfinite(ub)]
            if len(finite_lb) == 0 or len(finite_ub) == 0:
                ymin, ymax = 0.1, 10
            else:
                ymin, ymax = finite_lb.min(), finite_ub.max()
            lb[np.isinf(lb)] = ymin
            ub[np.isinf(ub)] = ymax

            ax = axs[i]
            ax.plot(rounds, est_arr, marker='o', color='blue', label=param_names[i])
            ax.fill_between(rounds, lb, ub, color='blue', alpha=0.3, label='95% CI')
            # Example: if you know a "true" param is 1, you could do:
            ax.axhline(y=1, color='red', linestyle='--', label='True Param=1')
            ax.set_ylabel('Estimate')
            ax.set_title(f'{param_names[i]} over Rounds')
            ax.grid(True)
            ax.legend(loc='best')
            ax.set_yscale('log')

        axs[-1].set_xlabel('Round')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    def pt_plot(self, filename, roundc):
        """
        Plot changes in P values, t values, and TRV across selected rounds,
        skipping any with broad covariance.

        Parameters
        ----------
        filename : str
            Filename to save the P/t plot.
        roundc : dict
            Additional design info for each round (key='Round X'), e.g.
            {'Round 2': {'design_type': 'classic design'}, ...}.
        """
        round_list = self._rounds_to_process(exclude_broad_cov=True)
        if not round_list:
            logger.info("No valid rounds for pt_plot (all broad or none selected).")
            return

        P_vals = []
        t_vals_all = []
        trv_vals_all = []
        used_rounds = []

        for (rnd_num, lbl) in round_list:
            solver_data = self.round_data[lbl]['result'].get(self.winner_solver, {})
            if 'P' not in solver_data or 't_values' not in solver_data:
                continue

            P_vals.append(solver_data['P'])
            t_vals_all.append(solver_data['t_values'])

            trv_data = self.round_data[lbl].get('trv', {})
            # Check if TRV is given for this model
            if self.winner_solver in trv_data:
                val = trv_data[self.winner_solver]
                if isinstance(val, float):
                    trv_vals_all.append([val])
                else:
                    trv_vals_all.append(val)
            else:
                trv_vals_all.append([np.nan])

            used_rounds.append(rnd_num)

        if not used_rounds:
            logger.info("No data with P or t_values for the chosen rounds.")
            return

        # Sort everything by round number
        used_rounds = np.array(used_rounds)
        sidx = np.argsort(used_rounds)
        used_rounds = used_rounds[sidx]
        P_vals = np.array(P_vals)[sidx]
        t_vals_all = [t_vals_all[i] for i in sidx]
        trv_vals_all = [trv_vals_all[i] for i in sidx]

        plt.figure(figsize=(16, 8))

        # Subplot 1: P values
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(used_rounds, P_vals, marker='o')
        ax1.set_title(f'P-Test {self.winner_solver}')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('P Value')
        ax1.set_xticks(used_rounds)

        # Highlight design types
        for i, round_num in enumerate(used_rounds):
            label_ = f'Round {round_num}'
            if label_ in roundc:
                d_type = roundc[label_].get('design_type', '')
                if d_type == 'preliminary':
                    ax1.axvspan(round_num - 0.5, round_num + 0.5, color='green', alpha=0.3)
                elif d_type == 'MBDOE_MD':
                    ax1.axvspan(round_num - 0.5, round_num + 0.5, color='red', alpha=0.3)
                elif d_type == 'MBDOE_PP':
                    ax1.axvspan(round_num - 0.5, round_num + 0.5, color='blue', alpha=0.3)

        # Legend patches
        classic_patch = Patch(color='green', alpha=0.3, label='preliminary')
        md_patch = Patch(color='red', alpha=0.3, label='MBDOE_MD')
        pp_patch = Patch(color='blue', alpha=0.3, label='MBDOE_PP')
        ax1.legend(handles=[classic_patch, md_patch, pp_patch], loc='best')

        # Subplot 2: T values vs. TRV
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_yscale('log')

        # Plot TRV
        trv_main = [vals[0] if len(vals) else np.nan for vals in trv_vals_all]
        ax2.plot(used_rounds, trv_main, 'k--', label='TRV reference')

        max_len_t = max(len(tv) for tv in t_vals_all)
        for i_col in range(max_len_t):
            param_tvals = [
                t_vals_all[i][i_col] if i_col < len(t_vals_all[i]) else np.nan
                for i in range(len(t_vals_all))
            ]
            ax2.plot(used_rounds, param_tvals, marker='o', label=f'T param {i_col + 1}')
            # Mark intersections
            for k, (tt, trv_0) in enumerate(zip(param_tvals, trv_main)):
                if np.isclose(tt, trv_0, atol=1e-2):
                    ax2.plot(used_rounds[k], tt, 'ro')

        ax2.set_title(f't-Test {self.winner_solver}')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('t value')
        ax2.set_xticks(used_rounds)
        ax2.legend(loc='best')

        # Again, highlight design types
        for round_num in used_rounds:
            label_ = f'Round {round_num}'
            if label_ in roundc:
                d_type = roundc[label_].get('design_type', '')
                if d_type == 'classic design':
                    ax2.axvspan(round_num - 0.5, round_num + 0.5, color='green', alpha=0.3)
                elif d_type == 'MBDOE_MD design':
                    ax2.axvspan(round_num - 0.5, round_num + 0.5, color='red', alpha=0.3)
                elif d_type == 'MBDOE_PP design':
                    ax2.axvspan(round_num - 0.5, round_num + 0.5, color='blue', alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    # def reporter(self, filename):
    #     """
    #     Write an Excel file for each round in self.selected_rounds (or all if none are specified).
    #     Each file will contain experimental data, model simulations, and input data (if available),
    #     one sheet per experiment for that round.
    #
    #     Parameters
    #     ----------
    #     filename : str
    #         Base filename. The round number is appended (e.g. "my_file_2.xlsx" for Round 2).
    #     """
    #     if not self.round_data:
    #         logger.warning("No round data found. Aborting reporter.")
    #         return
    #
    #     # Decide which rounds to use
    #     if not self.selected_rounds:
    #         all_lbls = self._get_round_labels()
    #         used_nums = [self._get_round_number(lbl) for lbl in all_lbls]
    #     else:
    #         used_nums = self.selected_rounds
    #
    #     used_nums.sort()
    #     for rnd_num in used_nums:
    #         round_label = f"Round {rnd_num}"
    #         if round_label not in self.round_data:
    #             logger.info(f"Skipping {round_label}: not found in round_data.")
    #             continue
    #
    #         solvers_data = self.round_data[round_label].get('result', {})
    #         if self.winner_solver not in solvers_data:
    #             logger.info(f"Skipping {round_label}: model {self.winner_solver} not found.")
    #             continue
    #
    #         solver_info = solvers_data[self.winner_solver]
    #         if 'data' not in solver_info:
    #             logger.info(f"Skipping {round_label}: no 'data' key in model info.")
    #             continue
    #
    #         exp_data_dict = solver_info['data']
    #         excel_filename = f"{filename}_{rnd_num}.xlsx"
    #         combined_dataframes = {}
    #
    #         # Check if 'tv_output_m' is available
    #         if 'tv_output_m' not in solver_info:
    #             logger.info(f"Skipping {round_label}: no 'tv_output_m' found in model info.")
    #             continue
    #
    #         # Build data for each experiment
    #         for exp_name, sheet_data in exp_data_dict.items():
    #             # Check if we have time-variable outputs for this experiment
    #             if exp_name not in solver_info['tv_output_m']:
    #                 logger.info(f"No 'tv_output_m' entry for {exp_name} in {round_label}. Skipping experiment.")
    #                 continue
    #
    #             variables = list(solver_info['tv_output_m'][exp_name].keys())
    #
    #             # 1) Experimental
    #             df_exp = pd.DataFrame()
    #             for var_name in variables:
    #                 mx_col = f"MES_X:{var_name}"
    #                 my_col = f"MES_Y:{var_name}"
    #                 if mx_col in sheet_data and my_col in sheet_data:
    #                     t_arr = np.array(sheet_data[mx_col], dtype=float)
    #                     y_arr = np.array(sheet_data[my_col], dtype=float)
    #                     df_exp[mx_col] = t_arr
    #                     df_exp[my_col] = y_arr
    #
    #             # 2) Model
    #             df_model = pd.DataFrame()
    #             if 't_m' in solver_info and exp_name in solver_info['t_m']:
    #                 t_model = np.array(solver_info['t_m'][exp_name], dtype=float)
    #                 df_model["SIM_t_model"] = t_model
    #                 for var_name in variables:
    #                     vals = solver_info['tv_output_m'][exp_name].get(var_name, [])
    #                     df_model[f"SIM_{var_name}"] = np.array(vals, dtype=float)
    #
    #             # 3) Input
    #             df_input = pd.DataFrame()
    #             if exp_name in solver_info.get('tv_input_m', {}):
    #                 input_vars = solver_info['tv_input_m'][exp_name]
    #                 # If we have model time, align inputs with model time
    #                 if not df_model.empty and 'SIM_t_model' in df_model.columns:
    #                     t_mod = df_model['SIM_t_model'].values
    #                     df_input['INP_t_model'] = t_mod
    #                     for iname, ivals in input_vars.items():
    #                         if 't_input' in sheet_data:
    #                             t_in = np.array(sheet_data['t_input'], dtype=float)
    #                         else:
    #                             t_in = np.linspace(t_mod[0], t_mod[-1], len(ivals))
    #                         ivals_arr = np.array(ivals, dtype=float)
    #                         f_int = interp1d(t_in, ivals_arr, kind='linear',
    #                                          bounds_error=False, fill_value='extrapolate')
    #                         df_input[f"INP_{iname}"] = f_int(t_mod)
    #                 else:
    #                     # Just store them raw if there's no model time
    #                     for iname, ivals in input_vars.items():
    #                         df_input[f"INP_{iname}"] = ivals
    #
    #             # Concatenate all data horizontally
    #             df_combined = pd.concat([df_exp, df_model, df_input], axis=1)
    #             combined_dataframes[exp_name] = df_combined
    #
    #         # Write each experiment's DataFrame to a separate sheet
    #         with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
    #             for exp_name, cdf in combined_dataframes.items():
    #                 cdf.to_excel(writer, sheet_name=exp_name, index=False)
    #
    #         logger.info(f"Reporter wrote data for {round_label} to '{excel_filename}'.")

    def reporter(self, filename):
        """
        Write an Excel file for each round and a summary text file for the last round.
        Excel files contain experimental data, model simulations, and input data.
        The text file contains scaled parameters, t-values, and R² metrics for the last round.

        Parameters
        ----------
        filename : str
            Base filename. Round numbers are appended to Excel files (e.g., "my_file_2.xlsx").
            The text summary is written to a file with "_summary.txt" appended.
        """
        if not self.round_data:
            logger.warning("No round data found. Aborting reporter.")
            return

        # Decide which rounds to use
        if not self.selected_rounds:
            all_lbls = self._get_round_labels()
            used_nums = [self._get_round_number(lbl) for lbl in all_lbls]
        else:
            used_nums = self.selected_rounds

        used_nums.sort()
        last_round_label = f"Round {used_nums[-1]}"
        summary_filename = f"{filename}_summary.txt"

        for rnd_num in used_nums:
            round_label = f"Round {rnd_num}"
            if round_label not in self.round_data:
                logger.info(f"Skipping {round_label}: not found in round_data.")
                continue

            solvers_data = self.round_data[round_label].get('result', {})
            if self.winner_solver not in solvers_data:
                logger.info(f"Skipping {round_label}: model {self.winner_solver} not found.")
                continue

            solver_info = solvers_data[self.winner_solver]
            if 'data' not in solver_info:
                logger.info(f"Skipping {round_label}: no 'data' key in model info.")
                continue

            exp_data_dict = solver_info['data']
            excel_filename = f"{filename}_{rnd_num}.xlsx"
            combined_dataframes = {}

            # Check if 'tv_output_m' is available
            if 'tv_output_m' not in solver_info:
                logger.info(f"Skipping {round_label}: no 'tv_output_m' found in model info.")
                continue

            # Build data for each experiment
            for exp_name, sheet_data in exp_data_dict.items():
                # Check if we have time-variable outputs for this experiment
                if exp_name not in solver_info['tv_output_m']:
                    logger.info(f"No 'tv_output_m' entry for {exp_name} in {round_label}. Skipping experiment.")
                    continue

                variables = list(solver_info['tv_output_m'][exp_name].keys())

                # 1) Experimental
                df_exp = pd.DataFrame()
                for var_name in variables:
                    mx_col = f"MES_X:{var_name}"
                    my_col = f"MES_Y:{var_name}"
                    if mx_col in sheet_data and my_col in sheet_data:
                        t_arr = np.array(sheet_data[mx_col], dtype=float)
                        y_arr = np.array(sheet_data[my_col], dtype=float)
                        df_exp[mx_col] = t_arr
                        df_exp[my_col] = y_arr

                # 2) Model
                df_model = pd.DataFrame()
                if 't_m' in solver_info and exp_name in solver_info['t_m']:
                    t_model = np.array(solver_info['t_m'][exp_name], dtype=float)
                    df_model["SIM_t_model"] = t_model
                    for var_name in variables:
                        vals = solver_info['tv_output_m'][exp_name].get(var_name, [])
                        df_model[f"SIM_{var_name}"] = np.array(vals, dtype=float)

                # 3) Input
                df_input = pd.DataFrame()
                if exp_name in solver_info.get('tv_input_m', {}):
                    input_vars = solver_info['tv_input_m'][exp_name]
                    # If we have model time, align inputs with model time
                    if not df_model.empty and 'SIM_t_model' in df_model.columns:
                        t_mod = df_model['SIM_t_model'].values
                        df_input['INP_t_model'] = t_mod
                        for iname, ivals in input_vars.items():
                            if 't_input' in sheet_data:
                                t_in = np.array(sheet_data['t_input'], dtype=float)
                            else:
                                t_in = np.linspace(t_mod[0], t_mod[-1], len(ivals))
                            ivals_arr = np.array(ivals, dtype=float)
                            f_int = interp1d(t_in, ivals_arr, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
                            df_input[f"INP_{iname}"] = f_int(t_mod)
                    else:
                        # Just store them raw if there's no model time
                        for iname, ivals in input_vars.items():
                            df_input[f"INP_{iname}"] = ivals

                # Concatenate all data horizontally
                df_combined = pd.concat([df_exp, df_model, df_input], axis=1)
                combined_dataframes[exp_name] = df_combined

            # Write each experiment's DataFrame to a separate sheet
            with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
                for exp_name, cdf in combined_dataframes.items():
                    cdf.to_excel(writer, sheet_name=exp_name, index=False)

            logger.info(f"Reporter wrote data for {round_label} to '{excel_filename}'.")

        # Write summary for the last round
        if last_round_label in self.round_data:
            last_round_data = self.round_data[last_round_label]
            with open(summary_filename, "w") as summary_file:
                for model_name, model_data in last_round_data.get('scaled_params', {}).items():
                    summary_file.write(f"Model: {model_name}\n")
                    summary_file.write(f"Scaled Parameters: {model_data}\n")

                    t_values = last_round_data['result'].get(model_name, {}).get('t_values', [])
                    summary_file.write(f"T-Values: {t_values}\n")

                    CI95 = last_round_data['result'].get(model_name, {}).get('CI', [])
                    summary_file.write(f"CI 95% confidence: {CI95}\n")

                    R2_overall = last_round_data['result'].get(model_name, {}).get('LS', "N/A")
                    summary_file.write(f"Overall R²: {R2_overall}\n")

                    Chi_overall = last_round_data['result'].get(model_name, {}).get('Chi', "N/A")
                    summary_file.write(f"Overall Chi-square: {Chi_overall}\n")

                    R2_responses = last_round_data['result'].get(model_name, {}).get('R2_responses_summary', {})
                    summary_file.write("R² by Response:\n")
                    for response, r2_value in R2_responses.items():
                        summary_file.write(f"  {response}: {r2_value}\n")

                    # Include ranking for each model
                    ranking = (last_round_data.get('ranking') or {}).get(model_name, {})
                    if isinstance(ranking, dict):  # Handle dictionary case
                        summary_file.write("Parameter Ranking:\n")
                        for param, rank in ranking.items():
                            summary_file.write(f"  {param}: {rank}\n")
                    elif isinstance(ranking, list):  # Handle list case
                        summary_file.write("Parameter Ranking (list):\n")
                        for idx, rank in enumerate(ranking):
                            summary_file.write(f"  Parameter {idx + 1}: {rank}\n")
                    else:  # Handle case where ranking is not available or unexpected type
                        summary_file.write("Parameter Ranking: N/A\n")

            logger.info(f"Summary written to '{summary_filename}'.")


    def pcomp_plot(self, save_path):
        """
        Plot P values for each selected round and save each plot with the round number in the filename.

        Parameters
        ----------
        save_path : str
            Path where the plots will be saved.

        Returns
        -------
        None
        """
        logger.info("Starting pcomp_plot.")
        rounds_to_plot = self._rounds_to_process()
        if not rounds_to_plot:
            logger.warning("No rounds selected for P value plotting.")
            return

        for round_number, round_label in rounds_to_plot:
            if round_label not in self.round_data:
                logger.warning(f"{round_label} does not exist in round data. Skipping.")
                continue

            model_names = []
            P_values = []

            # Collect P values for the round
            round_data = self.round_data[round_label]['result']
            for model_name, model_data in round_data.items():
                P_value = model_data.get('P', None)
                if P_value is not None:
                    model_names.append(model_name)
                    P_values.append(P_value)

            if not P_values:
                logger.warning(f"No P values found for {round_label}. Skipping plot.")
                continue

            # Plot P values for the current round
            plt.figure(figsize=(5, 5))
            plt.bar(model_names, P_values, color='black')
            plt.xlabel('Model')
            plt.ylabel('P Value')
            plt.title(f'P Values for {round_label}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save the plot with the round number in the filename
            save_file_path = os.path.join(save_path, f'p_values_{round_label.replace(" ", "_").lower()}.png')
            plt.savefig(save_file_path)
            plt.show()

            logger.info(f"P values plot for {round_label} saved to {save_file_path}.")

    def tcomp_plot(self, save_path):
        """
        Plot t values for each selected round and save each plot with the round number in the filename.

        Parameters
        ----------
        save_path : str
            Path where the plots will be saved.

        Returns
        -------
        None
        """
        logger.info("Starting tcomp_plot.")
        rounds_to_plot = self._rounds_to_process()
        if not rounds_to_plot:
            logger.warning("No rounds selected for t value plotting.")
            return

        for round_number, round_label in rounds_to_plot:
            if round_label not in self.round_data:
                logger.warning(f"{round_label} does not exist in round data. Skipping.")
                continue

            round_data = self.round_data[round_label]['result']
            trv_data = self.round_data[round_label].get('trv', {})
            t_reference_value = next(iter(trv_data.values()), None)

            model_names = set()
            theta_names_per_model = {}
            t_values_per_model = {}

            for model_name, model_data in round_data.items():
                t_value_list = model_data.get('t_values', [])
                if len(t_value_list) > 0:
                    model_names.add(model_name)
                    theta_names_per_model[model_name] = [f'theta {i}' for i in range(len(t_value_list))]
                    t_values_per_model[model_name] = t_value_list

            if not t_values_per_model:
                logger.warning(f"No t values found for {round_label}. Skipping plot.")
                continue

            # Plot t values for the current round
            plt.figure(figsize=(5, 5))
            bar_width = 0.5
            gap_between_models = 1
            bar_position = 0
            model_mid_positions = []

            for model_name in sorted(model_names):
                t_values = t_values_per_model.get(model_name, [])
                num_thetas = len(t_values)

                indices = np.arange(bar_position, bar_position + num_thetas)
                plt.bar(indices, t_values, bar_width, label=model_name)

                mid_position = bar_position + (num_thetas - 1) / 2
                model_mid_positions.append(mid_position)

                bar_position += num_thetas + gap_between_models

            if t_reference_value is not None:
                plt.axhline(y=t_reference_value, color='red', linestyle='--', label=f'Reference t (TRV = {t_reference_value:.2f})')

            plt.xlabel('Model - Theta')
            plt.ylabel('t Value')
            plt.title(f't Values for {round_label}')
            plt.xticks(model_mid_positions, sorted(model_names), rotation=45, ha='right')
            plt.legend(title='Models')
            plt.tight_layout()

            # Save the plot with the round number in the filename
            save_file_path = os.path.join(save_path, f't_values_{round_label.replace(" ", "_").lower()}.png')
            plt.savefig(save_file_path)
            plt.show()

            logger.info(f"T values plot for {round_label} saved to {save_file_path}.")


def _initialize_dictionaries(models, iden_opt):
    """
    Initialize dictionaries for modelling settings and estimation settings.

    Parameters:
    models (dict): Dictionary containing modelling settings, including 'theta_parameters', 'bound_min', and 'bound_max'.
    iden_opt (dict): Dictionary containing estimation settings.

    Returns:
    None

    Raises:
    KeyError: If required keys are missing in models or iden_opt.
    """
    # Check if keys are missing in models and iden_opt
    keys_to_check = ['original_positions', 'masked_positions']
    if any(key not in models for key in keys_to_check) or 'x0' not in iden_opt:
        # Generate x0_dict with random initialization within bounds
        if iden_opt['init'] == 'rand':
            x0_dict = {
                solver: np.array([
                    np.random.uniform(
                        models['t_l'][solver][i],
                        models['t_u'][solver][i]
                    )
                    for i in range(len(params))
                ])
                for solver, params in models['theta'].items()
                if solver in models['t_l'] and solver in models['t_u']
            }
        else:
            x0_dict = {
                solver: [1] * len(params)  # Replace all elements in the list with 1
                for solver, params in models['theta'].items()
                if solver in models['t_l'] and solver in models['t_u']
            }

        # Populate original_positions and masked_positions
        original_positions = {
            solver: list(range(len(params)))
            for solver, params in models['theta'].items()
        }
        masked_positions = {
            solver: list(range(len(params)))
            for solver, params in models['theta'].items()
        }

        # Remove any existing empty placeholders and update with generated values
        models.pop('original_positions', None)
        models.pop('masked_positions', None)
        iden_opt.pop('x0', None)

        models['original_positions'] = original_positions
        models['masked_positions'] = masked_positions
        iden_opt['x0'] = x0_dict

    # Ensure V_matrix and mutation dictionaries are initialized
    v_matrix_dict = models.get('V_matrix', {})
    mutation_dict = models.get('mutation', {})

    # Loop through each theta in 'theta_parameters'
    for key, theta_values in models['theta'].items():
        length = len(theta_values)

        # Create V_matrix if not available
        if key not in v_matrix_dict:
            v_matrix_dict[key] = [[1e-50 if i == j else 0 for j in range(length)] for i in range(length)]

        # Create mutation list if not available
        if key not in mutation_dict:
            mutation_dict[key] = [True] * length

    # Update models with the new V_matrix and mutation dictionaries
    models['V_matrix'] = v_matrix_dict
    models['mutation'] = mutation_dict


def validation_R2(prediction_metric, validation_metric, reference_metric, case):
    """
    Plot R² prediction, R² validation, and R² reference for each model as an envelope plot.
    Also, create a bar plot with average R² values for prediction, validation, and all the data for each model.

    Parameters:
    prediction_R2 (dict): Dictionary containing R² prediction values for each fold and model.
    validation_R2 (dict): Dictionary containing R² validation values for each fold and model.
    R2_ref (dict): Dictionary containing R² reference values for each model.

    Returns:
    None
    """
    solvers = next(iter(prediction_metric.values())).keys()
    n_folds = len(prediction_metric)

    for solver in solvers:
        if case == 'R2':
            label = 'R²'
        elif case == 'MSE':
            label = 'MSE'

        pred_r2_values = [prediction_metric[i][solver] for i in range(1, n_folds + 1)]
        val_r2_values = [validation_metric[i][solver] for i in range(1, n_folds + 1)]
        ref_r2_value = reference_metric[solver]

        # Plot R² prediction, validation, and reference as an envelope plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_folds + 1), pred_r2_values, marker='o', linestyle='-', color='b', label=f'{label} Prediction')
        plt.plot(range(1, n_folds + 1), val_r2_values, marker='o', linestyle='-', color='r', label=f'{label} Validation')
        plt.fill_between(range(1, n_folds + 1), pred_r2_values, val_r2_values, color='gray', alpha=0.2)
        plt.axhline(y=ref_r2_value, color='g', linestyle='--', label=f'R² Reference = {ref_r2_value:.4f}')
        plt.title(f'{label} Prediction, Validation, and all data for Solver: {solver}')
        plt.xlabel('Fold')
        plt.ylabel(f'{label} Value')
        plt.xticks(range(1, n_folds + 1))  # Ensure x-axis ticks are integers
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        validation_dir = os.path.join(os.getcwd(), 'validation')
        os.makedirs(validation_dir, exist_ok=True)
        full_path = validation_dir
        plot_filename = os.path.join(full_path, f'{label} of validation for {solver}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.show()

        # Calculate average and standard deviation for prediction and validation R² values
        pred_r2_mean = np.mean(pred_r2_values)
        pred_r2_std = np.std(pred_r2_values)
        val_r2_mean = np.mean(val_r2_values)
        val_r2_std = np.std(val_r2_values)

        # Create bar plot with error bars
        plt.figure(figsize=(8, 6))
        bar_width = 0.3  # Adjusted bar width for thinner bars
        bars = ['Prediction', 'Validation', 'All Data']
        y_values = [pred_r2_mean, val_r2_mean, ref_r2_value]
        y_errors = [pred_r2_std, val_r2_std, 0]  # No error bar for reference

        # Bar plot with error bars
        plt.bar(bars, y_values, yerr=y_errors, capsize=5, color=['blue', 'red', 'green'], width=bar_width)

        # Calculate y-axis limits
        min_error = min(y_values[i] - y_errors[i] for i in range(len(y_values)) if y_errors[i] != 0)
        max_error = max(y_values[i] + y_errors[i] for i in range(len(y_values)) if y_errors[i] != 0)
        all_data_value = ref_r2_value
        y_min = min(min_error, all_data_value) - 0.1 * (max_error - min_error)  # Add padding below
        y_max = max(max_error, all_data_value) + 0.1 * (max_error - min_error)  # Add padding above
        y_range = y_max - y_min

        # Ensure that the error bars or R² values span 80% of the y-range
        y_min_adjusted = y_min - 0.1 * y_range
        y_max_adjusted = y_max + 0.1 * y_range
        plt.ylim(y_min_adjusted, y_max_adjusted)

        # Add labels, grid, and save plot
        plt.title(f'Average {label} Values for Solver: {solver}')
        plt.ylabel(f'{label} Value')
        # plt.grid(True, linestyle='--', linewidth=0.7)
        plt.tight_layout()
        bar_plot_filename = os.path.join(full_path, f'Average {label} Values for {solver}.png')
        plt.savefig(bar_plot_filename, dpi=300)
        plt.show()


def validation_params(parameters, ref_params):
    """
    Plot normalized parameters for each model in each fold, divided by the corresponding member in ref_params.

    Parameters:
    parameters (dict): Dictionary containing parameter values for each fold and model. (one data out at a time based cross-validation)
    ref_params (dict): Dictionary containing reference parameter values for each model. (all data used estimation)

    Returns:
    None
    """
    solvers = next(iter(parameters.values())).keys()
    n_folds = len(parameters)

    for solver in solvers:
        param_trends = {}
        for i in range(n_folds):
            if isinstance(parameters[i + 1][solver], dict):
                for param_name, param_value in parameters[i + 1][solver].items():
                    if param_name not in param_trends:
                        param_trends[param_name] = []
                    normalized_value = param_value / ref_params[solver].get(param_name, 1)
                    param_trends[param_name].append(normalized_value)
            elif isinstance(parameters[i + 1][solver], list):
                for idx, param_value in enumerate(parameters[i + 1][solver]):
                    param_name = f'param_{idx}'
                    if param_name not in param_trends:
                        param_trends[param_name] = []
                    normalized_value = param_value / ref_params[solver][idx]
                    param_trends[param_name].append(normalized_value)
            else:
                raise TypeError(f"Unexpected type for parameters[{i + 1}][{solver}]: {type(parameters[i + 1][solver])}")

        # Plotting each parameter trend
        plt.figure(figsize=(10, 6))
        for param_name, values in param_trends.items():
            plt.plot(range(1, n_folds + 1), values, marker='o', linestyle='-', label=param_name)

        plt.title(f'Normalized Parameter Trends for Solver: {solver}')
        plt.xlabel('Fold')
        plt.ylabel('Normalized Parameter Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        validation_dir = os.path.join(os.getcwd(), 'validation')
        os.makedirs(validation_dir, exist_ok=True)
        full_path = validation_dir
        plot_filename = os.path.join(full_path, f'Normalized Parameter Trends for {solver}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.show()


def run_postprocessing(round_data, solvers, selected_rounds,
                       plot_global_p_and_t=True,
                       plot_confidence_spaces=True,
                       plot_p_and_t_tests=True,
                       export_excel_reports=True,
                       plot_estimability=True):
    """
    Run post-processing analysis for specified solvers and rounds with plot/excel export options.

    Parameters
    ----------
    round_data : dict
        Data containing results from parameter estimation rounds.
    solvers : list of str
        Solvers to include in the post-analysis (e.g. ['f20', 'f21']).
    selected_rounds : list of int
        Rounds to include in the plots (e.g. [1, 2, 3]).
    plot_global_p_and_t : bool
        If True, generates global P and t comparison plots across all models and rounds.
    plot_confidence_spaces : bool
        If True, generates confidence space, ellipsoid volume, std dev, and parameter estimate plots.
    plot_p_and_t_tests : bool
        If True, generates P-value and t-value evolution plots over rounds per model.
    export_excel_reports : bool
        If True, generates Excel files with experimental, simulation, and input data + summary.
    plot_estimability : bool
        If True, generates rCC vs k estimability plots per model.
    """
    import os

    postprocessing_dir = os.path.join(os.getcwd(), "post_processing")
    os.makedirs(postprocessing_dir, exist_ok=True)

    if plot_global_p_and_t:
        overall_plotter = Plotting_FinalResults(round_data, None, [])
        overall_plotter.pcomp_plot(postprocessing_dir)
        overall_plotter.tcomp_plot(postprocessing_dir)

    for solver_name in solvers:
        print(f"Post-processing model: {solver_name}")
        plotter = Plotting_FinalResults(round_data, solver_name, selected_rounds)

        if plot_confidence_spaces:
            plotter.conf_plot(
                filename=os.path.join(postprocessing_dir, f"{solver_name}_confidence_space.png"),
                ellipsoid_volume_filename=os.path.join(postprocessing_dir, f"{solver_name}_ellipsoid_volume.png"),
                std_devs_filename=os.path.join(postprocessing_dir, f"{solver_name}_parameter_std_devs.png"),
                parameter_estimates_filename=os.path.join(postprocessing_dir, f"{solver_name}_parameter_estimates.png"),
            )

        if plot_p_and_t_tests:
            plotter.pt_plot(
                filename=os.path.join(postprocessing_dir, f"{solver_name}_pt_tests.png"),
                roundc=round_data
            )

        if export_excel_reports:
            plotter.reporter(
                filename=os.path.join(postprocessing_dir, f"{solver_name}_report")
            )

        if plot_estimability:
            Plot_estimability(
                round_data=round_data,
                path=postprocessing_dir,
                solver=solver_name
            )

    print(f"Post-processing completed for: {', '.join(solvers)}")


