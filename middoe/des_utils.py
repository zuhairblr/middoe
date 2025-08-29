from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import pandas as pd


def _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling):
    """
    Slice optimization variables into structured design variables including time-invariant inputs,
    switching points, and sampling times. Handles shared sampling groups and fixed sampling points.

    Parameters
    ----------
    x : list
        Flat optimization vector containing all design variables.
    index_dict : dict
        Dictionary with index mappings for different variable types:
        - 'ti': Time-invariant input variables.
        - 'swps': Switching points for time-variant inputs.
        - 'st': Sampling times for outputs.
    tlin : array
        Time grid used for snapping switching and sampling times to valid values.
    tv_ophi_forcedsamples : dict
        Dictionary of fixed sampling times for each output variable.
    tv_ophi_sampling : dict
        Mapping of output variables to their respective sampling groups.

    Returns
    -------
    tuple
        A tuple containing:
        - ti : dict
            Time-invariant input variables.
        - swps : dict
            Switching points for time-variant inputs.
        - St : dict
            Sampling times for each output variable.
    """
    ti = {}
    for var, idx_list in index_dict['ti'].items():
        ti[var] = x[idx_list[0]]

    swps = {}
    for key, idx_list in index_dict['swps'].items():
        if key.endswith('l'):
            values = [x[i] for i in idx_list]
            values.append(values[-1])
            swps[key] = values
        elif key.endswith('t'):
            raw_times = [x[i] for i in idx_list]
            snapped = [float(tlin[np.argmin(np.abs(tlin - t))]) for t in raw_times]
            snapped = [0.0] + snapped + [1.0]
            swps[key] = snapped

    sampling_groups = {}
    for var, group in tv_ophi_sampling.items():
        sampling_groups.setdefault(group, []).append(var)

    St = {}
    processed_groups = set()
    for var, idx_list in index_dict['st'].items():
        group = tv_ophi_sampling[var]
        if group in processed_groups:
            continue

        shared_vars = sampling_groups[group]
        group_idx_lists = [index_dict['st'][v] for v in shared_vars]
        forced = tv_ophi_forcedsamples[shared_vars[0]]
        raw_times = [x[i] for i in group_idx_lists[0]]
        snapped = [float(tlin[np.argmin(np.abs(tlin - t))]) for t in raw_times]
        full_times = forced[:1] + snapped + forced[1:]

        for v in shared_vars:
            St[v] = full_times

        processed_groups.add(group)

    return ti, swps, St

def _reporter(phi, phit, swps, St, performance_metric_value, t, tv_ophi, ti_ophi,
              tv_iphi_vars, tv_iphi_max,
              ti_iphi_vars, ti_iphi_max,
              tf, design_criteria, round, pltshow, core_number):
    """
    Report the design results by scaling variables, plotting designs, and saving the plots.

    This function processes the results of a design optimization by scaling the input variables,
    switching points, and sampling times to their respective ranges. It then generates plots
    to visualize the design and saves the plots and data for further analysis.

    Parameters
    ----------
    phi : dict
        Dictionary of time-invariant input variables.
    phit : dict
        Dictionary of time-variant input variables.
    swps : dict
        Dictionary of switching points for time-variant inputs.
    St : dict
        Dictionary of sampling times for output variables.
    performance_metric_value : float
        The value of the performance metric for the design.
    t : array-like
        Time vector used in the design.
    tv_ophi : dict
        Dictionary of time-variant output variables.
    ti_ophi : dict
        Dictionary of time-invariant output variables.
    tv_iphi_vars : list
        List of time-variant input variable names.
    tv_iphi_max : list
        List of maximum values for time-variant input variables.
    ti_iphi_vars : list
        List of time-invariant input variable names.
    ti_iphi_max : list
        List of maximum values for time-invariant input variables.
    tf : float
        Final time for scaling time-related variables.
    design_criteria : str
        The design criterion used (e.g., 'D-optimality').
    round : int
        The current round of the design process.
    pltshow : bool
        Whether to display the generated plots.
    core_number : int
        The core number used for parallel processing.

    Returns
    -------
    tuple
        A tuple containing the scaled and processed design variables:
        - phi : dict
            Scaled time-invariant input variables.
        - phit : dict
            Scaled time-variant input variables.
        - swps : dict
            Scaled switching points for time-variant inputs.
        - St : dict
            Scaled sampling times for output variables.
    """

    for var in ti_iphi_vars:
        if var in phi:
            max_val = ti_iphi_max[ti_iphi_vars.index(var)]
            phi[var] = np.array(phi[var]) * max_val

    def convert_or_scale_phit(data):
        """
        Recursively process tvi variables:
        - If data is a dict, recursively convert its elements.
        - If data is array-like, convert to NumPy array.
        """
        if isinstance(data, dict):
            return {k: convert_or_scale_phit(v) for k, v in data.items()}
        else:
            # Assume it's array-like, convert to NumPy array
            return np.array(data, dtype=float)

    for var in phit:
        phit[var] = convert_or_scale_phit(phit[var])

    for var in swps:
        if var.endswith('l'):  # Level scaling
            var_name = var[:-1]  # Remove 'l' to get the variable name
            if var_name in tv_iphi_vars:
                max_val = tv_iphi_max[tv_iphi_vars.index(var_name)]
                swps[var] = (np.array(swps[var]) * max_val).tolist()
        elif var.endswith('t'):  # Time scaling
            swps[var] = (np.array(swps[var]) * tf).tolist()

    for var in St:
        St[var] = (np.array(St[var]) * tf).tolist()

    t = np.array(t) * tf

    _plot_designs(phi, phit, swps, St, performance_metric_value, t,
                  tv_ophi, ti_ophi, design_criteria, round, core_number, pltshow)

    for var in phi:
        phi[var] = phi[var].tolist()

    def convert_phit_to_list(data):
        """
        Recursively convert NumPy arrays in tvi back to lists.
        """
        if isinstance(data, dict):
            return {k: convert_phit_to_list(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    for var in phit:
        phit[var] = convert_phit_to_list(phit[var])

    return phi, phit, swps, St


def _plot_designs(phi, phit, swps, St, performance_metric, t,
                  tv_ophi, ti_ophi, design_criteria, round, core_number, pltshow):
    """
    Plot MBDoE results and save corresponding data to Excel file.
    """

    if design_criteria in ['HR', 'BFF']:
        performance_metric_name = "T-optimality"
    elif design_criteria in ['A', 'D', 'E', 'ME']:
        performance_metric_name = "PP-optimality"
    else:
        performance_metric_name = "Unknown"

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    phi_text = ', '.join([f'{var}: {val:.5e}' for var, val in phi.items()])
    performance_metric_text = (
        f'Round {round} - {performance_metric_name} of {design_criteria}: '
        f'{performance_metric:.20e}'
    )
    fig.suptitle(f'{phi_text}\n{performance_metric_text}\n{performance_metric_name}')

    colors = ['k', 'b', 'r', 'c', 'm', 'y', 'g']
    linestyles = ['-', '--', '-.', ':']

    # --- Subplot 1: Time-variant output variables ---
    ax1 = axs[0]
    ax1.set_title("Time-variant output variables")
    ax1.set_xlabel('Time (s)')
    ax_list_1 = [ax1]

    for solver_idx, (solver_name, solver_tv_ophi) in enumerate(tv_ophi.items()):
        for var_idx, (var_name, y_values) in enumerate(solver_tv_ophi.items()):
            color = colors[var_idx % len(colors)]
            linestyle = linestyles[solver_idx % len(linestyles)]
            if var_idx == 0:
                ax1.plot(t, y_values, color=color, linestyle=linestyle, label=f'{solver_name} - {var_name}')
                ax1.set_ylabel(f'{var_name} [-]', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
            else:
                ax_new = ax1.twinx()
                ax_new.spines['left'].set_position(('outward', 60 * (var_idx - 1)))
                ax_new.plot(t, y_values, color=color, linestyle=linestyle, label=f'{solver_name} - {var_name}')
                ax_new.set_ylabel(f'{var_name} [-]', color=color)
                ax_new.tick_params(axis='y', labelcolor=color)
                ax_list_1.append(ax_new)
            if isinstance(swps, dict) and var_name in swps:
                for s_time in swps[var_name]:
                    ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.5)
                    ax1.annotate(f'{s_time:.2f}', xy=(s_time, ax1.get_ylim()[0]),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', rotation=90, color=color)

    # Sampling times
    if isinstance(St, dict):
        first_solver = next(iter(tv_ophi)) if tv_ophi else None
        var_list = list(tv_ophi[first_solver].keys()) if first_solver else []
        for var_name, sampling_times in St.items():
            color = colors[var_list.index(var_name) % len(colors)] if var_name in var_list else 'k'
            for s_time in sampling_times:
                ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.7)
                ax1.annotate(f'{s_time:.2f}', xy=(s_time, ax1.get_ylim()[1]),
                             xytext=(0, -15), textcoords='offset points',
                             ha='center', va='bottom', rotation=90, color=color)
    else:
        for s_time in St:
            ax1.axvline(x=s_time, color='k', linestyle='-', alpha=0.7)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    for ax in ax_list_1[1:]:
        h, l = ax.get_legend_handles_labels()
        handles_1 += h
        labels_1 += l
    if handles_1:
        ax1.legend(handles_1, labels_1, loc='upper right')

    # --- Subplot 2: Time-variant input variables (tvi) ---
    ax2 = axs[1]
    ax2.set_title("Time-variant input variables (tvi)")
    ax2.set_xlabel('Time (s)')
    ax_list_2 = [ax2]

    for var_idx, (var_name, numeric_value) in enumerate(phit.items()):
        color = colors[var_idx % len(colors)]
        linestyle = linestyles[var_idx % len(linestyles)]
        if var_idx == 0:
            ax2.plot(t, numeric_value, color=color, linestyle=linestyle, label=f'{var_name}', linewidth=2.5)
            ax2.set_ylabel(f'{var_name} [-]', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            ax_new = ax2.twinx()
            ax_new.spines['right'].set_position(('outward', 60 * (var_idx - 1)))
            ax_new.plot(t, numeric_value, color=color, linestyle=linestyle, label=f'{var_name}', linewidth=2.5)
            ax_new.set_ylabel(f'{var_name} [-]', color=color)
            ax_new.tick_params(axis='y', labelcolor=color)
            ax_list_2.append(ax_new)

    # Switching points for tvi
    if isinstance(swps, dict):
        for var_name, switching_times in swps.items():
            if not var_name.endswith('t'):
                continue
            base_var_name = var_name[:-1]
            color = 'k'
            if isinstance(phit, dict) and base_var_name in phit:
                idx = list(phit.keys()).index(base_var_name)
                color = colors[idx % len(colors)]
            for s_time in switching_times:
                ax2.axvline(x=s_time, color=color, linestyle='--', alpha=0.2)
                ax2.annotate(f'{s_time:.2f}', xy=(s_time, ax2.get_ylim()[0]),
                             xytext=(0, 5), textcoords='offset points',
                             ha='center', va='bottom', rotation=90, color=color)

    handles_2, labels_2 = ax2.get_legend_handles_labels()
    for ax in ax_list_2[1:]:
        h, l = ax.get_legend_handles_labels()
        handles_2 += h
        labels_2 += l
    if handles_2:
        ax2.legend(handles_2, labels_2, loc='upper right')

    plt.tight_layout()

    # === Save plot and Excel data ===
    design_folder = Path.cwd() / 'design'
    design_folder.mkdir(parents=True, exist_ok=True)
    filename_base = f"{round} (round) by {core_number} core"
    plot_filename = design_folder / f"{filename_base}.png"
    excel_filename = design_folder / f"{filename_base}.xlsx"

    # Save the plot
    plt.savefig(plot_filename, dpi=300)
    if pltshow:
        from IPython.display import Image, display
        display(Image(plot_filename))
        plt.show()
    plt.close()

    # === Save data to Excel ===
    excel_data = {'time': t}
    for solver_name, solver_tv_ophi in tv_ophi.items():
        for var_name, y_values in solver_tv_ophi.items():
            excel_data[f'{solver_name}_{var_name}'] = y_values

    if isinstance(St, dict):
        for var_name, times in St.items():
            excel_data[f'Sampling_{var_name}'] = pd.Series(times)
    else:
        excel_data['Sampling'] = pd.Series(St)

    for var_name, values in phi.items():
        excel_data[f'phi_{var_name}'] = pd.Series(values)

    for var_name, values in phit.items():
        if isinstance(values, dict):
            for subname, subvals in values.items():
                excel_data[f'{var_name}_{subname}'] = subvals
        else:
            excel_data[f'tvi_{var_name}'] = values

    for var_name, sw_values in swps.items():
        excel_data[f'switch_{var_name}'] = pd.Series(sw_values)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in excel_data.items()]))
    df.to_excel(excel_filename, index=False)


def get_var_info(var, var_groups):
    """
    Shared helper, same as in des-md.
    """
    for group_key, group_data in var_groups.items():
        if var in group_data["vars"]:
            i = group_data["vars"].index(var)
            return group_key, i
    raise ValueError(f"Variable '{var}' not found in any var list!")


def configure_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add a handler that goes to stdout (not stderr)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger

