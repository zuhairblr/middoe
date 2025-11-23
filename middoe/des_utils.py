# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# import logging
# import sys
# import pandas as pd
#
# def _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling):
#     """
#     Slice optimization variables into structured design variables including time-invariant inputs,
#     switching points, and sampling times. Handles shared sampling groups and fixed sampling points.
#
#     Parameters
#     ----------
#     x : list
#         Flat optimization vector containing all design variables.
#     index_dict : dict
#         Dictionary with index mappings for different variable types:
#         - 'ti': Time-invariant input variables.
#         - 'swps': Switching points for time-variant inputs.
#         - 'st': Sampling times for outputs.
#     tlin : array
#         Time grid used for snapping switching and sampling times to valid values.
#     tv_ophi_forcedsamples : dict
#         Dictionary of fixed sampling times for each output variable.
#     tv_ophi_sampling : dict
#         Mapping of output variables to their respective sampling groups.
#
#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - ti : dict
#             Time-invariant input variables.
#         - swps : dict
#             Switching points for time-variant inputs.
#         - St : dict
#             Sampling times for each output variable.
#     """
#     ti = {}
#     for var, idx_list in index_dict['ti'].items():
#         ti[var] = x[idx_list[0]]
#
#     swps = {}
#     for key, idx_list in index_dict['swps'].items():
#         if key.endswith('l'):
#             values = [x[i] for i in idx_list]
#             values.append(values[-1])
#             swps[key] = values
#         elif key.endswith('t'):
#             raw_times = [x[i] for i in idx_list]
#             snapped = [float(tlin[np.argmin(np.abs(tlin - t))]) for t in raw_times]
#             snapped = [0.0] + snapped + [1.0]
#             swps[key] = snapped
#
#     sampling_groups = {}
#     for var, group in tv_ophi_sampling.items():
#         sampling_groups.setdefault(group, []).append(var)
#
#     St = {}
#     processed_groups = set()
#     for var, idx_list in index_dict['st'].items():
#         group = tv_ophi_sampling[var]
#         if group in processed_groups:
#             continue
#
#         shared_vars = sampling_groups[group]
#         group_idx_lists = [index_dict['st'][v] for v in shared_vars]
#         forced = tv_ophi_forcedsamples[shared_vars[0]]
#         raw_times = [x[i] for i in group_idx_lists[0]]
#         snapped = [float(tlin[np.argmin(np.abs(tlin - t))]) for t in raw_times]
#         full_times = forced[:1] + snapped + forced[1:]
#
#         for v in shared_vars:
#             St[v] = full_times
#
#         processed_groups.add(group)
#
#     return ti, swps, St
#
# def _reporter(phi, phit, swps, St, performance_metric_value, t, tv_ophi, ti_ophi,
#               tv_iphi_vars, tv_iphi_max,
#               ti_iphi_vars, ti_iphi_max,
#               tf, design_criteria, round, pltshow, core_number):
#     """
#     Report the design results by scaling variables, plotting designs, and saving the plots.
#
#     This function processes the results of a design optimization by scaling the input variables,
#     switching points, and sampling times to their respective ranges. It then generates plots
#     to visualize the design and saves the plots and data for further analysis.
#
#     Parameters
#     ----------
#     phi : dict
#         Dictionary of time-invariant input variables.
#     phit : dict
#         Dictionary of time-variant input variables.
#     swps : dict
#         Dictionary of switching points for time-variant inputs.
#     St : dict
#         Dictionary of sampling times for output variables.
#     performance_metric_value : float
#         The value of the performance metric for the design.
#     t : array-like
#         Time vector used in the design.
#     tv_ophi : dict
#         Dictionary of time-variant output variables.
#     ti_ophi : dict
#         Dictionary of time-invariant output variables.
#     tv_iphi_vars : list
#         List of time-variant input variable names.
#     tv_iphi_max : list
#         List of maximum values for time-variant input variables.
#     ti_iphi_vars : list
#         List of time-invariant input variable names.
#     ti_iphi_max : list
#         List of maximum values for time-invariant input variables.
#     tf : float
#         Final time for scaling time-related variables.
#     design_criteria : str
#         The design criterion used (e.g., 'D-optimality').
#     round : int
#         The current round of the design process.
#     pltshow : bool
#         Whether to display the generated plots.
#     core_number : int
#         The core number used for parallel processing.
#
#     Returns
#     -------
#     tuple
#         A tuple containing the scaled and processed design variables:
#         - phi : dict
#             Scaled time-invariant input variables.
#         - phit : dict
#             Scaled time-variant input variables.
#         - swps : dict
#             Scaled switching points for time-variant inputs.
#         - St : dict
#             Scaled sampling times for output variables.
#     """
#
#     for var in ti_iphi_vars:
#         if var in phi:
#             max_val = ti_iphi_max[ti_iphi_vars.index(var)]
#             phi[var] = np.array(phi[var]) * max_val
#
#     def convert_or_scale_phit(data):
#         """
#         Recursively process tvi variables:
#         - If data is a dict, recursively convert its elements.
#         - If data is array-like, convert to NumPy array.
#         """
#         if isinstance(data, dict):
#             return {k: convert_or_scale_phit(v) for k, v in data.items()}
#         else:
#             # Assume it's array-like, convert to NumPy array
#             return np.array(data, dtype=float)
#
#     for var in phit:
#         phit[var] = convert_or_scale_phit(phit[var])
#
#     for var in swps:
#         if var.endswith('l'):  # Level scaling
#             var_name = var[:-1]  # Remove 'l' to get the variable name
#             if var_name in tv_iphi_vars:
#                 max_val = tv_iphi_max[tv_iphi_vars.index(var_name)]
#                 swps[var] = (np.array(swps[var]) * max_val).tolist()
#         elif var.endswith('t'):  # Time scaling
#             swps[var] = (np.array(swps[var]) * tf).tolist()
#
#     for var in St:
#         St[var] = (np.array(St[var]) * tf).tolist()
#
#     t = np.array(t) * tf
#
#     _plot_designs(phi, phit, swps, St, performance_metric_value, t,
#                   tv_ophi, ti_ophi, design_criteria, round, core_number, pltshow)
#
#     for var in phi:
#         phi[var] = phi[var].tolist()
#
#     def convert_phit_to_list(data):
#         """
#         Recursively convert NumPy arrays in tvi back to lists.
#         """
#         if isinstance(data, dict):
#             return {k: convert_phit_to_list(v) for k, v in data.items()}
#         elif isinstance(data, np.ndarray):
#             return data.tolist()
#         else:
#             return data
#
#     for var in phit:
#         phit[var] = convert_phit_to_list(phit[var])
#
#     return phi, phit, swps, St
#
# def _plot_designs(phi, phit, swps, St, performance_metric, t,
#                   tv_ophi, ti_ophi, design_criteria, round, core_number, pltshow):
#     """
#     Plot MBDoE results and save corresponding data to Excel file.
#     """
#
#     if design_criteria in ['HR', 'BFF']:
#         performance_metric_name = "T-optimality"
#     elif design_criteria in ['A', 'D', 'E', 'ME']:
#         performance_metric_name = "PP-optimality"
#     else:
#         performance_metric_name = "Unknown"
#
#     fig, axs = plt.subplots(2, 1, figsize=(12, 12))
#     phi_text = ', '.join([f'{var}: {val:.5e}' for var, val in phi.items()])
#     performance_metric_text = (
#         f'Round {round} - {performance_metric_name} of {design_criteria}: '
#         f'{performance_metric:.20e}'
#     )
#     fig.suptitle(f'{phi_text}\n{performance_metric_text}\n{performance_metric_name}')
#
#     colors = ['k', 'b', 'r', 'c', 'm', 'y', 'g']
#     linestyles = ['-', '--', '-.', ':']
#
#     # --- Subplot 1: Time-variant output variables ---
#     ax1 = axs[0]
#     ax1.set_title("Time-variant output variables")
#     ax1.set_xlabel('Time (s)')
#     ax_list_1 = [ax1]
#
#     for solver_idx, (solver_name, solver_tv_ophi) in enumerate(tv_ophi.items()):
#         for var_idx, (var_name, y_values) in enumerate(solver_tv_ophi.items()):
#             color = colors[var_idx % len(colors)]
#             linestyle = linestyles[solver_idx % len(linestyles)]
#             if var_idx == 0:
#                 ax1.plot(t, y_values, color=color, linestyle=linestyle, label=f'{solver_name} - {var_name}')
#                 ax1.set_ylabel(f'{var_name} [-]', color=color)
#                 ax1.tick_params(axis='y', labelcolor=color)
#             else:
#                 ax_new = ax1.twinx()
#                 ax_new.spines['left'].set_position(('outward', 60 * (var_idx - 1)))
#                 ax_new.plot(t, y_values, color=color, linestyle=linestyle, label=f'{solver_name} - {var_name}')
#                 ax_new.set_ylabel(f'{var_name} [-]', color=color)
#                 ax_new.tick_params(axis='y', labelcolor=color)
#                 ax_list_1.append(ax_new)
#             if isinstance(swps, dict) and var_name in swps:
#                 for s_time in swps[var_name]:
#                     ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.5)
#                     ax1.annotate(f'{s_time:.2f}', xy=(s_time, ax1.get_ylim()[0]),
#                                  xytext=(0, 5), textcoords='offset points',
#                                  ha='center', va='bottom', rotation=90, color=color)
#
#     # Sampling times
#     if isinstance(St, dict):
#         first_solver = next(iter(tv_ophi)) if tv_ophi else None
#         var_list = list(tv_ophi[first_solver].keys()) if first_solver else []
#         for var_name, sampling_times in St.items():
#             color = colors[var_list.index(var_name) % len(colors)] if var_name in var_list else 'k'
#             for s_time in sampling_times:
#                 ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.7)
#                 ax1.annotate(f'{s_time:.2f}', xy=(s_time, ax1.get_ylim()[1]),
#                              xytext=(0, -15), textcoords='offset points',
#                              ha='center', va='bottom', rotation=90, color=color)
#     else:
#         for s_time in St:
#             ax1.axvline(x=s_time, color='k', linestyle='-', alpha=0.7)
#
#     handles_1, labels_1 = ax1.get_legend_handles_labels()
#     for ax in ax_list_1[1:]:
#         h, l = ax.get_legend_handles_labels()
#         handles_1 += h
#         labels_1 += l
#     if handles_1:
#         ax1.legend(handles_1, labels_1, loc='upper right')
#
#     # --- Subplot 2: Time-variant input variables (tvi) ---
#     ax2 = axs[1]
#     ax2.set_title("Time-variant input variables (tvi)")
#     ax2.set_xlabel('Time (s)')
#     ax_list_2 = [ax2]
#
#     for var_idx, (var_name, numeric_value) in enumerate(phit.items()):
#         color = colors[var_idx % len(colors)]
#         linestyle = linestyles[var_idx % len(linestyles)]
#         if var_idx == 0:
#             ax2.plot(t, numeric_value, color=color, linestyle=linestyle, label=f'{var_name}', linewidth=2.5)
#             ax2.set_ylabel(f'{var_name} [-]', color=color)
#             ax2.tick_params(axis='y', labelcolor=color)
#         else:
#             ax_new = ax2.twinx()
#             ax_new.spines['right'].set_position(('outward', 60 * (var_idx - 1)))
#             ax_new.plot(t, numeric_value, color=color, linestyle=linestyle, label=f'{var_name}', linewidth=2.5)
#             ax_new.set_ylabel(f'{var_name} [-]', color=color)
#             ax_new.tick_params(axis='y', labelcolor=color)
#             ax_list_2.append(ax_new)
#
#     # Switching points for tvi
#     if isinstance(swps, dict):
#         for var_name, switching_times in swps.items():
#             if not var_name.endswith('t'):
#                 continue
#             base_var_name = var_name[:-1]
#             color = 'k'
#             if isinstance(phit, dict) and base_var_name in phit:
#                 idx = list(phit.keys()).index(base_var_name)
#                 color = colors[idx % len(colors)]
#             for s_time in switching_times:
#                 ax2.axvline(x=s_time, color=color, linestyle='--', alpha=0.2)
#                 ax2.annotate(f'{s_time:.2f}', xy=(s_time, ax2.get_ylim()[0]),
#                              xytext=(0, 5), textcoords='offset points',
#                              ha='center', va='bottom', rotation=90, color=color)
#
#     handles_2, labels_2 = ax2.get_legend_handles_labels()
#     for ax in ax_list_2[1:]:
#         h, l = ax.get_legend_handles_labels()
#         handles_2 += h
#         labels_2 += l
#     if handles_2:
#         ax2.legend(handles_2, labels_2, loc='upper right')
#
#     plt.tight_layout()
#
#     # === Save plot and Excel data ===
#     design_folder = Path.cwd() / 'design'
#     design_folder.mkdir(parents=True, exist_ok=True)
#     filename_base = f"{round} (round) by {core_number} core"
#     plot_filename = design_folder / f"{filename_base}.png"
#     excel_filename = design_folder / f"{filename_base}.xlsx"
#
#     # Save the plot
#     plt.savefig(plot_filename, dpi=300)
#     if pltshow:
#         from IPython.display import Image, display
#         display(Image(plot_filename))
#         plt.show()
#     plt.close()
#
#     # === Save data to Excel ===
#     excel_data = {'time': t}
#     for solver_name, solver_tv_ophi in tv_ophi.items():
#         for var_name, y_values in solver_tv_ophi.items():
#             excel_data[f'{solver_name}_{var_name}'] = y_values
#
#     if isinstance(St, dict):
#         for var_name, times in St.items():
#             excel_data[f'Sampling_{var_name}'] = pd.Series(times)
#     else:
#         excel_data['Sampling'] = pd.Series(St)
#
#     for var_name, values in phi.items():
#         excel_data[f'phi_{var_name}'] = pd.Series(values)
#
#     for var_name, values in phit.items():
#         if isinstance(values, dict):
#             for subname, subvals in values.items():
#                 excel_data[f'{var_name}_{subname}'] = subvals
#         else:
#             excel_data[f'tvi_{var_name}'] = values
#
#     for var_name, sw_values in swps.items():
#         excel_data[f'switch_{var_name}'] = pd.Series(sw_values)
#
#     df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in excel_data.items()]))
#     df.to_excel(excel_filename, index=False)
#
# def _get_var_info(var, var_groups):
#     """
#     Shared helper, same as in des-md.
#     """
#     for group_key, group_data in var_groups.items():
#         if var in group_data["vars"]:
#             i = group_data["vars"].index(var)
#             return group_key, i
#     raise ValueError(f"Variable '{var}' not found in any var list!")
#
# def _configure_logger(name=None, level=logging.INFO):
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#
#     # Remove existing handlers to prevent duplication
#     if logger.hasHandlers():
#         logger.handlers.clear()
#
#     # Add a handler that goes to stdout (not stderr)
#     handler = logging.StreamHandler(sys.stdout)
#     formatter = logging.Formatter('%(levelname)s: %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#
#     logger.propagate = False
#     return logger

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import pandas as pd


def _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling):
    r"""
    Slice optimisation decision vector into structured design variables.

    This function decodes the flat optimisation vector into physically meaningful
    design variables: time-invariant inputs, switching points (times and levels)
    for time-variant inputs, and sampling times for outputs. It handles shared
    sampling groups and merges free sampling times with forced sampling points.

    Parameters
    ----------
    x : list or np.ndarray
        Flat optimisation vector containing all design variables in the following order:
            1. Time-invariant input levels (normalised to [0, 1])
            2. Time-variant input levels for each segment (normalised)
            3. Switching times between segments (normalised to [0, 1])
            4. Free sampling times for outputs (normalised to [0, 1])
    index_dict : dict
        Dictionary with index mappings for different variable types:
            - 'ti' : dict[str, list[int]]
                Time-invariant input variable names mapped to their indices in x.
            - 'swps' : dict[str, list[int]]
                Switching point variable names (with suffixes 'l' for levels,
                't' for times) mapped to their indices in x.
            - 'st' : dict[str, list[int]]
                Sampling time variable names (output variables) mapped to their
                indices in x.
    tlin : np.ndarray
        Time grid (normalised, typically [0, 1]) used for snapping switching and
        sampling times to valid discretised values. This ensures all times align
        with the simulation grid.
    tv_ophi_forcedsamples : dict[str, list[float]]
        Dictionary of fixed (forced) sampling times for each output variable.
        Keys are output variable names, values are lists of normalised times.
        Forced times are typically at boundaries (e.g., [0.0, 1.0]).
    tv_ophi_sampling : dict[str, str]
        Mapping of output variables to their respective sampling synchronisation groups.
        Variables in the same group share identical sampling times.

    Returns
    -------
    ti : dict[str, float]
        Time-invariant input variables (normalised values).
    swps : dict[str, list[float]]
        Switching points for time-variant inputs:
            - Keys ending with 'l': level values (normalised).
            - Keys ending with 't': switching times (normalised, snapped to tlin).
    St : dict[str, list[float]]
        Sampling times for each output variable (normalised, snapped to tlin).
        Variables in the same synchronisation group have identical sampling times.

    Notes
    -----
    **Decision Variable Ordering**:
    The flat vector x is structured as:
        1. Time-invariant inputs (one value per variable)
        2. Time-variant levels (seg-1 values per variable, where seg = number of segments)
        3. Switching times (seg-2 values per variable, boundaries are fixed)
        4. Free sampling times (grouped by synchronisation)

    **Time Snapping**:
    Switching and sampling times are snapped to the nearest value in tlin to ensure
    consistency with the simulation grid. This prevents interpolation errors and
    ensures times are feasible for the solver.

    **Sampling Groups**:
    Outputs sharing a sampling group (same value in tv_ophi_sampling) use the same
    set of sampling times. This reduces decision variables and enforces synchronisation
    when measurements are taken simultaneously (e.g., multi-sensor instruments).

    **Forced Sampling**:
    Forced sampling times (typically boundaries) are merged with free sampling times.
    The convention used here assumes forced[0] is the initial time and forced[1:] are
    final/boundary times.

    See Also
    --------
    _reporter : Scales and reports decoded design variables.
    _optimiser : Constructs index_dict and decision vector.

    Examples
    --------
    >>> x = [0.5, 0.3, 0.7, 0.2, 0.4, 0.6]  # Flat decision vector
    >>> index_dict = {
    ...     'ti': {'T_init': [0]},
    ...     'swps': {'u1l': [1, 2], 'u1t': [3]},
    ...     'st': {'y1': [4, 5]}
    ... }
    >>> tlin = np.linspace(0, 1, 101)
    >>> tv_ophi_forcedsamples = {'y1': [0.0, 1.0]}
    >>> tv_ophi_sampling = {'y1': 'group1'}
    >>> ti, swps, St = _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling)
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
    r"""
    Scale, visualise, and report optimised experimental design results.

    This function denormalises all design variables (inputs, switching points,
    sampling times) from the optimisation space [0, 1] to physical units, generates
    comprehensive visualisation plots, saves results to disk, and returns the
    scaled design for experimental execution.

    Parameters
    ----------
    phi : dict[str, float]
        Time-invariant input variables (normalised values from optimisation).
    phit : dict[str, np.ndarray or dict]
        Time-variant input profiles (normalised values). May be nested dictionaries
        for complex control structures.
    swps : dict[str, list[float]]
        Switching points for time-variant inputs (normalised):
            - Keys ending with 'l': level values.
            - Keys ending with 't': switching times.
    St : dict[str, list[float]] or list[float]
        Sampling times for output variables (normalised). Can be a dictionary
        (per-variable sampling) or a single list (shared sampling).
    performance_metric_value : float
        Value of the optimisation objective (e.g., det(FIM), T-optimality).
    t : np.ndarray or list[float]
        Normalised time vector used for simulation (typically [0, 1]).
    tv_ophi : dict[str, dict[str, np.ndarray]]
        Time-variant output predictions:
            - Outer keys: model/solver names.
            - Inner keys: output variable names.
            - Values: predicted trajectories over time.
    ti_ophi : dict[str, dict[str, float]]
        Time-invariant output predictions (terminal or steady-state values).
    tv_iphi_vars : list[str]
        Names of time-variant input variables.
    tv_iphi_max : list[float]
        Maximum (scaling) values for time-variant input variables.
    ti_iphi_vars : list[str]
        Names of time-invariant input variables.
    ti_iphi_max : list[float]
        Maximum (scaling) values for time-invariant input variables.
    tf : float
        Final time for denormalising time-related variables (in physical units, e.g., seconds).
    design_criteria : str
        Optimisation criterion used ('D', 'A', 'E', 'ME', 'HR', 'BFF').
    round : int
        Current experimental design round (for labelling and file naming).
    pltshow : bool
        If True, display plots interactively (in Jupyter/IPython environments).
    core_number : int
        Core/process identifier for parallel runs (for file naming).

    Returns
    -------
    phi : dict[str, list[float]]
        Scaled time-invariant input variables (denormalised, converted to lists).
    phit : dict[str, list or dict]
        Scaled time-variant input profiles (denormalised, converted to lists).
    swps : dict[str, list[float]]
        Scaled switching points (denormalised).
    St : dict[str, list[float]] or list[float]
        Scaled sampling times (denormalised).

    Notes
    -----
    **Denormalisation**:
    All normalised variables (range [0, 1]) are scaled back to physical units:
        - Levels: multiplied by corresponding max values.
        - Times: multiplied by tf.

    **Visualisation**:
    Two subplots are generated:
        1. Time-variant outputs with sampling times (vertical lines).
        2. Time-variant inputs with switching times (vertical lines).

    Each subplot supports multiple y-axes (twinx) for variables with different scales.

    **File Outputs**:
    Results are saved in a './design/' folder:
        - PNG plot: '{round} (round) by {core_number} core.png'
        - Excel file: '{round} (round) by {core_number} core.xlsx'

    The Excel file contains all time-series data, sampling times, and design variables.

    **Data Type Conversions**:
    All NumPy arrays are converted to lists for JSON serialisation and downstream
    compatibility with non-NumPy systems.

    See Also
    --------
    _plot_designs : Core plotting logic.
    _slicer : Decodes optimisation vector into design variables.

    Examples
    --------
    >>> phi, phit, swps, St = _reporter(
    ...     phi, phit, swps, St, pp_obj, t_values,
    ...     tv_ophi_dict, ti_ophi_dict,
    ...     tv_iphi_vars, tv_iphi_max,
    ...     ti_iphi_vars, ti_iphi_max,
    ...     tf, design_criteria, round, pltshow=True, core_number=0
    ... )
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
    r"""
    Generate and save visualisation plots for MBDoE experimental designs.

    This function creates a two-panel figure showing:
        1. Time-variant output trajectories with sampling times.
        2. Time-variant input profiles with switching times.

    Results are saved as high-resolution PNG images and Excel files in a './design/' folder.

    Parameters
    ----------
    phi : dict[str, float or np.ndarray]
        Time-invariant input variables (denormalised).
    phit : dict[str, np.ndarray or dict]
        Time-variant input profiles (denormalised).
    swps : dict[str, list[float]]
        Switching points (times and levels, denormalised).
    St : dict[str, list[float]] or list[float]
        Sampling times (denormalised).
    performance_metric : float
        Optimisation objective value (e.g., det(FIM), T-optimality).
    t : np.ndarray
        Time vector (denormalised, physical units).
    tv_ophi : dict[str, dict[str, np.ndarray]]
        Time-variant output predictions (model -> variable -> trajectory).
    ti_ophi : dict[str, dict[str, float]]
        Time-invariant output predictions (model -> variable -> value).
    design_criteria : str
        Optimisation criterion ('D', 'A', 'E', 'ME', 'HR', 'BFF').
    round : int
        Current experimental design round.
    core_number : int
        Core identifier for parallel runs.
    pltshow : bool
        If True, display plot interactively (requires IPython environment).

    Notes
    -----
    **Plot Structure**:
        - **Subplot 1** (top): Time-variant outputs with sampling time markers.
        - **Subplot 2** (bottom): Time-variant inputs with switching time markers.

    Each subplot uses multiple y-axes (twinx) to accommodate variables with different
    scales. Colors and linestyles distinguish models and variables.

    **Performance Metric Mapping**:
        - 'HR', 'BFF' → "T-optimality" (model discrimination)
        - 'D', 'A', 'E', 'ME' → "PP-optimality" (parameter precision)

    **File Outputs**:
        - **PNG**: High-resolution plot (300 DPI) saved as '{round} (round) by {core_number} core.png'.
        - **Excel**: Tabular data (time, outputs, inputs, sampling times, switching points)
          saved as '{round} (round) by {core_number} core.xlsx'.

    **Annotations**:
        - Sampling times are marked as vertical dashed lines with time labels (top of subplot 1).
        - Switching times are marked as vertical dashed lines with time labels (bottom of subplot 2).

    **Color Scheme**:
        - Colors: ['k', 'b', 'r', 'c', 'm', 'y', 'g'] (cycled for variables).
        - Linestyles: ['-', '--', '-.', ':'] (cycled for models/solvers).

    See Also
    --------
    _reporter : Calls this function after denormalising variables.
    matplotlib.pyplot.subplots : Used for figure creation.
    pandas.DataFrame.to_excel : Saves tabular data.

    Examples
    --------
    >>> _plot_designs(
    ...     phi, phit, swps, St, 1.234e5, t_values,
    ...     tv_ophi_dict, ti_ophi_dict,
    ...     'D', round=1, core_number=0, pltshow=True
    ... )
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


def _get_var_info(var, var_groups):
    r"""
    Retrieve group and index information for a variable from grouped variable definitions.

    This utility function searches through grouped variable dictionaries to find
    which group a variable belongs to and its position within that group.

    Parameters
    ----------
    var : str
        Variable name to search for.
    var_groups : dict[str, dict]
        Dictionary of variable groups. Each group contains:
            - 'vars' : list[str]
                List of variable names in the group.
            - Additional metadata (e.g., bounds, constraints).

    Returns
    -------
    group_key : str
        The key of the group containing the variable.
    index : int
        Zero-based index of the variable within its group's 'vars' list.

    Raises
    ------
    ValueError
        If the variable is not found in any group.

    Notes
    -----
    This function is typically used for model discrimination (MBDoE-MD) when
    variables are grouped by synchronisation or sampling constraints.

    See Also
    --------
    _slicer : Uses variable grouping information.

    Examples
    --------
    >>> var_groups = {
    ...     'group1': {'vars': ['y1', 'y2'], 'constraint': 'sync'},
    ...     'group2': {'vars': ['y3'], 'constraint': 'independent'}
    ... }
    >>> group, idx = _get_var_info('y2', var_groups)
    >>> print(group, idx)
    group1 1
    """
    for group_key, group_data in var_groups.items():
        if var in group_data["vars"]:
            i = group_data["vars"].index(var)
            return group_key, i
    raise ValueError(f"Variable '{var}' not found in any var list!")


def configure_logger(name=None, level=logging.INFO):
    r"""
    Configure a logger with standard output formatting for MBDoE operations.

    This function creates or reconfigures a logger with a consistent format and
    ensures output is directed to stdout (not stderr). Existing handlers are
    removed to prevent duplicate log messages.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns the root logger (default: None).
    level : int, optional
        Logging level (default: logging.INFO).
        Common levels:
            - logging.DEBUG (10): Detailed diagnostic information.
            - logging.INFO (20): General informational messages.
            - logging.WARNING (30): Warning messages.
            - logging.ERROR (40): Error messages.
            - logging.CRITICAL (50): Critical errors.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.

    Notes
    -----
    **Handler Configuration**:
        - A StreamHandler is attached to sys.stdout (not sys.stderr).
        - Format: '%(levelname)s: %(message)s' (e.g., "INFO: Optimisation started").

    **Handler Clearing**:
        If the logger already has handlers, they are removed before adding a new one.
        This prevents duplicate messages when the function is called multiple times.

    **Propagation**:
        Logger propagation is disabled (logger.propagate = False) to prevent messages
        from being passed to parent loggers.

    See Also
    --------
    logging.getLogger : Standard library logger retrieval.
    logging.StreamHandler : Standard library stream handler.

    Examples
    --------
    >>> logger = configure_logger('mbdoe', level=logging.DEBUG)
    >>> logger.info("Starting optimisation")
    INFO: Starting optimisation

    >>> logger.debug("Current objective value: 1.234e5")
    DEBUG: Current objective value: 1.234e5
    """
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