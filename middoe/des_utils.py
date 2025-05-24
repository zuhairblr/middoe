from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

import pandas as pd


def _slicer(x, index_dict, tlin, tv_ophi_forcedsamples, tv_ophi_sampling):
    """
    Slice optimization variables into structured design variables including time-invariant inputs, switching points, and sampling times.
    Handles shared sampling groups and fixed sampling points.

    Parameters:
    x (list): Flat optimization vector.
    index_dict (dict): Dictionary with index mappings for 'ti', 'swps', and 'st'.
    tlin (array): Time grid for snapping.
    tv_ophi_forcedsamples (dict): Dict of fixed sampling times per output.
    tv_ophi_sampling (dict): Sampling group index per output.
    tv_ophi_matching (dict): 1 or 0 flag per output (whether matching is required).

    Returns:
    tuple: ti (dict), swps (dict), St (dict of sampling times per output).
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
    """

    ########################################################################
    # 1) Scale the time-invariant input variables (tii)
    ########################################################################
    for var in ti_iphi_vars:
        if var in phi:
            max_val = ti_iphi_max[ti_iphi_vars.index(var)]
            phi[var] = np.array(phi[var]) * max_val

    ########################################################################
    # 2) Scale the time-variant input variables (tvi)
    ########################################################################
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

    ########################################################################
    # 3) Scale the switching points in `swps`
    ########################################################################
    for var in swps:
        if var.endswith('l'):  # Level scaling
            var_name = var[:-1]  # Remove 'l' to get the variable name
            if var_name in tv_iphi_vars:
                max_val = tv_iphi_max[tv_iphi_vars.index(var_name)]
                swps[var] = (np.array(swps[var]) * max_val).tolist()
        elif var.endswith('t'):  # Time scaling
            swps[var] = (np.array(swps[var]) * tf).tolist()

    ########################################################################
    # 4) Scale the sampling times (St)
    ########################################################################
    for var in St:
        St[var] = (np.array(St[var]) * tf).tolist()

    ########################################################################
    # 5) Scale the time vector `t`
    ########################################################################
    t = np.array(t) * tf

    ########################################################################
    # 6) Plot results using the `_plot_designs` function
    ########################################################################
    _plot_designs(phi, phit, swps, St, performance_metric_value, t,
                  tv_ophi, ti_ophi, design_criteria, round, core_number, pltshow)

    ########################################################################
    # 7) Convert `tii` and `tvi` back to lists for returning
    ########################################################################
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

# def _plot_designs(phi, phit, swps, St, performance_metric, t,
#                   tv_ophi, ti_ophi, design_criteria, round, core_number, pltshow):
#     """
#     Plot MBDoE (model-based design of experiments) results with multiple y-axes
#     for time-variant outputs and inputs, including color-coded switching and sampling times.
#     """
#
#     # Decide if we are in MBDOE_MD or MBDOE_PP
#     if design_criteria in ['HR', 'BFF']:
#         performance_metric_name = "T-optimality"
#     elif design_criteria in ['A', 'D', 'E', 'ME']:
#         performance_metric_name = "PP-optimality"
#     else:
#         performance_metric_name = "Unknown"
#
#     # Create subplots
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
#     # ----------------------------------------------------------------------
#     # SUBPLOT 1: Time-variant output variables
#     # ----------------------------------------------------------------------
#     ax1 = axs[0]
#     ax1.set_title("Time-variant output variables")
#     ax1.set_xlabel('Time (s)')
#
#     ax_list_1 = [ax1]  # manage multiple y-axes
#
#     for solver_idx, (solver_name, solver_tv_ophi) in enumerate(tv_ophi.items()):
#         for var_idx, (var_name, y_values) in enumerate(solver_tv_ophi.items()):
#             color = colors[var_idx % len(colors)]
#             linestyle = linestyles[solver_idx % len(linestyles)]
#
#             if var_idx == 0:
#                 ax1.plot(
#                     t, y_values, color=color, linestyle=linestyle,
#                     label=f'{solver_name} - {var_name}'
#                 )
#                 ax1.set_ylabel(f'{var_name} [-]', color=color)
#                 ax1.tick_params(axis='y', labelcolor=color)
#             else:
#                 ax_new = ax1.twinx()
#                 ax_new.spines['left'].set_position(('outward', 60 * (var_idx - 1)))
#                 ax_new.plot(
#                     t, y_values, color=color, linestyle=linestyle,
#                     label=f'{solver_name} - {var_name}'
#                 )
#                 ax_new.set_ylabel(f'{var_name} [-]', color=color)
#                 ax_new.tick_params(axis='y', labelcolor=color)
#                 ax_list_1.append(ax_new)
#
#             # Plot switching times for the current variable
#             if isinstance(swps, dict) and var_name in swps:
#                 for s_time in swps[var_name]:
#                     ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.5)
#                     ax1.annotate(
#                         f'{s_time:.2f}',
#                         xy=(s_time, ax1.get_ylim()[0]),
#                         xytext=(0, 5),
#                         textcoords='offset points',
#                         ha='center', va='bottom', rotation=90, color=color
#                     )
#
#     # Plot sampling times on the first subplot
#     if isinstance(St, dict):
#         # E.g. St = { 'y1': [times], 'y2': [times], ... }
#         # We'll guess the color by matching var_name in tv_ophi of the first model
#         # or just do a single color if that is simpler.
#         first_solver = next(iter(tv_ophi)) if tv_ophi else None
#         if first_solver:
#             var_list = list(tv_ophi[first_solver].keys())  # e.g. ["y1", "y2", ...]
#         else:
#             var_list = []
#
#         for var_name, sampling_times in St.items():
#             # Try to find color index by var_name in var_list:
#             if var_name in var_list:
#                 color_idx = var_list.index(var_name) % len(colors)
#                 color = colors[color_idx]
#             else:
#                 color = 'k'  # fallback color
#
#             for s_time in sampling_times:
#                 ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.7)
#                 ax1.annotate(
#                     f'{s_time:.2f}',
#                     xy=(s_time, ax1.get_ylim()[1]),
#                     xytext=(0, -15),
#                     textcoords='offset points',
#                     ha='center', va='bottom', rotation=90, color=color
#                 )
#     else:
#         # If St is just a list
#         for s_time in St:
#             ax1.axvline(x=s_time, color='k', linestyle='-', alpha=0.7)
#
#     # Add legend for the first subplot
#     handles_1, labels_1 = ax1.get_legend_handles_labels()
#     for ax in ax_list_1[1:]:
#         h, l = ax.get_legend_handles_labels()
#         handles_1 += h
#         labels_1 += l
#     if handles_1:
#         ax1.legend(handles_1, labels_1, loc='upper right')
#     # ax1.grid(True)
#
#     # ----------------------------------------------------------------------
#     # SUBPLOT 2: "Time-invariant input" or "tvi"
#     # ----------------------------------------------------------------------
#     ax2 = axs[1]
#     ax2.set_title("Time-invariant input variables (tvi)")
#     ax2.set_xlabel('Time (s)')
#
#     ax_list_2 = [ax2]
#
#     # Now that we have a dictionary, do the usual loop
#     for var_idx, (var_name, numeric_value) in enumerate(phit.items()):
#         color = colors[var_idx % len(colors)]
#         linestyle = linestyles[var_idx % len(linestyles)]
#
#         if var_idx == 0:
#             ax2.plot(
#                 t, numeric_value, color=color, linestyle=linestyle,
#                 label=f'{var_name}', linewidth=2.5  # Line thickness
#             )
#             ax2.set_ylabel(f'{var_name} [-]', color=color)
#             ax2.tick_params(axis='y', labelcolor=color)
#         else:
#             ax_new = ax2.twinx()
#             ax_new.spines['right'].set_position(('outward', 60 * (var_idx - 1)))
#             ax_new.plot(
#                 t, numeric_value, color=color, linestyle=linestyle,
#                 label=f'{var_name}', linewidth=2.5  # Line thickness
#             )
#             ax_new.set_ylabel(f'{var_name} [-]', color=color)
#             ax_new.tick_params(axis='y', labelcolor=color)
#             ax_list_2.append(ax_new)
#
#     # Plot switching times on the second subplot
#     if isinstance(swps, dict):
#         for var_name, switching_times in swps.items():
#             # Typically we only plot "time" switching points here
#             if not var_name.endswith('t'):
#                 continue
#             base_var_name = var_name[:-1]  # Remove trailing 't'
#             # Find a color from tvi if possible
#             color = 'k'
#             if isinstance(phit, dict):
#                 # For each design, check if base_var_name is in its dictionary
#                 for design_name, subdict in phit.items():
#                     if isinstance(subdict, dict) and (base_var_name in subdict):
#                         idx = list(subdict.keys()).index(base_var_name)
#                         color = colors[idx % len(colors)]
#                         break
#
#             # Plot the switching lines
#             for s_time in switching_times:
#                 ax2.axvline(x=s_time, color=color, linestyle='--', alpha=0.2)
#                 ax2.annotate(
#                     f'{s_time:.2f}',
#                     xy=(s_time, ax2.get_ylim()[0]),
#                     xytext=(0, 5),
#                     textcoords='offset points',
#                     ha='center', va='bottom', rotation=90, color=color
#                 )
#
#     # Add legend for the second subplot
#     handles_2, labels_2 = ax2.get_legend_handles_labels()
#     for ax in ax_list_2[1:]:
#         h, l = ax.get_legend_handles_labels()
#         handles_2 += h
#         labels_2 += l
#     if handles_2:
#         ax2.legend(handles_2, labels_2, loc='upper right')
#     # ax2.grid(True)
#
#     plt.tight_layout()
#
#     # Save the figure
#     # Create the 'design' subfolder in the current project directory if it doesn't exist
#     design_folder = Path.cwd() / 'design'
#     design_folder.mkdir(parents=True, exist_ok=True)
#
#     # Build the filename
#     final_filename = design_folder / f"{round} (round) by {core_number} core.png"
#
#     plt.savefig(final_filename, dpi=300)
#     if pltshow == True:
#         from IPython.display import Image, display
#         display(Image(final_filename))
#         plt.show()
#     plt.close()

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

def _par_update(theta_parameters, estimations):
    """
    Multiplies corresponding elements in lists or arrays of two dictionaries for matching keys.

    Parameters:
    -----------
    theta_parameters (dict) : dict of model parameters (lists or NumPy arrays).
    estimations (dict)      : dict of normalized parameters (lists or NumPy arrays).

    Returns:
    --------
    dict : A dictionary where each key has values that are element-wise multiplications
           of the corresponding lists or arrays in theta_parameters and estimations.
    """
    result = {}
    for key in theta_parameters:
        if key in estimations:
            param_values = np.array(theta_parameters[key])
            estimation_values = np.array(estimations[key])
            result[key] = (param_values * estimation_values).tolist()
    return result

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

