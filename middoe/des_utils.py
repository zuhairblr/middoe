import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint

def _segmenter(tv_iphi_vars, tv_iphi_seg, tv_iphi_max, tv_iphi_min, tv_iphi_const, tv_iphi_offsett, tv_iphi_offsetl,
        ti_iphi_vars, ti_iphi_max, ti_iphi_min, tv_ophi_vars, tv_ophi_seg, tv_ophi_offsett, tv_ophi_matching, tf, ti):
    """
    Segment the input variables for the design space.

    Parameters:
    tv_iphi_vars (list): List of time-variant input variable names.
    tv_iphi_seg (list): List of segment counts for time-variant input variables (segmenting inputs by piecewise).
    tv_iphi_max (list): List of maximum design space values for time-variant input variables.
    tv_iphi_min (list): List of minimum design space values for time-variant input variables.
    tv_iphi_const (list): List of constraints for time-variant input variables: 'rel' (relaxed), 'dec' (decreasing), 'inc' (increasing).
    ti_iphi_vars (list): List of time-invariant input variable names.
    ti_iphi_max (list): List of maximum design space values for time-invariant input variables.
    ti_iphi_min (list): List of minimum design space values for time-invariant input variables.
    tv_ophi_seg (list): List of segment counts for time-variant output variables (segmenting responses by sampling).
    tf (float): Final time value.
    ti (float): Sampling constraint (minimum time interval since batch start to allow sampling).
    offsett (float): Minimum time interval between two consecutive switching times in a piecewise function for time-variant inputs.
    offsetl (float): Minimum level difference between two consecutive switching levels in a piecewise function for time-variant inputs.

    Returns:
    tuple: A tuple containing bounds for design decisions, initial estimation points for design decisions,
    dictionaries of index pairs for constraints (increasing or decreasing constraints) for levels and times,
    and a dictionary containing index ranges for each type of variable.
    """

    # Initialize the index pairs for constraints as dictionaries
    index_pairs_levels = {}  # For level constraints (increasing or decreasing)
    index_pairs_times = {}   # For time constraints (always increasing)

    # Initialize bounds per variable for levels and times
    tvl_bounds_per_var = []
    tvt_bounds_per_var = []
    bounds_for_samples_per_var= []
    tvl_bounds = []
    tvt_bounds = []
    st_bounds = []
    for i, var in enumerate(tv_iphi_vars):
        max_val, min_val, seg_count = tv_iphi_max[i], tv_iphi_min[i], tv_iphi_seg[i]
        # Levels bounds for this variable
        bounds_for_levels = [(min_val / max_val, 1) for _ in range(seg_count - 1)]
        tvl_bounds_per_var.append(bounds_for_levels)
        tvl_bounds.extend(bounds_for_levels)
        # Times bounds for this variable
        bounds_for_times = [((ti / tf), (1 - (ti / tf))) for _ in range(seg_count - 2)]
        tvt_bounds_per_var.append(bounds_for_times)
        tvt_bounds.extend(bounds_for_times)

    # Time-invariant input bounds
    ti_bounds = [(ti_min / ti_max, 1) for ti_min, ti_max in zip(ti_iphi_min, ti_iphi_max)]

    # Sampling times bounds
    for i, var in enumerate(tv_ophi_vars):
        seg_count = tv_ophi_seg[i]
        bounds_for_samples = [((ti / tf), (1 - (ti / tf))) for _ in range(seg_count - 2)]
        bounds_for_samples_per_var.append(bounds_for_samples)
        st_bounds.extend(bounds_for_samples)

    # Combine all bounds
    bounds = ti_bounds + tvl_bounds + tvt_bounds + st_bounds

    # Function to generate random initial guesses within the feasible region
    def generate_feasible_initial_guesses(tv_iphi_const, tv_iphi_seg, tv_iphi_offsett, tv_iphi_offsetl, tv_ophi_vars, tv_ophi_offsett):
        """
        Generate initial guesses for the optimization variables that are within the feasible region.

        Parameters:
        tv_iphi_const (list): Constraints for time-variant inputs.
        tv_iphi_seg (list): Number of segments for each time-variant input variable.

        Returns:
        np.array: Initial guesses for the optimization variables.
        """
        x0 = []

        # Generate random guesses for time-invariant inputs within bounds
        for bound in ti_bounds:
            x0.append(np.random.uniform(bound[0], bound[1]))

        # Generate random feasible levels for time-variant inputs
        for i, const in enumerate(tv_iphi_const):
            seg_count = tv_iphi_seg[i]
            bounds_for_levels = tvl_bounds_per_var[i]
            min_bound = np.array([b[0] for b in bounds_for_levels])
            max_bound = np.array([b[1] for b in bounds_for_levels])

            if const == 'rel':
                # No specific ordering constraint
                levels = np.random.uniform(min_bound, max_bound)
            else:
                # Constraints exist ('inc' or 'dec')
                # Generate feasible levels respecting the ordering constraints and offsetl
                levels = []
                if const == 'inc':
                    current_min = min_bound[0]
                    for j in range(seg_count - 1):
                        max_level = max_bound[j]
                        feasible_max = max_level
                        feasible_min = max(current_min + tv_iphi_offsetl[i], min_bound[j])
                        if feasible_min > feasible_max:
                            # Adjust feasible_min if it exceeds feasible_max
                            feasible_min = feasible_max
                        level = np.random.uniform(feasible_min, feasible_max)
                        levels.append(level)
                        current_min = level
                elif const == 'dec':
                    current_max = max_bound[0]
                    for j in range(seg_count - 1):
                        min_level = min_bound[j]
                        feasible_min = min_level
                        feasible_max = min(current_max - tv_iphi_offsetl[i], max_bound[j])
                        if feasible_max < feasible_min:
                            # Adjust feasible_max if it falls below feasible_min
                            feasible_max = feasible_min
                        level = np.random.uniform(feasible_min, feasible_max)
                        levels.append(level)
                        current_max = level
                levels = np.array(levels)
            x0.extend(levels)

        # Generate feasible increasing times for each time-variant input variable
        for i, var in enumerate(tv_iphi_vars):
            seg_count = tv_iphi_seg[i]
            num_times = seg_count - 2
            if num_times > 0:
                feasible_times = []
                bounds_for_times = tvt_bounds_per_var[i]
                min_bound = np.array([b[0] for b in bounds_for_times])
                max_bound = np.array([b[1] for b in bounds_for_times])

                current_min = min_bound[0]
                for j in range(num_times):
                    feasible_min = max(current_min + tv_iphi_offsett[i] / tf, min_bound[j])
                    feasible_max = max_bound[j]
                    if feasible_min > feasible_max:
                        # Adjust feasible_min if it exceeds feasible_max
                        feasible_min = feasible_max
                    time_point = np.random.uniform(feasible_min, feasible_max)
                    feasible_times.append(time_point)
                    current_min = time_point
                x0.extend(feasible_times)

        # # Generate feasible increasing sampling times (st)
        # num_st = len(st_bounds)
        # if num_st > 0:
        #     feasible_st_times = []
        #     min_bound = np.array([b[0] for b in st_bounds])
        #     max_bound = np.array([b[1] for b in st_bounds])
        #
        #     current_min = min_bound[0]
        #     for j in range(num_st):
        #         feasible_min = max(current_min + offsett / tf, min_bound[j])
        #         feasible_max = max_bound[j]
        #         if feasible_min > feasible_max:
        #             # Adjust feasible_min if it exceeds feasible_max
        #             feasible_min = feasible_max
        #         st_time = np.random.uniform(feasible_min, feasible_max)
        #         feasible_st_times.append(st_time)
        #         current_min = st_time
        #     x0.extend(feasible_st_times)

        # Generate feasible increasing times for each time-variant input variable
        for i, var in enumerate(tv_ophi_vars):
            seg_count = tv_ophi_seg[i]
            num_times = seg_count - 2
            if num_times > 0:
                feasible_times = []
                bounds_for_times = bounds_for_samples_per_var[i]
                min_bound = np.array([b[0] for b in bounds_for_times])
                max_bound = np.array([b[1] for b in bounds_for_times])

                current_min = min_bound[0]
                for j in range(num_times):
                    feasible_min = max(current_min + tv_ophi_offsett[i] / tf, min_bound[j])
                    feasible_max = max_bound[j]
                    if feasible_min > feasible_max:
                        # Adjust feasible_min if it exceeds feasible_max
                        feasible_min = feasible_max
                    time_point = np.random.uniform(feasible_min, feasible_max)
                    feasible_times.append(time_point)
                    current_min = time_point
                x0.extend(feasible_times)

        return np.array(x0)

    # Generate the initial guesses with the updated function
    x0 = generate_feasible_initial_guesses(tv_iphi_const, tv_iphi_seg, tv_iphi_offsett, tv_iphi_offsetl, tv_ophi_vars, tv_ophi_offsett)

    # Calculate starting indices for different variable segments
    start_index_ti = 0  # Start index for time-invariant inputs
    start_index_tvl = start_index_ti + len(ti_bounds)  # Start index for time-variant levels
    start_index_tvt = start_index_tvl + len(tvl_bounds)  # Start index for time-variant times
    start_index_st = start_index_tvt + len(tvt_bounds)  # Start index for sampling times

    # Create a dictionary to track the index ranges for each type of variable
    index_dict = {
        'ti': {},    # Time-invariant inputs
        'swps': {},  # Switching points for levels ('l') and times ('t')
        'st': {}     # Sampling times
    }

    # Time-invariant inputs
    for i, var in enumerate(ti_iphi_vars):
        index_dict['ti'][var] = [start_index_ti + i]

    # Time-variant inputs (levels and times)
    idx_l = start_index_tvl
    idx_t = start_index_tvt
    for i, var in enumerate(tv_iphi_vars):
        seg_count = tv_iphi_seg[i]
        index_dict['swps'][var + 'l'] = list(range(idx_l, idx_l + seg_count - 1))  # 'l' for levels
        index_dict['swps'][var + 't'] = list(range(idx_t, idx_t + seg_count - 2))  # 't' for times
        idx_l += seg_count - 1
        idx_t += seg_count - 2

    # Sampling times
    idx_st = start_index_st
    for i, var in enumerate(tv_ophi_vars):
        seg_count = tv_ophi_seg[i]
        index_dict['st'][var] = list(range(idx_st, idx_st + seg_count - 2))  #  for times
        idx_st += seg_count - 2

    # Apply constraints based on the 'const' settings for time-variant input levels
    for i, const in enumerate(tv_iphi_const):
        var = tv_iphi_vars[i]
        seg_count = tv_iphi_seg[i]
        var_level_indices = index_dict['swps'][var + 'l']
        pairs = []
        if const == 'inc':
            for j in range(seg_count - 2):
                idx1 = var_level_indices[j]
                idx2 = var_level_indices[j + 1]
                pairs.append((idx1, idx2))
        elif const == 'dec':
            for j in range(seg_count - 2):
                idx1 = var_level_indices[j + 1]
                idx2 = var_level_indices[j]
                pairs.append((idx1, idx2))
        index_pairs_levels[var] = pairs

    # Constraints for time segments (tvt) to always increase
    for i, var in enumerate(tv_iphi_vars):
        seg_count = tv_iphi_seg[i]
        var_time_indices = index_dict['swps'][var + 't']
        num_times = len(var_time_indices)
        pairs = []
        for j in range(num_times - 1):
            idx1 = var_time_indices[j]
            idx2 = var_time_indices[j + 1]
            pairs.append((idx1, idx2))  # Ensure time segments are increasing
        index_pairs_times[var] = pairs

    # Constraints for sampling times (st) to always increase
    for i, var in enumerate(tv_ophi_vars):
        seg_count = tv_iphi_seg[i]
        var_time_indices = index_dict['st'][var]
        num_times = len(var_time_indices)
        pairs = []
        for j in range(num_times - 1):
            idx1 = var_time_indices[j]
            idx2 = var_time_indices[j + 1]
            pairs.append((idx1, idx2))  # Ensure time segments are increasing
        index_pairs_times[var] = pairs

    # idx_st = index_dict['st']
    # pairs_st = []
    # for i in range(len(idx_st) - 1):
    #     idx1 = idx_st[i]
    #     idx2 = idx_st[i + 1]
    #     pairs_st.append((idx1, idx2))
    # index_pairs_times['st'] = pairs_st

    return bounds, x0, index_pairs_levels, index_pairs_times, index_dict

def _slicer(x, index_dict):
    """
    Slice the optimization variables into time-invariant inputs, switching points, and sampling times (a index base decomposer of design decision concated list

    Parameters:
    x (list): Current values of the optimization variables.
    index_dict (dict): Dictionary containing index ranges for each type of variable (time-invariant inputs, switching points, and sampling times).

    Returns:
    tuple: A tuple containing:
        - ti (dict): Time-invariant inputs.
        - swps (dict): Switching points for levels and times.
        - St (list): Sampling times.
    """
    # Extract time-invariant inputs
    ti = {}
    for var, idx_list in index_dict['ti'].items():
        # ti[var] = [x[i] for i in idx_list]  # Single value for time-invariant inputs
        ti[var] = x[idx_list[0]]  # Assign the single value directly, not as a list

    # Extract switching points (levels and times)
    swps = {}
    for key, idx_list in index_dict['swps'].items():
        if key.endswith('l'):  # Handle level-based variables (e.g., 'Pl', 'Tl')
            values = [x[i] for i in idx_list]
            values.append(values[-1])  # Append the last value again (repetition)
            swps[key] = values
        elif key.endswith('t'):  # Handle time-based variables (e.g., 'Pt', 'Tt')
            values = [0] + [x[i] for i in idx_list] + [1]  # Add 0 at the start and 1 at the end
            swps[key] = values

    # Extract sampling times (St)
    # St = [0] + [x[i] for i in index_dict['st']] + [1]  # Add 0 at the start and 1 at the end
    St = {}
    for key, idx_list in index_dict['st'].items():
            values = [0] + [x[i] for i in idx_list] + [1]  # Add 0 at the start and 1 at the end
            St[key] = values

    return ti, swps, St

def _reporter(phi, phit, swps, St, performance_metric_value, t, tv_ophi, ti_ophi,
              tv_iphi_vars, tv_iphi_max,
              ti_iphi_vars, ti_iphi_max,
              tf, filename, design_criteria, round, core_number):
    """
    Report the design results by scaling variables, plotting designs, and saving the plots.
    """

    ########################################################################
    # 1) Scale the time-invariant input variables (phi)
    ########################################################################
    for var in ti_iphi_vars:
        if var in phi:
            max_val = ti_iphi_max[ti_iphi_vars.index(var)]
            phi[var] = np.array(phi[var]) * max_val

    ########################################################################
    # 2) Scale the time-variant input variables (phit)
    ########################################################################
    def convert_or_scale_phit(data):
        """
        Recursively process phit variables:
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
                  tv_ophi, ti_ophi, filename, design_criteria, round, core_number)

    ########################################################################
    # 7) Convert `phi` and `phit` back to lists for returning
    ########################################################################
    for var in phi:
        phi[var] = phi[var].tolist()

    def convert_phit_to_list(data):
        """
        Recursively convert NumPy arrays in phit back to lists.
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
#                   tv_ophi, ti_ophi, filename, design_criteria, round, core_number):
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
#     # Subplot 1: Time-variant outputs (already fine, keeping it as is)
#     ax1 = axs[0]
#     ax1.set_title("Time-variant output variables")
#     ax1.set_xlabel('Time (s)')
#
#     ax_list_1 = [ax1]  # List to manage multiple y-axes
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
#             if var_name in swps:
#                 for s_time in swps[var_name]:
#                     ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.5)
#                     ax1.annotate(
#                         f'{s_time:.2f}', xy=(s_time, ax1.get_ylim()[0]),
#                         xytext=(0, 5), textcoords='offset points',
#                         ha='center', va='bottom', rotation=90, color=color
#                     )
#
#     # Plot sampling times on the first subplot
#     if isinstance(St, dict):
#         for var_name, sampling_times in St.items():
#             color = colors[list(tv_ophi[next(iter(tv_ophi))]).index(var_name) % len(colors)]
#             for s_time in sampling_times:
#                 ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.7)
#                 ax1.annotate(
#                     f'{s_time:.2f}', xy=(s_time, ax1.get_ylim()[1]),
#                     xytext=(0, -15), textcoords='offset points',
#                     ha='center', va='bottom', rotation=90, color=color
#                 )
#     else:
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
#     ax1.grid(True)
#
#     # Subplot 2: Time-variant input variables (phit)
#     ax2 = axs[1]
#     ax2.set_title("Time-invariant input variables")
#     ax2.set_xlabel('Time (s)')
#
#     ax_list_2 = [ax2]  # List to manage multiple y-axes
#
#     for model_idx, (design_name, maybe_array_or_dict) in enumerate(phit.items()):
#         for var_idx, (var_name, numeric_value) in enumerate(maybe_array_or_dict.items()):
#             color = colors[var_idx % len(colors)]
#             linestyle = linestyles[model_idx % len(linestyles)]
#             if var_idx == 0:
#                 ax2.plot(
#                     t, numeric_value, color=color, linestyle=linestyle,
#                     label=f'{design_name}-{var_name}'
#                 )
#                 ax2.set_ylabel(f'{var_name} [-]', color=color)
#                 ax2.tick_params(axis='y', labelcolor=color)
#             else:
#                 ax_new = ax2.twinx()
#                 ax_new.spines['right'].set_position(('outward', 60 * (var_idx - 1)))
#                 ax_new.plot(
#                     t, numeric_value, color=color, linestyle=linestyle,
#                     label=f'{design_name}-{var_name}'
#                 )
#                 ax_new.set_ylabel(f'{var_name} [-]', color=color)
#                 ax_new.tick_params(axis='y', labelcolor=color)
#                 ax_list_2.append(ax_new)
#
#     # Plot switching times on the second subplot
#     if isinstance(swps, dict):
#         for var_name, switching_times in swps.items():
#             if var_name.endswith('t'):  # Only consider 'var+t' keys
#                 base_var_name = var_name[:-1]  # Remove the trailing 't'
#
#                 # Find the corresponding color from the nested phit structure
#                 color = None
#                 for model_name, variables in phit.items():
#                     if base_var_name in variables:
#                         color = colors[list(variables.keys()).index(base_var_name) % len(colors)]
#                         break
#
#                 if color is None:
#                     print(f"Warning: Variable '{base_var_name}' in swps not found in phit.")
#                     continue
#
#                 # Plot the switching times with annotations
#                 for s_time in switching_times:
#                     ax2.axvline(x=s_time, color=color, linestyle='--', alpha=0.7)
#                     ax2.annotate(
#                         f'{s_time:.2f}', xy=(s_time, ax2.get_ylim()[0]),
#                         xytext=(0, 5), textcoords='offset points',
#                         ha='center', va='bottom', rotation=90, color=color
#                     )
#
#     # Add legend for the second subplot
#     handles_2, labels_2 = ax2.get_legend_handles_labels()
#     for ax in ax_list_2[1:]:
#         h, l = ax.get_legend_handles_labels()
#         handles_2 += h
#         labels_2 += l
#     if handles_2:
#         ax2.legend(handles_2, labels_2, loc='upper right')
#     ax2.grid(True)
#
#     plt.tight_layout()
#
#     # Save the figure
#     base_path = filename
#     modelling_folder = 'design'
#     full_path = os.path.join(base_path, modelling_folder)
#     os.makedirs(full_path, exist_ok=True)
#     final_filename = os.path.join(full_path, f'{round} (round) by {core_number} core.png')
#     plt.savefig(final_filename, dpi=300)
#     plt.close()

def _plot_designs(phi, phit, swps, St, performance_metric, t,
                  tv_ophi, ti_ophi, filename, design_criteria, round, core_number):
    """
    Plot MBDoE (model-based design of experiments) results with multiple y-axes
    for time-variant outputs and inputs, including color-coded switching and sampling times.
    """

    # Decide if we are in MBDOE_MD or MBDOE_PP
    if design_criteria in ['HR', 'BFF']:
        performance_metric_name = "T-optimality"
    elif design_criteria in ['A', 'D', 'E', 'ME']:
        performance_metric_name = "PP-optimality"
    else:
        performance_metric_name = "Unknown"

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    phi_text = ', '.join([f'{var}: {val:.5e}' for var, val in phi.items()])
    performance_metric_text = (
        f'Round {round} - {performance_metric_name} of {design_criteria}: '
        f'{performance_metric:.20e}'
    )
    fig.suptitle(f'{phi_text}\n{performance_metric_text}\n{performance_metric_name}')

    colors = ['k', 'b', 'r', 'c', 'm', 'y', 'g']
    linestyles = ['-', '--', '-.', ':']

    # ----------------------------------------------------------------------
    # SUBPLOT 1: Time-variant output variables
    # ----------------------------------------------------------------------
    ax1 = axs[0]
    ax1.set_title("Time-variant output variables")
    ax1.set_xlabel('Time (s)')

    ax_list_1 = [ax1]  # manage multiple y-axes

    for solver_idx, (solver_name, solver_tv_ophi) in enumerate(tv_ophi.items()):
        for var_idx, (var_name, y_values) in enumerate(solver_tv_ophi.items()):
            color = colors[var_idx % len(colors)]
            linestyle = linestyles[solver_idx % len(linestyles)]

            if var_idx == 0:
                ax1.plot(
                    t, y_values, color=color, linestyle=linestyle,
                    label=f'{solver_name} - {var_name}'
                )
                ax1.set_ylabel(f'{var_name} [-]', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
            else:
                ax_new = ax1.twinx()
                ax_new.spines['left'].set_position(('outward', 60 * (var_idx - 1)))
                ax_new.plot(
                    t, y_values, color=color, linestyle=linestyle,
                    label=f'{solver_name} - {var_name}'
                )
                ax_new.set_ylabel(f'{var_name} [-]', color=color)
                ax_new.tick_params(axis='y', labelcolor=color)
                ax_list_1.append(ax_new)

            # Plot switching times for the current variable
            if isinstance(swps, dict) and var_name in swps:
                for s_time in swps[var_name]:
                    ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.5)
                    ax1.annotate(
                        f'{s_time:.2f}',
                        xy=(s_time, ax1.get_ylim()[0]),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom', rotation=90, color=color
                    )

    # Plot sampling times on the first subplot
    if isinstance(St, dict):
        # E.g. St = { 'y1': [times], 'y2': [times], ... }
        # We'll guess the color by matching var_name in tv_ophi of the first solver
        # or just do a single color if that is simpler.
        first_solver = next(iter(tv_ophi)) if tv_ophi else None
        if first_solver:
            var_list = list(tv_ophi[first_solver].keys())  # e.g. ["y1", "y2", ...]
        else:
            var_list = []

        for var_name, sampling_times in St.items():
            # Try to find color index by var_name in var_list:
            if var_name in var_list:
                color_idx = var_list.index(var_name) % len(colors)
                color = colors[color_idx]
            else:
                color = 'k'  # fallback color

            for s_time in sampling_times:
                ax1.axvline(x=s_time, color=color, linestyle='--', alpha=0.7)
                ax1.annotate(
                    f'{s_time:.2f}',
                    xy=(s_time, ax1.get_ylim()[1]),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center', va='bottom', rotation=90, color=color
                )
    else:
        # If St is just a list
        for s_time in St:
            ax1.axvline(x=s_time, color='k', linestyle='-', alpha=0.7)

    # Add legend for the first subplot
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    for ax in ax_list_1[1:]:
        h, l = ax.get_legend_handles_labels()
        handles_1 += h
        labels_1 += l
    if handles_1:
        ax1.legend(handles_1, labels_1, loc='upper right')
    # ax1.grid(True)

    # ----------------------------------------------------------------------
    # SUBPLOT 2: "Time-invariant input" or "phit"
    # ----------------------------------------------------------------------
    ax2 = axs[1]
    ax2.set_title("Time-invariant input variables (phit)")
    ax2.set_xlabel('Time (s)')

    ax_list_2 = [ax2]

    # for model_idx, (design_name, maybe_array_or_dict) in enumerate(phit.items()):
    #     # If 'maybe_array_or_dict' is just an array, then we have no .items().
    #     # We'll interpret that as a single variable named design_name
    #     if isinstance(maybe_array_or_dict, np.ndarray):
    #         # Wrap it into a single entry dict
    #         maybe_array_or_dict = {design_name: maybe_array_or_dict}
    #     elif not isinstance(maybe_array_or_dict, dict):
    #         # It's neither array nor dict => skip or log a warning
    #         print(f"Warning: '{design_name}' is neither dict nor array. Skipping.")
    #         continue

    # Now that we have a dictionary, do the usual loop
    for var_idx, (var_name, numeric_value) in enumerate(phit.items()):
        color = colors[var_idx % len(colors)]
        linestyle = linestyles[var_idx % len(linestyles)]

        if var_idx == 0:
            ax2.plot(
                t, numeric_value, color=color, linestyle=linestyle,
                label=f'{var_name}', linewidth=2.5  # Line thickness
            )
            ax2.set_ylabel(f'{var_name} [-]', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            ax_new = ax2.twinx()
            ax_new.spines['right'].set_position(('outward', 60 * (var_idx - 1)))
            ax_new.plot(
                t, numeric_value, color=color, linestyle=linestyle,
                label=f'{var_name}', linewidth=2.5  # Line thickness
            )
            ax_new.set_ylabel(f'{var_name} [-]', color=color)
            ax_new.tick_params(axis='y', labelcolor=color)
            ax_list_2.append(ax_new)

    # Plot switching times on the second subplot
    if isinstance(swps, dict):
        for var_name, switching_times in swps.items():
            # Typically we only plot "time" switching points here
            if not var_name.endswith('t'):
                continue
            base_var_name = var_name[:-1]  # Remove trailing 't'
            # Find a color from phit if possible
            color = 'k'
            if isinstance(phit, dict):
                # For each design, check if base_var_name is in its dictionary
                for design_name, subdict in phit.items():
                    if isinstance(subdict, dict) and (base_var_name in subdict):
                        idx = list(subdict.keys()).index(base_var_name)
                        color = colors[idx % len(colors)]
                        break

            # Plot the switching lines
            for s_time in switching_times:
                ax2.axvline(x=s_time, color=color, linestyle='--', alpha=0.2)
                ax2.annotate(
                    f'{s_time:.2f}',
                    xy=(s_time, ax2.get_ylim()[0]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', rotation=90, color=color
                )

    # Add legend for the second subplot
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    for ax in ax_list_2[1:]:
        h, l = ax.get_legend_handles_labels()
        handles_2 += h
        labels_2 += l
    if handles_2:
        ax2.legend(handles_2, labels_2, loc='upper right')
    # ax2.grid(True)

    plt.tight_layout()

    # Save the figure
    base_path = filename
    modelling_folder = 'design'
    full_path = os.path.join(base_path, modelling_folder)
    os.makedirs(full_path, exist_ok=True)
    final_filename = os.path.join(full_path, f'{round} (round) by {core_number} core.png')
    plt.savefig(final_filename, dpi=300)
    plt.close()


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


def build_var_groups(tv_iphi_vars, tv_iphi_offsetl, tv_iphi_offsett, tv_iphi_const,
                     tv_ophi_vars, tv_ophi_offsett_ophi):
    """
    Same style as in des-md to keep the code structure consistent.
    """
    return {
        "tv_iphi": {
            "vars": tv_iphi_vars,
            "offset_l": tv_iphi_offsetl,
            "offset_t": tv_iphi_offsett,
            "const": tv_iphi_const
        },
        "tv_ophi": {
            "vars": tv_ophi_vars,
            "offset_t": tv_ophi_offsett_ophi,
        }
    }


def get_var_info(var, var_groups):
    """
    Shared helper, same as in des-md.
    """
    for group_key, group_data in var_groups.items():
        if var in group_data["vars"]:
            i = group_data["vars"].index(var)
            return group_key, i
    raise ValueError(f"Variable '{var}' not found in any var list!")


def build_linear_constraints(x_len,
                             index_pairs_levels, index_pairs_times,
                             var_groups):
    """
    Same approach as in des-md for building linear constraints.
    """
    A_level = []
    lb_level = []
    ub_level = []

    for var, pairs in index_pairs_levels.items():
        group_key, i = get_var_info(var, var_groups)
        group_data = var_groups[group_key]
        const_type = group_data["const"][i] if "const" in group_data else "inc"

        if "offset_l" not in group_data:
            raise ValueError(
                f"Group '{group_key}' missing 'offset_l' for var '{var}'."
            )
        offset_l = group_data["offset_l"][i]

        for (idx1, idx2) in pairs:
            row = np.zeros(x_len)
            if const_type in ("inc", "dec"):
                row[idx2] = 1
                row[idx1] = -1
            else:
                raise ValueError(f"Unknown constraint type: {const_type}")
            A_level.append(row)
            lb_level.append(offset_l)
            ub_level.append(np.inf)

    A_time = []
    lb_time = []
    ub_time = []

    for var, pairs in index_pairs_times.items():
        group_key, i = get_var_info(var, var_groups)
        group_data = var_groups[group_key]

        if "offset_t" not in group_data:
            raise ValueError(
                f"Group '{group_key}' missing 'offset_t' for var '{var}'."
            )
        offset_t = group_data["offset_t"][i]

        for (idx1, idx2) in pairs:
            row = np.zeros(x_len)
            row[idx2] = 1
            row[idx1] = -1
            A_time.append(row)
            lb_time.append(offset_t)
            ub_time.append(np.inf)

    constraints_list = []
    if A_level:
        A_level = np.array(A_level)
        lb_level = np.array(lb_level)
        ub_level = np.array([np.inf]*len(lb_level))  # or a parallel list
        constraints_list.append(LinearConstraint(A_level, lb_level, ub_level))

    if A_time:
        A_time = np.array(A_time)
        lb_time = np.array(lb_time)
        ub_time = np.array([np.inf]*len(lb_time))
        constraints_list.append(LinearConstraint(A_time, lb_time, ub_time))

    return (A_level, lb_level, A_time, lb_time, constraints_list)


def constraint_violation(x, A_level, lb_level, A_time, lb_time):
    """
    A quick violation check, re-used from des-md.
    """
    vio_sum = 0.0
    x_arr = np.array(x, dtype=float)

    if A_level is not None and len(A_level) > 0:
        AxL = A_level.dot(x_arr)
        for (lb_val, ax_val) in zip(lb_level, AxL):
            diff = lb_val - ax_val
            if diff > 0:
                vio_sum += diff

    if A_time is not None and len(A_time) > 0:
        AxT = A_time.dot(x_arr)
        for (lb_val, ax_val) in zip(lb_time, AxT):
            diff = lb_val - ax_val
            if diff > 0:
                vio_sum += diff

    return vio_sum

def penalized_objective(
    x,
    obj_args,        # dict of everything needed by the objective function
    constraint_args, # dict of everything needed by the constraint violation
    penalty_weight=1e4
):
    """
    Combine the objective (md_objective_function) with a penalty for
    constraint violation, for use in differential_evolution.
    """
    # Unpack arguments for objective
    objective_fun = obj_args['objective_fun']

    # Evaluate base objective
    base_cost = objective_fun(x)

    # Evaluate constraint violation
    violation_fun = constraint_args['violation_fun']
    vio = violation_fun(x)

    # Combine
    return base_cost + penalty_weight * vio

