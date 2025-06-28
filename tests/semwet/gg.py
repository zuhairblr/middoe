import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'

# Define data for means, variances, and true parameter values
parameters = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$']
parameters1 = [r'$\hat{\theta}_1$', r'$\hat{\theta}_2$', r'$\hat{\theta}_3$', r'$\hat{\theta}_4$', r'$\hat{\theta}_5$']

scenarios = ['Sc1', 'Sc2']
true_params = [0.31, 0.11, 0.65, 0.25, 5.0]

means = {
    'Sc2': [0.303, 0.109, 0.644, 0.248, 5.026],
    'Sc1': [0.308, None, 0.658, 0.247, None],
}

variances = {
    'Sc2': [0.026, 0.0532, 0.023, 0.004, 0.001],
    'Sc1': [0.114, None, 0.124, 0.019, None],
}

colors = {'Sc2': 'red',
          'Sc1': 'blue'}

# Additional information for estimations and t-values
estimation_info = {
    'Sc2': {
        r'$\theta_1$': ('0.303', '3.020'),
        r'$\theta_2$': ('0.109', '2.120'),
        r'$\theta_3$': ('0.644', '3.214'),
        r'$\theta_4$': ('0.248', '7.950'),
        r'$\theta_5$': ('5.026', '15.155')
    },
    'Sc1': {
        r'$\theta_1$': ('0.308', '1.426'),
        r'$\theta_2$': ('Fixed to 0.109', '-'),
        r'$\theta_3$': ('0.658', '1.365'),
        r'$\theta_4$': ('0.247', '3.508'),
        r'$\theta_5$': ('Fixed to 5.027', '-')
    }
}

# Custom xlim and ylim for each subplot
xlims = {
    r'$\theta_1$': (-1, 1.5),
    r'$\theta_2$': (-0.7, 1),
    r'$\theta_3$': (-0.5, 2),
    r'$\theta_4$': (-0.2, 0.7),
    r'$\theta_5$': (4.9, 5.2)
}

ylims = {
    r'$\theta_1$': (-0.2, 2.6),
    r'$\theta_2$': (-0.2, 1.8),
    r'$\theta_3$': (-0.2, 2.8),
    r'$\theta_4$': (-0.5, 7),
    r'$\theta_5$': (-0.5, 13)
}

# Gaussian function to generate the density
def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


# Create the subplots
fig, axes = plt.subplots(5, 1, figsize=(5, 14), sharex=False, dpi=300)

for i, (param, param1) in enumerate(zip(parameters, parameters1)):
    ax = axes[i]
    # ax.set_title(f"{param}", fontsize=12)

    # # Define x-axis range centered around the true parameter value
    # x = np.linspace(true_params[i] - 1.5, true_params[i] + 1.5, 5000)
    # Dynamically compute x range based on widest std-dev across scenarios
    x_min = float('inf')
    x_max = float('-inf')

    for scenario in scenarios:
        mean = means[scenario][i]
        var = variances[scenario][i]
        if mean is not None and var is not None and var > 0:
            std = np.sqrt(var)
            x_min = min(x_min, mean - 15 * std)
            x_max = max(x_max, mean + 15 * std)

    # Apply fixed subplot margin if you want (optional)
    margin = 0.05 * (x_max - x_min)
    x = np.linspace(x_min - margin, x_max + margin, 50000)

    # Plot each scenario's distribution
    for scenario in scenarios:
        mean = means[scenario][i]
        var = variances[scenario][i]

        if mean is not None and var is not None and var > 0:
            y = gaussian(x, mean, var)

            est, tval = estimation_info[scenario][param]
            formatted_est = f"{float(est):.2f}" if est.replace('.', '',
                                                               1).isdigit() else est  # Keep 'Fixed to...' as is
            formatted_tval = f"{float(tval):.2f}" if tval.replace('.', '',
                                                                  1).isdigit() else tval  # Handle '-' gracefully
            ax.plot(x, y, label=f"{scenario}:{param1}={formatted_est},t-value: {formatted_tval}", color=colors[scenario],
                    linewidth=2)

    # Add a vertical dashed line for the true parameter value
    ax.axvline(true_params[i], color='green', linestyle='--', linewidth=2,
               label=f'{param} ={true_params[i]:.2f}')

    # Set x and y limits
    if param in xlims:
        left, right = xlims[param]
        ax.set_xlim(left, right + 0.7 * (right - left))
    if param in ylims:
        ax.set_ylim(ylims[param])

    ax.set_ylabel('Density')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.6)

# Label the x-axis for the last subplot
axes[-1].set_xlabel('Parameter Value')

plt.tight_layout()
plt.subplots_adjust(top=0.99)

# Save the figure as a TIFF file
plt.savefig("C:/Users/Tadmin/Documents/parameter_estimates_with_custom_limits2.svg", format='svg', dpi=300)

plt.show()