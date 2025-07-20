import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'

# Define data
parameters = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$']
parameters1 = [r'$\hat{\theta}_1$', r'$\hat{\theta}_2$', r'$\hat{\theta}_3$', r'$\hat{\theta}_4$', r'$\hat{\theta}_5$']
scenarios = ['Sc1', 'Sc2']
true_params = [0.31, 0.11, 0.65, 0.25, 5.0]

means = {
    'Sc2': [0.3206, 0.1307, 0.4979, 0.2354, 4.0673],
    'Sc1': [0.399, None, 0.9749, 0.2779, None],
}

variances = {
    'Sc2': [0.003038, 0.033950, 0.001944, 2.46e-05, 0.143386],
    'Sc1': [0.00050, None, 0.00346, 0.00005, None],
}

colors = {'Sc1': 'red', 'Sc2': 'blue'}

estimation_info = {
    'Sc2': {
        r'$\theta_1$': ('0.3205', '2.779'),
        r'$\theta_2$': ('0.1307', '0.339'),
        r'$\theta_3$': ('0.4979', '5.396'),
        r'$\theta_4$': ('0.235', '22.692'),
        r'$\theta_5$': ('4.067', '5.132')
    },
    'Sc1': {
        r'$\theta_1$': ('0.399', '8.606'),
        r'$\theta_2$': ('Fixed to 0.188', '-'),
        r'$\theta_3$': ('0.658', '7.974'),
        r'$\theta_4$': ('0.247', '18.286'),
        r'$\theta_5$': ('Fixed to 5.027', '-')
    }
}


def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


fig, axes = plt.subplots(5, 1, figsize=(5, 13), sharex=False, dpi=300)

for i, (param, param1) in enumerate(zip(parameters, parameters1)):
    ax = axes[i]

    # Dynamically determine x range
    x_min, x_max = float('inf'), float('-inf')
    max_y = 0

    for scenario in scenarios:
        mean = means[scenario][i]
        var = variances[scenario][i]
        if mean is not None and var is not None and var > 0:
            std = np.sqrt(var)
            x_min = min(x_min, mean - 7 * std)
            x_max = max(x_max, mean + 7 * std)

    margin = 0.05 * (x_max - x_min)
    x = np.linspace(x_min - margin, x_max + margin, 10000)

    for scenario in scenarios:
        mean = means[scenario][i]
        var = variances[scenario][i]

        if mean is not None and var is not None and var > 0:
            y = gaussian(x, mean, var)
            max_y = max(max_y, max(y))

            est, tval = estimation_info[scenario][param]
            formatted_est = f"{float(est):.2f}" if est.replace('.', '', 1).isdigit() else est
            formatted_tval = f"{float(tval):.2f}" if tval.replace('.', '', 1).isdigit() else tval

            ax.plot(x, y, label=f"{scenario}: {param1}={formatted_est}, t-value: {formatted_tval}",
                    color=colors[scenario], linewidth=2)

    ax.axvline(true_params[i], color='green', linestyle='--', linewidth=2,
               label=f'{param} = {true_params[i]:.2f}')

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(-0.05 * max_y, 1.2 * max_y)
    ax.set_ylabel('Density')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.6)

axes[-1].set_xlabel('Parameter Value')
plt.tight_layout()
plt.subplots_adjust(top=0.99)
plt.savefig("C:/Users/Tadmin/Documents/parameter_estimates_auto_limits.svg", format='svg', dpi=300)
plt.show()
