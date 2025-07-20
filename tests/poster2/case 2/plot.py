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
    'Sc2': [0.33359123890949904, 0.17045230802742248, 0.619022445962234, 0.23039874051767348, 4.527266978818574],
    'Sc1': [0.4174235633972362, None, 0.9738718861343159, 0.27867738913713375, None],
}

variances = {
    'Sc2': [8.52520252584895e-05, 0.0012811272202549197, 0.00035600509797354443, 7.214796340769345e-06, 0.0021090578583854414],
    'Sc1': [0.0004975041559578311, None, 0.003456187334654482, 5.344097620939989e-05, None],
}

colors = {'Sc1': 'red', 'Sc2': 'blue'}

estimation_info = {
    'Sc2': {
        r'$\theta_1$': ('0.33359123890949904', '17.915241436871565'),
        r'$\theta_2$': ('0.17045230802742248', '2.361386839026271'),
        r'$\theta_3$': ('0.619022445962234', '16.2681687990876'),
        r'$\theta_4$': ('0.23039874051767348', '42.53323778200584'),
        r'$\theta_5$': ('4.527266978818574', '48.88236851147831')
    },
    'Sc1': {
        r'$\theta_1$': ('0.4174235633972362', '8.606785551672685'),
        r'$\theta_2$': ('Fixed to 0.188', '-'),
        r'$\theta_3$': ('0.9738718861343159', '7.974252259528415'),
        r'$\theta_4$': ('0.27867738913713375', '18.286394264019023'),
        r'$\theta_5$': ('Fixed to 5.563', '-')
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

    x_min = min(x_min, true_params[i])
    x_max = max(x_max, true_params[i])
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
