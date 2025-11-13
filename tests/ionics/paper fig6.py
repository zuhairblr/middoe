import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
from matplotlib.transforms import Bbox


# Custom legend handler to draw line, marker, and vertical error bar with caps
class HandlerErrorBarWithCap(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Calculate center for positioning
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2

        # Line: horizontal centered
        line = mlines.Line2D(
            [xdescent, xdescent + width], [center_y, center_y],
            color=orig_handle['line'].get_color(),
            linewidth=orig_handle['line'].get_linewidth(),
            transform=trans)

        # Marker at center
        marker = mlines.Line2D(
            [center_x], [center_y],
            marker=orig_handle['marker'].get_marker(),
            markersize=orig_handle['marker'].get_markersize(),
            markeredgewidth=orig_handle['marker'].get_markeredgewidth(),
            markeredgecolor=orig_handle['marker'].get_markeredgecolor(),
            markerfacecolor=orig_handle['marker'].get_markerfacecolor(),
            linestyle='None',
            color=orig_handle['marker'].get_color(),
            transform=trans)

        # Error bar vertical line length
        err_len = height * 0.6

        # Vertical error bar centered and caps length
        err_line = mlines.Line2D(
            [center_x, center_x], [center_y - err_len / 2, center_y + err_len / 2],
            color=orig_handle['line'].get_color(),
            linewidth=orig_handle['line'].get_linewidth(),
            transform=trans)

        cap_width = width * 0.1  # Horizontal caps width

        # Caps top and bottom horizontal lines
        cap_top = mlines.Line2D(
            [center_x - cap_width / 2, center_x + cap_width / 2],
            [center_y + err_len / 2, center_y + err_len / 2],
            color=orig_handle['line'].get_color(),
            linewidth=orig_handle['line'].get_linewidth(),
            transform=trans)

        cap_bottom = mlines.Line2D(
            [center_x - cap_width / 2, center_x + cap_width / 2],
            [center_y - err_len / 2, center_y - err_len / 2],
            color=orig_handle['line'].get_color(),
            linewidth=orig_handle['line'].get_linewidth(),
            transform=trans)

        return [line, marker, err_line, cap_top, cap_bottom]


# Setup font
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'

# Input data
data = {
    "Time (min)": [0, 10, 20, 40, 80, 120, 180],
    "25°C": [0, 0.2331513961, 0.3427964018, 0.4947052308, 0.703131352, 0.7247250931, 0.8036112646],
    "25°C (2)": [0, 0.2274965098, 0.3148759546, np.nan, 0.701356515, 0.7449237576, 0.7547319555],
    "45°C": [0, 0.2134312931, 0.3373302717, 0.4677618033, 0.688, 0.7720860675, 0.7987906507],
    "65°C": [0, 0.2298323145, 0.3121567119, 0.5414545089, 0.714, 0.7471155162, 0.8138997541],
    "85°C": [0, 0.2673016863, 0.3365458044, 0.4884239925, 0.676, 0.7271083331, 0.7480283813]
}

df = pd.DataFrame(data)

# Calculate mean and std for 25°C replicates
df['25°C_mean'] = df[['25°C', '25°C (2)']].mean(axis=1, skipna=True)
df['25°C_std'] = df[['25°C', '25°C (2)']].std(axis=1, skipna=True, ddof=0)

# 80 min replicate sets for other temps
temps = ['45°C', '65°C', '85°C']
replicates_80 = {
    '45°C': [0.7098781588, 0.688],
    '65°C': [0.6819594359, 0.714],
    '85°C': [0.7148681496, 0.676]
}
means_80 = {k: np.mean(v) for k, v in replicates_80.items()}
stds_80 = {k: np.std(v, ddof=0) for k, v in replicates_80.items()}

for temp in temps:
    idx_80 = df.index[df['Time (min)'] == 80][0]
    df.loc[idx_80, temp] = means_80[temp]
    df[temp + '_std'] = 0.0
    df.loc[idx_80, temp + '_std'] = stds_80[temp]

# Colors and markers
temp_styles = {
    '25°C': {'marker': 'o', 'color': 'black'},
    '45°C': {'marker': 's', 'color': '#0072B2'},
    '65°C': {'marker': '^', 'color': '#009E73'},
    '85°C': {'marker': 'D', 'color': '#D55E00'},
}

plt.figure(figsize=(4.5, 4.5), dpi=300)
plt.subplots_adjust(left=0.13, bottom=0.12)

# Plot lines with error bars
line_objects = {}
for temp in ['25°C'] + temps:
    if temp == '25°C':
        y = df['25°C_mean']
        yerr = df['25°C_std']
    else:
        y = df[temp]
        yerr = df[temp + '_std']
    line, = plt.plot(df['Time (min)'], y,
                     marker=temp_styles[temp]['marker'],
                     color=temp_styles[temp]['color'],
                     markersize=5,
                     markeredgecolor='white',
                     markeredgewidth=0.6,
                     linewidth=1.2,
                     label=temp)
    plt.errorbar(df['Time (min)'] if temp == '25°C' else [80],
                 y if temp == '25°C' else [y.loc[df['Time (min)'] == 80].values[0]],
                 yerr=yerr if temp == '25°C' else [yerr.loc[df['Time (min)'] == 80].values[0]],
                 fmt='none',
                 ecolor=temp_styles[temp]['color'],
                 capsize=4,
                 zorder=5)
    line_objects[temp] = {
        'line': line,
        'marker': mlines.Line2D([], [], color=temp_styles[temp]['color'], marker=temp_styles[temp]['marker'],
                                linestyle='None', markersize=7, markeredgecolor='white', markeredgewidth=0.6)
    }

# Create custom legend handles dictionary
legend_handles = []

for temp in ['25°C'] + temps:
    orig_handles = line_objects[temp]
    legend_handles.append(orig_handles)


# Create proxy artists combining line and marker for the legend, along with error bar [custom handler]
class CustomHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2

        # Horizontal line across the legend box width
        line = mlines.Line2D([xdescent, xdescent + width], [center_y, center_y],
                             color=orig_handle['line'].get_color(),
                             linewidth=orig_handle['line'].get_linewidth(),
                             transform=trans)

        # Marker at center
        marker = mlines.Line2D([center_x], [center_y],
                               marker=orig_handle['marker'].get_marker(),
                               color=orig_handle['marker'].get_color(),
                               markerfacecolor=orig_handle['marker'].get_markerfacecolor(),
                               markeredgecolor=orig_handle['marker'].get_markeredgecolor(),
                               markeredgewidth=orig_handle['marker'].get_markeredgewidth(),
                               markersize=orig_handle['marker'].get_markersize(),
                               linestyle='None',
                               transform=trans)

        # Vertical error bar length
        err_len = height * 1.2
        cap_width = width * 0.15

        # Vertical line of error bar
        err_line = mlines.Line2D([center_x, center_x], [center_y - err_len / 2, center_y + err_len / 2],
                                 color=orig_handle['line'].get_color(),
                                 linewidth=orig_handle['line'].get_linewidth(),
                                 transform=trans)

        # Caps - top and bottom horizontal lines
        cap_top = mlines.Line2D([center_x - cap_width / 2, center_x + cap_width / 2],
                                [center_y + err_len / 2, center_y + err_len / 2],
                                color=orig_handle['line'].get_color(),
                                linewidth=orig_handle['line'].get_linewidth(),
                                transform=trans)
        cap_bottom = mlines.Line2D([center_x - cap_width / 2, center_x + cap_width / 2],
                                   [center_y - err_len / 2, center_y - err_len / 2],
                                   color=orig_handle['line'].get_color(),
                                   linewidth=orig_handle['line'].get_linewidth(),
                                   transform=trans)

        return [line, marker, err_line, cap_top, cap_bottom]


# Generate the legend using the custom handler
plt.legend(legend_handles,
           ['25°C', '45°C', '65°C', '85°C'],
           handler_map={dict: CustomHandler()},
           frameon=False,
           fontsize=14)

plt.xticks([0, 10, 20, 40, 80, 120, 180], fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel(r'Time ($t$) [min]', fontsize=15)
plt.ylabel(r'Carbonation Efficiency ($X$)', fontsize=15)
plt.xlim(-2, 182)
plt.ylim(-0.01, 0.85)

ax = plt.gca()
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(1)

plt.savefig(r'C:\Users\bolosey94542\Desktop\to transfer\CVAL\reports\papers\rcf wet\data\carbonation_plot3.svg',
            format='svg')

plt.show()
