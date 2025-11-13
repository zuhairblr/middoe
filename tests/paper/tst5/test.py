import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 8.314  # J/(mol*K)

# Only 205°C for both systems
T_value = 205  # Celsius
T_values_RCF = np.array([T_value])
T_values_RCP = np.array([T_value])

# RCF parameters
P_RCF = 0.15
t_RCF = np.linspace(0, 180, 200)
A_RCF = 1.3414E-06
Ea_RCF = 20252
theta_RCF = 5.9263

# RCP parameters
P_RCP = 1
t_RCP = np.linspace(0, 60, 100)
A_RCP = 1.4890E-08
Ea_RCP = 24191
theta_RCP = 4.1206

# Calculations for RCF
De_values_RCF = A_RCF * np.exp(-Ea_RCF / (R * (T_values_RCF + 273.15)))
CCO2_values_RCF = P_RCF * 1000 * 0.035 * np.exp(2400 * (1 / (T_values_RCF + 273.15) - 1 / 298.15))
ka_values_RCF = (4 * CCO2_values_RCF * De_values_RCF) / (27090 * (0.0000698**2))
phi_values_RCF = np.array([np.exp(-ka * theta_RCF * t_RCF * 60) for ka in ka_values_RCF])

# Calculations for RCP
De_values_RCP = A_RCP * np.exp(-Ea_RCP / (R * (T_values_RCP + 273.15)))
CCO2_values_RCP = P_RCP * 1000 * 0.035 * np.exp(2400 * (1 / (T_values_RCP + 273.15) - 1 / 298.15))
ka_values_RCP = (8 * CCO2_values_RCP * De_values_RCP) / (27090 * (0.0000084**2))
phi_values_RCP = np.array([np.exp(-ka * theta_RCP * t_RCP * 60) for ka in ka_values_RCP])

# Time-dependent diffusivity
De_time_RCF = np.array([De_values_RCF[i] * phi_values_RCF[i] for i in range(len(T_values_RCF))])
De_time_RCP = np.array([De_values_RCP[i] * phi_values_RCP[i] for i in range(len(T_values_RCP))])

# Font
import matplotlib as mpl

mpl.rcParams["font.family"] = "Calibri"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Calibri'
mpl.rcParams['mathtext.it'] = 'Calibri:italic'


# --- Plot De(t) ---
fig, ax1 = plt.subplots(figsize=(4.5, 4.5))
plt.subplots_adjust(bottom=0.15, right=0.88)

# RCF on left axis (black)
ax1.set_ylabel("Diffusion Coefficient ($D$) [m²/s] - RCFs", color="black", fontsize=12)
ax1.tick_params(axis="y", labelcolor="black", labelsize=10)
ax1.tick_params(axis="x", labelsize=9)
ax1.plot(t_RCF, De_time_RCF[0], linestyle='-', label=f"RCFs - T = 205 °C", color='black')
ax1.fill_between(t_RCF, De_time_RCF[0]*0.95, De_time_RCF[0]*1.05, alpha=0.3, color='black')

# RCP on right axis (red)
ax2 = ax1.twinx()
ax2.set_ylabel("Diffusion Coefficient ($D$) [m²/s] - RCP", color="red", fontsize=12)
ax2.tick_params(axis="y", labelcolor="red", color="red", labelsize=10)
ax2.spines["right"].set_color("red")
ax2.plot(t_RCP, De_time_RCP[0], linestyle='-', label=f"RCP - T = 205 °C", color='red')
ax2.fill_between(t_RCP, De_time_RCP[0]*0.95, De_time_RCP[0]*1.05, alpha=0.3, color='red')

ax1.set_xlabel("Time ($t$) [min]", fontsize=12)
ax1.set_xlim([-2, 182])
ax1.set_xticks(np.arange(0, 181, 30))
ax1.set_ylim(bottom=-1e-11)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_lines = lines1 + lines2
all_labels = labels1 + labels2
ax2.legend(all_lines, all_labels, loc="upper right", frameon=False, fontsize=10)
ax2.set_ylim(bottom=-1e-14)

output_De = r"C:\Users\bolosey94542\Desktop\to transfer\CVAL\reports\papers\rcf wet\interp\interp_Detest.svg"
plt.savefig(output_De, format="svg", dpi=300)
plt.show()
