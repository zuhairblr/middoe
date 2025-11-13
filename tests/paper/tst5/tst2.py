import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Constants
R = 8.314  # J/(mol*K)
# RCF parameters
T_values_RCF = np.array([25, 45, 65, 85])
P_RCF = 0.15
t_RCF = np.linspace(0, 180, 200)
A_RCF = 1.3414E-06
Ea_RCF = 20252
theta_RCF = 5.9263
# RCP parameters
T_values_RCP = np.array([25, 40, 80])
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
# New line styles for distinction by temperature
line_styles_RCF = ['-', '--', '-.', ':']
line_styles_RCP = ['-', '--', '-.']
# Font
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# --- Plot φ ---
plt.figure(figsize=(5, 5))
for i, T in enumerate(T_values_RCF):
    plt.plot(t_RCF, phi_values_RCF[i], linestyle=line_styles_RCF[i % len(line_styles_RCF)],
             label=f"RCFs - T = {T} °C", color='black')
    plt.fill_between(t_RCF, phi_values_RCF[i]*0.95, phi_values_RCF[i]*1.05, alpha=0.3, color='black')
for i, T in enumerate(T_values_RCP):
    plt.plot(t_RCP, phi_values_RCP[i], linestyle=line_styles_RCP[i % len(line_styles_RCP)],
             label=f"RCP - T = {T} °C", color='red')
    plt.fill_between(t_RCP, phi_values_RCP[i]*0.95, phi_values_RCP[i]*1.05, alpha=0.3, color='red')
plt.xlabel("Time [minutes]\n$\it{a}$", fontsize=9)
plt.ylabel("φ", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(frameon=False, fontsize=7)
plt.tight_layout()
# output_phi = r"C:\Users\bolosey94542\Desktop\to transfer\CVAL\reports\papers\rcf wet\interp\interp_phi.svg"
# plt.savefig(output_phi, format="svg", dpi=300)
# plt.show()
# --- Plot De(t) ---
fig, ax1 = plt.subplots(figsize=(4.5, 4.5))
plt.subplots_adjust(bottom=0.15, right=0.88)
# RCF on left axis (black)
ax1.set_ylabel("Diffusion Coefficient ($D$) [m²/s] - RCFs", color="black", fontsize=12)
ax1.tick_params(axis="y", labelcolor="black", labelsize=10)
ax1.tick_params(axis="x", labelsize=9)
for i, T in enumerate(T_values_RCF):
    ax1.plot(t_RCF, De_time_RCF[i], linestyle=line_styles_RCF[i % len(line_styles_RCF)],
             label=f"RCFs - T = {T} °C", color='black')
    ax1.fill_between(t_RCF, De_time_RCF[i]*0.95, De_time_RCF[i]*1.05, alpha=0.3, color='black')
# RCP on right axis (red)
ax2 = ax1.twinx()
ax2.set_ylabel("Diffusion Coefficient ($D$) [m²/s] - RCP", color="red", fontsize=12)
ax2.tick_params(axis="y", labelcolor="red", color="red", labelsize=10)
ax2.spines["right"].set_color("red")
for i, T in enumerate(T_values_RCP):
    ax2.plot(t_RCP, De_time_RCP[i], linestyle=line_styles_RCP[i % len(line_styles_RCP)],
             label=f"RCP - T = {T} °C", color='red')
    ax2.fill_between(t_RCP, De_time_RCP[i]*0.95, De_time_RCP[i]*1.05, alpha=0.3, color='red')
ax1.set_xlabel("Time ($t$) [min]\n$\it{a}$", fontsize=12)
ax1.set_xlabel("Time ($t$) [min]", fontsize=12)
ax1.set_xlim([-2, 182])
ax1.set_xticks(np.arange(0, 181, 30))  # Add this line for better x-axis ticks
ax1.set_ylim(bottom=-0.00000000001)
# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_lines = lines1 + lines2
all_labels = labels1 + labels2
ax2.legend(all_lines[:7], all_labels[:7], loc="upper right", frameon=False, fontsize=10)
ax2.set_ylim(bottom=-0.00000000000001)
# plt.tight_layout()
output_De = r"C:\Users\bolosey94542\Desktop\to transfer\CVAL\reports\papers\rcf wet\interp\interp_Detest.svg"
plt.savefig(output_De, format="svg", dpi=300)
plt.show()

import pandas as pd

# Prepare RCF full data table
rcf_dict = {
    'Time (min)': np.tile(t_RCF, len(T_values_RCF)),
    'Temperature (°C)': np.repeat(T_values_RCF, len(t_RCF)),
    'Phi': np.concatenate(phi_values_RCF),
    'De (m2/s)': np.concatenate(De_time_RCF)
}
rcf_df = pd.DataFrame(rcf_dict)
rcf_df.to_csv('RCF_full_data.csv', index=False)

# Prepare RCP full data table
rcp_dict = {
    'Time (min)': np.tile(t_RCP, len(T_values_RCP)),
    'Temperature (°C)': np.repeat(T_values_RCP, len(t_RCP)),
    'Phi': np.concatenate(phi_values_RCP),
    'De (m2/s)': np.concatenate(De_time_RCP)
}
rcp_df = pd.DataFrame(rcp_dict)
rcp_df.to_csv('RCP_full_data.csv', index=False)

print("RCF and RCP full data saved to 'RCF_full_data.csv' and 'RCP_full_data.csv' respectively.")

