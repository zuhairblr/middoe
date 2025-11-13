import numpy as np
import pandas as pd

# Constants
R = 8.314  # J/(mol*K)

# RCF parameters
T_values_RCF = np.array([25, 45, 65, 85])  # temperatures in °C
P_RCF = 0.15
A_RCF = 1.3414E-06
Ea_RCF = 20252
theta_RCF = 5.9263

# Time of interest: 180 mins
t_180 = 180

# Calculate diffusion coefficient (De) for each temperature
De_values_RCF = A_RCF * np.exp(-Ea_RCF / (R * (T_values_RCF +273.15)))

# Calculate CO2 solubility (CCO2) for each temperature (mmol/L)
CCO2_values_RCF = P_RCF * 1000 * 0.035 * np.exp(2400 * (1 / (T_values_RCF + 273.15) - 1 / 298.15))

# Calculate rate constant (ka) for each temperature
ka_values_RCF = (4 * CCO2_values_RCF * De_values_RCF) / (27090 * (0.0000698**2))

# Calculate phi value at 180 minutes (converted to seconds)
phi_at_180 = np.exp(-ka_values_RCF * theta_RCF * t_180 * 60)

# Calculate time-dependent diffusion coefficient De(t) = De * phi
De_time_180 = De_values_RCF * phi_at_180

# Create pandas DataFrame for reporting
report_dict = {
    "Temperature (°C)": T_values_RCF,
    "Diffusion Coefficient (De) [m^2/s]": De_time_180,
    "CO2 Solubility (CCO2) [mmol/L]": CCO2_values_RCF
}

df_report = pd.DataFrame(report_dict)

# Save to CSV
df_report.to_csv("RCF_diffusivity_CO2_solubility_180min.csv", index=False)

print("File saved: RCF_diffusivity_CO2_solubility_180min.csv")



