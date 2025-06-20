"""
Carbonation Model with CO2 Injection — Full Code with Inline and Structural Comments
Phenomena Layers:
(1) Gas-Liquid Mass Transfer
(2) Aqueous Ionic Reactions
(3) Solid Dissolution and Transport
(4) CaCO3 Precipitation
(5) pH and Ionic Strength Calculation
(6) Diffusion-Controlled CSH Carbonation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ---------------------------------------------
# Function: carbonation_model
# ---------------------------------------------
def carbonation_model(t, y0, tii, tvi, theta):
    """
    Mechanistic carbonation model simulating dissolution, precipitation,
    ionic equilibria, and CO2 transfer in a cementitious aqueous system.

    Parameters
    ----------
    t : ndarray
        Time vector [s]
    y0 : list or None
        Initial concentrations and states (see below for units)
    tii : dict
        Time-invariant inputs (e.g., 'n0': initial Ca(OH)₂ [mol/L])
    tvi : dict
        Time-variant inputs (e.g., 'P' [bar], 'T' [K])
    theta : list
        Model parameters (see unpacking below for units)

    Returns
    -------
    dict
        Simulation results with concentrations [mol/L], pH, ionic strength [mol/L],
        reaction rates [mol/(L·s)], and activity coefficients [–]
    """

    # === Debye–Hückel constants and ion sizes ===
    A, B = 0.037, 1.04e8                     # Debye–Hückel constants [dimensionless]
    a_Ca, a_OH, a_H, a_HCO3, a_CO3 = 1e-13, 3e-10, 9e-11, 4e-10, 4e-10  # Ion sizes [m]

    # === Equilibrium & solubility constants ===
    Ksp = 5.5e-6           # Ca(OH)₂ solubility product [(mol/L)³]
    KspCaCO3 = 2.8e-9      # CaCO₃ solubility product [(mol/L)²]
    Keq_CO2_HCO3 = 10**-6.35     # CO₂ + H₂O ⇌ HCO₃⁻ + H⁺ [(mol/L)]
    Keq_HCO3_CO3 = 10**-10.33    # HCO₃⁻ ⇌ CO₃²⁻ + H⁺ [(mol/L)]
    Keq_CaCO3 = 10**(2.108 + 7)  # Ca²⁺ + CO₃²⁻ ⇌ CaCO₃ [(mol/L)²]
    Keq_H_OH = 10**14            # H⁺ + OH⁻ ⇌ H₂O [(mol/L)²]
    Keq_CO2_OH = 10**7.66        # CO₂ + OH⁻ ⇌ HCO₃⁻ [(mol/L)]
    Keq_HCO3_OH = 10**4.3        # HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O [(mol/L)]

    # === process properties ===
    d_p = 8e-6   # Particle size [m]
    V_liq = 0.5 # Liquid volume [L]
    rho_p = 2468.3 # Particle density [kg/m³]


    # === Initial Conditions ===
    if y0 is None:
        y0 = [
            1e-10,           # CCa_s: Surface Ca²⁺ [mol/L]
            0.0,             # CCa_b: Bulk Ca²⁺ [mol/L]
            tii['n0'],       # nCH: Solid Ca(OH)₂ [mol/L]
            1e-7,            # COH_s: Surface OH⁻ [mol/L]
            1e-2,            # COH_b: Bulk OH⁻ [mol/L]
            0.0,             # CCO3: CO₃²⁻ [mol/L]
            1e-6,            # CH: H⁺ [mol/L]
            0.0,             # CHCO3: HCO₃⁻ [mol/L]
            0.0,             # CCaCO3_CH: CaCO₃ via CH path [mol/L]
            tii['CSH_fast0'],# nCSH: Reactive CSH [mol/L]
            0.0,             # CCaCO3_CSH: CaCO₃ via CSH path [mol/L]
            0.0              # CO2_aq: Aqueous CO₂ [mol/L]
        ]

    # === Activity Coefficient Calculation Function ===
    def calc_gamma(I, z, a_i):
        # I: Ionic strength [mol/L]
        # z: Ion charge [–]
        # a_i: Effective ion size [m]
        I3 = I * 1e3               # Convert to mol/m³
        sqrtI3 = np.sqrt(I3)
        lnGamma = -A * z**2 * sqrtI3 / (1 + a_i * B * sqrtI3)
        return np.clip(np.exp(lnGamma), 0.1, 1.0)  # gamma: dimensionless

    # === Define the ODE System ===
    def ode_rhs(t_local, y, tii, tvi, theta, te):
        """
        Right-hand side of the carbonation model ODE system.
        Computes time derivatives of concentrations for all species.
        """

        # === Unpack kinetic and transport parameters ===
        (
            k1a,  # [1/s]       CO₂(aq) + H₂O ⇌ HCO₃⁻ + H⁺
            k2a,  # [1/s]       HCO₃⁻ ⇌ CO₃²⁻ + H⁺
            k1b,  # [L/mol·s]   CO₂ + OH⁻ ⇌ HCO₃⁻
            k2b,  # [L/mol·s]   HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O
            k_dis_CH,  # [1/s]       Ca(OH)₂ dissolution
            k_con_Ca,  # [1/s]       Ca²⁺ convection surface ↔ bulk
            k_con_OH,  # [1/s]       OH⁻ convection surface ↔ bulk
            D0,  # [m²/s]      Initial diffusivity of CO₂
            decayratio,  # [–]         Time-dependent diffusion decay factor
            kLa,  # [1/s]       CO₂ gas–liquid mass transfer rate
            k3,  # [1/s]       Ca²⁺ + CO₃²⁻ ⇌ CaCO₃
            k4  # [L/mol·s]   H⁺ + OH⁻ ⇌ H₂O
        ) = theta

        # === Unpack state variables (concentrations in [mol/L]) ===
        CCa_s, CCa_b, nCH, COH_s, COH_b, CCO3, CH, CHCO3, CCaCO3_CH, nCSH, CCaCO3_CSH, CO2_aq = y

        # Prevent negative concentrations (numerical safeguard)
        CCa_b, COH_b, CH, CHCO3, CCO3, CO2_aq = map(lambda x: max(x, 1e-20), [CCa_b, COH_b, CH, CHCO3, CCO3, CO2_aq])

        # === Interpolate time-variant pressure [Pa] and temperature [K] ===
        P = np.interp(t_local, te, tvi['P'])
        T = np.interp(t_local, te, tvi['T'])

        # === Calculate CO₂ solubility [mol/L] ===
        CO2_sat = 0.0 if t_local < tii['inj'] else P * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

        # === Calculate ionic strength I [mol/L] ===
        I = max(2 * CCa_b + 0.5 * COH_b + 2 * CCO3 + 0.5 * CH + 0.5 * CHCO3, 1e-4)

        # === Activity coefficients using Debye–Hückel ===
        gCa = calc_gamma(I, 2, a_Ca)  # Ca²⁺
        gOH = calc_gamma(I, 1, a_OH)  # OH⁻
        gH = calc_gamma(I, 1, a_H)  # H⁺
        gHCO3 = calc_gamma(I, 1, a_HCO3)  # HCO₃⁻
        gCO3 = calc_gamma(I, 2, a_CO3)  # CO₃²⁻

        # === Reaction rates [mol/(L·s)] ===
        r1a = k1a * CO2_aq - (k1a / Keq_CO2_HCO3) * CHCO3 * gHCO3 * CH * gH  # CO₂ hydration
        r2a = k2a * CHCO3 * gHCO3 - (k2a / Keq_HCO3_CO3) * CCO3 * gCO3 * CH * gH  # HCO₃⁻ dissociation
        r1b = k1b * CO2_aq * COH_b * gOH - (k1b / Keq_CO2_OH) * CHCO3 * gHCO3  # CO₂ + OH⁻
        r2b = k2b * CHCO3 * gHCO3 * COH_b * gOH - (
                    k2b / Keq_HCO3_OH) * CCO3 * gCO3  # HCO₃⁻ + OH⁻
        r4 = k4 * CH * gH * COH_b * gOH - (k4 / Keq_H_OH)  # Water dissociation
        r3 = k3 * ((CCa_b * gCa) * (CCO3 * gCO3) / KspCaCO3 - 1)  # CaCO₃ precipitation

        # === CH dissolution driving force [–] ===
        S = (CCa_s * gCa) * (COH_s * gOH) ** 2 / Ksp

        # === ODEs (species balances) ===
        dnCH_dt = -k_dis_CH * (1 - S) * nCH  # Solid CH [mol/(L·s)]
        dCCa_s_dt = -dnCH_dt - k_con_Ca * (CCa_s - CCa_b)  # Surface Ca²⁺
        dCOH_s_dt = 2 * (-dnCH_dt) - k_con_OH * (COH_s - COH_b)  # Surface OH⁻
        dCCa_b_dt = k_con_Ca * (CCa_s - CCa_b) - r3  # Bulk Ca²⁺
        dCOH_b_dt = k_con_OH * (COH_s - COH_b) - r4 - r1b - r2b  # Bulk OH⁻
        dCH_dt = r1a + r2a - r4  # H⁺
        dCHCO3_dt = r1a - r2a + r1b - r2b  # HCO₃⁻
        dCCO3_dt = r2a - r3 + r2b  # CO₃²⁻

        CSH0 = tii['CSH_fast0']
        X_CSH = np.clip((CSH0 - nCSH) / (CSH0 + 1e-10), 0.0, 1.0)
        thick = max(d_p * (1 - np.sqrt(1 - X_CSH)) / 2, 1e-12)
        Dt = D0 * np.exp(-decayratio * t_local)
        k_diff = Dt / thick
        a_v = 6 / d_p
        r7 = -nCSH * k_diff * a_v * CO2_aq

        dCSH_dt = r7
        dCaCO3_CSH_dt = -r7
        dCaCO3_CH_dt = r3
        dCO2_aq_dt = kLa * (CO2_sat - CO2_aq) - r1a - r1b - r7

        # === Return time derivatives of all state variables ===
        return [
            dCCa_s_dt,  # d[CCa_s]/dt
            dCCa_b_dt,  # d[CCa_b]/dt
            dnCH_dt,  # d[nCH]/dt
            dCOH_s_dt,  # d[COH_s]/dt
            dCOH_b_dt,  # d[COH_b]/dt
            dCCO3_dt,  # d[CO3²⁻]/dt
            dCH_dt,  # d[H⁺]/dt
            dCHCO3_dt,  # d[HCO3⁻]/dt
            dCaCO3_CH_dt,  # d[CCaCO3_CH]/dt
            dCSH_dt,  # d[nCSH]/dt
            dCaCO3_CSH_dt,  # d[CCaCO3_CSH]/dt
            dCO2_aq_dt  # d[CO₂_aq]/dt
        ]

    # === Solve the ODE System ===
    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(tii, tvi, theta, t), t_eval=t, method='Radau', rtol=1e-6, atol=1e-9)

    # === Postprocess ===
    yout = sol.y
    CH = yout[6]           # [mol/L]
    CB = yout[1]           # [mol/L]
    OH = yout[4]           # [mol/L]
    CQ = yout[5]           # [mol/L]
    CHCO3 = yout[7]        # [mol/L]
    CO2_aq = yout[11]      # [mol/L]
    CSH_fast = yout[9]     # [mol/L]
    CaCO3_CH = yout[8]     # [mol/L]
    CaCO3_CSH = yout[10]   # [mol/L]
    CaCO3 = CaCO3_CH + CaCO3_CSH  # [mol/L]
    n = yout[2]            # [mol/L] Solid Ca(OH)₂
    pH = -np.log10(np.maximum(CH, 1e-12))  # [–]
    I = np.maximum(2*CB + 0.5*OH + 2*CQ + 0.5*CH + 0.5*CHCO3, 1e-4)  # [mol/L]

    # === Unpack kinetic and transport parameters ===
    (
        k1a,  # [1/s]       CO₂(aq) + H₂O ⇌ HCO₃⁻ + H⁺
        k2a,  # [1/s]       HCO₃⁻ ⇌ CO₃²⁻ + H⁺
        k1b,  # [L/mol·s]   CO₂ + OH⁻ ⇌ HCO₃⁻
        k2b,  # [L/mol·s]   HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O
        k_dis_CH,  # [1/s]       Ca(OH)₂ dissolution
        k_con_Ca,  # [1/s]       Ca²⁺ convection surface ↔ bulk
        k_con_OH,  # [1/s]       OH⁻ convection surface ↔ bulk
        D0,  # [m²/s]      Initial diffusivity of CO₂
        decayratio,  # [–]         Time-dependent diffusion decay factor
        kLa,  # [1/s]       CO₂ gas–liquid mass transfer rate
        k3,  # [1/s]       Ca²⁺ + CO₃²⁻ ⇌ CaCO₃
        k4  # [L/mol·s]   H⁺ + OH⁻ ⇌ H₂O
    ) = theta

    gCa = calc_gamma(I, 2, a_Ca)
    gOH = calc_gamma(I, 1, a_OH)
    gH = calc_gamma(I, 1, a_H)
    gHCO3 = calc_gamma(I, 1, a_HCO3)
    gCO3 = calc_gamma(I, 2, a_CO3)

    r1a = k1a * CO2_aq - (k1a / Keq_CO2_HCO3) * CHCO3 * gHCO3 * CH * gH
    r2a = k2a * CHCO3 * gHCO3 - (k2a / Keq_HCO3_CO3) * CQ * gCO3 * CH * gH
    r1b = k1b * CO2_aq * OH * gOH - (k1b / Keq_CO2_OH) * CHCO3 * gHCO3
    r2b = k2b * CHCO3 * gHCO3 * OH * gOH - (k2b / Keq_HCO3_OH) * CQ * gCO3
    r4 = k4 * CH * gH * OH * gOH - (k4 / Keq_H_OH)
    r3 = k3 * ((CB * gCa) * (CQ * gCO3) / KspCaCO3 - 1)

    # === Mechanistic CSH carbonation: thickness and Dt over time ===
    X_CSH = np.clip((tii['CSH_fast0'] - CSH_fast) / (tii['CSH_fast0'] + 1e-10), 0.0, 1.0)
    thickness = np.maximum(d_p * (1 - np.sqrt(1 - X_CSH)) / 2, 1e-12)
    Dt = theta[8] * np.exp(-theta[9] * t)
    k_diff = Dt / thickness
    a_v = 6 / d_p
    r7 = -CSH_fast * k_diff * a_v * CO2_aq

    return {
        'time': t,                 # [s]
        'nCH': n,                    # [mol/L]
        'OH': OH,                  # [mol/L]
        'HCO3': CHCO3,             # [mol/L]
        'CO3': CQ,                 # [mol/L]
        'CaCO3': CaCO3,            # [mol/L]
        'CO2_aq': CO2_aq,          # [mol/L]
        'CCa_b': CB,                  # [mol/L]
        'nCSH': CSH_fast,      # [mol/L]
        'pH': pH,                  # [–]
        'I': I,                    # [mol/L]
        'r1a': r1a, 'r2a': r2a, 'r1b': r1b, 'r2b': r2b, 'r4': r4, 'r3': r3, 'r7': r7,  # [mol/(L·s)]
        'CCaCO3_CH': CaCO3_CH,      # [mol/L]
        'CCaCO3_CSH': CaCO3_CSH,    # [mol/L]
        'gamma_Ca': gCa, 'gamma_OH': gOH, 'gamma_H': gH,            # [–]
        'gamma_HCO3': gHCO3, 'gamma_CO3': gCO3                      # [–]
    }



# ===============================
# SIMULATION SETUP & EXECUTION
# ===============================
# Define time vector (0 to 3600 s)
t = np.linspace(0, 10800, 300)

# Initial inputs
tii = {
    'n0': 0.425,  # initial Ca(OH)2 [mol/L]
    'CSH_fast0': 0.6,  # initial fast-reacting CSH [mol/L]
    'inj': 300
}
tvi = {'T': np.full_like(t, 298.15), 'P': np.full_like(t, 1.01325)}

# Model parameters
theta = [
    2.836907e-04,  # k_CO2_HCO3: merged CO₂(aq)+H₂O ⇌ HCO₃⁻+H⁺ rate constant [1/s]
    2.311080e+00,  # k3f: HCO₃⁻ ⇌ CO₃²⁻ + H⁺ forward rate constant [1/s]
    1.064109e+01,  # k6f: CO₂ + OH⁻ ⇌ HCO₃⁻ forward rate constant [L/(mol·s)]
    5.495592e+06,  # k7f: HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O forward rate constant [L/(mol·s)]
    1.967433e-03,  # KL: Ca(OH)₂ dissolution rate constant [1/s]
    5.169879e-01,  # KO: convection rate constant for Ca²⁺ from surface to bulk [1/s]
    5.169879e-01,  # KO_OH: convection rate constant for OH⁻ from surface to bulk [1/s]
    8e-9,  # D0: initial diffusion coefficient of CO₂ in product layer [m²/s]
    3,  # alpha: diffusion resistance parameter for CaCO₃ layer [dimensionless]
    4.777403e-02,  # kLa: volumetric mass transfer coefficient for CO₂ [1/s]
    7.069680e-02,  # k4f: Ca²⁺ + CO₃²⁻ ⇌ CaCO₃ forward rate constant [L/(mol·s)]
    1.400000e+11,  # k5f: H⁺ + OH⁻ ⇌ H₂O forward rate constant [L/(mol·s)]
]



# Run model
out = carbonation_model(t, None, tii, tvi, theta)

time_min = out['time'] / 60.0  # convert to minutes

# Define a consistent color palette
colors = {
    'red': [0.8500, 0.3250, 0.0980],
    'blue': [0, 0.4470, 0.7410],
    'green': [0.4660, 0.6740, 0.1880],
    'purple': [0.4940, 0.1840, 0.5560],
    'orange': [0.9290, 0.6940, 0.1250],
    'cyan': [0.3010, 0.7450, 0.9330],
    'darkblue': [0, 0.2, 0.5],
    'darkgreen': [0.2, 0.5, 0.2],
    'magenta': [0.9, 0.1, 0.9]
}


# ============================================================
# 1. INDIVIDUAL CONCENTRATION PLOTS
# ============================================================

def plot_single(x, y, title, ylabel, color):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color=color, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time (min)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_single(time_min, out['n'], 'Ca(OH)₂ Concentration', 'mol/L', colors['blue'])
plot_single(time_min, out['OH'], 'OH⁻ Concentration', 'mol/L', colors['cyan'])
plot_single(time_min, out['HCO3'], 'HCO₃⁻ Concentration', 'mol/L', colors['green'])
plot_single(time_min, out['CO3'], 'CO₃²⁻ Concentration', 'mol/L', colors['purple'])
plot_single(time_min, out['CaCO3'], 'Total CaCO₃ Concentration', 'mol/L', colors['orange'])
plot_single(time_min, out['CO2_aq'], 'Dissolved CO₂ Concentration', 'mol/L', colors['darkgreen'])
plot_single(time_min, out['pH'], 'pH Evolution', '', colors['red'])
plot_single(time_min, out['I'], 'Ionic Strength', 'mol/L', colors['darkblue'])

# ============================================================
# 2. CSH CARBONATION PATHWAY
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(time_min, out['CSH_fast'], color=colors['blue'], label='CSH Concentration')
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("CSH (mol/L)", color=colors['blue'])
ax1.tick_params(axis='y', labelcolor=colors['blue'])

ax2 = ax1.twinx()
ax2.plot(time_min, out['r7'], color=colors['orange'], label='r₇: CaCO₃ via CSH')
ax2.set_ylabel("Rate (mol/L/s)", color=colors['orange'])
ax2.tick_params(axis='y', labelcolor=colors['orange'])

fig.suptitle('CSH Carbonation Pathway')
fig.tight_layout()
plt.show()

# ============================================================
# 3. CaCO₃ COMPONENT CONTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time_min, out['CaCO3_CH'], label='From Ca(OH)₂', color=colors['blue'])
ax.plot(time_min, out['CaCO3_CSH'], label='From CSH', color=colors['green'])
ax.plot(time_min, out['CaCO3'], label='Total', linestyle='--', color=colors['red'])
ax.set_xlabel("Time (min)")
ax.set_ylabel("CaCO₃ (mol/L)")
ax.set_title("CaCO₃ Accumulation from Different Sources")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 4. ACTIVITY COEFFICIENTS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time_min, out['gamma_Ca'], label='γ(Ca²⁺)', color=colors['blue'])
ax.plot(time_min, out['gamma_OH'], label='γ(OH⁻)', color=colors['cyan'])
ax.plot(time_min, out['gamma_H'], label='γ(H⁺)', color=colors['red'])
ax.plot(time_min, out['gamma_HCO3'], label='γ(HCO₃⁻)', color=colors['green'])
ax.plot(time_min, out['gamma_CO3'], label='γ(CO₃²⁻)', color=colors['purple'])
ax.set_xlabel("Time (min)")
ax.set_ylabel("Activity Coefficient")
ax.set_title("Ion Activity Coefficients Over Time")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 5. DIFFUSIVITY OF CO₂ IN PRODUCT LAYER
# ============================================================

Dt = theta[8] * np.exp(-theta[9] * t)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time_min, Dt, color=colors['darkgreen'])
ax.set_xlabel("Time (min)")
ax.set_ylabel("Effective Diffusivity D(t) [m²/s]")
ax.set_title("Time-Dependent CO₂ Diffusivity")
ax.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 6. CARBON SPECIES DISTRIBUTION & pH OVER TIME
# ============================================================

# Calculate molar fractions for CO₂(aq), HCO₃⁻, CO₃²⁻
total_C = out['CO2_aq'] + out['HCO3'] + out['CO3'] + 1e-10
frac_CO2 = out['CO2_aq'] / total_C
frac_HCO3 = out['HCO3'] / total_C
frac_CO3 = out['CO3'] / total_C

# Mask pre-injection values
mask = t < tii['inj']
frac_CO2[mask] = np.nan
frac_HCO3[mask] = np.nan
frac_CO3[mask] = np.nan

# A. Carbon species with pH overlay
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.stackplot(time_min, frac_CO2, frac_HCO3, frac_CO3,
              colors=[colors['darkgreen'], colors['cyan'], colors['purple']],
              labels=['CO₂(aq)', 'HCO₃⁻', 'CO₃²⁻'])
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Molar Fraction")
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(time_min, out['pH'], color=colors['red'], label='pH')
ax2.set_ylabel("pH", color=colors['red'])
ax2.tick_params(axis='y', labelcolor=colors['red'])
fig.suptitle('Carbon Speciation (Post-Injection) and pH')
fig.tight_layout()
plt.show()

# B. Split view
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax3.stackplot(time_min, frac_CO2, frac_HCO3, frac_CO3,
              colors=[colors['darkgreen'], colors['cyan'], colors['purple']],
              labels=['CO₂(aq)', 'HCO₃⁻', 'CO₃²⁻'])
ax3.set_ylabel("Molar Fraction")
ax3.set_title("Carbon Speciation (After CO₂ Injection)")
ax3.legend(loc='upper right')
ax3.grid(True)

ax4.plot(time_min, out['pH'], color=colors['red'])
ax4.set_xlabel("Time (min)")
ax4.set_ylabel("pH")
ax4.set_title("pH Evolution")
ax4.grid(True)
fig.tight_layout()
plt.show()

# ============================================================
# 7. REACTION RATE DYNAMICS
# ============================================================

plot_single(time_min, out['r1'], 'r₁: CO₂ + H₂O ⇌ HCO₃⁻ + H⁺', 'mol/L/s', colors['blue'])
plot_single(time_min, out['r2'], 'r₂: HCO₃⁻ ⇌ CO₃²⁻ + H⁺', 'mol/L/s', colors['purple'])
plot_single(time_min, out['r3'], 'r₃: CO₂ + OH⁻ ⇌ HCO₃⁻', 'mol/L/s', colors['green'])
plot_single(time_min, out['r4'], 'r₄: HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O', 'mol/L/s', colors['darkgreen'])
plot_single(time_min, out['r5'], 'r₅: H⁺ + OH⁻ ⇌ H₂O', 'mol/L/s', colors['magenta'])
plot_single(time_min, out['r6'], 'r₆: Ca²⁺ + CO₃²⁻ ⇌ CaCO₃', 'mol/L/s', colors['orange'])
plot_single(time_min, out['r7'], 'r₇: CaCO₃ via CSH', 'mol/L/s', colors['red'])