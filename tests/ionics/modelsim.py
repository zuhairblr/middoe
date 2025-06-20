import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def carbonation_model(t, y0, tii, tvi=None, theta=None):
    """
    Runs the carbonation ODE system with Debye–Hückel activity corrections,
    mirroring the MATLAB implementation.

    Parameters
    ----------
    t : 1d array
        Time vector [s].
    y0 : list or None
        Initial state vector [CS, CB, n, COH_surface, COH_bulk, CQ, CH, CHCO3,
        CaCO3_CH, CSH_fast, CaCO3_CSH, CO2_aq]. If None, defaults are used.
    tii : dict
        Time-invariant inputs, must include:
            'n0' : initial Ca(OH)_2 concentration [mol/L],
            'CSH_fast0' : initial fast-reacting CSH concentration [mol/L].
    tvi : dict, optional
        (unused placeholder for time-variant inputs if ever needed).
    theta : list of floats
        Model parameters (in the following order):
        [k_CO2_HCO3,  # merged CO2 dissociation rate constant [1/s]
         k3f,         # HCO3- ⇌ CO3^2- forward rate constant [1/s]
         k6f,         # CO2 + OH- ⇌ HCO3- forward rate constant [L/(mol·s)]
         k7f,         # HCO3- + OH- ⇌ CO3^2- forward rate constant [L/(mol·s)]
         KL,          # dissolution rate constant for Ca(OH)_2 [1/s]
         KO,          # convection rate constant for Ca^2+ to bulk [1/s]
         KO_OH,       # convection rate constant for OH- to bulk [1/s]
         kCSH_fast,   # rate constant for fast-reacting CSH fraction [L/(mol·s)]
         D0,          # initial diffusion coefficient of CO2 in product layer [m²/s]
         alpha,       # diffusion resistance parameter for CaCO3 layer [dimensionless]
         alpha_diff,  # secondary diffusion resistance parameter [dimensionless]
         kLa,         # volumetric mass transfer coefficient for CO2 [1/s]
         k4f,         # Ca^2+ + CO3^2- ⇌ CaCO3 forward rate constant [L/(mol·s)]
         k5f,         # H+ + OH- = H2O forward rate constant [L/(mol·s)]
         p,           # partial pressure of CO2 [Pa]
         initial_pH   # initial pH value (unitless)
        ]

    Returns
    -------
    dict
        'time'       : t (same as input),
        'n'          : Ca(OH)_2 concentration [mol/L],
        'OH'         : bulk OH- concentration [mol/L],
        'HCO3'       : HCO3- concentration [mol/L],
        'CO3'        : CO3^2- concentration [mol/L],
        'CaCO3'      : total CaCO3 concentration [mol/L],
        'CO2_aq'     : dissolved CO2 concentration [mol/L],
        'CB'         : bulk Ca^2+ concentration [mol/L],
        'CSH_fast'   : fast-reacting CSH concentration [mol/L],
        'pH'         : pH (unitless),
        'I'          : ionic strength [mol/L],
        'r1','r3','r4','r5','r6','r7','r8' : reaction rates [mol/(L·s)]
    """
    # Universal constants
    R, T = 8.314, 298.0  # Gas constant [J/(mol·K)] and temperature [K]

    # Unpack theta parameters
    (
        k_CO2_HCO3,  # merged CO2 dissociation: CO2(aq) + H2O ⇌ HCO3- + H+
        k3f,         # HCO3- ⇌ CO3^2- + H+
        k6f,         # CO2 + OH- ⇌ HCO3-
        k7f,         # HCO3- + OH- ⇌ CO3^2- + H2O
        KL,          # dissolution of Ca(OH)_2
        KO,          # convection of Ca^2+ to bulk
        KO_OH,       # convection of OH- to bulk
        kCSH_fast,   # rate constant for fast-reacting CSH fraction
        D0,          # initial diffusion coefficient of CO2 in product layer
        alpha,       # diffusion resistance parameter
        alpha_diff,  # secondary diffusion resistance parameter
        kLa,         # volumetric mass transfer coefficient for CO2
        k4f,         # Ca^2+ + CO3^2- ⇌ CaCO3 forward
        k5f,         # H+ + OH- = H2O forward
        p,           # partial pressure of CO2 [Pa]
        initial_pH   # starting pH (unitless)
    ) = theta

    # Debye–Hückel parameters (from MATLAB)
    A = 0.037
    B = 1.04e8
    a_Ca = 1.0e-13
    a_OH = 3.0e-10
    a_H = 9.0e-11
    a_HCO3 = 4.0e-10
    a_CO3 = 4.0e-10

    # Solubility products
    Ksp = 5.5e-6       # Ca(OH)_2 Ksp [mol^3/L^3]
    KspCaCO3 = 2.8e-9  # CaCO3 Ksp [mol^2/L^2]

    # Equilibrium constants (dimensionless)
    Keq_CO2_HCO3 = 10 ** (-6.35)     # CO2(aq) + H2O ⇌ HCO3- + H+
    Keq3 = 10 ** (-10.33)            # HCO3- ⇌ CO3^2- + H+
    Keq4 = 10 ** (2.108 + 7)         # Ca^2+ + CO3^2- ⇌ CaCO3
    Keq5 = 10 ** 14                  # H+ + OH- = H2O
    Keq6 = 10 ** 7.66                # CO2 + OH- ⇌ HCO3-
    Keq7 = 10 ** 4.3                 # HCO3- + OH- ⇌ CO3^2- + H2O

    # Henry's law constant for CO2 dissolution
    KH = 2.82e6 * np.exp(-2044.0 / T)  # [Pa·m^3/mol]
    CO2_sat = p / KH / 1000.0          # saturation CO2 concentration [mol/L]

    # If no initial state provided, set defaults using tii
    if y0 is None:
        OH_bulk_0 = 10 ** (initial_pH - 14.0)  # compute initial OH- from initial pH

        y0 = [
            1e-10,                # CS initial
            0.0,                  # CB initial
            tii['n0'],            # n initial (0.425)
            1e-7,                 # COH_s initial (surface OH-)
            OH_bulk_0,            # COH_b initial (bulk OH-)
            0.0,                  # CQ initial (CO3^2-)
            1e-6,                 # CH initial (H+)
            0.0,                  # CHCO3 initial (HCO3-)
            0.0,                  # CaCO3_CH initial
            tii['CSH_fast0'],     # CSH_fast initial (0.6)
            0.0,                  # CaCO3_CSH initial
            0.0                   # CO2_aq initial
        ]

    def calc_gamma(I, z, a_i):
        """
        Debye–Hückel activity coefficient for ion of charge z, size a_i,
        given ionic strength I.
        """
        I3 = I * 1e3
        sqrtI3 = np.sqrt(I3)
        lnGamma = (
            -A * z**2 * sqrtI3 / (1 + a_i * B * sqrtI3)
            + (0.2 - 4.17e-5 * I3) * A * z**2 * I3 / np.sqrt(1000)
        )
        gamma = np.exp(lnGamma)
        # Clip between 0.1 and 1.0
        return np.minimum(np.maximum(gamma, 0.1), 1.0)

    def ode_rhs(t_local, y):
        """
        Time derivatives for each state variable at time t_local,
        including activity corrections.
        """
        # Unpack state variables
        CS, CB, n, COH_s, COH_b, CQ, CH, CHCO3, CaCO3_CH, CSH_fast, CaCO3_CSH, CO2_aq = y

        # Enforce non-negativity to avoid log or divide errors
        CB = max(CB, 1e-20)
        COH_b = max(COH_b, 1e-20)
        CH = max(CH, 1e-20)
        CHCO3 = max(CHCO3, 1e-20)
        CQ = max(CQ, 1e-20)
        CO2_aq = max(CO2_aq, 1e-20)

        # -----------------------------
        # Calculate ionic strength (MATLAB formula)
        # -----------------------------
        I = 2 * CB + 0.5 * COH_b + 2 * CQ + 0.5 * CH + 0.5 * CHCO3
        I = max(I, 1e-4)

        # -----------------------------
        # Activity coefficients via Debye–Hückel
        # -----------------------------
        gamma_Ca = calc_gamma(I, 2, a_Ca)
        gamma_OH = calc_gamma(I, 1, a_OH)
        gamma_H = calc_gamma(I, 1, a_H)
        gamma_HCO3 = calc_gamma(I, 1, a_HCO3)
        gamma_CO3 = calc_gamma(I, 2, a_CO3)

        # -----------------------------
        # Reaction rates with activity corrections
        # -----------------------------
        # r1: CO2(aq) + H2O ⇌ HCO3- + H+
        r1 = k_CO2_HCO3 * CO2_aq - (k_CO2_HCO3 / Keq_CO2_HCO3) * (CHCO3 * gamma_HCO3) * (CH * gamma_H)

        # r3: HCO3- ⇌ CO3^2- + H+
        r3 = k3f * (CHCO3 * gamma_HCO3) - (k3f / Keq3) * (CQ * gamma_CO3) * (CH * gamma_H)

        # r4: Ca^2+ + CO3^2- ⇌ CaCO3
        r4 = k4f * (((CB * gamma_Ca) * (CQ * gamma_CO3)) / KspCaCO3 - 1.0)

        # r5: H+ + OH- = H2O
        r5 = k5f * ((CH * gamma_H) * (COH_b * gamma_OH)) - (k5f / Keq5)

        # r6: CO2 + OH- ⇌ HCO3-
        r6 = k6f * (CO2_aq * (COH_b * gamma_OH)) - (k6f / Keq6) * (CHCO3 * gamma_HCO3)

        # r7: HCO3- + OH- ⇌ CO3^2- + H2O
        r7 = k7f * ((CHCO3 * gamma_HCO3) * (COH_b * gamma_OH)) - (k7f / Keq7) * (CQ * gamma_CO3)

        # -----------------------------
        # Saturation index for Ca(OH)2 with activity corrections
        # -----------------------------
        S = (CS * gamma_Ca) * ((COH_s * gamma_OH)**2) / Ksp

        # -----------------------------
        # Ca(OH)2 dissolution (CH phase)
        # -----------------------------
        dn_dt = -KL * (1.0 - S) * n  # negative when unsaturated

        # Surface Ca2+ balance: dissolution minus convection to bulk
        dCS_dt = -dn_dt - KO * (CS - CB)

        # Surface OH- balance: produced 2× per dissolution, minus convection to bulk
        dCOH_s_dt = 2.0 * (-dn_dt) - KO_OH * (COH_s - COH_b)

        # -----------------------------
        # Carbonate speciation dynamics
        # -----------------------------
        dCHCO3_dt = r1 - r3 + r6 - r7
        dCQ_dt = r3 - r4 + r7

        # -----------------------------
        # Proton / bulk OH- balances
        # -----------------------------
        dCH_dt = r1 + r3 - r5
        dCOH_b_dt = KO_OH * (COH_s - COH_b) - r5 - r6 - r7

        # -----------------------------
        # Bulk Ca^2+ dynamics
        # -----------------------------
        dCB_dt = KO * (CS - CB) - r4

        # -----------------------------
        # CSH carbonation (solid CSH → CaCO3)
        # -----------------------------
        Dt = D0 * np.exp(-alpha * t_local)  # time‐dependent diffusion coefficient
        diff_res = 1.0 + alpha_diff * (CaCO3_CSH + CaCO3_CH) / (CSH_fast + 1e-10)
        dCSH_dt = -kCSH_fast * CSH_fast * Dt * CO2_aq / diff_res
        dCaCO3_CSH_dt = -dCSH_dt

        # CaCO3 formation from Ca(OH)2 (CH path)
        dCaCO3_CH_dt = r4

        # -----------------------------
        # Dissolved CO2 balance
        # -----------------------------
        dCO2_aq_dt = kLa * (CO2_sat - CO2_aq) - r1 - r6 - (-dCSH_dt)

        # Return derivatives in the same order as y
        return [
            dCS_dt,
            dCB_dt,
            dn_dt,
            dCOH_s_dt,
            dCOH_b_dt,
            dCQ_dt,
            dCH_dt,
            dCHCO3_dt,
            dCaCO3_CH_dt,
            dCSH_dt,
            dCaCO3_CSH_dt,
            dCO2_aq_dt
        ]

    # Solve the ODE system over the time span t
    sol = solve_ivp(
        ode_rhs,               # RHS function
        (t[0], t[-1]),         # time interval
        y0,                    # initial state
        t_eval=t,              # output at specified times
        method='Radau',        # stiff solver
        rtol=1e-6,             # relative tolerance
        atol=1e-9              # absolute tolerance
    )

    # Extract solution time series
    yout = sol.y
    CaCO3_CH = yout[8, :]
    CaCO3_CSH = yout[10, :]
    n = yout[2, :]                   # Ca(OH)2 [mol/L]
    OH = yout[4, :]                  # bulk OH- [mol/L]
    HCO3 = yout[7, :]                # HCO3- [mol/L]
    CO3 = yout[5, :]                 # CO3^2- [mol/L]
    CaCO3 = yout[8, :] + yout[10, :] # total CaCO3 [mol/L]
    CO2_aq = yout[11, :]             # dissolved CO2 [mol/L]
    CB = yout[1, :]                  # bulk Ca^2+ [mol/L]
    CSH_fast = yout[9, :]            # fast-reacting CSH [mol/L]
    CH = yout[6, :]                  # H+ [mol/L]
    CHCO3 = yout[7, :]               # HCO3- [mol/L]
    CS = yout[0, :]                  # surface Ca^2+ [mol/L]
    COH_s = yout[3, :]               # surface OH- [mol/L]

    # -----------------------------
    # pH calculation with buffer
    # -----------------------------
    buffer_effect = 1.0 + 0.1 * n / (tii['n0'] + 1e-10)
    pH = 14.0 + np.log10(np.maximum(OH, 1e-12)) * buffer_effect

    # -----------------------------
    # Ionic strength (MATLAB formula) for output
    # -----------------------------
    I = 2 * CB + 0.5 * OH + 2 * CO3 + 0.5 * CH + 0.5 * HCO3
    I = np.maximum(I, 1e-4)

    # -----------------------------
    # Activity coefficients array for output
    # -----------------------------
    gamma_Ca = calc_gamma(I, 2, a_Ca)
    gamma_OH = calc_gamma(I, 1, a_OH)
    gamma_H = calc_gamma(I, 1, a_H)
    gamma_HCO3 = calc_gamma(I, 1, a_HCO3)
    gamma_CO3 = calc_gamma(I, 2, a_CO3)

    # -----------------------------
    # Recompute reaction rates for output (with γ)
    # -----------------------------
    r1 = k_CO2_HCO3 * CO2_aq - (k_CO2_HCO3 / Keq_CO2_HCO3) * (HCO3 * gamma_HCO3) * (CH * gamma_H)
    r3 = k3f * (HCO3 * gamma_HCO3) - (k3f / Keq3) * (CO3 * gamma_CO3) * (CH * gamma_H)
    r4 = k4f * (((CB * gamma_Ca) * (CO3 * gamma_CO3)) / KspCaCO3 - 1.0)
    r5 = k5f * ((CH * gamma_H) * (OH * gamma_OH)) - (k5f / Keq5)
    r6 = k6f * (CO2_aq * (OH * gamma_OH)) - (k6f / Keq6) * (HCO3 * gamma_HCO3)
    r7 = k7f * ((HCO3 * gamma_HCO3) * (OH * gamma_OH)) - (k7f / Keq7) * (CO3 * gamma_CO3)

    # -----------------------------
    # CSH carbonation rate for output
    # -----------------------------
    Dt = D0 * np.exp(-alpha * t)
    diff_res = 1.0 + alpha_diff * (yout[10, :] + yout[8, :]) / (CSH_fast + 1e-10)
    r8 = -kCSH_fast * CSH_fast * Dt * CO2_aq / diff_res

    # -----------------------------
    # Package results into dictionary
    # -----------------------------
    return {
        'time': t,
        'n': n,
        'OH': OH,
        'HCO3': HCO3,
        'CO3': CO3,
        'CaCO3': CaCO3,
        'CO2_aq': CO2_aq,
        'CB': CB,
        'CSH_fast': CSH_fast,
        'pH': pH,
        'I': I,
        'r1': r1,
        'r3': r3,
        'r4': r4,
        'r5': r5,
        'r6': r6,
        'r7': r7,
        'r8': r8,
        'CaCO3_CH': CaCO3_CH,
        'CaCO3_CSH': CaCO3_CSH,
        'gamma_Ca': gamma_Ca,
        'gamma_OH': gamma_OH,
        'gamma_H': gamma_H,
        'gamma_HCO3': gamma_HCO3,
        'gamma_CO3': gamma_CO3,
    }


# ===============================
# SIMULATION SETUP & EXECUTION
# ===============================
# Define time vector (0 to 3600 s)
t = np.linspace(0, 10800, 500)

# Initial inputs
tii = {
    'n0': 0.425,  # initial Ca(OH)2 [mol/L]
    'CSH_fast0': 0.6  # initial fast-reacting CSH [mol/L]
}

# Model parameters (fixed)
theta = [
    2.836907e-04,  # k_CO2_HCO3
    2.311080e+00,  # k3f
    1.064109e+01,  # k6f
    5.495592e+06,  # k7f
    1.967433e-03,  # KL
    5.169879e-01,  # KO
    5.169879e-01,  # KO_OH
    6.315715e+01,  # kCSH_fast
    2.417100e-04,  # D0
    4.487054e-04,  # alpha
    1.681085e+00,  # alpha_diff
    4.777403e-02,  # kLa
    7.069680e-02,  # k4f
    1.400000e+11,  # k5f
    1.013250e+05,  # p (Pa)
    11.53  # initial_pH
]

# Run model
out = carbonation_model(t, None, tii, theta=theta)
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



# ===============================
# 1. CARBONATE SPECIES EVOLUTION
# ===============================

# A. Single figure with twin y-axis
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(time_min, out['HCO3'], color=colors['cyan'], label='HCO₃⁻')
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("HCO₃⁻ (mol/L)", color=colors['cyan'])
ax1.tick_params(axis='y', labelcolor=colors['cyan'])

ax2 = ax1.twinx()
ax2.plot(time_min, out['CO3'], color=colors['purple'], label='CO₃²⁻')
ax2.set_ylabel("CO₃²⁻ (mol/L)", color=colors['purple'])
ax2.tick_params(axis='y', labelcolor=colors['purple'])

fig.suptitle('Carbonate Species Evolution')
fig.tight_layout()
plt.show()

# B. Two stacked subplots
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax3.plot(time_min, out['HCO3'], color=colors['cyan'])
ax3.set_ylabel("HCO₃⁻ (mol/L)")
ax3.set_title("HCO₃⁻ Concentration")
ax3.grid(True)

ax4.plot(time_min, out['CO3'], color=colors['purple'])
ax4.set_xlabel("Time (min)")
ax4.set_ylabel("CO₃²⁻ (mol/L)")
ax4.set_title("CO₃²⁻ Concentration")
ax4.grid(True)

fig.tight_layout()
plt.show()


# ===========================
# 2. CSH CARBONATION PROCESS
# ===========================

# A. Twin y-axis on one figure
fig, ax5 = plt.subplots(figsize=(8, 4))
ax5.plot(time_min, out['CSH_fast'], color=colors['blue'], label='CSH')
ax5.set_xlabel("Time (min)")
ax5.set_ylabel("CSH (mol/L)", color=colors['blue'])
ax5.tick_params(axis='y', labelcolor=colors['blue'])

ax6 = ax5.twinx()
ax6.plot(time_min, out['r8'], color=colors['orange'], label='r₈: CaCO₃ via CSH')
ax6.set_ylabel("Rate (mol/L/s)", color=colors['orange'])
ax6.tick_params(axis='y', labelcolor=colors['orange'])

fig.suptitle('CSH Carbonation Process')
fig.tight_layout()
plt.show()

# B. Separate subplots
fig, (ax7, ax8) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax7.plot(time_min, out['CSH_fast'], color=colors['blue'])
ax7.set_ylabel("CSH (mol/L)")
ax7.set_title("CSH Concentration Over Time")
ax7.grid(True)

ax8.plot(time_min, out['r8'], color=colors['orange'])
ax8.set_xlabel("Time (min)")
ax8.set_ylabel("Rate (mol/L/s)")
ax8.set_title("CaCO₃ Formation Rate via CSH")
ax8.grid(True)

fig.tight_layout()
plt.show()


# ===============================
# 3. pH BUFFERING BY Ca(OH)₂
# ===============================

# A. Twin y-axis
fig, ax9 = plt.subplots(figsize=(8, 4))
ax9.plot(time_min, out['pH'], color=colors['red'], label='pH')
ax9.set_xlabel("Time (min)")
ax9.set_ylabel("pH", color=colors['red'])
ax9.tick_params(axis='y', labelcolor=colors['red'])

ax10 = ax9.twinx()
ax10.plot(time_min, out['n'], color=colors['blue'], label='Ca(OH)₂')
ax10.set_ylabel("Ca(OH)₂ (mol/L)", color=colors['blue'])
ax10.tick_params(axis='y', labelcolor=colors['blue'])

fig.suptitle('pH Buffering by Ca(OH)₂')
fig.tight_layout()
plt.show()

# B. Two subplots
fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax11.plot(time_min, out['pH'], color=colors['red'])
ax11.set_ylabel("pH")
ax11.set_title("pH Evolution")
ax11.grid(True)

ax12.plot(time_min, out['n'], color=colors['blue'])
ax12.set_xlabel("Time (min)")
ax12.set_ylabel("Ca(OH)₂ (mol/L)")
ax12.set_title("Ca(OH)₂ Concentration")
ax12.grid(True)

fig.tight_layout()
plt.show()


# =======================================
# 4. pH vs. Carbonate Species (Three series)
# =======================================

# A. Two y-axes (pH on left, carbonates on right)
fig, ax13 = plt.subplots(figsize=(8, 4))
ax13.plot(time_min, out['pH'], color=colors['red'], label='pH')
ax13.set_xlabel("Time (min)")
ax13.set_ylabel("pH", color=colors['red'])
ax13.tick_params(axis='y', labelcolor=colors['red'])

ax14 = ax13.twinx()
ax14.plot(time_min, out['HCO3'], color=colors['cyan'], linestyle='-', label='HCO₃⁻')
ax14.plot(time_min, out['CO3'], color=colors['purple'], linestyle='--', label='CO₃²⁻')
ax14.set_ylabel("Concentration (mol/L)", color='k')
ax14.tick_params(axis='y', labelcolor='k')

lines_1, labels_1 = ax13.get_legend_handles_labels()
lines_2, labels_2 = ax14.get_legend_handles_labels()
ax14.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

fig.suptitle('pH Effect on Carbonate Speciation')
fig.tight_layout()
plt.show()

# B. Three stacked subplots
fig, (ax15, ax16, ax17) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

ax15.plot(time_min, out['pH'], color=colors['red'])
ax15.set_ylabel("pH")
ax15.set_title("pH Evolution")
ax15.grid(True)

ax16.plot(time_min, out['HCO3'], color=colors['cyan'])
ax16.set_ylabel("HCO₃⁻ (mol/L)")
ax16.set_title("HCO₃⁻ Concentration")
ax16.grid(True)

ax17.plot(time_min, out['CO3'], color=colors['purple'])
ax17.set_xlabel("Time (min)")
ax17.set_ylabel("CO₃²⁻ (mol/L)")
ax17.set_title("CO₃²⁻ Concentration")
ax17.grid(True)

fig.tight_layout()
plt.show()


# ===========================================
# 5. COMBINED REACTION PATHWAYS (r₁, r₆, r₇, r₅)
# ===========================================

# A. Two y-axes (r₁, r₆, r₇ on one; r₅ on the other)
fig, ax18 = plt.subplots(figsize=(8, 5))
ax18.plot(time_min, out['r1'], label='r₁: CO₂+H₂O⇌HCO₃⁻+H⁺', color=colors['blue'])
ax18.plot(time_min, out['r6'], label='r₆: CO₂+OH⁻⇌HCO₃⁻', color=colors['red'])
ax18.plot(time_min, out['r7'], label='r₇: HCO₃⁻+OH⁻⇌CO₃²⁻+H₂O', color=colors['green'])
ax18.set_xlabel("Time (min)")
ax18.set_ylabel("Rate (mol/L/s)", color='k')
ax18.grid(True)

ax19 = ax18.twinx()
ax19.plot(time_min, out['r5'], label='r₅: H⁺+OH⁻⇌H₂O', color=colors['magenta'], linestyle=':')
ax19.set_ylabel("r₅ Rate (mol/L/s)", color=colors['magenta'])
ax19.tick_params(axis='y', labelcolor=colors['magenta'])

lines_3, labels_3 = ax18.get_legend_handles_labels()
lines_4, labels_4 = ax19.get_legend_handles_labels()
ax18.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')

fig.suptitle('Reaction Pathways for CO₂ and OH⁻')
fig.tight_layout()
plt.show()

# B. Two subplots (r₁, r₆, r₇ in top; r₅ in bottom)
fig, (ax20, ax21) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax20.plot(time_min, out['r1'], label='r₁', color=colors['blue'])
ax20.plot(time_min, out['r6'], label='r₆', color=colors['red'])
ax20.plot(time_min, out['r7'], label='r₇', color=colors['green'])
ax20.set_ylabel("Rate (mol/L/s)")
ax20.set_title("r₁, r₆, r₇ Reaction Rates")
ax20.legend(loc='upper right')
ax20.grid(True)

ax21.plot(time_min, out['r5'], label='r₅', color=colors['magenta'])
ax21.set_xlabel("Time (min)")
ax21.set_ylabel("Rate (mol/L/s)")
ax21.set_title("r₅: H⁺+OH⁻⇌H₂O")
ax21.legend(loc='upper right')
ax21.grid(True)

fig.tight_layout()
plt.show()


# ============================================================
# 6. SPECIES DISTRIBUTIONS (STACKED AREAS) + OPTIONAL pH OVERLAY
# ============================================================

# Calculate fractions for carbon species
total_C = out['CO2_aq'] + out['HCO3'] + out['CO3']
frac_CO2 = out['CO2_aq'] / (total_C + 1e-10)
frac_HCO3 = out['HCO3'] / (total_C + 1e-10)
frac_CO3 = out['CO3'] / (total_C + 1e-10)

# A. Stacked area + pH on second y-axis
fig, ax22 = plt.subplots(figsize=(8, 4))
ax22.stackplot(
    time_min,
    frac_CO2, frac_HCO3, frac_CO3,
    colors=[colors['darkgreen'], colors['cyan'], colors['purple']],
    labels=['CO₂(aq)', 'HCO₃⁻', 'CO₃²⁻']
)
ax22.set_xlabel("Time (min)")
ax22.set_ylabel("Molar Fraction")
ax22.legend(loc='upper left')
ax22.grid(True)

ax23 = ax22.twinx()
ax23.plot(time_min, out['pH'], color=colors['red'], label='pH', linewidth=1)
ax23.set_ylabel("pH", color=colors['red'])
ax23.tick_params(axis='y', labelcolor=colors['red'])
ax23.legend(loc='upper right')

fig.suptitle('Carbon Species Distribution & pH')
fig.tight_layout()
plt.show()

# B. Separate subplots: one for distribution, one for pH
fig, (ax24, ax25) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Top subplot: stacked fractions
ax24.stackplot(
    time_min,
    frac_CO2, frac_HCO3, frac_CO3,
    colors=[colors['darkgreen'], colors['cyan'], colors['purple']],
    labels=['CO₂(aq)', 'HCO₃⁻', 'CO₃²⁻']
)
ax24.set_ylabel("Molar Fraction")
ax24.set_title("Carbon Species Distribution")
ax24.legend(loc='upper right')
ax24.grid(True)

# Bottom subplot: pH alone
ax25.plot(time_min, out['pH'], color=colors['red'])
ax25.set_xlabel("Time (min)")
ax25.set_ylabel("pH")
ax25.set_title("pH Evolution")
ax25.grid(True)

fig.tight_layout()
plt.show()


# ========================================================
# 7. CALCULATE AND PLOT TIME-DEPENDENT DIFFUSION COEFFICIENT
# ========================================================

# D0_val and alpha_val from theta array
D0_val = theta[8]
alpha_val = theta[9]
Dt = D0_val * np.exp(-alpha_val * t)  # assuming t is in seconds; convert to minutes if needed

fig, ax26 = plt.subplots(figsize=(8, 4))
ax26.plot(time_min, Dt, color=colors['darkgreen'])
ax26.set_title('Time‐Dependent Diffusion Coefficient')
ax26.set_xlabel("Time (min)")
ax26.set_ylabel("Diffusion Coefficient (m²/s)")
ax26.grid(True)
fig.tight_layout()
plt.show()


# ========================================
# 8. OVERALL CARBONATION DEGREE
# ========================================

# Assuming tii['n0'] is initial Ca(OH)₂ and tii['CSH_fast0'] is initial CSH
carbonation_degree = 100.0 * out['CaCO3'] / (tii['n0'] + tii['CSH_fast0'])

fig, ax27 = plt.subplots(figsize=(8, 4))
ax27.plot(time_min, carbonation_degree, color=colors['orange'])
ax27.set_title('Overall Carbonation Degree')
ax27.set_xlabel("Time (min)")
ax27.set_ylabel("Carbonation Degree (%)")
ax27.grid(True)
fig.tight_layout()
plt.show()


# ========================================
# 9. CALCIUM SPECIES EVOLUTION (Separate axes)
# ========================================

# A. Twin y-axis for Ca²⁺ vs. CaCO₃
fig, ax28 = plt.subplots(figsize=(8, 4))
ax28.plot(time_min, out['CB'], color=colors['red'], label='Ca²⁺')
ax28.set_xlabel("Time (min)")
ax28.set_ylabel("Ca²⁺ (mol/L)", color=colors['red'])
ax28.tick_params(axis='y', labelcolor=colors['red'])

ax29 = ax28.twinx()
ax29.plot(time_min, out['CaCO3'], color=colors['green'], label='CaCO₃')
ax29.set_ylabel("CaCO₃ (mol/L)", color=colors['green'])
ax29.tick_params(axis='y', labelcolor=colors['green'])

fig.suptitle('Calcium Species Evolution')
fig.tight_layout()
plt.show()

# B. Two subplots
fig, (ax30, ax31) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax30.plot(time_min, out['CB'], color=colors['red'])
ax30.set_ylabel("Ca²⁺ (mol/L)")
ax30.set_title("Ca²⁺ Concentration")
ax30.grid(True)

ax31.plot(time_min, out['CaCO3'], color=colors['green'])
ax31.set_xlabel("Time (min)")
ax31.set_ylabel("CaCO₃ (mol/L)")
ax31.set_title("CaCO₃ (Total) Concentration")
ax31.grid(True)

fig.tight_layout()
plt.show()


# ================================================================
# 10. IONIC STRENGTH, OH⁻, CO₂(aq) (Separate subplots)
# ================================================================

# OH⁻ Concentration
fig, ax32 = plt.subplots(figsize=(8, 4))
ax32.plot(time_min, out['OH'], color=colors['cyan'])
ax32.set_title("OH⁻ Concentration")
ax32.set_xlabel("Time (min)")
ax32.set_ylabel("OH⁻ (mol/L)")
ax32.grid(True)
fig.tight_layout()
plt.show()

# CO₂(aq) Concentration
fig, ax33 = plt.subplots(figsize=(8, 4))
ax33.plot(time_min, out['CO2_aq'], color=colors['darkgreen'])
ax33.set_title("CO₂(aq) Concentration")
ax33.set_xlabel("Time (min)")
ax33.set_ylabel("CO₂(aq) (mol/L)")
ax33.grid(True)
fig.tight_layout()
plt.show()

# Ionic Strength
fig, ax34 = plt.subplots(figsize=(8, 4))
ax34.plot(time_min, out['I'], color=colors['darkblue'])
ax34.set_title("Ionic Strength Evolution")
ax34.set_xlabel("Time (min)")
ax34.set_ylabel("Ionic Strength (mol/L)")
ax34.grid(True)
fig.tight_layout()
plt.show()


# ================================================================
# 11. SINGLE-SPECIES CONCENTRATION PLOTS (for any remaining species)
# ================================================================

def plot_single(x, y, title, ylabel, color):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color=color, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time (min)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example calls (you can add more if needed):
plot_single(time_min, out['n'], 'Ca(OH)₂ Concentration', 'mol/L', colors['blue'])
plot_single(time_min, out['OH'], 'OH⁻ Concentration', 'mol/L', colors['cyan'])
plot_single(time_min, out['HCO3'], 'HCO₃⁻ Concentration', 'mol/L', colors['green'])
plot_single(time_min, out['CO3'], 'CO₃²⁻ Concentration', 'mol/L', colors['purple'])
plot_single(time_min, out['CaCO3'], 'CaCO₃ (Total) Concentration', 'mol/L', colors['orange'])
plot_single(time_min, out['CO2_aq'], 'CO₂(aq) Concentration', 'mol/L', colors['darkgreen'])
plot_single(time_min, out['pH'], 'pH Evolution', 'pH', colors['red'])
plot_single(time_min, out['I'], 'Ionic Strength Evolution', 'mol/L', colors['darkblue'])

# ================================================================
# 12. OTHER REACTION RATES (each separately)
# ================================================================
plot_single(time_min, out['r1'], 'r₁: CO₂+H₂O ⇌ HCO₃⁻+H⁺', 'mol/L/s', colors['blue'])
plot_single(time_min, out['r3'], 'r₃: HCO₃⁻ ⇌ CO₃²⁻+H⁺', 'mol/L/s', colors['green'])
plot_single(time_min, out['r4'], 'r₄: Ca²⁺+CO₃²⁻ → CaCO₃', 'mol/L/s', colors['darkblue'])
plot_single(time_min, out['r5'], 'r₅: H⁺+OH⁻ ⇌ H₂O', 'mol/L/s', colors['magenta'])
plot_single(time_min, out['r6'], 'r₆: CO₂+OH⁻ ⇌ HCO₃⁻', 'mol/L/s', colors['red'])
plot_single(time_min, out['r7'], 'r₇: HCO₃⁻+OH⁻ ⇌ CO₃²⁻+H₂O', 'mol/L/s', colors['purple'])
plot_single(time_min, out['r8'], 'r₈: CaCO₃ via CSH Path', 'mol/L/s', colors['orange'])


# ================================================================
# 13. ACTIVITY COEFFICIENTS OVER TIME
# ================================================================
fig, ax_g = plt.subplots(figsize=(8, 4))
ax_g.plot(time_min, out['gamma_Ca'],   label='γ₍Ca²⁺₎',   color=colors['blue'])
ax_g.plot(time_min, out['gamma_OH'],   label='γ₍OH⁻₎',    color=colors['cyan'])
ax_g.plot(time_min, out['gamma_H'],    label='γ₍H⁺₎',     color=colors['red'])
ax_g.plot(time_min, out['gamma_HCO3'], label='γ₍HCO₃⁻₎',  color=colors['green'])
ax_g.plot(time_min, out['gamma_CO3'],  label='γ₍CO₃²⁻₎', color=colors['purple'])
ax_g.set_xlabel("Time (min)")
ax_g.set_ylabel("Activity Coefficient")
ax_g.set_title("Activity Coefficients vs. Time")
ax_g.grid(True)
ax_g.legend(loc='best')
fig.tight_layout()
plt.show()


# ============================================================
# 14. CaCO₃ FORMATION PATHWAYS: CH vs CSH vs TOTAL
# ============================================================
fig, ax_cp = plt.subplots(figsize=(8, 4))
ax_cp.plot(time_min, out['CaCO3_CH'],  label='CaCO₃ from Ca(OH)₂',  color=colors['orange'])
ax_cp.plot(time_min, out['CaCO3_CSH'], label='CaCO₃ from CSH',      color=colors['blue'])
ax_cp.plot(time_min, out['CaCO3'],     linestyle='--', label='Total CaCO₃',  color=colors['red'])
ax_cp.set_xlabel("Time (min)")
ax_cp.set_ylabel("CaCO₃ (mol/L)")
ax_cp.set_title("CaCO₃ Formation Pathways")
ax_cp.grid(True)
ax_cp.legend(loc='best')
fig.tight_layout()
plt.show()