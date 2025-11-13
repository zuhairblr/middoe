import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

experimental_data_pH = np.array([
    [10, 8],
    [320, 11.16],
    [620, 11.17],
    [920, 10.63],
    [1220, 10.24],
    [1520, 9.75],
    [1820, 9.4],
    [2120, 9.1],
    [2420, 8.41],
    [2720, 7.29],
    [3020, 6.33],
    [3320, 6.24],
    [3620, 6.23],
])

pH_values = experimental_data_pH[:, 1]
H_concentration = 10 ** (-pH_values)
experimental_data_H = np.column_stack((experimental_data_pH[:, 0], H_concentration))
timeExp_C_proton_mol_l = experimental_data_H[:, 0]
C_proton_mol_lExp = experimental_data_H[:, 1]

import numpy as np

def try_integration_with_retries(t, tii, tvi, bounds, max_retries=1000, min_time=1000):
    rng = np.random.default_rng()
    for attempt in range(max_retries):
        # Generate a random theta inside bounds
        theta = np.array([rng.uniform(low, high) for (low, high) in bounds])
        print(f"Attempt {attempt +1}: Trying parameters = {theta}")

        sol = solve_modeli(t, None, tii, tvi, theta)
        if sol is None or 't' not in sol:
            print("Integration failed or returned None, retrying...")
            continue

        final_time = sol['t'][-1]
        print(f"Integration finished at time {final_time:.2f} s")

        if final_time >= min_time:
            print("Feasible solution found.")
            return theta, sol
        else:
            print("Integration stopped early, retrying...")

    print("No feasible solution found after max retries.")
    return None, None



def solve_modeli(t, y0, tii, tvi, theta):
    """
    Model simulating aqueous carbonation kinetics with proper particle-scale
    reaction-diffusion coupling following Equation 4a and 4d.

    n1 (solid phase) is uniform in space, only varies with time.
    All units included as comments next to variables/equations.
    Stoichiometry and fluxes explicitly balanced.
    """
    t = np.array(t, dtype=np.float64)
    theta = np.array(theta, dtype=np.float64)
    tii = {k: float(v) for k, v in tii.items()}
    T_profile = np.array(tvi['T'], dtype=np.float64)  # [K]
    P_profile = np.array(tvi['P'], dtype=np.float64)  # [bar]
    R_gas = 8.314  # Universal gas constant [J/(mol·K)]

    uncvar = {
        'CaO_leachable_solid_fr': 0.32061,      # [dimensionless] fraction
        'CaO_unleachable_solid_fr': 0.3434,     # [dimensionless] fraction
        'porosity': 0.3,                        # [dimensionless]
        'CaCO3_porosity': 0.3,                  # [dimensionless]
        'rho_solid_material': 2260,             # [kg/m³]
        'rho_solid_CaCO3': 2710,                # [kg/m³]
        'slurry_volume': 0.5,                   # [L]
        'LOI': 0.01903,                         # [dimensionless]
    }

    const = {
        'Mw_CaO': 56e-3,         # [kg/mol]
        'Mw_CaOH2': 74e-3,       # [kg/mol]
        'Mw_CaCO3': 100.1e-3,    # [kg/mol]
        'Mw_CO2': 44e-3,         # [kg/mol]
        'k_R4f': 1.4e11,         # [L/(mol·s)]
        'T_ref': 298.15,         # [K]
        'Keq_R1_ref': 10 ** (-6.357), # [dimensionless]
        'Keq_R2_ref': 10 ** (-10.33),
        'Keq_R3_ref': 10 ** (9.108),
        'Keq_R4_ref': 10 ** (14),
        'Keq_R5_ref': 10 ** (7.66),
        'deltaH_R1': -20000,     # [J/mol]
        'deltaH_R2': 14400,      # [J/mol]
        'deltaH_R3': -12100,     # [J/mol]
        'deltaH_R4': 55800,      # [J/mol]
        'deltaH_R5': -41400,     # [J/mol]
    }

    # Fixed initial conditions [mol/L]
    C_proton_mol_l = 1e-8      # [mol/L]
    C_OH_mol_l = 1e-6          # [mol/L] (from autoionization: Kw / [H+])

    dp = tii['dp']             # Particle diameter [m]
    SLR = tii['SLR']           # Slurry loading ratio [kg/L]

    CaO_leachable_solid_fr = uncvar['CaO_leachable_solid_fr']
    CaO_unleachable_solid_fr = uncvar['CaO_unleachable_solid_fr']
    LOI = uncvar['LOI']
    porosity = uncvar['porosity']
    rho_solid_material = uncvar['rho_solid_material'] * 1e-3  # [kg/L]
    slurry_volume = uncvar['slurry_volume']                  # [L]
    rho_solid_CaCO3 = uncvar['rho_solid_CaCO3'] * 1e-3       # [kg/L]
    CaCO3_porosity = uncvar['CaCO3_porosity']

    Mw_CaO = const['Mw_CaO']          # [kg/mol]
    Mw_CaOH2 = const['Mw_CaOH2']
    Mw_CaCO3 = const['Mw_CaCO3']
    Mw_CO2 = const['Mw_CO2']

    k_R4f = const['k_R4f']
    T_ref = const['T_ref']

    Keq_R1_ref = const['Keq_R1_ref']
    Keq_R2_ref = const['Keq_R2_ref']
    Keq_R3_ref = const['Keq_R3_ref']
    Keq_R4_ref = const['Keq_R4_ref']
    Keq_R5_ref = const['Keq_R5_ref']

    deltaH_R1 = const['deltaH_R1']
    deltaH_R2 = const['deltaH_R2']
    deltaH_R3 = const['deltaH_R3']
    deltaH_R4 = const['deltaH_R4']
    deltaH_R5 = const['deltaH_R5']

    # Particle geometry calculations
    molar_vol_CaCO3 = Mw_CaCO3 / rho_solid_CaCO3       # [L/mol]
    V_particle = (4 / 3) * np.pi * (dp / 2) ** 3 * 1000  # [L]
    V_solid_per_particle = (1 - porosity) * V_particle   # [L]
    mass_particle = rho_solid_material * V_solid_per_particle  # [kg]
    total_mass_kg = SLR * slurry_volume                      # [kg]
    dry_mass_kg = total_mass_kg * (1 - LOI)                  # [kg]
    N_particles = total_mass_kg / mass_particle              # [dimensionless]

    CH_solid_mol = total_mass_kg * CaO_leachable_solid_fr / Mw_CaO    # [mol]
    CSH_solid_mol = total_mass_kg * CaO_unleachable_solid_fr / Mw_CaO # [mol]

    # Ion parameters (for activity coefficients)
    aCa = 1.0e-13      # ion size [m]
    aOH = 3.0e-10
    aH = 9.0e-11
    aHCO3 = 4.0e-10
    aCO3 = 4.0e-10
    A_DH = 0.037       # Debye-Hückel A parameter [dimensionless]
    B_DH = 1.04e8      # Debye-Hückel B parameter [dimensionless]

    # Radial discretization [m]
    Nr = 30
    r_particle = dp / 2
    r_grid = np.linspace(0, r_particle, Nr)
    dr = r_grid[1] - r_grid[0]

    def vanthoff(K_eq_ref, deltaH, T):
        # Van't Hoff eq: K_eq(T) = K_eq_ref * exp(-ΔH/R [1/T - 1/Tref])
        return K_eq_ref * np.exp(-deltaH / R_gas * (1 / T - 1 / T_ref))

    def ksps(T):
        # Ca(OH)2 solubility, empirical [dimensionless]
        deltaH = -16730
        deltaS = -157.86
        lnKsp_T = (-deltaH / R_gas) * (1 / T) + (deltaS / R_gas)
        return np.exp(lnKsp_T)

    def T_at_time(tt):
        # Linear interpolation for temperature profile [K]
        return np.interp(tt, t, T_profile)

    def P_at_time(tt):
        # Linear interpolation for pressure profile [bar]
        return np.interp(tt, t, P_profile)

    def calcGamma(I, z, a_i, A, B):
        # Extended Debye-Hückel equation
        # I: ionic strength [mol/L], z: charge, a_i: ion size [m], A/B: empirical parameters
        Ip = I * 1e3  # convert [mol/L] → [mol/m³]
        lnGamma = (-A * z ** 2 * np.sqrt(Ip) / (1 + a_i * B * np.sqrt(Ip)) +
                   (0.2 - 4.17e-5 * Ip) * A * z ** 2 * Ip / np.sqrt(1000))
        gamma = np.exp(lnGamma)
        return np.clip(gamma, 0.01, 10.0)  # Activity coefficient bounds [dimensionless]

    def kspCaCO3(T):
        # CaCO3 solubility [mol^2/L^2], empirical
        return 10 ** (-(0.01183 * (T - 273.15) + 8.03))

    # Initial profiles and bulk variables [mol/L]
    Ca_profile_init = np.full(Nr, 1e-30)         # Ca2+ initial profile
    OH_profile_init = np.full(Nr, C_OH_mol_l)    # OH- initial profile

    # Initial scalar variable vector [mol/L]; indices indicated in comments
    y0 = np.concatenate([
        Ca_profile_init,          # 0:Nr [mol/L]
        OH_profile_init,          # Nr:2*Nr [mol/L]
        [1e-30,                   # 2*Nr: C_CH_bulk_mol_l [mol/L]
         CH_solid_mol,            # n1_total [mol] (solid phase moles)
         C_OH_mol_l,              # C_OH_bulk_mol_l [mol/L]
         1e-30,                   # C_bicarbonate_mol_l [mol/L]
         C_proton_mol_l,          # C_proton_mol_l [mol/L]
         1e-30,                   # C_carbonate_mol_l [mol/L]
         1e-30,                   # C_CaCO3_precipitated_mol_l [mol/L]
         CSH_solid_mol,           # M_CSH_solid_mol [mol]
         1e-30,                   # C_CaCO3_shrinked_mol_l [mol/L]
         1e-30,                   # C_CO2_aq_mol_l [mol/L]
         1e-30,                   # alpha [kg_CO2/kg_dry]
         dp,                      # particle_diameter [m]
         dp]                      # core_diameter [m]
    ])

    def odes(tt, y):
        # Extract spatial profiles [mol/L]
        Ca_profile = y[0:Nr].copy()
        OH_profile = y[Nr:2 * Nr].copy()

        # Extract scalar bulk variables
        idx = 2 * Nr
        (C_CH_bulk_mol_l, n1_total, C_OH_bulk_mol_l,
         C_bicarbonate_mol_l, C_proton_mol_l, C_carbonate_mol_l,
         C_CaCO3_precipitated_mol_l, M_CSH_solid_mol, C_CaCO3_shrinked_mol_l,
         C_CO2_aq_mol_l, alpha, particle_diameter, core_diameter) = y[idx:]

        # Bound all scalar variables by physical values
        scalar_vars = [C_CH_bulk_mol_l, n1_total, C_OH_bulk_mol_l,
                       C_bicarbonate_mol_l, C_proton_mol_l, C_carbonate_mol_l,
                       C_CaCO3_precipitated_mol_l, M_CSH_solid_mol, C_CaCO3_shrinked_mol_l,
                       C_CO2_aq_mol_l]
        scalar_vars = [np.clip(v, 1e-30, 10) for v in scalar_vars]
        (C_CH_bulk_mol_l, n1_total, C_OH_bulk_mol_l,
         C_bicarbonate_mol_l, C_proton_mol_l, C_carbonate_mol_l,
         C_CaCO3_precipitated_mol_l, M_CSH_solid_mol, C_CaCO3_shrinked_mol_l,
         C_CO2_aq_mol_l) = scalar_vars

        Ca_profile = np.clip(Ca_profile, 1e-30, 10)
        OH_profile = np.clip(OH_profile, 1e-30, 10)
        particle_diameter = np.clip(particle_diameter, dp, dp * 2)
        core_diameter = np.clip(core_diameter, 0, particle_diameter)

        T_loc = T_at_time(tt)     # [K]
        P_loc = P_at_time(tt)     # [bar]

        # Rate constants [user input or from theta]
        k_R1f = theta[0]         # [L/(mol·s)]
        k_R2f = theta[1]
        k_R5f = theta[2]
        k_R3f = theta[3]
        kL = theta[4]            # Ca(OH)2 dissolution [1/s]
        D = theta[5]             # Diffusion coefficient [m²/s]
        kLa = theta[6]           # CO2 transfer [1/s]

        # Equilibrium constants @ temperature [dimensionless]
        Keq_R1 = vanthoff(Keq_R1_ref, deltaH_R1, T_loc)
        Keq_R2 = vanthoff(Keq_R2_ref, deltaH_R2, T_loc)
        Keq_R3 = vanthoff(Keq_R3_ref, deltaH_R3, T_loc)
        Keq_R4 = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
        Keq_R5 = vanthoff(Keq_R5_ref, deltaH_R5, T_loc)
        Ksp_CaCO3 = kspCaCO3(T_loc)           # [mol²/L²]
        Ksp_CaOH2 = ksps(T_loc)               # [mol³/L³]

        # Ionic strength:
        # I [mol/L] = 1/2 Σ c_i z_i^2 = 1/2(4*Ca + 1*OH + 1*HCO3 + 1*H + 4*CO3)
        I = 0.5 * (4 * C_CH_bulk_mol_l + 1 * C_OH_bulk_mol_l +
                   1 * C_bicarbonate_mol_l + 1 * C_proton_mol_l +
                   4 * C_carbonate_mol_l)        # [mol/L]

        # Activity coefficients
        gamma_Ca = calcGamma(I, 2, aCa, A_DH, B_DH)
        gamma_OH = calcGamma(I, 1, aOH, A_DH, B_DH)
        gamma_CO3 = calcGamma(I, 2, aCO3, A_DH, B_DH)
        gamma_H = calcGamma(I, 1, aH, A_DH, B_DH)
        gamma_HCO3 = calcGamma(I, 1, aHCO3, A_DH, B_DH)

        c_std = 1  # Standard concentration for consistency

        # Bulk reactions [mol/(L·s)]
        # R1: CO₂ hydration
        r_R1 = k_R1f * C_CO2_aq_mol_l - (k_R1f / Keq_R1) * ((C_bicarbonate_mol_l / c_std) * gamma_HCO3) * \
                    ((C_proton_mol_l / c_std) * gamma_H)
        # R2: Bicarbonate dissociation
        r_R2 = k_R2f * ((C_bicarbonate_mol_l / c_std) * gamma_HCO3) - (k_R2f / Keq_R2) * \
                    ((C_carbonate_mol_l / c_std) * gamma_CO3) * ((C_proton_mol_l / c_std) * gamma_H)
        # R3: CaCO₃ precipitation (supersaturation term, Eq. S = [Ca][CO3]/Ksp)
        supersat_ratio = ((C_CH_bulk_mol_l / c_std) * gamma_Ca * (C_carbonate_mol_l / c_std) * gamma_CO3) / Ksp_CaCO3
        supersat_term = max(0, min(supersat_ratio - 1, 1e6))
        r_R3 = k_R3f * supersat_term
        # R4: Water autoprotolysis
        r_R4 = k_R4f * ((C_proton_mol_l / c_std) * gamma_H) * ((C_OH_bulk_mol_l / c_std) * gamma_OH)
        # R5: Direct carbonate formation
        r_R5 = k_R5f * C_CO2_aq_mol_l * ((C_OH_bulk_mol_l / c_std) * gamma_OH) - (k_R5f / Keq_R5) * \
                    ((C_bicarbonate_mol_l / c_std) * gamma_HCO3)

        # **SURFACE BOUNDARY CONDITIONS**
        Ca_profile[-1] = C_CH_bulk_mol_l
        OH_profile[-1] = C_OH_bulk_mol_l

        # Uniform solid dissolution [mol/s]
        S_surface = ((Ca_profile[-1] * gamma_Ca) * ((OH_profile[-1] * gamma_OH) ** 2)) / Ksp_CaOH2
        if S_surface < 1 and n1_total > 1e-6:
            dn1_dt = -kL * (1 - S_surface) * n1_total
            dn1_dt = max(dn1_dt, -n1_total / 10.0)  # Limit max dissolution
        else:
            dn1_dt = 0

        # Volumetric molar rate [mol/(L·s)]
        V_particle_current = (4 / 3) * np.pi * (particle_diameter / 2) ** 3 * 1e3  # [L per particle]
        V_particles_total = V_particle_current * N_particles  # [L total particles]
        V_pore_total = porosity * V_particles_total  # [L pore liquid inside particles]

        if V_pore_total > 1e-15:
            dn1_dt_pore = dn1_dt / V_pore_total  # [mol/L_pore/s]
        else:
            dn1_dt_pore = 0.0

        # **PDE for radial Ca2+/OH- diffusion + reaction**
        dCa_dt = np.zeros(Nr)
        dOH_dt = np.zeros(Nr)

        # for i in range(Nr):
        #     if i == 0:
        #         # Center node: symmetry
        #         d2C_dr2_Ca = 2 * (Ca_profile[1] - Ca_profile[0]) / dr ** 2
        #         d2C_dr2_OH = 2 * (OH_profile[1] - OH_profile[0]) / dr ** 2
        #         diffusion_term_Ca = D * d2C_dr2_Ca                      # [mol/(L·s)]
        #         diffusion_term_OH = D * d2C_dr2_OH
        #         # Dissolution term [mol/(L·s)]; factor from solid/void ratio
        #         dCa_dt[i] = diffusion_term_Ca - ((1 - porosity) / porosity) * dn1_dt_volumetric
        #         dOH_dt[i] = diffusion_term_OH - ((1 - porosity) / porosity) * 2 * dn1_dt_volumetric  # Ca(OH)2: 2 OH⁻/mol
        #     elif i == Nr - 1:
        #         # Surface: strongly coupled to bulk [mol/(L·s)]
        #         kmt = 1e5
        #         dCa_dt[i] = kmt * (C_CH_bulk_mol_l - Ca_profile[i])
        #         dOH_dt[i] = kmt * (C_OH_bulk_mol_l - OH_profile[i])
        #     else:
        #         r_i = r_grid[i]
        #         if r_i > 1e-15:
        #             dC_dr_Ca = (Ca_profile[i + 1] - Ca_profile[i - 1]) / (2 * dr)
        #             d2C_dr2_Ca = (Ca_profile[i + 1] - 2 * Ca_profile[i] + Ca_profile[i - 1]) / (dr ** 2)
        #             diffusion_term_Ca = D * (d2C_dr2_Ca + (2 / r_i) * dC_dr_Ca)
        #
        #             dC_dr_OH = (OH_profile[i + 1] - OH_profile[i - 1]) / (2 * dr)
        #             d2C_dr2_OH = (OH_profile[i + 1] - 2 * OH_profile[i] + OH_profile[i - 1]) / (dr ** 2)
        #             diffusion_term_OH = D * (d2C_dr2_OH + (2 / r_i) * dC_dr_OH)
        #
        #             dCa_dt[i] = diffusion_term_Ca - ((1 - porosity) / porosity) * dn1_dt_volumetric
        #             dOH_dt[i] = diffusion_term_OH - ((1 - porosity) / porosity) * 2 * dn1_dt_volumetric
        #         else:
        #             dCa_dt[i] = dCa_dt[0]
        #             dOH_dt[i] = dOH_dt[0]
        #
        # # **Bulk concentration evolution from flux (Eq. 4d)**
        # dC_dr_Ca_surface = (Ca_profile[-1] - Ca_profile[-2]) / dr
        # dC_dr_OH_surface = (OH_profile[-1] - OH_profile[-2]) / dr
        #
        # # Flux [mol/(L·s)]
        # flux_term_Ca = -4 * np.pi * (r_particle ** 2) * porosity * D * dC_dr_Ca_surface * N_particles / (slurry_volume * 1e-3)
        # flux_term_OH = -4 * np.pi * (r_particle ** 2) * porosity * D * dC_dr_OH_surface * N_particles / (slurry_volume * 1e-3)
        #
        # flux_term_Ca = np.clip(flux_term_Ca, -1e6, 1e6)
        # flux_term_OH = np.clip(flux_term_OH, -1e6, 1e6)
        #
        # dC_CH_bulk_mol_l_dt = flux_term_Ca - r_R3
        # dC_OH_bulk_mol_l_dt = flux_term_OH - r_R4 - r_R5



        # Constants for Stokes-Einstein scaling
        r_Ca = 6.0e-10  # m (hydrated radius of Ca2+)
        r_OH = 3.0e-10  # m (hydrated radius of OH-)

        # Calculate diffusion coefficient for OH- based on Ca2+ diffusion (D)
        D_OH = D * (r_Ca / r_OH)  # Scale diffusion coefficient inversely by radius

        for i in range(Nr):
            if i == 0:
                d2C_dr2_Ca = 2 * (Ca_profile[1] - Ca_profile[0]) / dr ** 2
                d2C_dr2_OH = 2 * (OH_profile[1] - OH_profile[0]) / dr ** 2

                diffusion_term_Ca = D * d2C_dr2_Ca  # [mol/(L·s)]
                diffusion_term_OH = D_OH * d2C_dr2_OH  # Use separate D for OH⁻

                dCa_dt[i] = diffusion_term_Ca - ((1 - porosity) / porosity) * dn1_dt_pore

                dOH_dt[i] = diffusion_term_OH -  2 * dn1_dt_pore
  # 2 for OH⁻ stoichiometry

            elif i == Nr - 1:
                kmt = 1e5
                dCa_dt[i] = kmt * (C_CH_bulk_mol_l - Ca_profile[i])
                dOH_dt[i] = kmt * (C_OH_bulk_mol_l - OH_profile[i])

            else:
                r_i = r_grid[i]
                if r_i > 1e-15:
                    dC_dr_Ca = (Ca_profile[i + 1] - Ca_profile[i - 1]) / (2 * dr)
                    d2C_dr2_Ca = (Ca_profile[i + 1] - 2 * Ca_profile[i] + Ca_profile[i - 1]) / (dr ** 2)
                    diffusion_term_Ca = D * (d2C_dr2_Ca + (2 / r_i) * dC_dr_Ca)

                    dC_dr_OH = (OH_profile[i + 1] - OH_profile[i - 1]) / (2 * dr)
                    d2C_dr2_OH = (OH_profile[i + 1] - 2 * OH_profile[i] + OH_profile[i - 1]) / (dr ** 2)
                    diffusion_term_OH = D_OH * (d2C_dr2_OH + (2 / r_i) * dC_dr_OH)  # Use separate D

                    dCa_dt[i] = diffusion_term_Ca -  dn1_dt_pore

                    dOH_dt[i] = diffusion_term_OH -  2 * dn1_dt_pore

                else:
                    dCa_dt[i] = dCa_dt[0]
                    dOH_dt[i] = dOH_dt[0]

        # **Bulk concentration evolution from flux (Eq. 4d)**
        dC_dr_Ca_surface = (Ca_profile[-1] - Ca_profile[-2]) / dr
        dC_dr_OH_surface = (OH_profile[-1] - OH_profile[-2]) / dr

        # Bulk flux terms likewise updated:
        flux_term_Ca = -4 * np.pi * (r_particle ** 2) * porosity * D * dC_dr_Ca_surface * N_particles / (
                    slurry_volume * 1e-3)
        flux_term_OH = -4 * np.pi * (r_particle ** 2) * porosity * D_OH * dC_dr_OH_surface * N_particles / (
                    slurry_volume * 1e-3)

        flux_term_Ca = np.clip(flux_term_Ca, -1e6, 1e6)
        flux_term_OH = np.clip(flux_term_OH, -1e6, 1e6)

        dC_CH_bulk_mol_l_dt = flux_term_Ca - r_R3
        dC_OH_bulk_mol_l_dt = flux_term_OH - r_R4 - r_R5


        # Other bulk variables [mol/L·s]
        dC_bicarbonate_mol_l_dt = r_R1 + r_R5 - r_R2   # [mol/L·s]
        dC_proton_mol_l_dt = r_R1 + r_R2 - r_R4
        dC_carbonate_mol_l_dt = r_R2 - r_R3
        dC_CaCO3_precipitated_mol_l_dt = r_R3

        # Shrinking core model (if n1 solid depleted)
        dM_CSH_solid_mol_dt = 0.0
        dcore_dt = 0.0

        if n1_total < 1e-4:
            delta = max((particle_diameter / 2) - (core_diameter / 2), 1e-12) # [m]
            if np.isfinite(delta) and delta > 0:
                dM_CSH_solid_mol_dt = -(4 * N_particles * np.pi * (particle_diameter / 2) * (core_diameter / 2) *
                    C_CO2_aq_mol_l * 1e3 * D) / delta    # [mol/s]
                dM_CSH_solid_mol_dt = np.clip(dM_CSH_solid_mol_dt, -1e6, 0)
                term = 1 - (M_CSH_solid_mol * (Mw_CO2 / Mw_CaO)) / dry_mass_kg
                term = max(term, 1e-12)
                if np.isfinite(dM_CSH_solid_mol_dt):
                    dcore_dt = (2 / 3) * (dp / 2) * term ** (2 / 3) * (
                        (dM_CSH_solid_mol_dt * (Mw_CO2 / Mw_CaO)) / dry_mass_kg)
        dC_CaCO3_shrinked_mol_l_dt = -dM_CSH_solid_mol_dt / slurry_volume  # [mol/L·s]

        # Particle growth by CaCO3 precipitation [m/s]
        if C_CaCO3_precipitated_mol_l > 1e-2:
            dV_deposit_dt = dC_CaCO3_precipitated_mol_l_dt * slurry_volume * molar_vol_CaCO3 / ((1 - CaCO3_porosity) * N_particles)
            dR_dt = (dV_deposit_dt * 1e-3) / (4 * np.pi * (particle_diameter / 2) ** 2) # [m/s]
            dR_dt = np.clip(dR_dt, -1e-10, 1e-10)
        else:
            dR_dt = 0.0
        dparticle_dt = 2 * dR_dt            # [m/s]

        # CO₂ exchange (Henry's Law saturation) [mol/L·s]
        CO2_sat = P_loc * 35 * np.exp(2400 * ((1 / T_loc) - (1 / T_ref))) * 1e-3
        dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5

        # Carbonation extent [kg CO2/kg dry]
        dalpha_dt = ((dC_CaCO3_precipitated_mol_l_dt + dC_CaCO3_shrinked_mol_l_dt) * slurry_volume *
                     (Mw_CO2 / Mw_CaCO3)) / dry_mass_kg

        # Assemble derivatives
        dy = np.zeros_like(y)
        dy[0:Nr]       = dCa_dt
        dy[Nr:2*Nr]    = dOH_dt
        dy[2*Nr]       = dC_CH_bulk_mol_l_dt
        dy[2*Nr + 1]   = dn1_dt
        dy[2*Nr + 2]   = dC_OH_bulk_mol_l_dt
        dy[2*Nr + 3]   = dC_bicarbonate_mol_l_dt
        dy[2*Nr + 4]   = dC_proton_mol_l_dt
        dy[2*Nr + 5]   = dC_carbonate_mol_l_dt
        dy[2*Nr + 6]   = dC_CaCO3_precipitated_mol_l_dt
        dy[2*Nr + 7]   = dM_CSH_solid_mol_dt
        dy[2*Nr + 8]   = dC_CaCO3_shrinked_mol_l_dt
        dy[2*Nr + 9]   = dCO2_aq_dt
        dy[2*Nr + 10]  = dalpha_dt
        dy[2*Nr + 11]  = dparticle_dt
        dy[2*Nr + 12]  = dcore_dt

        # NaN/Inf safety
        if not np.all(np.isfinite(dy)):
            print(f"Warning: Non-finite derivatives at t={tt:.2f}")
            dy = np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

        return dy

    sol = solve_ivp(
        fun=odes,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='BDF',
        rtol=1e-5,
        atol=1e-8,
        max_step=100
    )

    print("Integrator success:", sol.success)
    print("Integrator message:", sol.message)
    print(f'Initial CH_solid_mol={CH_solid_mol:.3e} [mol]')

    y_output = np.zeros((15, len(sol.t)))
    y_output[0]  = sol.y[Nr - 1, :]        # Ca2+ surface [mol/L]
    y_output[1]  = sol.y[2 * Nr, :]        # Ca2+ bulk [mol/L]
    y_output[2]  = sol.y[2 * Nr + 1, :]    # n1_total (solid) [mol]
    y_output[3]  = sol.y[2 * Nr - 1, :]    # OH- surface [mol/L]
    y_output[4]  = sol.y[2 * Nr + 2, :]    # OH- bulk [mol/L]
    y_output[5]  = sol.y[2 * Nr + 3, :]    # HCO3- bulk [mol/L]
    y_output[6]  = sol.y[2 * Nr + 4, :]    # H+ bulk [mol/L]
    y_output[7]  = sol.y[2 * Nr + 5, :]    # CO3^2- bulk [mol/L]
    y_output[8]  = sol.y[2 * Nr + 6, :]    # CaCO3 precipitated [mol/L]
    y_output[9]  = sol.y[2 * Nr + 7, :]    # CSH solid [mol]
    y_output[10] = sol.y[2 * Nr + 8, :]    # CaCO3 shrinked [mol/L]
    y_output[11] = sol.y[2 * Nr + 9, :]    # CO2(aq) [mol/L]
    y_output[12] = sol.y[2 * Nr + 10, :]   # alpha [kg CO2/kg dry]
    y_output[13] = sol.y[2 * Nr + 11, :]   # particle diameter [m]
    y_output[14] = sol.y[2 * Nr + 12, :]   # core diameter [m]

    return {
        't': sol.t,
        'y': y_output,
        'particle_diameter': y_output[13],
        'core_diameter': y_output[14],
        'sol_full': sol,
        'Nr': Nr,
    }


# Initialize counter outside of the function
iteration_counter = 0


def scaled_objective_theta(theta, t, tii, tvi,
                           exp_times_C_proton, exp_values_C_proton,
                           bounds):
    global iteration_counter
    iteration_counter += 1

    # Print iteration number and theta
    print(f"Iteration {iteration_counter}: theta = {theta}")

    import numpy as np
    penalty = 1e8

    # Bounds check
    theta = np.asarray(theta)
    bounds_arr = np.array(bounds)
    if np.any(theta < bounds_arr[:, 0]) or np.any(theta > bounds_arr[:, 1]):
        return penalty

    try:
        result = solve_modeli(t, None, tii, tvi, theta)
        if result is None or 't' not in result or 'y' not in result:
            return penalty

        model_times = result['t']
        y_out = result['y']

        if not isinstance(y_out, np.ndarray) or y_out.shape[0] < 7:
            return penalty

        model_vals = np.interp(exp_times_C_proton, model_times, y_out[6])
        model_vals = np.clip(model_vals, 0, None)

        if np.any(np.isnan(model_vals)) or np.any(np.isinf(model_vals)):
            return penalty

        ss_res = np.sum((exp_values_C_proton - model_vals) ** 2)
        ss_tot = np.sum((exp_values_C_proton - np.mean(exp_values_C_proton)) ** 2)
        # Edge case
        if ss_tot == 0:
            return penalty

        r2 = 1 - ss_res / ss_tot
        obj = 1 - r2
        if obj < 0:
            obj = 0.0

        print(f"Current 1 - R2: {obj:.3e}")
        return obj

    except ValueError as e:
        print(f"Integration failed for theta={theta} due to: {e}")
        # Return large penalty to signal optimizer this parameter set is invalid
        return penalty

    except Exception as e:
        print(f"Exception in objective: {e}")
        return penalty


def run_simple_estimation():
    # bounds = [
    #     (1e-8, 1e6), (1e-8, 1e10), (1e-8, 1e10), (1e-8, 1e10),
    #     (1e-8, 1e4), (1e-8, 1e6), (1e-8, 1e8), (1e-8, 1e6),
    #
    #     (10000, 80000), (10000, 80000), (10000, 80000), (10000, 80000),
    #     (10000, 80000), (10000, 80000), (10000, 80000), (10000, 80000),
    # ]
    bounds = [
        (1e4, 3e4),  # for k_R1f
        (1e2, 1e3),  # for k_R2f
        (1, 50),  # for k_R5f
        (1, 10),  # for k_R3f
        (9.63e-23, 1.77e10),  # for kL
        (9.63e-23, 1.77e10),  # for D
        (9.63e-23, 1.77e10)  # for kLa
    ]

    t = np.linspace(0, 3620, 1000)
    T_profile = np.full_like(t, 298.15)
    P_profile = np.full_like(t, 1.01325)
    tii = {'SLR': 0.1, 'dp': 10e-6, 't_CO2_inject': 0.0}
    tvi = {'T': T_profile, 'P': P_profile}

    # # x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
    # x0 = [
    #     1.205e4, 4.813e7, 1.904e8, 3.340e5,
    #     9.158e1, 4.694e2, 7.059e-4, 9.585e1,
    #     36715, 15807, 39536, 47039,
    #     38801, 32132, 87550, 20037
    # ]

    x0 = [
        1e4,         # k_R1f: used in r_R1 = k_R1f * C_CO2_aq - (k_R1f / Keq_R1) * (C_carbonate * gamma_HCO3) * (C_proton * gamma_H) (CO2 hydration/dissociation rate)
        2e4,       # k_R2f: used in r_R2 = k_R2f * (C_carbonate * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate * gamma_CO3) * (C_proton * gamma_H) (Bicarbonate dissociation rate)
        1e4,           # k_R5f: used in r_R5 = k_R5f * C_CO2_aq * (C_OH_bulk * gamma_OH) - (k_R5f / Keq_R5) * (C_carbonate * gamma_HCO3) (Hydroxycarbonate formation rate)
        1e7,        # k_R3f: used in r_R3 = k_R3f * (((C_CH_bulk * gamma_Ca) * (C_bicarbonate * gamma_CO3)) / Ksp_CaCO3 - 1) (Calcite precipitation rate)
        0.014,         # kL: used in dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol (Ca(OH)2 dissolution rate)
        3.23e-13,        # D: used in dM_CSH_solid_mol_dt = -(4 * N_particles * pi * ...) * C_CO2_aq_mol_l * D / delta (Diffusion parameter in shrinking core)
        0.0266           # kLa: used in dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5 (Gas-liquid CO2 mass transfer rate)
    ]

    # from scipy.optimize import minimize
    #
    # res = minimize(
    #     scaled_objective_theta,
    #     x0,
    #     args=(t, tii, tvi, timeExp_C_proton_mol_l, C_proton_mol_lExp, bounds),
    #     method='trust-constr',
    #     bounds=bounds,
    #     options={'maxiter': 1000, 'disp': True}
    # )

    # # Try multiple initial guesses to get a feasible starting point
    # theta_init, sol_init = try_integration_with_retries(t, tii, tvi, bounds, max_retries=20000, min_time=300)
    #
    # if theta_init is None:
    #     print("Failed to find feasible initial parameters for optimization.")
    #     return



    from scipy.optimize import differential_evolution

    # res = differential_evolution(
    #     scaled_objective_theta,
    #     bounds=bounds,
    #     args=(t, tii, tvi, timeExp_C_proton_mol_l, C_proton_mol_lExp, bounds),
    #     maxiter=10000,
    #     disp=True,
    #     polish=False,  # optional: polish final solution by local optimization
    #     tol=1e-30
    # )


    res = minimize(
        scaled_objective_theta,
        x0,
        args=(t, tii, tvi, timeExp_C_proton_mol_l, C_proton_mol_lExp, bounds),
        method='trust-constr',
        bounds=bounds,
        options={'maxiter': 1000000000, 'disp': True}
    )


    theta_opt = res.x
    if res.success:
        print("Optimized parameters (original scale):\n", res.x)
        print("Final objective error:", res.fun)
        result = solve_modeli(t, None, tii, tvi, res.x)
        if result is not None:
            plot_results(result['t'], result['y'], res.x)
        else:
            print("Integration failed after optimization.")
    else:
        print("Optimization failed or terminated early:", res.message)

    result = solve_modeli(t, None, tii, tvi, theta_opt)
    if result is not None:
        plot_results(result['t'], result['y'], theta_opt)
    else:
        print("Integration failed after optimization.")





import numpy as np
from scipy.optimize import minimize
import concurrent.futures

def generate_random_initial_guesses(bounds, n_guesses):
    bounds_arr = np.array(bounds)
    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    rng = np.random.default_rng()
    initial_guesses = rng.uniform(low=low, high=high, size=(n_guesses, len(bounds)))
    return initial_guesses.tolist()

# def run_simple_estimation():
#     # bounds = [
#     #     (9.63e-23, 1.77e4),  # for k_R1f
#     #     (9.63e-23, 1.77e8),  # for k_R2f
#     #     (9.63e-23, 1.77e8),  # for k_R5f
#     #     (9.63e-23, 1.77e8),  # for k_R3f
#     #     (9.63e-23, 1.77e2),  # for kL
#     #     (9.63e-23, 1.77e4),  # for kO
#     #     (9.63e-23, 1.77e6),  # for D
#     #     (9.63e-23, 1.77e4)  # for kLa
#     # ]
#     bounds = [
#         (1e4, 3e4),  # for k_R1f
#         (1e2, 1e3),  # for k_R2f
#         (1, 50),  # for k_R5f
#         (1, 10),  # for k_R3f
#         (9.63e-23, 1.77e2),  # for kL
#         (9.63e-23, 1.77e4),  # for kO
#         (9.63e-23, 1.77e6),  # for D
#         (9.63e-23, 1.77e4)  # for kLa
#     ]
#
#     t = np.linspace(0, 3600, 300)
#     T_profile = np.full_like(t, 298.15)
#     P_profile = np.full_like(t, 1.01325)
#     tii = {'SLR': 0.1, 'dp': 10e-6, 't_CO2_inject': 0.0}
#     tvi = {'T': T_profile, 'P': P_profile}
#
#     # Generate 10 random initial guesses inside bounds
#     initial_guesses = generate_random_initial_guesses(bounds, n_guesses=10)
#
#     def optimize_from_x0(x0):
#         res = minimize(
#             scaled_objective_theta,
#             x0,
#             args=(t, tii, tvi, timeExp_C_proton_mol_l, C_proton_mol_lExp, bounds),
#             method='trust-constr',
#             bounds=bounds,
#             options={'maxiter': 1000, 'disp': False}
#         )
#         return res
#
#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(optimize_from_x0, x0) for x0 in initial_guesses]
#         for future in concurrent.futures.as_completed(futures):
#             res = future.result()
#             results.append(res)
#
#     # Filter only successful converged results
#     converged = [r for r in results if r.success]
#
#     if not converged:
#         print("No optimization run converged.")
#         return
#
#     best_res = min(converged, key=lambda r: r.fun)
#
#     print("Best optimized parameters (original scale):\n", best_res.x)
#     print("Final objective error:", best_res.fun)
#
#     result = solve_modeli(t, None, tii, tvi, best_res.x)
#     if result is not None:
#         plot_results(result['t'], result['y'], best_res.x)
#     else:
#         print("Integration failed after optimization.")



def plot_results(t_out, y_out, theta):
    t_min = t_out / 60
    colors = {
        'pH': '#d62728',
        'exp_pH': '#ff7f0e'
    }

    # Model pH calculation from proton concentration (mol/L to mol/m³)
    pH_model = -np.log10(np.maximum(y_out[6], 1e-14))
    pH_exp = -np.log10(np.maximum(C_proton_mol_lExp, 1e-14))

    plt.figure(figsize=(10, 6))
    plt.plot(t_min, pH_model, color=colors['pH'], label='Model pH')
    plt.scatter(timeExp_C_proton_mol_l / 60, pH_exp, color=colors['exp_pH'], marker='x', label='Experimental pH', s=80)
    plt.xlabel('Time (min)')
    plt.ylabel('pH')
    plt.title('pH Evolution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(3, 12)
    plt.show()

if __name__ == "__main__":
    run_simple_estimation()