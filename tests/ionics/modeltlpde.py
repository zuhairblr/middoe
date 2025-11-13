import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Global constants and reaction numbering
# -----------------------------

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


import numpy as np
from scipy.integrate import solve_ivp

def solve_model(t, y0, tii, tvi, theta):
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
        Ds = theta[7]            # shrinking core D [m²/s]

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
                    C_CO2_aq_mol_l * 1e3 * Ds) / delta    # [mol/s]
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
        rtol=1e-8,
        atol=1e-11,
        # max_step=100
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



experimental_data_pH = np.array([
    [0, 8],
    [20, 11.53],
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
timeExp_pH = experimental_data_pH[:, 0]
pHExp = experimental_data_pH[:, 1]

experimental_data_CaCO3 = np.array([
    [0, 0],
    [320, 0.207972],
    [620, 0.348255],
    [1820, 0.530721],
    [3620, 0.595467],
])
timeExp_CaCO3 = experimental_data_CaCO3[:, 0]
CaCO3Exp = experimental_data_CaCO3[:, 1]

# experimental_data_others = np.array([
#     [0, 0.430844595],
#     [300, 0.291648649],
#     [600, 0.046398649],
#     [1800, 0],
# ])

experimental_data_others = np.array([
    [0, 0.28521292],
    [320, 0.19306721],
    [620, 0.03071524],
    [1820, 0.0],
])

timeExp_others = experimental_data_others[:, 0]
nExp = experimental_data_others[:, 1]

colors = {
    'CaOH2': '#1f77b4',
    'pH': '#d62728',
    'CaCO3': '#2ca02c',
    'exp_pH': '#ff7f0e',
    'exp_CaCO3': '#9467bd'
}


def run_and_plot():
    t = np.linspace(0,3620,100)
    T_profile = np.full_like(t,298.15)
    P_profile = np.full_like(t,1.01325)
    tii = {
        'SLR': 0.1,               # Solid-to-liquid ratio (kg/L)
        'dp': 10e-6,              # Diameter of solid particles (m)
        't_CO2_inject': 0.0       # Time when CO2 injection starts in the system (s), allows for switching CO2 source on/off in model
    }

    # theta = [
    #     1.205e4, 4.813e7, 1.904e8, 3.340e5,
    #     5.458e-7, 4.694e2, 7.059e10, 9.585e1,
    #     36715, 15807, 39536, 47039,
    #     35801, 32132, 27550, 20037,
    #     1
    # ]

    # theta = [
    #     1.205e4, 4.813e7, 1.904e8, 3.340e5,
    #     3.458e+3, 4.694e2, 7.059e8, 9.585e1,
    #     36715, 15807, 39536, 47039,
    #     28801, 32132, 27550, 20037
    # ]

    # theta = [
    #     1e4,         # k_R1f: used in r_R1 = k_R1f * C_CO2_aq - (k_R1f / Keq_R1) * (C_carbonate * gamma_HCO3) * (C_proton * gamma_H) (CO2 hydration/dissociation rate)
    #     2e4,       # k_R2f: used in r_R2 = k_R2f * (C_carbonate * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate * gamma_CO3) * (C_proton * gamma_H) (Bicarbonate dissociation rate)
    #     1e4,           # k_R5f: used in r_R5 = k_R5f * C_CO2_aq * (C_OH_bulk * gamma_OH) - (k_R5f / Keq_R5) * (C_carbonate * gamma_HCO3) (Hydroxycarbonate formation rate)
    #     1e7,        # k_R3f: used in r_R3 = k_R3f * (((C_CH_bulk * gamma_Ca) * (C_bicarbonate * gamma_CO3)) / Ksp_CaCO3 - 1) (Calcite precipitation rate)
    #     0.014,         # kL: used in dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol (Ca(OH)2 dissolution rate)
    #     3.23e-13,        # D: used in dM_CSH_solid_mol_dt = -(4 * N_particles * pi * ...) * C_CO2_aq_mol_l * D / delta (Diffusion parameter in shrinking core)
    #     0.0266           # kLa: used in dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5 (Gas-liquid CO2 mass transfer rate)
    # ]


    theta = [
        1e4,         # k_R1f: used in r_R1 = k_R1f * C_CO2_aq - (k_R1f / Keq_R1) * (C_carbonate * gamma_HCO3) * (C_proton * gamma_H) (CO2 hydration/dissociation rate)
        2e4,       # k_R2f: used in r_R2 = k_R2f * (C_carbonate * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate * gamma_CO3) * (C_proton * gamma_H) (Bicarbonate dissociation rate)
        1e4,           # k_R5f: used in r_R5 = k_R5f * C_CO2_aq * (C_OH_bulk * gamma_OH) - (k_R5f / Keq_R5) * (C_carbonate * gamma_HCO3) (Hydroxycarbonate formation rate)
        1e7,        # k_R3f: used in r_R3 = k_R3f * (((C_CH_bulk * gamma_Ca) * (C_bicarbonate * gamma_CO3)) / Ksp_CaCO3 - 1) (Calcite precipitation rate)
        0.004,         # kL: used in dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol (Ca(OH)2 dissolution rate)
        3.23e-16,        # D: used in dM_CSH_solid_mol_dt = -(4 * N_particles * pi * ...) * C_CO2_aq_mol_l * D / delta (Diffusion parameter in shrinking core)
        0.0266,           # kLa: used in dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5 (Gas-liquid CO2 mass transfer rate)
        3.23e-25          # Ds: used in dM_CSH_solid_mol_dt = -(4 * N_particles * pi * ...) * C_CO2_aq_mol_l * Ds / delta (Diffusion parameter in shrinking core)
    ]



    # theta = [
    #     1.205e4, 4.813e7, 1.904e8, 3.340e5,
    #     9.158e1, 4.694e2, 7.059e-4, 9.585e1,
    #     36715, 15807, 39536, 47039,
    #     38801, 32132, 87550, 20037
    # ]



    tvi = {'T':T_profile, 'P':P_profile}

    result = solve_model(t, None, tii, tvi, theta)

    model_times = result['t']
    y_out = result['y']
    experimental_data_pH = np.array([
        [0, 8],
        [10, 11.53],
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

    # Interpolating the model concentration y_out[6] (proton concentration) at experimental time points
    model_times = result['t']  # times in seconds from model solution
    model_proton = y_out[6]  # proton concentration array from model solution
    from scipy.interpolate import interp1d
    # Create interpolation function
    interp_func = interp1d(model_times, model_proton, kind='linear', fill_value='extrapolate')
    model_proton_at_exp_times = interp_func(timeExp_C_proton_mol_l)

    # Calculate model pH at experimental times
    pH_model_at_exp_times = -np.log10(np.maximum(model_proton_at_exp_times, 1e-14))
    pH_exp = -np.log10(np.maximum(C_proton_mol_lExp, 1e-14))

    print(f'times = {timeExp_C_proton_mol_l}')
    print(f'exp_values = {pH_exp}')
    print(f'model_vals = {pH_model_at_exp_times}')




    t_out = result['t']
    y_out = result['y']
    y_out = result['y']
    t_out = result['t']

    R_evol = y_out[13]  # particle diameter (m)
    R0 = R_evol[0]  # initial particle diameter (m)
    core_diam = y_out[14]  # core diameter (m)
    t_min = t_out/60

    # Plotting
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.plot(t_min, y_out[2], color=colors['CaOH2'], label='Model Ca(OH)$_2$')
    plt.scatter(timeExp_others/60, nExp, facecolors='none', edgecolors='black', label='Exp Ca(OH)$_2$', s=70)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (mol/L)')
    plt.title('Ca(OH)$_2$ Concentration')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.subplot(1, 3, 2)
    pH_calc = -np.log10(np.maximum(y_out[6],1e-14))
    plt.plot(t_min, pH_calc, color=colors['pH'], label='Model pH')
    plt.scatter(timeExp_pH/60, pHExp, color=colors['exp_pH'], marker='x', label='Exp pH', s=80)
    plt.xlabel('Time (min)')
    plt.ylabel('pH')
    plt.title('pH Evolution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.ylim(3,12)

    plt.subplot(1, 3, 3)
    plt.plot(t_min, y_out[12], color=colors['CaCO3'], label='Model CaCO$_3$')
    plt.scatter(timeExp_CaCO3/60, CaCO3Exp, color=colors['exp_CaCO3'], marker='s', label='Exp CaCO$_3$', s=60)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (mol/L)')
    plt.title('CaCO$_3$ Formation')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')

    plt.suptitle('Temperature-Dependent Carbonation Model Simulation', fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    plot_12_species(result['t'], result['y'],
                    sol_full=result['sol_full'],
                    dp=tii['dp'],
                    Nr=result.get('Nr', 30))



    #
    # # Obtain particle and core diameters from solution matrix
    # particle_diameter_array = y_out[13]  # particle diameter in meters
    # core_diameter_array = y_out[14]  # core diameter in meters
    #
    # # Convert to microns
    # particle_diameter_microns = 1e6 * particle_diameter_array
    # core_diameter_microns = 1e6 * core_diameter_array
    #
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(t_min, particle_diameter_microns, label='Particle diameter')
    # plt.plot(t_min, core_diameter_microns, label='Core diameter', linestyle='--')
    # plt.xlabel('Time (min)')
    # plt.ylabel('Diameter (µm)')
    # plt.title('Particle and Core Diameter Evolution')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # Reporting summary
    # print(f'Initial particle diameter: {particle_diameter_microns[0]:.2f} µm')
    # print(f'Final particle diameter: {particle_diameter_microns[-1]:.2f} µm')
    # print(f'Final core diameter: {core_diameter_microns[-1]:.2f} µm')
    # print(f'Final product layer thickness: {(particle_diameter_microns[-1] - core_diameter_microns[-1]) / 2:.2f} µm')
    #
    # # Ionic species concentrations
    # plt.figure(figsize=(12,6))
    # plt.plot(t_min, y_out[4], label='OH⁻')
    # plt.plot(t_min, y_out[7], label='HCO₃⁻')
    # plt.plot(t_min, y_out[5], label='CO₃²⁻')
    # plt.xlabel('Time (min)')
    # plt.ylabel('Concentration (mol/L)')
    # plt.title('Ionic Species Concentrations')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # Activity coefficients
    # def calcGamma_np(I,z,a_i,A=0.037,B=1.04e8):
    #     lnGamma = (-A*z**2*np.sqrt(I*1e3)/(1+a_i*B*np.sqrt(I*1e3))
    #             + (0.2 -4.17e-5*I*1e3)*A*z**2*I*1e3/np.sqrt(1000))
    #     gamma = np.exp(lnGamma)
    #     return np.clip(gamma,0.1,1.0)
    #
    # I_calc = 2*y_out[1] + 0.5*y_out[4] + 2*y_out[5] + 0.5*y_out[6] + 0.5*y_out[7]
    # I_calc = np.maximum(I_calc,1e-4)
    # gamma_dict = {
    #     'Ca$^{2+}$': calcGamma_np(I_calc, 2, 1.0e-13),
    #     'OH$^-$': calcGamma_np(I_calc, 1, 3.0e-10),
    #     'H$^+$': calcGamma_np(I_calc, 1, 9.0e-11),
    #     'HCO$_3^-$': calcGamma_np(I_calc, 1, 4.0e-10),
    #     'CO$_3^{2-}$': calcGamma_np(I_calc, 2, 4.0e-10)
    # }
    # plt.figure(figsize=(10,5))
    # for ion, val in gamma_dict.items():
    #     plt.plot(t_min, val, label=f'γ({ion})')
    # plt.xlabel('Time (min)')
    # plt.ylabel('Activity Coefficient')
    # plt.title('Ion Activity Coefficients Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # Carbon speciation and pH stacked plot
    # CO2_aq = y_out[11]
    # HCO3 = y_out[7]
    # CO3 = y_out[5]
    # pH_vals = pH_calc
    # totalC = CO2_aq + HCO3 + CO3 + 1e-10
    # frac_CO2 = CO2_aq / totalC
    # frac_HCO3 = HCO3 / totalC
    # frac_CO3 = CO3 / totalC
    #
    # fig, ax1 = plt.subplots(figsize=(10,5))
    # ax1.stackplot(t_min, frac_CO2, frac_HCO3, frac_CO3,
    #               colors=['#2ca02c', '#17becf', '#9467bd'],
    #               labels=['CO₂(aq)', 'HCO₃⁻', 'CO₃²⁻'])
    # ax1.set_xlabel('Time (min)')
    # ax1.set_ylabel('Molar Fraction')
    # ax1.legend(loc='upper left')
    # ax1.grid(True)
    #
    # ax2 = ax1.twinx()
    # ax2.plot(t_min, pH_vals, 'r-', label='pH')
    # ax2.set_ylabel('pH', color='r')
    # ax2.tick_params(axis='y', labelcolor='r')
    # plt.title('Carbon Speciation and pH Evolution')
    # plt.tight_layout()
    # plt.show()


def plot_12_species(t, y, sol_full=None, dp=None, Nr=30):
    """
    Plot species concentrations over time, including contour plots for Ca and OH profiles

    Parameters:
    -----------
    t : array
        Time points (seconds)
    y : array
        Extracted species concentrations (15 x len(t))
    sol_full : solve_ivp solution object, optional
        Full solution including spatial profiles
    dp : float, optional
        Particle diameter (m)
    Nr : int, optional
        Number of radial discretization points (default: 30)
    """
    species_names = [
        'C_CH_void', 'C_CH_bulk', 'M_CH_solid',
        'C_OH_void', 'C_OH_bulk',
        'C_bicarbonate', 'C_proton',
        'C_carbonate', 'C_CaCO3_precipitated',
        'M_CSH_solid', 'C_CaCO3_shrinked',
        'C_CO2_aq'
    ]

    # Plot 1: Standard 12 species time series (SAME AS BEFORE)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    t_min = t / 60  # convert seconds to minutes

    for i, ax in enumerate(axes):
        ax.plot(t_min, y[i])
        ax.set_title(species_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Concentration (mol/L) or mol')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Species Concentrations Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot 2: NEW - Contour plots for Ca and OH profiles
    if sol_full is not None and dp is not None:
        # Extract Ca and OH profiles
        Ca_profiles = sol_full.y[0:Nr, :]
        OH_profiles = sol_full.y[Nr:2 * Nr, :]

        # Create radial grid
        r_particle = dp / 2
        r_grid = np.linspace(0, r_particle, Nr)
        r_grid_microns = r_grid * 1e6

        # Time grid in minutes
        t_grid_min = sol_full.t / 60

        # Create meshgrid for contour plots
        T_mesh, R_mesh = np.meshgrid(t_grid_min, r_grid_microns)

        # Create contour plots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Ca2+ contour plot
        ax1 = axes[0]
        contour1 = ax1.contourf(T_mesh, R_mesh, Ca_profiles, levels=20, cmap='viridis')
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('Ca²⁺ Concentration (mol/L)', fontsize=12)
        ax1.set_xlabel('Time (min)', fontsize=12)
        ax1.set_ylabel('Radial Position (μm)', fontsize=12)
        ax1.set_title('Ca²⁺ Concentration Profile in Particle Pores', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        contour_lines1 = ax1.contour(T_mesh, R_mesh, Ca_profiles, levels=10,
                                     colors='white', linewidths=0.5, alpha=0.5)
        ax1.clabel(contour_lines1, inline=True, fontsize=8, fmt='%.2e')

        # OH- contour plot
        ax2 = axes[1]
        contour2 = ax2.contourf(T_mesh, R_mesh, OH_profiles, levels=20, cmap='plasma')
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('OH⁻ Concentration (mol/L)', fontsize=12)
        ax2.set_xlabel('Time (min)', fontsize=12)
        ax2.set_ylabel('Radial Position (μm)', fontsize=12)
        ax2.set_title('OH⁻ Concentration Profile in Particle Pores', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        contour_lines2 = ax2.contour(T_mesh, R_mesh, OH_profiles, levels=10,
                                     colors='white', linewidths=0.5, alpha=0.5)
        ax2.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.2e')

        plt.suptitle('Spatial-Temporal Evolution of Ion Concentrations in Particle',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Plot 3: NEW - Radial profiles at selected times
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_times = len(t_grid_min)
        time_indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, -1]
        colors_time = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

        # Ca2+ radial profiles
        ax1 = axes[0]
        for idx, color in zip(time_indices, colors_time):
            ax1.plot(r_grid_microns, Ca_profiles[:, idx], 'o-', color=color,
                     label=f't = {t_grid_min[idx]:.1f} min', linewidth=2, markersize=4)
        ax1.set_xlabel('Radial Position (μm)', fontsize=12)
        ax1.set_ylabel('Ca²⁺ Concentration (mol/L)', fontsize=12)
        ax1.set_title('Ca²⁺ Radial Profiles', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # OH- radial profiles
        ax2 = axes[1]
        for idx, color in zip(time_indices, colors_time):
            ax2.plot(r_grid_microns, OH_profiles[:, idx], 's-', color=color,
                     label=f't = {t_grid_min[idx]:.1f} min', linewidth=2, markersize=4)
        ax2.set_xlabel('Radial Position (μm)', fontsize=12)
        ax2.set_ylabel('OH⁻ Concentration (mol/L)', fontsize=12)
        ax2.set_title('OH⁻ Radial Profiles', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Radial Profiles (0=center, max=surface)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print statistics
        print("\n" + "=" * 80)
        print("PARTICLE CONCENTRATION STATISTICS")
        print("=" * 80)
        print(f"Particle diameter: {dp * 1e6:.2f} μm")
        print(f"Particle radius: {r_particle * 1e6:.2f} μm")
        print(f"\nCa²⁺ concentration:")
        print(f"  Initial (center): {Ca_profiles[0, 0]:.3e} mol/L")
        print(f"  Final (center): {Ca_profiles[0, -1]:.3e} mol/L")
        print(f"  Final (surface): {Ca_profiles[-1, -1]:.3e} mol/L")
        print(f"\nOH⁻ concentration:")
        print(f"  Initial (center): {OH_profiles[0, 0]:.3e} mol/L")
        print(f"  Final (center): {OH_profiles[0, -1]:.3e} mol/L")
        print(f"  Final (surface): {OH_profiles[-1, -1]:.3e} mol/L")
        print("=" * 80)

        if sol_full is not None and dp is not None:
            Ca_profiles = sol_full.y[0:Nr, :]
            OH_profiles = sol_full.y[Nr:2 * Nr, :]
            r_particle = dp / 2
            r_grid = np.linspace(0, r_particle, Nr)
            r_grid_microns = r_grid * 1e6
            t_grid_min = sol_full.t / 60

            # Define time indices based on your time segmentation request
            times_short = [min(range(len(t_grid_min)), key=lambda i: abs(t_grid_min[i] - m)) for m in
                           range(0, 6)]  # 0-5 min every 1 min
            times_mid = [min(range(len(t_grid_min)), key=lambda i: abs(t_grid_min[i] - m)) for m in
                         range(10, 61, 10)]  # 5-60 min every 10 min
            times_long = [min(range(len(t_grid_min)), key=lambda i: abs(t_grid_min[i] - m)) for m in
                          range(70, int(t_grid_min[-1]) + 10, 20)]  # rest every 20 min
            time_indices = times_short + times_mid + times_long

            def plot_half_circle_profiles(r_grid, Ca_prof, OH_prof, ax, title):
                theta_right = np.linspace(-np.pi / 2, np.pi / 2, 150)
                theta_left = np.linspace(np.pi / 2, 3 * np.pi / 2, 150)

                # Swap arguments here to fix shape issue: theta first, radius second
                R_right, T_right = np.meshgrid(r_grid, theta_right)
                Ca_map = np.tile(Ca_prof, (len(theta_right), 1))
                x_right = R_right * np.cos(T_right)
                y_right = R_right * np.sin(T_right)

                R_left, T_left = np.meshgrid(r_grid, theta_left)
                OH_map = np.tile(OH_prof, (len(theta_left), 1))
                x_left = R_left * np.cos(T_left)
                y_left = R_left * np.sin(T_left)

                cf1 = ax.contourf(x_right, y_right, Ca_map, levels=20, cmap='Blues')
                cf2 = ax.contourf(x_left, y_left, OH_map, levels=20, cmap='Reds')

                cn1 = ax.contour(x_right, y_right, Ca_map, levels=10, colors='black', linewidths=0.5)
                cn2 = ax.contour(x_left, y_left, OH_map, levels=10, colors='black', linewidths=0.5)

                # Add colorbars next to plot for clarity
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                cbar1 = plt.colorbar(cf1, cax=cax1)
                cbar1.set_label('Ca²⁺ (mol/L)')

                cax2 = divider.append_axes("left", size="5%", pad=0.05)
                cbar2 = plt.colorbar(cf2, cax=cax2)
                cbar2.set_label('OH⁻ (mol/L)')
                cbar2.ax.yaxis.set_ticks_position('left')
                cbar2.ax.yaxis.set_label_position('left')

                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title(title)
                ax.add_patch(plt.Circle((0, 0), r_grid[-1], fill=False, color='k'))

            # Create 3-row plot layout
            n_plots = len(time_indices)
            fig, axes = plt.subplots(3, (n_plots + 2) // 3, figsize=(4 * (n_plots // 3), 12))
            axes = axes.flatten()
            for ax, idx in zip(axes, time_indices):
                plot_half_circle_profiles(r_grid_microns / 1e6, Ca_profiles[:, idx], OH_profiles[:, idx], ax,
                                          f"t = {t_grid_min[idx]:.1f} min")
            for ax in axes[len(time_indices):]:
                ax.axis('off')

            plt.suptitle('Particle Half-Circle Concentration Profiles\nLeft=OH⁻ (red), Right=Ca²⁺ (blue)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


# After solve_model call in run_and_plot:
# Example usage:



if __name__ == "__main__":
    run_and_plot()
