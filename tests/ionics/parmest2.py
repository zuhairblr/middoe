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
    Model simulating aqueous carbonation kinetics and equilibria:
    Ca(OH)2 dissolution, carbonate speciation, CaCO3 precipitation,
    with temperature and pressure variations.
    Reference equilibrium constants at T_ref = 298 K (unitless)
        Reaction numbering for clarity:
    R1: CO2 hydration/dissociation: CO2 + H2O ⇌ H+ + HCO3-
    R2: Bicarbonate dissociation: HCO3- ⇌ CO3^2- + H+ (pKa2)
    R3: Calcite precipitation inverse solubility: Ca2+ + CO3^2- ⇌ CaCO3(s)
    R4: Water ion recombination: H+ + OH- ⇌ H2O
    R5: Hydroxycarbonate formation: CO2 + OH- ⇌ HCO3-

    Parameters:
        t          : array_like, time points (s)
        y0         : initial conditions for species (mol/L), or None for defaults
        tii : dict of constants, physical and material (time-invariant)
        tvi: dict of time-dependent profiles (temperature K, pressure Pa)
        theta      : array_like, kinetic parameters (Arrhenius pre-exponentials,
                     activation energies (J/mol), diffusivity params (m^2/s, dimensionless))

    Returns:
        dict: keys 't' (time steps, s), 'y' (species concentrations matrix, mol/L),
              'R_particle' (particle radius evolution, m), 'R_particle0' (initial radius, m)
    """
    t = np.array(t, dtype=np.float64)                     # time array, s
    theta = np.array(theta, dtype=np.float64)             # kinetic parameters
    tii = {k: float(v) for k, v in tii.items()}            # time-invariant constants dict
    T_profile = np.array(tvi['T'], dtype=np.float64)       # temperature profile, K
    P_profile = np.array(tvi['P'], dtype=np.float64)       # pressure profile, bar
    R_gas = 8.314  # J/(mol·K) universal gas constant
    # # Extract kinetics parameters
    # A_arrh = theta[0:8]      # Arrhenius pre-exponentials (units depend on rate law, e.g. 1/s or m³/(mol·s))
    # Ea_arrh = theta[8:16]    # Activation energies (J/mol)
    uncvar = {                                   # Uncontrollable variables
        'CaO_leachable_solid_fr': 0.32061,       # Solid mass fraction of CaO in raw material (dimensionless)
        'CaO_unleachable_solid_fr': 0.3434,      # Solid mass fraction of equivalent CaO in raw material (dimensionless)
        'porosity': 0.3,                         # Solid mineral porosity (dimensionless)
        'CaCO3_porosity': 0.3,                   # Porosity of CaCO3 deposit (dimensionless)
        'rho_solid_material': 2260,              # Density solid material (kg/m³)
        'rho_solid_CaCO3': 2710,                 # Density CaCO3 (kg/m³)
        'slurry_volume': 0.5,                    # Liquid volume (l)
        'LOI': 0.01903,                          # LOI of mineral in weight fraction (dimensionless)
    }
    const = {                       # Constants (SI units)
        'Mw_CaO': 56e-3,            # Molecular weight CaO (kg/mol)
        'Mw_CaOH2': 74e-3,          # Molecular weight Ca(OH)2 (kg/mol)
        'Mw_CaCO3': 100.1e-3,       # Molecular weight CaCO3 (kg/mol)
        'Mw_CO2': 44e-3,            # Molecular weight CO2 (kg/mol)
        'k_R4f': 1.4e11,        # Fixed rate constant R4 (m³/(mol·s))
        'T_ref': 298.15,               # Reference temperature (K)
        # Equilibrium constants at T_ref (unitless)
        'Keq_R1_ref': 10 ** (-6.357),
        'Keq_R2_ref': 10 ** (-10.33),
        'Keq_R3_ref': 10 ** (9.108),
        'Keq_R4_ref': 10 ** (14),
        'Keq_R5_ref': 10 ** (7.66),
        # Enthalpy changes for van't Hoff (J/mol)
        'deltaH_R1': -20000,
        'deltaH_R2': 14400,
        'deltaH_R3': -12100,
        'deltaH_R4': 55800,
        'deltaH_R5': -41400,
    }
    C_proton_mol_l = 10 ** (-8)  # H+ initial concentration (mol/l)
    print(f'Initial C_proton_mol_l: {C_proton_mol_l}')
    C_OH_mol_l = 10 ** (-14) / C_proton_mol_l  # OH- concentration (mol/l)
    # Controllable variables
    dp = tii['dp']                               # Particle diameter (m)
    SLR = tii['SLR']                             # Solid-to-liquid ratio (kg.l)
    t_CO2_inject = tii.get('t_CO2_inject', 0.0)  # CO2 injection time (s)
    # Uncontrollable variables unpacked
    CaO_leachable_solid_fr = uncvar['CaO_leachable_solid_fr']         # dimensionless
    CaO_unleachable_solid_fr = uncvar['CaO_unleachable_solid_fr']     # dimensionless
    LOI = uncvar['LOI']                                               # dimensionless
    porosity = uncvar['porosity']                                     # dimensionless
    rho_solid_material = uncvar['rho_solid_material']*1e-3            # kg/l
    slurry_volume = uncvar['slurry_volume']                           # l
    rho_solid_CaCO3 = uncvar['rho_solid_CaCO3'] *1e-3                 # kg/l
    CaCO3_porosity = uncvar['CaCO3_porosity']                         # dimensionless
    # Constants unpacked
    Mw_CaO = const['Mw_CaO']                       # kg/mol
    Mw_CaOH2 = const['Mw_CaOH2']                   # kg/mol
    Mw_CaCO3 = const['Mw_CaCO3']                   # kg/mol
    Mw_CO2 = const['Mw_CO2']                       # kg/mol
    k_R4f = const['k_R4f']                 # l/(mol·s) for R4 forward
    T_ref = const['T_ref']                         # K
    Keq_R1_ref = const['Keq_R1_ref']               # unitless
    Keq_R2_ref = const['Keq_R2_ref']               # unitless
    Keq_R3_ref = const['Keq_R3_ref']               # unitless
    Keq_R4_ref = const['Keq_R4_ref']               # unitless
    Keq_R5_ref = const['Keq_R5_ref']               # unitless
    deltaH_R1 = const['deltaH_R1']                 # J/mol
    deltaH_R2 = const['deltaH_R2']                 # J/mol
    deltaH_R3 = const['deltaH_R3']                 # J/mol
    deltaH_R4 = const['deltaH_R4']                 # J/mol
    deltaH_R5 = const['deltaH_R5']                 # J/mol
    # Particle geometry and count
    molar_vol_CaCO3 = Mw_CaCO3 / rho_solid_CaCO3                 # Molar volume CaCO3 (l/mol)
    V_particle = (4/3) * np.pi * (dp / 2)**3*1000                # Volume per particle (l)
    V_solid_per_particle = (1 - porosity) * V_particle           # Solid volume per particle (l)
    mass_particle = rho_solid_material * V_solid_per_particle    # Mass per particle (kg)
    total_mass_kg = SLR * slurry_volume                          # Total solid mass (kg)
    dry_mass_kg = total_mass_kg * (1 - LOI)                      # Dry solid mass (kg)
    N_particles = total_mass_kg / mass_particle                  # Number of particles (dimensionless)

    CH_solid_mol = total_mass_kg * CaO_leachable_solid_fr / Mw_CaO  # Total reactable material mass (mol)
    CSH_solid_mol = total_mass_kg * CaO_unleachable_solid_fr / Mw_CaO  # hard to leach CaO equivalent mass in raw material (mol)
    # Ion radii and constants for activity coefficients
    aCa = 1.0e-13
    aOH = 3.0e-10
    aH = 9.0e-11
    aHCO3 = 4.0e-10
    aCO3 = 4.0e-10
    A_DH = 0.037
    B_DH = 1.04e8
    def vanthoff(K_eq_ref, deltaH, T):
        # Calculate temperature-dependent equilibrium constant (unitless)
        return K_eq_ref * np.exp(-deltaH / R_gas * (1/T - 1/T_ref))

    def ksps(T):
        # Calculate temperature-dependent solubility product (unitless)
        R_gas = 8.314  # J/(mol·K)
        deltaH= -16730
        deltaS= -157.86
        lnKsp_T = (-deltaH / R_gas) * (1 /T) + (deltaS / R_gas)
        return np.exp(lnKsp_T)

    # def arrhenius(A, Ea, T):
    #     # Arrhenius rate constant (units depend on reaction)
    #     return A * np.exp(-Ea / (R_gas * T))
    def T_at_time(tt):
        # Interpolate temperature at time tt (K)
        return np.interp(tt, t, T_profile)
    def P_at_time(tt):
        # Interpolate pressure at time tt (bar)
        return np.interp(tt, t, P_profile)
    def calcGamma(I, z, a_i, A, B):
        # Ion activity coefficient (dimensionless)
        Ip=I*1e3
        lnGamma = (-A * z**2 * np.sqrt(Ip) /
                   (1 + a_i * B * np.sqrt(Ip)) +
                   (0.2 - 4.17e-5 * Ip) * A * z**2 * Ip / np.sqrt(1000))
        gamma = np.exp(lnGamma)
        # return np.clip(gamma, 0.1, 1.0)
        return gamma


    def kspCaCO3(T):  # T in K
        # Solubility product of CaCO3 (unitless)
        return 10**(-(0.01183*(T-273.15) + 8.03))
    def kspCaOH2(T):  # T in K
        # Solubility product of Ca(OH)2 (unitless)
        S_CH = -0.01087 * (T-273.15) + 1.7465  # solubility g/L
        Mw = 74.093                   # g/mol
        S_CH_molL = (S_CH / 1000) / (Mw / 1000)  # mol/L
        return 4 * S_CH_molL**3

    y0 = [1e-30, 1e-30, CH_solid_mol,                                                    # initial concentrations mol/L or mol
          1e-30, C_OH_mol_l,
          1e-30, C_proton_mol_l, 1e-30, 1e-30,
          CSH_solid_mol, 1e-30, 1e-30, 1e-30, dp, dp]  # initial particle diameter and core diameter (m)



    def odes(tt, y):
        (C_CH_void_mol_l, C_CH_bulk_mol_l, M_CH_solid_mol,
         C_OH_void_mol_l, C_OH_bulk_mol_l,
         C_bicarbonate_mol_l, C_proton_mol_l,
         C_carbonate_mol_l, C_CaCO3_precipitated_mol_l,
         M_CSH_solid_mol, C_CaCO3_shrinked_mol_l,
         C_CO2_aq_mol_l, alpha,
         particle_diameter, core_diameter) = y
        # Enforce minimum positive concentrations for stability
        species = [C_CH_void_mol_l, C_CH_bulk_mol_l, M_CH_solid_mol,
                   C_OH_void_mol_l, C_OH_bulk_mol_l,
                   C_bicarbonate_mol_l, C_proton_mol_l,
                   C_carbonate_mol_l, C_CaCO3_precipitated_mol_l,
                   M_CSH_solid_mol, C_CaCO3_shrinked_mol_l]
        species = np.maximum(species, 1e-30)
        (C_CH_void_mol_l, C_CH_bulk_mol_l, M_CH_solid_mol,
         C_OH_void_mol_l, C_OH_bulk_mol_l,
         C_bicarbonate_mol_l, C_proton_mol_l,
         C_carbonate_mol_l, C_CaCO3_precipitated_mol_l,
         M_CSH_solid_mol, C_CaCO3_shrinked_mol_l) = species
        T_loc = T_at_time(tt)
        P_loc = P_at_time(tt)
        # Rate constants & equilibrium constants (Arrhenius & van't Hoff)


        # k_R1f = arrhenius(A_arrh[0], Ea_arrh[0], T_loc)
        # k_R2f = arrhenius(A_arrh[1], Ea_arrh[1], T_loc)
        # k_R5f = arrhenius(A_arrh[2], Ea_arrh[2], T_loc)
        # k_R3f = arrhenius(A_arrh[3], Ea_arrh[3], T_loc)
        # kL = arrhenius(A_arrh[4], Ea_arrh[4], T_loc)
        # kO = arrhenius(A_arrh[5], Ea_arrh[5], T_loc)
        # D = arrhenius(A_arrh[6], Ea_arrh[6], T_loc)
        # kLa = arrhenius(A_arrh[7], Ea_arrh[7], T_loc)

        k_R1f = theta [0]
        k_R2f = theta [1]
        k_R5f = theta [2]
        k_R3f = theta [3]
        kL = theta [4]
        kO = theta [5]
        D = theta [6]
        kLa = theta [7]

        Keq_R1 = vanthoff(Keq_R1_ref, deltaH_R1, T_loc)
        Keq_R2 = vanthoff(Keq_R2_ref, deltaH_R2, T_loc)
        Keq_R3 = vanthoff(Keq_R3_ref, deltaH_R3, T_loc)
        Keq_R4 = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
        Keq_R5 = vanthoff(Keq_R5_ref, deltaH_R5, T_loc)
        # Ksp_CaOH2 = kspCaOH2(T_loc)
        Ksp_CaCO3 = kspCaCO3(T_loc)
        Ksp_CaOH2_25C = 7.9e-6  # Reference Ksp for Ca(OH)2 at 25°C
        Ksp_CaCO3_25C = 3.8e-9  # Reference Ksp for CaCO3 at 25°C
        deltaH_CaOH2 = -16000  # J/mol (example, actual value may vary)
        deltaH_CaCO3 = 9000  # J/mol (example, actual value may vary)
        R_gas = 8.314  # J/mol·K
        T_ref = 298.15  # 25°C in Kelvin

        # For your temperature T_loc (in K):
        Ksp_CaOH2 = ksps(T_loc)


        solid_liquid_total_surface = N_particles * 4 * np.pi * (particle_diameter / 2) ** 2
        I = max(
            2 * C_CH_bulk_mol_l + 0.5 * C_OH_bulk_mol_l + 0.5 * C_bicarbonate_mol_l + 0.5 * C_proton_mol_l + 2 * C_carbonate_mol_l,
            1e-4)

        gamma_Ca = calcGamma(I, 2, aCa, A_DH, B_DH)
        gamma_OH = calcGamma(I, 1, aOH, A_DH, B_DH)
        gamma_CO3 = calcGamma(I, 2, aCO3, A_DH, B_DH)
        gamma_H = calcGamma(I, 1, aH, A_DH, B_DH)
        gamma_HCO3 = calcGamma(I, 1, aHCO3, A_DH, B_DH)
        c_std = 1  # standard state concentration: 1 mol/l
        r_R1 = k_R1f * C_CO2_aq_mol_l - (k_R1f / Keq_R1) * ((C_bicarbonate_mol_l / c_std) * gamma_HCO3) * (
                    (C_proton_mol_l / c_std) * gamma_H)
        r_R2 = k_R2f * ((C_bicarbonate_mol_l / c_std) * gamma_HCO3) - (k_R2f / Keq_R2) * (
                    (C_carbonate_mol_l / c_std) * gamma_CO3) * ((C_proton_mol_l / c_std) * gamma_H)

        supersat_ratio = ((C_CH_bulk_mol_l / c_std) * gamma_Ca *
                          (C_carbonate_mol_l / c_std) * gamma_CO3) / Ksp_CaCO3
        supersat_term = max(0, supersat_ratio - 1)
        r_R3 = k_R3f * supersat_term

        r_R4 = k_R4f * ((C_proton_mol_l / c_std) * gamma_H) * (
                    (C_OH_bulk_mol_l / c_std) * gamma_OH)
        r_R5 = k_R5f * C_CO2_aq_mol_l * ((C_OH_bulk_mol_l / c_std) * gamma_OH) - (k_R5f / Keq_R5) * (
                    (C_bicarbonate_mol_l / c_std) * gamma_HCO3)

        # print(f'C_CH_bulk_mol_l={C_CH_bulk_mol_l}, gamma_Ca={gamma_Ca}, C_bicarbonate_mol_l={C_bicarbonate_mol_l}, gamma_CO3={gamma_CO3}, Ksp_CaCO3={Ksp_CaCO3}, r_R3={r_R3}')
        ((C_proton_mol_l / c_std) * gamma_H) * ((C_OH_bulk_mol_l / c_std) * gamma_OH) == Keq_R4
        # Saturation ratio for Ca(OH)2 dissolution
        S = ((C_CH_void_mol_l) * gamma_Ca) * (((C_OH_void_mol_l) * gamma_OH) ** 2) / Ksp_CaOH2


        # r_R1 = k_R1f * C_CO2_aq_mol_l - (k_R1f / Keq_R1) * ((C_bicarbonate_mol_l / c_std)) * (
        #             (C_proton_mol_l / c_std) )
        # r_R2 = k_R2f * ((C_bicarbonate_mol_l / c_std)) - (k_R2f / Keq_R2) * (
        #             (C_carbonate_mol_l / c_std) ) * ((C_proton_mol_l / c_std) )
        #
        # # Calculate supersaturation ratio
        # supersat_ratio = ((C_CH_bulk_mol_l / c_std) * (C_carbonate_mol_l / c_std)) / Ksp_CaCO3
        #
        # # Enforce no precipitation if system is undersaturated (supersaturation ≤ 1)
        # supersat_term = max(0, supersat_ratio - 1)
        #
        # # Optional: apply reaction order (e.g., n=1 or n=2)
        #
        # r_R3 = k_R3f * supersat_term
        #
        # print(f'time={tt:.2f}s, r_R3={r_R3:.3e}, Ksp_CaCO3={Ksp_CaCO3}, C_CaCO3_precipitated_mol_l={C_CaCO3_precipitated_mol_l} ')
        # r_R4 = k_R4f * ((C_proton_mol_l / c_std)) * (
        #             (C_OH_bulk_mol_l / c_std))
        # r_R5 = k_R5f * C_CO2_aq_mol_l * ((C_OH_bulk_mol_l / c_std)) - (k_R5f / Keq_R5) * (
        #             (C_bicarbonate_mol_l / c_std) )
        #
        # # print(f'C_CH_bulk_mol_l={C_CH_bulk_mol_l}, gamma_Ca={gamma_Ca}, C_bicarbonate_mol_l={C_bicarbonate_mol_l}, gamma_CO3={gamma_CO3}, Ksp_CaCO3={Ksp_CaCO3}, r_R3={r_R3}')
        # # ((C_proton_mol_l / c_std) ) * ((C_OH_bulk_mol_l / c_std) ) == Keq_R4
        # # Saturation ratio for Ca(OH)2 dissolution
        # S = ((C_CH_void_mol_l) ) * (((C_OH_void_mol_l) ) ** 2) / Ksp_CaOH2


        CO2_sat = P_loc * 35 * np.exp(2400 * ((1 / T_loc) - (1 / T_ref)))*1e-3
        dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5
        threshold_bulk_CH = 1e-4
        threshold_surface_CH = 1e-4
        # dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol
        # dM_CH_solid_mol_dt = min(dM_CH_solid_mol_dt, 0.0)
        if S<1:
            dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol
            dM_CH_solid_mol_dt = min(dM_CH_solid_mol_dt, 0.0)
        elif S>= 1:
            dM_CH_solid_mol_dt = 0
        # print (f'S = {S}, C_CH_void_mol_l = {C_CH_void_mol_l}, gamma_Ca={gamma_Ca}, C_OH_void_mol_l={C_OH_void_mol_l}, gamma_OH={gamma_OH}, Ksp_CaOH2= {Ksp_CaOH2}, dM_CH_solid_mol_dt = {dM_CH_solid_mol_dt}')
        # print(f'tt={tt}, C_proton_mol_l={C_proton_mol_l}')

        dM_CSH_solid_mol_dt = 0.0
        dcore_dt = 0.0
        dC_CaCO3_precipitated_mol_l_dt = r_R3
        # Shrinking core regime
        if M_CH_solid_mol < 1e-4:
            # Calculate product layer thickness
            delta = max((particle_diameter / 2) - (core_diameter / 2), 1e-12)
            dM_CSH_solid_mol_dt = -(4 * N_particles * np.pi * (particle_diameter / 2) * (
                    core_diameter / 2) * C_CO2_aq_mol_l*1e3 * D) / delta
            # dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5 - (dM_CSH_solid_mol_dt / slurry_volume)
            term = 1 - (M_CSH_solid_mol * (Mw_CO2 / Mw_CaO)) / dry_mass_kg
            term = max(term, 1e-12)  # Avoid zero or negative for fractional power
            dcore_dt = (2 / 3) * (dp/2) * term ** (2 / 3) * (
                    (dM_CSH_solid_mol_dt * (Mw_CO2 / Mw_CaO)) / dry_mass_kg)

        V_particle_new = (4 / 3) * np.pi * (particle_diameter / 2) ** 3 * 1e3
        factor = 1 / (porosity * V_particle_new * N_particles)
        dC_CH_void_mol_l_dt = -factor * dM_CH_solid_mol_dt -  kO * solid_liquid_total_surface * (
                C_CH_void_mol_l - C_CH_bulk_mol_l)
        dC_CH_bulk_mol_l_dt =  kO * solid_liquid_total_surface * (C_CH_void_mol_l - C_CH_bulk_mol_l) - r_R3
        dC_OH_void_mol_l_dt = -2 * factor * dM_CH_solid_mol_dt -  kO * solid_liquid_total_surface * (
                C_OH_void_mol_l - C_OH_bulk_mol_l)
        dC_OH_bulk_mol_l_dt =  kO * solid_liquid_total_surface * (
                C_OH_void_mol_l - C_OH_bulk_mol_l) - r_R4 - r_R5
        dC_carbonate_mol_l_dt = r_R2 - r_R3
        dC_bicarbonate_mol_l_dt = r_R1 + r_R5 - r_R2
        dC_proton_mol_l_dt = r_R1 + r_R2 - r_R4

        dC_CaCO3_shrinked_mol_l_dt = -dM_CSH_solid_mol_dt / slurry_volume
        if C_CaCO3_precipitated_mol_l > 1e-2:
            dV_deposit_dt = dC_CaCO3_precipitated_mol_l_dt * slurry_volume * molar_vol_CaCO3 / (
                    (1 - CaCO3_porosity) * N_particles)
            dR_dt = (dV_deposit_dt*1e-3) / (4 * np.pi * (particle_diameter / 2) ** 2)
        else:
            dR_dt = 0.0
        dparticle_dt = 2 * dR_dt
        # dalpha_dt = (((4 / 3) * np.pi * N_particles * (particle_diameter / 2) ** 3) - ((4 / 3) * np.pi * N_particles * (core_diameter / 2) ** 3)) / ((4 / 3) * np.pi * N_particles * (dp / 2) ** 3)
        dalpha_dt = ((dC_CaCO3_precipitated_mol_l_dt+dC_CaCO3_shrinked_mol_l_dt)* slurry_volume * (Mw_CO2/Mw_CaCO3))/ dry_mass_kg

        dy = [
            dC_CH_void_mol_l_dt, dC_CH_bulk_mol_l_dt, dM_CH_solid_mol_dt,
            dC_OH_void_mol_l_dt, dC_OH_bulk_mol_l_dt,
            dC_bicarbonate_mol_l_dt, dC_proton_mol_l_dt, dC_carbonate_mol_l_dt,
            dC_CaCO3_precipitated_mol_l_dt, dM_CSH_solid_mol_dt,
            dC_CaCO3_shrinked_mol_l_dt, dCO2_aq_dt, dalpha_dt, dparticle_dt, dcore_dt
        ]
        # Convert complex derivatives to real if close
        dy = [np.real_if_close(d) for d in dy]
        dM_CH_solid_mol_dt = min(dM_CH_solid_mol_dt, 0.0)
        # print(
        #     f"Time={tt:.2f}s: MCH= {M_CH_solid_mol}, MCSH = {M_CSH_solid_mol}, dMCSHdt= {dM_CSH_solid_mol_dt}, particle_diameter= {particle_diameter}, core_diameter = {core_diameter}")
        # pH = -np.log10(np.maximum(C_proton_mol_l, 1e-14))
        # pOH = -np.log10(np.maximum(C_OH_bulk_mol_l, 1e-14))
        # sum_pH_pOH = pH + pOH
        # print(
        #     f"Time={tt:.2f}s: pH={pH:.3f}, pOH={pOH:.3f}, sum={sum_pH_pOH:.3f}")

        # print(f'tt={tt:.2f}s, C_CH_bulk={C_CH_bulk_mol_l:.3e} mol/L, M_CH_solid={M_CH_solid_mol:.3e} mol, C_CaCO3_precipitated={C_CaCO3_precipitated_mol_l:.3e} mol/L, particle_diameter={particle_diameter*1e6:.3f} um, core_diameter={core_diameter*1e6:.3f} um')

        # # Print or log magnitudes of derivatives
        # for i, deriv in enumerate(dy):
        #     print(f"Time={tt:.2f}s, derivative {i} magnitude = {abs(deriv):.3e}")

        return dy

    sol = solve_ivp(
        fun=odes,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='BDF',
        rtol=1e-8,
        atol=1e-11,
        jac=None  # Use finite-difference approx Jacobian for stiff system
    )

    print(sol)

    print("Integrator success:", sol.success)
    print("Integrator message:", sol.message)
    print(f'CH_solid_mol={CH_solid_mol}')
    # Print final values of all integrated variables
    final_values = sol.y[:, -1]  # last column of solution array, all variables at final time
    variable_names = [
        'C_CH_void', 'C_CH_bulk', 'M_CH_solid',
        'C_OH_void', 'C_OH_bulk',
        'C_bicarbonate', 'C_proton', 'C_carbonate',
        'C_CaCO3_precipitated', 'M_CSH_solid',
        'C_CaCO3_shrinked', 'CO2_aq',
        'alpha', 'particle_diameter', 'core_diameter'
    ]

    print("Final values of integrated variables at time {:.2f}s:".format(sol.t[-1]))
    for name, value in zip(variable_names, final_values):
        print(f"{name}: {value}")

    return {
        't': sol.t,
        'y': sol.y,
        'particle_diameter': sol.y[13],  # absolute particle diameter (m)
        'core_diameter': sol.y[14],  # absolute core diameter (m)
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
        (9.63e-5, 1.77e10),  # for kO
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
        0.445,
        # k_R1f: used in r_R1 = k_R1f * C_CO2_aq - (k_R1f / Keq_R1) * (C_carbonate * gamma_HCO3) * (C_proton * gamma_H) (CO2 hydration/dissociation rate)
        818450.57,
        # k_R2f: used in r_R2 = k_R2f * (C_carbonate * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate * gamma_CO3) * (C_proton * gamma_H) (Bicarbonate dissociation rate)
        22.54,
        # k_R5f: used in r_R5 = k_R5f * C_CO2_aq * (C_OH_bulk * gamma_OH) - (k_R5f / Keq_R5) * (C_carbonate * gamma_HCO3) (Hydroxycarbonate formation rate)
        0.00192,
        # k_R3f: used in r_R3 = k_R3f * (((C_CH_bulk * gamma_Ca) * (C_bicarbonate * gamma_CO3)) / Ksp_CaCO3 - 1) (Calcite precipitation rate)
        5.46e+6,  # kL: used in dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol (Ca(OH)2 dissolution rate)
        0.000510,
        # kO: used in mass transfer terms between void and bulk phases (e.g., kO * solid_liquid_total_surface * (C_CH_void - C_CH_bulk))
        3.23e-19,
        # D: used in dM_CSH_solid_mol_dt = -(4 * N_particles * pi * ...) * C_CO2_aq_mol_l * D / delta (Diffusion parameter in shrinking core)
        0.0296
        # kLa: used in dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5 (Gas-liquid CO2 mass transfer rate)
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

    res = differential_evolution(
        scaled_objective_theta,
        bounds=bounds,
        args=(t, tii, tvi, timeExp_C_proton_mol_l, C_proton_mol_lExp, bounds),
        maxiter=10000,
        disp=True,
        polish=False,  # optional: polish final solution by local optimization
        tol=1e-30
    )


    # res = minimize(
    #     scaled_objective_theta,
    #     x0,
    #     args=(t, tii, tvi, timeExp_C_proton_mol_l, C_proton_mol_lExp, bounds),
    #     method='trust-constr',
    #     bounds=bounds,
    #     options={'maxiter': 1000000000, 'disp': True}
    # )


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