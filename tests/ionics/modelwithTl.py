import numpy as np
from mypy.checker import CheckerScope
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Global constants and reaction numbering
# -----------------------------

def solve_model(t, y0, tii, tvi, theta):
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

    # Extract kinetics parameters
    A_arrh = theta[0:8]      # Arrhenius pre-exponentials (units depend on rate law, e.g. 1/s or m³/(mol·s))
    Ea_arrh = theta[8:16]    # Activation energies (J/mol)
    etha = theta[16]   # Diffusivity decay coefficient (dimensionless)

    uncvar = {                                   # Uncontrollable variables
        'CaO_leachable_solid_fr': 0.26061,       # Solid mass fraction of CaO in raw material (dimensionless)
        'CaO_unleachable_solid_fr': 0.4034,      # Solid mass fraction of equivalent CaO in raw material (dimensionless)
        'porosity': 0.3,                         # Solid mineral porosity (dimensionless)
        'CaCO3_porosity': 0.3,                   # Porosity of CaCO3 deposit (dimensionless)
        'rho_solid_material': 2260,              # Density solid material (kg/m³)
        'rho_solid_CaCO3': 2710,                 # Density CaCO3 (kg/m³)
        'slurry_volume': 5e-4,                    # Liquid volume (m³)
        'LOI': 0.01903,                          # LOI of mineral in weight fraction (dimensionless)
    }

    const = {                       # Constants (SI units)
        'Mw_CaO': 56e-3,            # Molecular weight CaO (kg/mol)
        'Mw_CaOH2': 74e-3,          # Molecular weight Ca(OH)2 (kg/mol)
        'Mw_CaCO3': 100.1e-3,         # Molecular weight CaCO3 (kg/mol)
        'Mw_CO2': 44e-3,            # Molecular weight CO2 (kg/mol)
        'k5f_fixed': 1.4e8,        # Fixed rate constant R4 (m³/(mol·s))
        'T_ref': 298,               # Reference temperature (K)
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

    C_proton_mol_m3 = 10 ** (-11.53) * 1000  # H+ initial concentration (mol/m³)

    # Controllable variables
    dp = tii['dp']                               # Particle diameter (m)
    SLR = tii['SLR']*1e3                         # Solid-to-liquid ratio (kg.m-3)
    t_CO2_inject = tii.get('t_CO2_inject', 0.0)  # CO2 injection time (s)

    # Uncontrollable variables unpacked
    CaO_leachable_solid_fr = uncvar['CaO_leachable_solid_fr']         # dimensionless
    LOI = uncvar['LOI']                                               # dimensionless
    porosity = uncvar['porosity']                                     # dimensionless
    rho_solid_material = uncvar['rho_solid_material']                 # kg/m³
    slurry_volume = uncvar['slurry_volume']                           # m³
    rho_solid_CaCO3 = uncvar['rho_solid_CaCO3']                       # kg/m³
    CaCO3_porosity = uncvar['CaCO3_porosity']                         # dimensionless

    # Constants unpacked
    Mw_CaO = const['Mw_CaO']                       # kg/mol
    Mw_CaOH2 = const['Mw_CaOH2']                   # kg/mol
    Mw_CaCO3 = const['Mw_CaCO3']                   # kg/mol
    Mw_CO2 = const['Mw_CO2']                       # kg/mol
    k5f_fixed = const['k5f_fixed']                 # m³/(mol·s) for R4 forward
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
    R_particle0 = dp / 2                                         # Particle radius (m)
    molar_vol_CaCO3 = Mw_CaCO3 / rho_solid_CaCO3                 # Molar volume CaCO3 (m³/mol)
    V_particle = (4/3) * np.pi * R_particle0**3                  # Volume per particle (m³)
    V_solid_per_particle = (1 - porosity) * V_particle           # Solid volume per particle (m³)
    mass_particle = rho_solid_material * V_solid_per_particle    # Mass per particle (kg)
    total_mass_kg = SLR * slurry_volume                          # Total solid mass (kg)
    dry_mass_kg = total_mass_kg * (1 - LOI)                      # Dry solid mass (kg)
    N_particles = total_mass_kg / mass_particle                  # Number of particles (dimensionless)

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

    def arrhenius(A, Ea, T):
        # Arrhenius rate constant (units depend on reaction)
        return A * np.exp(-Ea / (R_gas * T))

    def T_at_time(tt):
        # Interpolate temperature at time tt (K)
        return np.interp(tt, t, T_profile)

    def P_at_time(tt):
        # Interpolate pressure at time tt (bar)
        return np.interp(tt, t, P_profile)

    def calcGamma(I, z, a_i, A, B):
        # Ion activity coefficient (dimensionless)
        Ip=I*1e-3
        lnGamma = (-A * z**2 * np.sqrt(Ip) /
                   (1 + a_i * B * np.sqrt(Ip)) +
                   (0.2 - 4.17e-5 * Ip) * A * z**2 * Ip / np.sqrt(1000))
        gamma = np.exp(lnGamma)
        return np.clip(gamma, 0.1, 1.0)


    if y0 is None:
        CaO_leachable_solid_mol = total_mass_kg * CaO_leachable_solid_fr / Mw_CaO            # free lime mass in raw material (mol)    # portlandite mass in raw material (mol)
        CH_solid_mol = CaO_leachable_solid_mol                   # Total reactable material mass (mol)
        solid_vol_m3 = N_particles * V_solid_per_particle                                    # total volume of particles (m³)
        CSH_solid_mol = total_mass_kg * uncvar['CaO_unleachable_solid_fr'] / Mw_CaO          # hard to leach CaO equivalent mass in raw material (mol)

        y0 = [1e-7, 1e-7, CH_solid_mol,                                                    # initial concentrations mol/L or mol
              1e-7, 3.33e3,
              1e-7, C_proton_mol_m3, 1e-7, 1e-7,
              CSH_solid_mol, 1e-7, 1e-7, 1e-7, dp, dp]  # initial particle diameter and core diameter (m)

    def kspCaCO3(T):  # T in K
        # Solubility product of CaCO3 (unitless)
        return 10**(-(0.01183*(T-273) + 8.03))

    def kspCaOH2(T):  # T in K
        # Solubility product of Ca(OH)2 (unitless)
        S_CH = -0.01087 * T + 1.7465  # solubility g/L
        Mw = 74.093                   # g/mol
        S_CH_molL = (S_CH / 1000) / (Mw / 1000)  # mol/L
        return 4 * S_CH_molL**3

    def odes(tt, y):
        (C_CH_void_mol_m3, C_CH_bulk_mol_m3, M_CH_solid_mol,
         C_OH_void_mol_m3, C_OH_bulk_mol_m3,
         C_bicarbonate_mol_m3, C_proton_mol_m3,
         C_carbonate_mol_m3, C_CaCO3_precipitated_mol_m3,
         M_CSH_solid_mol, C_CaCO3_shrinked_mol_m3,
         C_CO2_aq_mol_m3, alpha,
         particle_diameter, core_diameter) = y

        # Enforce minimum positive concentrations for stability
        species = [C_CH_void_mol_m3, C_CH_bulk_mol_m3, M_CH_solid_mol,
                   C_OH_void_mol_m3, C_OH_bulk_mol_m3,
                   C_bicarbonate_mol_m3, C_proton_mol_m3,
                   C_carbonate_mol_m3, C_CaCO3_precipitated_mol_m3,
                   M_CSH_solid_mol, C_CaCO3_shrinked_mol_m3]
        species = np.maximum(species, 1e-7)
        (C_CH_void_mol_m3, C_CH_bulk_mol_m3, M_CH_solid_mol,
         C_OH_void_mol_m3, C_OH_bulk_mol_m3,
         C_bicarbonate_mol_m3, C_proton_mol_m3,
         C_carbonate_mol_m3, C_CaCO3_precipitated_mol_m3,
         M_CSH_solid_mol, C_CaCO3_shrinked_mol_m3) = species

        T_loc = T_at_time(tt)
        P_loc = P_at_time(tt)

        # Rate constants & equilibrium constants (Arrhenius & van't Hoff)
        k_R1f = arrhenius(A_arrh[0], Ea_arrh[0], T_loc)
        k_R2f = arrhenius(A_arrh[1], Ea_arrh[1], T_loc)*1e-3
        k_R5f = arrhenius(A_arrh[2], Ea_arrh[2], T_loc)
        k_R3f = arrhenius(A_arrh[3], Ea_arrh[3], T_loc)*1e-3
        kL = arrhenius(A_arrh[4], Ea_arrh[4], T_loc)
        kO = arrhenius(A_arrh[5], Ea_arrh[5], T_loc)*1e-3
        D = arrhenius(A_arrh[6], Ea_arrh[6], T_loc)
        kLa = arrhenius(A_arrh[7], Ea_arrh[7], T_loc)

        Keq_R1 = vanthoff(Keq_R1_ref, deltaH_R1, T_loc)
        Keq_R2 = vanthoff(Keq_R2_ref, deltaH_R2, T_loc)
        Keq_R3 = vanthoff(Keq_R3_ref, deltaH_R3, T_loc)
        Keq_R4 = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
        Keq_R5 = vanthoff(Keq_R5_ref, deltaH_R5, T_loc)

        # Ksp_CaOH2 = kspCaOH2(T_loc)*1e-9
        # Ksp_CaCO3 = kspCaCO3(T_loc)*1e-6
        Ksp_CaOH2 = kspCaOH2(T_loc)
        Ksp_CaCO3 = kspCaCO3(T_loc)

        solid_liquid_total_surface = N_particles * 4 * np.pi * (particle_diameter / 2) ** 2

        I = max(
            2 * C_CH_bulk_mol_m3 + 0.5 * C_OH_bulk_mol_m3 + 2 * C_bicarbonate_mol_m3 + 0.5 * C_proton_mol_m3 + 0.5 * C_carbonate_mol_m3,
            1e-4)

        gamma_Ca = calcGamma(I, 2, aCa, A_DH, B_DH)
        gamma_OH = calcGamma(I, 1, aOH, A_DH, B_DH)
        gamma_CO3 = calcGamma(I, 2, aCO3, A_DH, B_DH)
        gamma_H = calcGamma(I, 1, aH, A_DH, B_DH)
        gamma_HCO3 = calcGamma(I, 1, aHCO3, A_DH, B_DH)

        # Aqueous reaction rates
        # r_R1 = k_R1f * C_CO2_aq_mol_m3 - (k_R1f / Keq_R1) * (C_carbonate_mol_m3 * gamma_HCO3) * (C_proton_mol_m3 * gamma_H)
        # r_R2 = k_R2f * (C_carbonate_mol_m3 * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate_mol_m3 * gamma_CO3) * (
        #         C_proton_mol_m3 * gamma_H)
        # r_R5 = k_R5f * C_CO2_aq_mol_m3 * (C_OH_bulk_mol_m3 * gamma_OH) - (k_R5f / Keq_R5) * (
        #         C_carbonate_mol_m3 * gamma_HCO3)
        # r_R3 = k_R3f * ((C_CH_void_mol_m3 * gamma_Ca) * (C_bicarbonate_mol_m3 * gamma_CO3) / Ksp_CaCO3 - 1)
        # r_recomb = k5f_fixed * (C_proton_mol_m3 * gamma_H) * (C_OH_bulk_mol_m3 * gamma_OH) - k5f_fixed / Keq_R4
        c_std = 1000.0  # standard state concentration mol/m³ (equivalent to 1 mol/L)

        r_R1 = k_R1f * C_CO2_aq_mol_m3 - (k_R1f / Keq_R1) * ((C_carbonate_mol_m3 / c_std) * gamma_HCO3) * (
                    (C_proton_mol_m3 / c_std) * gamma_H)

        r_R2 = k_R2f * ((C_carbonate_mol_m3 / c_std) * gamma_HCO3) - (k_R2f / Keq_R2) * (
                    (C_bicarbonate_mol_m3 / c_std) * gamma_CO3) * ((C_proton_mol_m3 / c_std) * gamma_H)

        r_R5 = k_R5f * C_CO2_aq_mol_m3 * ((C_OH_bulk_mol_m3 / c_std) * gamma_OH) - (k_R5f / Keq_R5) * (
                    (C_carbonate_mol_m3 / c_std) * gamma_HCO3)

        r_R3 = k_R3f * (((C_CH_bulk_mol_m3 / c_std) * gamma_Ca) * (
                    (C_bicarbonate_mol_m3 / c_std) * gamma_CO3) / Ksp_CaCO3 - 1)

        r_recomb = k5f_fixed * ((C_proton_mol_m3 / c_std) * gamma_H) * (
                    (C_OH_bulk_mol_m3 / c_std) * gamma_OH) - k5f_fixed / Keq_R4

        # Saturation ratio for Ca(OH)2 dissolution
        S = ((C_CH_void_mol_m3/1000) * gamma_Ca) * (((C_OH_void_mol_m3/1000) * gamma_OH) ** 2) / Ksp_CaOH2

        CO2_sat = P_loc * 35 * np.exp(2400 * ((1 / T_loc) - (1 / T_ref)))
        dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_m3) - r_R1 - r_R5


        threshold_bulk_CH = 1e-4
        threshold_surface_CH = 1e-4
        dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol
        dM_CH_solid_mol_dt = min(dM_CH_solid_mol_dt, 0.0)

        dM_CSH_solid_mol_dt = 0.0
        dcore_dt = 0.0
        dC_CaCO3_precipitated_mol_m3_dt = r_R3



        # Shrinking core regime
        if M_CH_solid_mol < 1e-6:
            # Calculate product layer thickness
            delta = max((particle_diameter / 2) - (core_diameter / 2), 1e-12)
            # Effective diffusivity
            D_e = D * np.exp(-etha * tt)
            # # Diffusion-controlled solid consumption rate (mol/s)
            # dM_CSH_solid_mol_dt = -(4 * N_particles * np.pi * (particle_diameter / 2) * (
            #         core_diameter / 2) * C_CO2_aq_mol_m3 * D_e) / delta
            # Diffusion-controlled solid consumption rate (mol/s)
            dM_CSH_solid_mol_dt = -(4 * N_particles * np.pi * (particle_diameter / 2) * (
                    core_diameter / 2) * C_CO2_aq_mol_m3 * D_e) / delta
            # Make sure dM_CSH_solid_mol_dt is negative or zero (solid consumption)
            # dM_CSH_solid_mol_dt = min(dM_CSH_solid_mol_dt, 0.0)
            dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_m3) - r_R1 - r_R5 - (dM_CSH_solid_mol_dt / slurry_volume)
            # Calculate core shrinkage rate
            term = 1 - (M_CSH_solid_mol * (Mw_CO2 / Mw_CaO)) / dry_mass_kg
            term = max(term, 1e-12)  # Avoid zero or negative for fractional power
            dcore_dt = (2 / 3) * R_particle0 * term ** (2 / 3) * (
                    (dM_CSH_solid_mol_dt * (Mw_CO2 / Mw_CaO)) / dry_mass_kg)
            dC_CaCO3_precipitated_mol_m3_dt = r_R3

        factor = 1 / (porosity * V_particle * N_particles)

        dC_CH_void_mol_m3_dt = -factor * dM_CH_solid_mol_dt - factor * kO * solid_liquid_total_surface * (
                C_CH_void_mol_m3 - C_CH_bulk_mol_m3)
        dC_CH_bulk_mol_m3_dt = factor * kO * solid_liquid_total_surface * (C_CH_void_mol_m3 - C_CH_bulk_mol_m3) - r_R3

        dC_OH_void_mol_m3_dt = -2 * factor * dM_CH_solid_mol_dt - factor * kO * solid_liquid_total_surface * (
                C_OH_void_mol_m3 - C_OH_bulk_mol_m3)
        dC_OH_bulk_mol_m3_dt = factor * kO * solid_liquid_total_surface * (
                C_OH_void_mol_m3 - C_OH_bulk_mol_m3) - r_recomb - r_R5

        dC_carbonate_mol_m3_dt = r_R1 - r_R2 + r_R5
        dC_bicarbonate_mol_m3_dt = r_R2 - r_R3
        dC_proton_mol_m3_dt = r_R1 + r_R2 - r_recomb
        dC_CaCO3_shrinked_mol_m3_dt = dM_CSH_solid_mol_dt / slurry_volume


        if C_CaCO3_precipitated_mol_m3 > 0:
            dV_deposit_dt = dC_CaCO3_precipitated_mol_m3_dt * slurry_volume * molar_vol_CaCO3 / (
                    (1 - CaCO3_porosity) * N_particles)
            dR_dt = dV_deposit_dt / (4 * np.pi * (particle_diameter / 2) ** 2)
        else:
            dR_dt = 0.0

        dparticle_dt = 2 * dR_dt
        dalpha_dt = (((4 / 3) * np.pi * N_particles * (particle_diameter / 2) ** 3) - ((4 / 3) * np.pi * N_particles * (core_diameter / 2) ** 3)) / ((4 / 3) * np.pi * N_particles * (dp / 2) ** 3)

        dy = [
            dC_CH_void_mol_m3_dt, dC_CH_bulk_mol_m3_dt, dM_CH_solid_mol_dt,
            dC_OH_void_mol_m3_dt, dC_OH_bulk_mol_m3_dt,
            dC_bicarbonate_mol_m3_dt, dC_proton_mol_m3_dt, dC_carbonate_mol_m3_dt,
            dC_CaCO3_precipitated_mol_m3_dt, dM_CSH_solid_mol_dt,
            dC_CaCO3_shrinked_mol_m3_dt, dCO2_aq_dt, dalpha_dt, dparticle_dt, dcore_dt
        ]

        # Convert complex derivatives to real if close
        dy = [np.real_if_close(d) for d in dy]
        dM_CH_solid_mol_dt = min(dM_CH_solid_mol_dt, 0.0)

        # (C_CH_void_mol_m3, C_CH_bulk_mol_m3, M_CH_solid_mol,
        #  C_OH_void_mol_m3, C_OH_bulk_mol_m3,
        #  C_bicarbonate_mol_m3, C_proton_mol_m3,
        #  C_carbonate_mol_m3, C_CaCO3_precipitated_mol_m3,
        #  M_CSH_solid_mol, C_CaCO3_shrinked_mol_m3,
        #  C_CO2_aq_mol_m3, alpha,
        #  particle_diameter, core_diameter) = y
        # print(
        #     f"Time={tt:.2f}s: particle_diameter= {particle_diameter}, core_diameter = {core_diameter}, dp= {dp} , alpha={alpha:.6f}")
        print(
            f"Time={tt:.2f}s: MCH= {M_CH_solid_mol}, MCSH = {M_CSH_solid_mol}, dMCSHdt= {dM_CSH_solid_mol_dt}")
        #
        # print(f'Time={tt:.2f}s: C_CH_void_mol_m3= {C_CH_void_mol_m3}, C_CH_bulk_mol_m3= {C_CH_bulk_mol_m3}, M_CH_solid_mol= {M_CH_solid_mol}, C_OH_void_mol_m3= {C_OH_void_mol_m3}, C_OH_bulk_mol_m3= {C_OH_bulk_mol_m3}, C_bicarbonate_mol_m3= {C_bicarbonate_mol_m3}, C_proton_mol_m3= {C_proton_mol_m3}, C_carbonate_mol_m3= {C_carbonate_mol_m3}, C_CaCO3_precipitated_mol_m3= {C_CaCO3_precipitated_mol_m3}, M_CSH_solid_mol= {M_CSH_solid_mol}, C_CaCO3_shrinked_mol_m3= {C_CaCO3_shrinked_mol_m3}, C_CO2_aq_mol_m3= {C_CO2_aq_mol_m3}, alpha= {alpha}, particle_diameter= {particle_diameter}, core_diameter= {core_diameter}')
        # print(f'Ksp_CaCO3={Ksp_CaCO3}, and r_R3={r_R3}')

        return dy

    sol = solve_ivp(
        fun=odes,
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method='BDF',
        rtol=1e-3,
        atol=1e-5,
        max_step=np.inf  # let the solver choose steps adaptively
    )

    print("Integrator success:", sol.success)
    print("Integrator message:", sol.message)
    return {
        't': sol.t,
        'y': sol.y,
        'particle_diameter': sol.y[13],  # absolute particle diameter (m)
        'core_diameter': sol.y[14],  # absolute core diameter (m)
    }

# def solve_model(t, y0, tii, tvi, theta):
#     """
#     Model simulating aqueous carbonation kinetics and equilibria:
#     Ca(OH)2 dissolution, carbonate speciation, CaCO3 precipitation,
#     with temperature and pressure variations.
#     Reference equilibrium constants at T_ref = 298 K (unitless)
#         Reaction numbering for clarity:
#     R1: CO2 hydration/dissociation: CO2 + H2O ⇌ H+ + HCO3-
#     R2: Bicarbonate dissociation: HCO3- ⇌ CO3^2- + H+ (pKa2)
#     R3: Calcite precipitation inverse solubility: Ca2+ + CO3^2- ⇌ CaCO3(s)
#     R4: Water ion recombination: H+ + OH- ⇌ H2O
#     R5: Hydroxycarbonate formation: CO2 + OH- ⇌ HCO3-
#
#     Parameters:
#         t          : array_like, time points (s)
#         y0         : initial conditions for species (mol/L), or None for defaults
#         tii : dict of constants, physical and material (time-invariant)
#         tvi: dict of time-dependent profiles (temperature K, pressure Pa)
#         theta      : array_like, kinetic parameters (Arrhenius pre-exponentials,
#                      activation energies (J/mol), diffusivity params (m^2/s, dimensionless))
#
#     Returns:
#         dict: keys 't' (time steps, s), 'y' (species concentrations matrix, mol/L),
#               'R_particle' (particle radius evolution, m), 'R_particle0' (initial radius, m)
#     """
#     t = np.array(t, dtype=np.float64)                     # time array, s
#     theta = np.array(theta, dtype=np.float64)             # kinetic parameters
#     tii = {k: float(v) for k, v in tii.items()}            # time-invariant constants dict
#     T_profile = np.array(tvi['T'], dtype=np.float64)       # temperature profile, K
#     P_profile = np.array(tvi['P'], dtype=np.float64)       # pressure profile, bar
#
#
#
#     R_gas = 8.314  # J/(mol·K) universal gas constant
#
#     # Extract kinetics parameters
#     A_arrh = theta[0:8]      # Arrhenius pre-exponentials (units depend on rate law, e.g. 1/s or m³/(mol·s))
#     Ea_arrh = theta[8:16]    # Activation energies (J/mol)
#     etha = theta[16]   # Diffusivity decay coefficient (dimensionless)
#
#     uncvar = {                                   # Uncontrollable variables
#         'CaO_leachable_solid_fr': 0.26061,        # Solid mass fraction of CaO in raw material (dimensionless)
#         'CaOH2_leachable_solid_fr': 0.0000001,   # Solid mass fraction of Ca(OH)2 in raw material (dimensionless)
#         'CaO_unleachable_solid_fr': 0.4034,       # Solid mass fraction of equivalent CaO in raw material (dimensionless)
#         'porosity': 0.3,                         # Solid mineral porosity (dimensionless)
#         'CaCO3_porosity': 0.3,                   # Porosity of CaCO3 deposit (dimensionless)
#         'rho_solid_material': 2260,              # Density solid material (kg/m³)
#         'rho_solid_CaCO3': 2710,                 # Density CaCO3 (kg/m³)
#         'slurry_volume': 0.5,                    # Liquid volume (L)
#         'LOI': 0.01903,                          # LOI of mineral in weight fraction (dimensionless)
#     }
#
#     const = {                       # Constants (SI units)
#         'Mw_CaO': 56e-3,            # Molecular weight CaO (kg/mol)
#         'Mw_CaOH2': 74e-3,          # Molecular weight Ca(OH)2 (kg/mol)
#         'Mw_CaCO3': 0.1001,         # Molecular weight CaCO3 (kg/mol)
#         'Mw_CO2': 44e-3,           # Molecular weight CO2 (kg/mol)
#         'k5f_fixed': 1.4e11,        # Fixed rate constant R4 (m³/(mol·s))
#         'T_ref': 298,               # Reference temperature (K)
#         # Equilibrium constants at T_ref (unitless)
#         'Keq_R1_ref': 10 ** (-6.357),
#         'Keq_R2_ref': 10 ** (-10.33),
#         'Keq_R3_ref': 10 ** (9.108),
#         'Keq_R4_ref': 10 ** (14),
#         'Keq_R5_ref': 10 ** (7.66),
#         # Enthalpy changes for van't Hoff (J/mol)
#         'deltaH_R1': -20000,
#         'deltaH_R2': 14400,
#         'deltaH_R3': -12100,
#         'deltaH_R4': 55800,
#         'deltaH_R5': -41400,
#     }
#
#     # Initial experimental pH (-log10[H+], dimensionless)
#     # pHExp_0 = 11.53
#     pHExp_0 = 11.53
#
#     # Controllable variables
#     dp = tii['dp']                               # Particle diameter (m)
#     SLR = tii['SLR']                             # Solid-to-liquid ratio (kg/L)
#     t_CO2_inject = tii.get('t_CO2_inject', 0.0)  # CO2 injection time (s)
#
#     # Uncontrollable variables unpacked
#     CaO_leachable_solid_fr = uncvar['CaO_leachable_solid_fr']         # dimensionless
#     CaOH2_leachable_solid_fr = uncvar['CaOH2_leachable_solid_fr']     # dimensionless
#     LOI = uncvar['LOI']                                               # dimensionless
#     porosity = uncvar['porosity']                                     # dimensionless
#     rho_solid_material = uncvar['rho_solid_material']                 # kg/m³
#     slurry_volume = uncvar['slurry_volume']                           # L
#     rho_solid_CaCO3 = uncvar['rho_solid_CaCO3']                       # kg/m³
#     CaCO3_porosity = uncvar['CaCO3_porosity']                         # dimensionless
#
#     # Constants unpacked
#     Mw_CaO = const['Mw_CaO']                       # kg/mol
#     Mw_CaOH2 = const['Mw_CaOH2']                   # kg/mol
#     Mw_CaCO3 = const['Mw_CaCO3']                   # kg/mol
#     Mw_CO2 = const['Mw_CO2']                       # kg/mol
#     k5f_fixed = const['k5f_fixed']                 # m³/(mol·s) for R4 forward
#     T_ref = const['T_ref']                         # K
#     Keq_R1_ref = const['Keq_R1_ref']               # unitless
#     Keq_R2_ref = const['Keq_R2_ref']               # unitless
#     Keq_R3_ref = const['Keq_R3_ref']               # unitless
#     Keq_R4_ref = const['Keq_R4_ref']               # unitless
#     Keq_R5_ref = const['Keq_R5_ref']               # unitless
#     deltaH_R1 = const['deltaH_R1']                 # J/mol
#     deltaH_R2 = const['deltaH_R2']                 # J/mol
#     deltaH_R3 = const['deltaH_R3']                 # J/mol
#     deltaH_R4 = const['deltaH_R4']                 # J/mol
#     deltaH_R5 = const['deltaH_R5']                 # J/mol
#
#     # Particle geometry and count
#     R_particle0 = dp / 2                                         # Particle radius (m)
#     molar_vol_CaCO3 = Mw_CaCO3 / rho_solid_CaCO3                 # Molar volume CaCO3 (m³/mol)
#     V_particle = (4/3) * np.pi * R_particle0**3                  # Volume per particle (m³)
#     V_solid_per_particle = (1 - porosity) * V_particle           # Solid volume per particle (m³)
#     mass_particle = rho_solid_material * V_solid_per_particle    # Mass per particle (kg)
#     total_mass_kg = SLR * slurry_volume                          # Total solid mass (kg)
#     dry_mass_kg = total_mass_kg * (1 - LOI)                      # Dry solid mass (kg)
#     N_particles = total_mass_kg / mass_particle                  # Number of particles (dimensionless)
#
#     # Ion radii and constants for activity coefficients
#     aCa = 1.0e-13
#     aOH = 3.0e-10
#     aH = 9.0e-11
#     aHCO3 = 4.0e-10
#     aCO3 = 4.0e-10
#     A_DH = 0.037
#     B_DH = 1.04e8
#
#     def vanthoff(K_eq_ref, deltaH, T):
#         # Calculate temperature-dependent equilibrium constant (unitless)
#         return K_eq_ref * np.exp(-deltaH / R_gas * (1/T - 1/T_ref))
#
#     def arrhenius(A, Ea, T):
#         # Arrhenius rate constant (units depend on reaction)
#         return A * np.exp(-Ea / (R_gas * T))
#
#     def T_at_time(tt):
#         # Interpolate temperature at time tt (K)
#         return np.interp(tt, t, T_profile)
#
#     def P_at_time(tt):
#         # Interpolate pressure at time tt (bar)
#         return np.interp(tt, t, P_profile)
#
#     def calcGamma(I, z, a_i, A, B):
#         # Ion activity coefficient (dimensionless)
#         lnGamma = (-A * z**2 * np.sqrt(I*1e3) /
#                    (1 + a_i * B * np.sqrt(I*1e3)) +
#                    (0.2 - 4.17e-5 * I*1e3) * A * z**2 * I*1e3 / np.sqrt(1000))
#         gamma = np.exp(lnGamma)
#         return np.clip(gamma, 0.1, 1.0)
#
#
#     if y0 is None:
#         CaO_leachable_solid_mol = total_mass_kg * CaO_leachable_solid_fr / Mw_CaO            # free lime mass in raw material (mol)
#         CaOH2_leachable_solid_mol = total_mass_kg * CaOH2_leachable_solid_fr / Mw_CaOH2      # portlandite mass in raw material (mol)
#         CH_solid_mol = CaOH2_leachable_solid_mol + CaO_leachable_solid_mol                   # Total reactable material mass (mol)
#         solid_vol_m3 = N_particles * V_solid_per_particle                                    # total volume of particles (m³)
#         solid_vol_L = solid_vol_m3 * 1000                                                    # total volume of particles (L)
#         # CH_solid_mol_L = CH_solid_mol / solid_vol_L                                        # Reactable material solid concentration (mol/L)
#         CSH_solid_mol = total_mass_kg * uncvar['CaO_unleachable_solid_fr'] / Mw_CaO          # hard to leach CaO equivalent mass in raw material (mol)
#         # CSH_solid_mol_L = CSH_solid_mol / solid_vol_L                                      # hard to leach CaO equivalent concentration in solid (mol/L)
#
#         y0 = [1e-10, 0.0, CH_solid_mol,                                                    # initial concentrations mol/L or mol
#               1e-7, 10**(pHExp_0 - 14),
#               0.0, 10**(-pHExp_0), 0.0, 0.0,
#               CSH_solid_mol, 0.0, 0.0, 0.0, dp, dp]  # initial particle diameter and core diameter (m)
#
#     def kspCaCO3(T):  # T in K
#         # Solubility product of CaCO3 (unitless)
#         return 10**(-(0.01183*(T-273) + 8.03))
#
#     def kspCaOH2(T):  # T in K
#         # Solubility product of Ca(OH)2 (unitless)
#         S_CH = -0.01087 * T + 1.7465  # solubility g/L
#         Mw = 74.093                   # g/mol
#         S_CH_molL = (S_CH / 1000) / (Mw / 1000)  # mol/L
#         return 4 * S_CH_molL**3
#
#     # def odes(tt, y):
#     #     # ODE system for species concentrations and surface processes
#     #     (C_CH_surface_mol_L, C_CH_bulk_mol_L, M_CH_solid_mol,
#     #      C_OH_surface_mol_L, C_OH_bulk_mol_L,
#     #      C_bicarbonate_mol_L, C_proton_mol_L,
#     #      C_carbonate_mol_L, C_CaCO3_precipitated_mol_L,
#     #      M_CSH_solid_mol, C_CaCO3_shrinked_mol_L,
#     #      C_CO2_aq_mol_L, alpha,
#     #      particle_diameter, core_diameter) = y
#     #
#     #     # Positivity enforcement
#     #     species = [C_CH_surface_mol_L, C_CH_bulk_mol_L, M_CH_solid_mol,
#     #                C_OH_surface_mol_L, C_OH_bulk_mol_L,
#     #                C_bicarbonate_mol_L, C_proton_mol_L,
#     #                C_carbonate_mol_L, C_CaCO3_precipitated_mol_L,
#     #                M_CSH_solid_mol, C_CaCO3_shrinked_mol_L]
#     #     species = np.maximum(species, 1e-20)
#     #     [C_CH_surface_mol_L, C_CH_bulk_mol_L, M_CH_solid_mol,
#     #      C_OH_surface_mol_L, C_OH_bulk_mol_L,
#     #      C_bicarbonate_mol_L, C_proton_mol_L,
#     #      C_carbonate_mol_L, C_CaCO3_precipitated_mol_L,
#     #      M_CSH_solid_mol, C_CaCO3_shrinked_mol_L] = species.tolist()
#     #
#     #     # if tt < t_CO2_inject:
#     #     #     C_CO2_aq_mol_L = 1e-20  # mol/L
#     #     # else:
#     #     #     C_CO2_aq_mol_L = max(C_CO2_aq_mol_L, 1e-20)
#     #
#     #     T_loc = T_at_time(tt)  # K
#     #     P_loc = P_at_time(tt)  # bar
#     #
#     #     # Rate constants (units vary by reaction, consistent with rate expressions)
#     #     k_R1f = arrhenius(A_arrh[0], Ea_arrh[0], T_loc)
#     #     k_R2f = arrhenius(A_arrh[1], Ea_arrh[1], T_loc)
#     #     k_R5f = arrhenius(A_arrh[2], Ea_arrh[2], T_loc)
#     #     k_R3f = arrhenius(A_arrh[3], Ea_arrh[3], T_loc)
#     #     kL = arrhenius(A_arrh[4], Ea_arrh[4], T_loc)
#     #     kO = arrhenius(A_arrh[5], Ea_arrh[5], T_loc)
#     #     D = arrhenius(A_arrh[6], Ea_arrh[6], T_loc)
#     #     kLa = arrhenius(A_arrh[7], Ea_arrh[7], T_loc)
#     #
#     #     # Equilibrium constants (unitless) adjusted for temperature
#     #     Keq_R1 = vanthoff(Keq_R1_ref, deltaH_R1, T_loc)
#     #     Keq_R2 = vanthoff(Keq_R2_ref, deltaH_R2, T_loc)
#     #     Keq_R3 = vanthoff(Keq_R3_ref, deltaH_R3, T_loc)
#     #     Keq_R4 = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
#     #     Keq_R5 = vanthoff(Keq_R5_ref, deltaH_R5, T_loc)
#     #
#     #     Ksp_CaOH2 = kspCaOH2(T_loc)  # unitless
#     #     Ksp_CaCO3 = kspCaCO3(T_loc)  # unitless
#     #
#     #
#     #
#     #
#     #     # # Particle volume and radius calculations for CaCO3 deposit growth
#     #     # C_CaCO3_total_mol_L = C_CaCO3_precipitated_mol_L + C_CaCO3_shrinked_mol_L  # total mol CaCO3 precipitated (mol.L-1)
#     #     # V_solid_CaCO3 = C_CaCO3_total_mol_L * slurry_volume * molar_vol_CaCO3      # solid volume CaCO3 (m³)
#     #     # V_deposit_CaCO3 = V_solid_CaCO3 / (1 - CaCO3_porosity)                     # deposited volume CaCO3 incl. pores (m³)
#     #     # V_particle_deposit = V_deposit_CaCO3 / N_particles                         # volume deposit per particle (m³)
#     #
#     #     # if V_particle_deposit > 1e-20:
#     #     #     coeffs = [(4/3) * np.pi, 4 * np.pi * R_particle0, 4 * np.pi * R_particle0**2, -V_particle_deposit]
#     #     #     roots = np.roots(coeffs)
#     #     #     roots_real = roots[np.isreal(roots)].real
#     #     #     roots_pos = roots_real[roots_real > 0]
#     #     #     dR = min(roots_pos) if roots_pos.size > 0 else 0
#     #     #     R_current = max(R_particle0, R_particle0 + dR)  # particle radius (m)
#     #     # else:
#     #     #     R_current = R_particle0  # m
#     #
#     #     specific_surface_area = N_particles * 4 * np.pi * (particle_diameter / 2)**2 / slurry_volume  # total particle surface area in reactor, m²/L  (slurry_volume in L)
#     #
#     #     # Ionic strength (mol/L)
#     #     I = max(2*C_CH_bulk_mol_L + 0.5*C_OH_bulk_mol_L + 2*C_bicarbonate_mol_L + 0.5*C_proton_mol_L + 0.5*C_carbonate_mol_L, 1e-4)
#     #
#     #     # Activity coefficients (dimensionless)
#     #     gamma_Ca = calcGamma(I, 2, aCa, A_DH, B_DH)
#     #     gamma_OH = calcGamma(I, 1, aOH, A_DH, B_DH)
#     #     gamma_CO3 = calcGamma(I, 2, aCO3, A_DH, B_DH)
#     #     gamma_H = calcGamma(I, 1, aH, A_DH, B_DH)
#     #     gamma_HCO3 = calcGamma(I, 1, aHCO3, A_DH, B_DH)
#     #
#     #     # Reaction rates (units consistent with rate laws: mol/(L·s), or equivalent)
#     #     r_R1 = (k_R1f * C_CO2_aq_mol_L - (k_R1f / Keq_R1) * (C_carbonate_mol_L * gamma_HCO3) * (C_proton_mol_L * gamma_H))
#     #     r_R2 = (k_R2f * (C_carbonate_mol_L * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate_mol_L * gamma_CO3) * (C_proton_mol_L * gamma_H))
#     #     r_R5 = (k_R5f * C_CO2_aq_mol_L * (C_OH_bulk_mol_L * gamma_OH) - (k_R5f / Keq_R5) * (C_carbonate_mol_L * gamma_HCO3))
#     #     if C_CH_surface_mol_L>0.0001:
#     #         r_R3 = k_R3f * ((C_CH_bulk_mol_L * gamma_Ca) * (C_bicarbonate_mol_L * gamma_CO3) / Ksp_CaCO3 - 1)
#     #     else:
#     #         r_R3 = 0
#     #
#     #     r_recomb = k5f_fixed * (C_proton_mol_L * gamma_H) * (C_OH_bulk_mol_L * gamma_OH) - k5f_fixed / Keq_R4
#     #
#     #     # Saturation ratio for Ca(OH)2 (dimensionless)
#     #     S = (C_CH_surface_mol_L * gamma_Ca) * ((C_OH_surface_mol_L * gamma_OH) ** 2) / Ksp_CaOH2
#     #
#     #     dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol  # leachable carbonable solid dissolution rate (mol/s)
#     #     if C_CH_bulk_mol_L>0.0001:
#     #         dM_CSH_solid_mol_dt = 0.0                            # uleachable carbonable solid consumption rate (mol/s)
#     #
#     #     bulk_CH_fading_concentration = 1e-6                  # small concentration threshold (mol/L) of CH in bulk where mechanism changes
#     #     dcore_dt = 0.0                                       # core diameter change rate (m/s)
#     #
#     #
#     #     if C_CH_surface_mol_L<0.0001:
#     #         # Product layer thickness and core/particle radii
#     #         delta = max((particle_diameter / 2) - (core_diameter / 2), 1e-12)  # product layer thickness [m]
#     #         # Effective diffusivity (time-dependent)
#     #         D_e = D * np.exp(-etha * tt)
#     #         dM_CSH_solid_mol_dt = -(4 * N_particles * np.pi * (particle_diameter / 2) * (core_diameter / 2) * (C_CO2_aq_mol_L) * D_e) / delta
#     #
#     #         # Update the solids according to total CaO consumption
#     #         dcore_dt = (-2 / 3) * R_particle0 * (1 - (M_CSH_solid_mol * (Mw_CO2 / Mw_CaO)) / dry_mass_kg) ** (
#     #                     -2 / 3) * ((dM_CSH_solid_mol_dt * (Mw_CO2 / Mw_CaO)) / dry_mass_kg)
#     #     else:
#     #         dcore_dt = 0.0
#     #
#     #     dC_CH_surface_mol_L_dt = -(1 / (porosity * V_particle * N_particles)) * dM_CH_solid_mol_dt - (1 / porosity * V_particle * N_particles) * kO * specific_surface_area * (C_CH_surface_mol_L - C_CH_bulk_mol_L)      # mol/(L·s)
#     #     dC_CH_bulk_mol_L_dt = (1 / porosity * V_particle * N_particles) * kO * specific_surface_area * (C_CH_surface_mol_L - C_CH_bulk_mol_L) - r_R3                                                                      # mol/(L·s)
#     #     dC_OH_surface_mol_L_dt = -2 * (1 / (porosity * V_particle * N_particles)) * dM_CH_solid_mol_dt - (1 / porosity * V_particle * N_particles) * kO * specific_surface_area * (C_OH_surface_mol_L - C_OH_bulk_mol_L)  # mol/(L·s)
#     #     dC_OH_bulk_mol_L_dt = (1 / porosity * V_particle * N_particles) * kO * specific_surface_area * (C_OH_surface_mol_L - C_OH_bulk_mol_L) - r_recomb - r_R5                                                           # mol/(L·s)
#     #
#     #     dC_carbonate_mol_L_dt = r_R1 - r_R2 + r_R5                                                            # mol/(L·s)
#     #     dC_bicarbonate_mol_L_dt = r_R2 - r_R3                                                                 # mol/(L·s)
#     #     dC_proton_mol_L_dt = r_R1 + r_R2 - r_recomb                                                           # mol/(L·s)
#     #
#     #     dC_CaCO3_precipitated_mol_L_dt = r_R3                                                                 # mol/(L·s)
#     #
#     #     CO2_sat = P_loc * 0.035 * np.exp(2400 * ((1 / T_loc) - (1 / T_ref)))                                  # Saturation CO2 (mol/L) in water
#     #     dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_L) - r_R1 - r_R5 - (dM_CSH_solid_mol_dt/slurry_volume)     # mol/(L·s)
#     #
#     #     dC_CaCO3_shrinked_mol_L_dt = dM_CSH_solid_mol_dt / slurry_volume
#     #
#     #     # Total rate of carbonation (precipitation + shrinking core)
#     #     dalpha_dt = (((dM_CSH_solid_mol_dt+dM_CH_solid_mol_dt)* (Mw_CO2/Mw_CaO))/dry_mass_kg)
#     #
#     #     if C_CaCO3_precipitated_mol_L > 0:
#     #         dV_deposit_dt = dC_CaCO3_precipitated_mol_L_dt * slurry_volume * molar_vol_CaCO3 / ((1 - CaCO3_porosity)*N_particles)
#     #         dR_dt = dV_deposit_dt / (4 * np.pi * ((particle_diameter / 2)) ** 2)
#     #     else:
#     #         dR_dt = 0.0
#     #
#     #     dparticle_dt = 2 * dR_dt
#     #
#     #     variables_to_check = [k_R1f, k_R2f, k_R3f, r_R1, r_R2, r_R3, S, dM_CH_solid_mol_dt, dM_CSH_solid_mol_dt]
#     #
#     #     for var in variables_to_check:
#     #         if np.iscomplex(var):
#     #
#     #             var = var.real  # Fix or alternative handling
#     #
#     #
#     #     dy = [
#     #         dC_CH_surface_mol_L_dt, dC_CH_bulk_mol_L_dt, dM_CH_solid_mol_dt,
#     #         dC_OH_surface_mol_L_dt, dC_OH_bulk_mol_L_dt,
#     #         dC_bicarbonate_mol_L_dt, dC_proton_mol_L_dt, dC_carbonate_mol_L_dt,
#     #         dC_CaCO3_precipitated_mol_L_dt, dM_CSH_solid_mol_dt,
#     #         dC_CaCO3_shrinked_mol_L_dt, dCO2_aq_dt, dalpha_dt, dparticle_dt, dcore_dt
#     #     ]
#     #
#     #     # Convert any complex derivative to real part if close
#     #     dy = [np.real_if_close(d) for d in dy]
#     #     # print(f"dv at={tt}: {dV_deposit_dt}, and dR is {dR_dt}, and dc is {dC_CaCO3_precipitated_mol_L_dt}")
#     #     print(f'mCH is {M_CH_solid_mol}, MCSH is {M_CSH_solid_mol}, C_CH_surf is {C_CH_surface_mol_L}, C_CH_bulk is {C_CH_bulk_mol_L}')
#     #
#     #     # print(f'CaCO3_precipitated_mol_L is {C_CaCO3_precipitated_mol_L}')
#     #
#     #     return dy
#
#     def odes(tt, y):
#         (C_CH_surface_mol_L, C_CH_bulk_mol_L, M_CH_solid_mol,
#          C_OH_surface_mol_L, C_OH_bulk_mol_L,
#          C_bicarbonate_mol_L, C_proton_mol_L,
#          C_carbonate_mol_L, C_CaCO3_precipitated_mol_L,
#          M_CSH_solid_mol, C_CaCO3_shrinked_mol_L,
#          C_CO2_aq_mol_L, alpha,
#          particle_diameter, core_diameter) = y
#
#         # Enforce minimum positive concentrations for stability
#         species = [C_CH_surface_mol_L, C_CH_bulk_mol_L, M_CH_solid_mol,
#                    C_OH_surface_mol_L, C_OH_bulk_mol_L,
#                    C_bicarbonate_mol_L, C_proton_mol_L,
#                    C_carbonate_mol_L, C_CaCO3_precipitated_mol_L,
#                    M_CSH_solid_mol, C_CaCO3_shrinked_mol_L]
#         species = np.maximum(species, 1e-20)
#         (C_CH_surface_mol_L, C_CH_bulk_mol_L, M_CH_solid_mol,
#          C_OH_surface_mol_L, C_OH_bulk_mol_L,
#          C_bicarbonate_mol_L, C_proton_mol_L,
#          C_carbonate_mol_L, C_CaCO3_precipitated_mol_L,
#          M_CSH_solid_mol, C_CaCO3_shrinked_mol_L) = species
#
#         T_loc = T_at_time(tt)
#         P_loc = P_at_time(tt)
#
#         # Rate constants & equilibrium constants (Arrhenius & van't Hoff)
#         k_R1f = arrhenius(A_arrh[0], Ea_arrh[0], T_loc)
#         k_R2f = arrhenius(A_arrh[1], Ea_arrh[1], T_loc)
#         k_R5f = arrhenius(A_arrh[2], Ea_arrh[2], T_loc)
#         k_R3f = arrhenius(A_arrh[3], Ea_arrh[3], T_loc)
#         kL = arrhenius(A_arrh[4], Ea_arrh[4], T_loc)
#         kO = arrhenius(A_arrh[5], Ea_arrh[5], T_loc)
#         D = arrhenius(A_arrh[6], Ea_arrh[6], T_loc)
#         kLa = arrhenius(A_arrh[7], Ea_arrh[7], T_loc)
#
#         Keq_R1 = vanthoff(Keq_R1_ref, deltaH_R1, T_loc)
#         Keq_R2 = vanthoff(Keq_R2_ref, deltaH_R2, T_loc)
#         Keq_R3 = vanthoff(Keq_R3_ref, deltaH_R3, T_loc)
#         Keq_R4 = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
#         Keq_R5 = vanthoff(Keq_R5_ref, deltaH_R5, T_loc)
#
#         Ksp_CaOH2 = kspCaOH2(T_loc)
#         Ksp_CaCO3 = kspCaCO3(T_loc)
#
#         specific_surface_area = N_particles * 4 * np.pi * (particle_diameter / 2) ** 2 / slurry_volume
#
#         I = max(
#             2 * C_CH_bulk_mol_L + 0.5 * C_OH_bulk_mol_L + 2 * C_bicarbonate_mol_L + 0.5 * C_proton_mol_L + 0.5 * C_carbonate_mol_L,
#             1e-4)
#         gamma_Ca = calcGamma(I, 2, aCa, A_DH, B_DH)
#         gamma_OH = calcGamma(I, 1, aOH, A_DH, B_DH)
#         gamma_CO3 = calcGamma(I, 2, aCO3, A_DH, B_DH)
#         gamma_H = calcGamma(I, 1, aH, A_DH, B_DH)
#         gamma_HCO3 = calcGamma(I, 1, aHCO3, A_DH, B_DH)
#
#         # Aqueous reaction rates
#         r_R1 = k_R1f * C_CO2_aq_mol_L - (k_R1f / Keq_R1) * (C_carbonate_mol_L * gamma_HCO3) * (C_proton_mol_L * gamma_H)
#         r_R2 = k_R2f * (C_carbonate_mol_L * gamma_HCO3) - (k_R2f / Keq_R2) * (C_bicarbonate_mol_L * gamma_CO3) * (
#                 C_proton_mol_L * gamma_H)
#         r_R5 = k_R5f * C_CO2_aq_mol_L * (C_OH_bulk_mol_L * gamma_OH) - (k_R5f / Keq_R5) * (
#                 C_carbonate_mol_L * gamma_HCO3)
#
#         if C_CH_surface_mol_L > 1e-4:
#             r_R3 = k_R3f * ((C_CH_surface_mol_L * gamma_Ca) * (C_bicarbonate_mol_L * gamma_CO3) / Ksp_CaCO3 - 1)
#         else:
#             r_R3 = 0.0
#
#         r_recomb = k5f_fixed * (C_proton_mol_L * gamma_H) * (C_OH_bulk_mol_L * gamma_OH) - k5f_fixed / Keq_R4
#
#         # Saturation ratio for Ca(OH)2 dissolution
#         S = (C_CH_surface_mol_L * gamma_Ca) * ((C_OH_surface_mol_L * gamma_OH) ** 2) / Ksp_CaOH2
#
#         threshold_bulk_CH = 1e-4
#         threshold_surface_CH = 1e-4
#         dcore_dt = 0.0
#
#         # Bulk dissolution regime
#         if C_CH_bulk_mol_L > threshold_bulk_CH:
#             dM_CH_solid_mol_dt = -kL * (1 - S) * M_CH_solid_mol
#             dM_CSH_solid_mol_dt = 0.0
#             dcore_dt = 0.0
#
#         # Surface reaction regime - modified to ensure consumption of M_CH_solid_mol
#         elif C_CH_bulk_mol_L <= threshold_bulk_CH and (
#                 C_CH_surface_mol_L > threshold_surface_CH or M_CH_solid_mol > 1e-6):
#             driving_force = max(1e-3, (1 - S))  # ensure nonzero driving force
#             dM_CH_solid_mol_dt = -kL * driving_force * M_CH_solid_mol
#
#             r_R3s = ((C_CH_surface_mol_L * gamma_Ca) * (C_bicarbonate_mol_L * gamma_CO3) / Ksp_CaCO3 - 1)
#             dM_CSH_solid_mol_dt = 0
#             dC_CaCO3_precipitated_mol_L_dt = r_R3s
#             dcore_dt = 0.0
#
#         # Shrinking core regime
#         else:
#             dM_CH_solid_mol_dt = 0.0
#
#             # Calculate product layer thickness
#             delta = max((particle_diameter / 2) - (core_diameter / 2), 1e-12)
#
#             # Effective diffusivity
#             D_e = D * np.exp(-etha * tt)
#
#             # Diffusion-controlled solid consumption rate (mol/s)
#             dM_CSH_solid_mol_dt = -(4 * N_particles * np.pi * (particle_diameter / 2) * (
#                     core_diameter / 2) * C_CO2_aq_mol_L * D_e) / delta
#
#             # Make sure dM_CSH_solid_mol_dt is negative or zero (solid consumption)
#             dM_CSH_solid_mol_dt = min(dM_CSH_solid_mol_dt, 0.0)
#
#             # Calculate core shrinkage rate
#             term = 1 - (M_CSH_solid_mol * (Mw_CO2 / Mw_CaO)) / dry_mass_kg
#             term = max(term, 1e-12)  # Avoid zero or negative for fractional power
#
#             dcore_dt = (-2 / 3) * R_particle0 * term ** (-2 / 3) * (
#                     (dM_CSH_solid_mol_dt * (Mw_CO2 / Mw_CaO)) / dry_mass_kg)
#
#         factor = 1 / (porosity * V_particle * N_particles)
#
#         dC_CH_surface_mol_L_dt = -factor * dM_CH_solid_mol_dt - factor * kO * specific_surface_area * (
#                 C_CH_surface_mol_L - C_CH_bulk_mol_L)
#         dC_CH_bulk_mol_L_dt = factor * kO * specific_surface_area * (C_CH_surface_mol_L - C_CH_bulk_mol_L) - r_R3
#
#         dC_OH_surface_mol_L_dt = -2 * factor * dM_CH_solid_mol_dt - factor * kO * specific_surface_area * (
#                 C_OH_surface_mol_L - C_OH_bulk_mol_L)
#         dC_OH_bulk_mol_L_dt = factor * kO * specific_surface_area * (
#                 C_OH_surface_mol_L - C_OH_bulk_mol_L) - r_recomb - r_R5
#
#         dC_carbonate_mol_L_dt = r_R1 - r_R2 + r_R5
#         dC_bicarbonate_mol_L_dt = r_R2 - r_R3
#         dC_proton_mol_L_dt = r_R1 + r_R2 - r_recomb
#
#         dC_CaCO3_precipitated_mol_L_dt = r_R3
#
#         CO2_sat = P_loc * 0.035 * np.exp(2400 * ((1 / T_loc) - (1 / T_ref)))
#         dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_L) - r_R1 - r_R5 - (dM_CSH_solid_mol_dt / slurry_volume)
#
#         dC_CaCO3_shrinked_mol_L_dt = dM_CSH_solid_mol_dt / slurry_volume
#
#         dalpha_dt = ((dM_CSH_solid_mol_dt + dM_CH_solid_mol_dt) * (Mw_CO2 / Mw_CaO)) / dry_mass_kg
#
#         if C_CaCO3_precipitated_mol_L > 0:
#             dV_deposit_dt = dC_CaCO3_precipitated_mol_L_dt * slurry_volume * molar_vol_CaCO3 / (
#                     (1 - CaCO3_porosity) * N_particles)
#             dR_dt = dV_deposit_dt / (4 * np.pi * (particle_diameter / 2) ** 2)
#         else:
#             dR_dt = 0.0
#
#         dparticle_dt = 2 * dR_dt
#
#         dy = [
#             dC_CH_surface_mol_L_dt, dC_CH_bulk_mol_L_dt, dM_CH_solid_mol_dt,
#             dC_OH_surface_mol_L_dt, dC_OH_bulk_mol_L_dt,
#             dC_bicarbonate_mol_L_dt, dC_proton_mol_L_dt, dC_carbonate_mol_L_dt,
#             dC_CaCO3_precipitated_mol_L_dt, dM_CSH_solid_mol_dt,
#             dC_CaCO3_shrinked_mol_L_dt, dCO2_aq_dt, dalpha_dt, dparticle_dt, dcore_dt
#         ]
#
#         # Convert complex derivatives to real if close
#         dy = [np.real_if_close(d) for d in dy]
#         dM_CH_solid_mol_dt = min(dM_CH_solid_mol_dt, 0.0)
#         dM_CSH_solid_mol_dt = min(dM_CSH_solid_mol_dt, 0.0)
#
#         print(
#             f"Time={tt:.2f}s: M_CH={M_CH_solid_mol:.6f}, M_CSH={M_CSH_solid_mol:.6f}, C_CH_surf={C_CH_surface_mol_L:.2e}, C_CH_bulk={C_CH_bulk_mol_L:.2e}")
#
#         return dy
#
#     sol = solve_ivp(odes, (t[0], t[-1]), y0, t_eval=t, method='BDF', rtol=1e-5, atol=1e-8)
#     print("Integrator success:", sol.success)
#     print("Integrator message:", sol.message)
#     return {
#         't': sol.t,
#         'y': sol.y,
#         'particle_diameter': sol.y[13],  # absolute particle diameter (m)
#         'core_diameter': sol.y[14],  # absolute core diameter (m)
#     }





# --------------------------------------------------
# Experimental data arrays and plotting colors dict used in run_and_plot (assumed as per original)
# --------------------------------------------------

experimental_data_pH = np.array([
    [1, 11.53],
    [300, 11.16],
    [600, 11.17],
    [900, 10.63],
    [1200, 10.24],
    [1500, 9.75],
    [1800, 9.4],
    [2100, 9.1],
    [2400, 8.41],
    [2700, 7.29],
    [3000, 6.33],
    [3300, 6.24],
    [3600, 6.23],
])
timeExp_pH = experimental_data_pH[:, 0]
pHExp = experimental_data_pH[:, 1]

experimental_data_CaCO3 = np.array([
    [0, 0],
    [300, 0.207972],
    [600, 0.348255],
    [1800, 0.530721],
    [3600, 0.595467],
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
    [300, 0.19306721],
    [600, 0.03071524],
    [1800, 0.0],
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
    t = np.linspace(0,3600,300)
    T_profile = np.full_like(t,298.15)
    P_profile = np.full_like(t,1.01325)
    tii = {
        'SLR': 0.1,               # Solid-to-liquid ratio (kg/L)
        'dp': 10e-6,              # Diameter of solid particles (m)
        't_CO2_inject': 0.0       # Time when CO2 injection starts in the system (s), allows for switching CO2 source on/off in model
    }

    theta = [
        1.205e4, 4.813e7, 1.904e8, 3.340e5,
        5.458e-2, 4.694e2, 7.059e10, 9.585e1,
        36715, 15807, 39536, 47039,
        35801, 32132, 27550, 20037,
        1
    ]



    # theta = [
    #     1.205e4, 4.813e7, 1.904e8, 3.340e5,
    #     3.458e3, 4.694e2, 7.059e4, 9.585e1,
    #     36715, 15807, 39536, 47039,
    #     38801, 32132, 87550, 20037,
    #     2
    # ]
    tvi = {'T':T_profile, 'P':P_profile}

    result = solve_model(t, None, tii, tvi, theta)
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
    pH_calc = -np.log10(np.maximum(y_out[6]*1e-3,1e-14))
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

    # Obtain particle and core diameters from solution matrix
    particle_diameter_array = y_out[13]  # particle diameter in meters
    core_diameter_array = y_out[14]  # core diameter in meters

    # Convert to microns
    particle_diameter_microns = 1e6 * particle_diameter_array
    core_diameter_microns = 1e6 * core_diameter_array


    plt.figure(figsize=(8, 4))
    plt.plot(t_min, particle_diameter_microns, label='Particle diameter')
    plt.plot(t_min, core_diameter_microns, label='Core diameter', linestyle='--')
    plt.xlabel('Time (min)')
    plt.ylabel('Diameter (µm)')
    plt.title('Particle and Core Diameter Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Reporting summary
    print(f'Initial particle diameter: {particle_diameter_microns[0]:.2f} µm')
    print(f'Final particle diameter: {particle_diameter_microns[-1]:.2f} µm')
    print(f'Final core diameter: {core_diameter_microns[-1]:.2f} µm')
    print(f'Final product layer thickness: {(particle_diameter_microns[-1] - core_diameter_microns[-1]) / 2:.2f} µm')

    # Ionic species concentrations
    plt.figure(figsize=(12,6))
    plt.plot(t_min, y_out[4], label='OH⁻')
    plt.plot(t_min, y_out[7], label='HCO₃⁻')
    plt.plot(t_min, y_out[5], label='CO₃²⁻')
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (mol/L)')
    plt.title('Ionic Species Concentrations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Activity coefficients
    def calcGamma_np(I,z,a_i,A=0.037,B=1.04e8):
        lnGamma = (-A*z**2*np.sqrt(I*1e3)/(1+a_i*B*np.sqrt(I*1e3))
                + (0.2 -4.17e-5*I*1e3)*A*z**2*I*1e3/np.sqrt(1000))
        gamma = np.exp(lnGamma)
        return np.clip(gamma,0.1,1.0)

    I_calc = 2*y_out[1] + 0.5*y_out[4] + 2*y_out[5] + 0.5*y_out[6] + 0.5*y_out[7]
    I_calc = np.maximum(I_calc,1e-4)
    gamma_dict = {
        'Ca$^{2+}$': calcGamma_np(I_calc, 2, 1.0e-13),
        'OH$^-$': calcGamma_np(I_calc, 1, 3.0e-10),
        'H$^+$': calcGamma_np(I_calc, 1, 9.0e-11),
        'HCO$_3^-$': calcGamma_np(I_calc, 1, 4.0e-10),
        'CO$_3^{2-}$': calcGamma_np(I_calc, 2, 4.0e-10)
    }
    plt.figure(figsize=(10,5))
    for ion, val in gamma_dict.items():
        plt.plot(t_min, val, label=f'γ({ion})')
    plt.xlabel('Time (min)')
    plt.ylabel('Activity Coefficient')
    plt.title('Ion Activity Coefficients Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Carbon speciation and pH stacked plot
    CO2_aq = y_out[11]
    HCO3 = y_out[7]
    CO3 = y_out[5]
    pH_vals = pH_calc
    totalC = CO2_aq + HCO3 + CO3 + 1e-10
    frac_CO2 = CO2_aq / totalC
    frac_HCO3 = HCO3 / totalC
    frac_CO3 = CO3 / totalC

    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.stackplot(t_min, frac_CO2, frac_HCO3, frac_CO3,
                  colors=['#2ca02c', '#17becf', '#9467bd'],
                  labels=['CO₂(aq)', 'HCO₃⁻', 'CO₃²⁻'])
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Molar Fraction')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t_min, pH_vals, 'r-', label='pH')
    ax2.set_ylabel('pH', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Carbon Speciation and pH Evolution')
    plt.tight_layout()
    plt.show()

    # Effective diffusivity placeholder plot
    t_mine = np.zeros_like(t_min)



    R_gas = 8.314
    A_D = theta[6]
    Ea_D = theta[14]
    etha = theta[16]

    D_t = A_D * np.exp(-Ea_D / (R_gas * T_profile))  # Should be (300,)
    D_eff = D_t * np.exp(-etha * t_mine)  # Should be (300,)

    plt.figure(figsize=(8,4))
    plt.plot(t_min, D_eff, color='#2ca02c')
    plt.xlabel('Time (min)')
    plt.ylabel('Effective Diffusivity D (m²/s)')
    plt.title('Time-Dependent CO₂ Diffusivity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Particle diameter evolution [microns]
    diameter_microns = 2 * R_evol * 1e6

    plt.figure(figsize=(8,4))
    plt.plot(t_min, diameter_microns, color='#9467bd')
    plt.xlabel('Time (min)')
    plt.ylabel('Particle Diameter (µm)')
    plt.title('Particle Diameter Evolution Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_and_plot()
