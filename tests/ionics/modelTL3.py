import numpy as np
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

TransformationFactory = pyo.TransformationFactory


def create_carbonation_pyomo_model(t_points, tii, theta, y0=None):
    model = pyo.ConcreteModel()

    # Continuous time domain
    model.t = ContinuousSet(bounds=(t_points[0], t_points[-1]))

    # Constants and parameters as Pyomo Params for clarity and symbolic access
    model.R_gas = pyo.Param(initialize=8.314, mutable=False)

    # Theta parameters split into individual Params for better symbolic use
    model.A_arrh = pyo.Param(range(8), initialize={i: theta[i] for i in range(8)}, mutable=False)
    model.Ea_arrh = pyo.Param(range(8), initialize={i: theta[i+8] for i in range(8)}, mutable=False)
    model.etha = pyo.Param(initialize=theta[16], mutable=False)

    # Fixed constants as Params or floats
    model.dp = pyo.Param(initialize=float(tii['dp']), mutable=False)
    model.SLR = pyo.Param(initialize=float(tii['SLR']) * 1e3, mutable=False)
    model.t_CO2_inject = pyo.Param(initialize=float(tii.get('t_CO2_inject', 0.0)), mutable=False)

    model.CaO_leachable_solid_fr = pyo.Param(initialize=0.26061, mutable=False)
    model.CaO_unleachable_solid_fr = pyo.Param(initialize=0.4034, mutable=False)
    model.porosity = pyo.Param(initialize=0.3, mutable=False)
    model.CaCO3_porosity = pyo.Param(initialize=0.3, mutable=False)
    model.rho_solid_material = pyo.Param(initialize=2260, mutable=False)
    model.rho_solid_CaCO3 = pyo.Param(initialize=2710, mutable=False)
    model.slurry_volume = pyo.Param(initialize=5e-4, mutable=False)
    model.LOI = pyo.Param(initialize=0.01903, mutable=False)

    model.Mw_CaO = pyo.Param(initialize=56e-3, mutable=False)
    model.Mw_CaOH2 = pyo.Param(initialize=74e-3, mutable=False)
    model.Mw_CaCO3 = pyo.Param(initialize=100.1e-3, mutable=False)
    model.Mw_CO2 = pyo.Param(initialize=44e-3, mutable=False)
    model.k5f_fixed = pyo.Param(initialize=1.4e8, mutable=False)
    model.T_ref = pyo.Param(initialize=298, mutable=False)

    model.Keq_R1_ref = pyo.Param(initialize=10 ** (-6.357), mutable=False)
    model.Keq_R2_ref = pyo.Param(initialize=10 ** (-10.33), mutable=False)
    model.Keq_R3_ref = pyo.Param(initialize=10 ** (9.108), mutable=False)
    model.Keq_R4_ref = pyo.Param(initialize=10 ** (14), mutable=False)
    model.Keq_R5_ref = pyo.Param(initialize=10 ** (7.66), mutable=False)

    model.deltaH_R1 = pyo.Param(initialize=-20000, mutable=False)
    model.deltaH_R2 = pyo.Param(initialize=14400, mutable=False)
    model.deltaH_R3 = pyo.Param(initialize=-12100, mutable=False)
    model.deltaH_R4 = pyo.Param(initialize=55800, mutable=False)
    model.deltaH_R5 = pyo.Param(initialize=-41400, mutable=False)

    # Additional constants for ion activity
    model.aCa = pyo.Param(initialize=1e-13, mutable=False)
    model.aOH = pyo.Param(initialize=3e-10, mutable=False)
    model.aH = pyo.Param(initialize=9e-11, mutable=False)
    model.aHCO3 = pyo.Param(initialize=4e-10, mutable=False)
    model.aCO3 = pyo.Param(initialize=4e-10, mutable=False)
    model.A_DH = pyo.Param(initialize=0.037, mutable=False)
    model.B_DH = pyo.Param(initialize=1.04e8, mutable=False)

    # Derived constants for particle geometry and counts (computed with Pyomo expressions)
    R_particle0_val = float(tii['dp']) / 2
    V_particle_val = 4/3 * 3.141592653589793 * (R_particle0_val ** 3)
    V_solid_per_particle_val = (1 - float(model.porosity)) * V_particle_val
    mass_particle_val = float(model.rho_solid_material) * V_solid_per_particle_val
    total_mass_kg_val = float(model.SLR) * float(model.slurry_volume)
    dry_mass_kg_val = total_mass_kg_val * (1 - float(model.LOI))
    N_particles_val = total_mass_kg_val / mass_particle_val

    # Expose these computed constants explicitly for rules
    model.R_particle0 = pyo.Param(initialize=R_particle0_val, mutable=False)
    model.V_particle = pyo.Param(initialize=V_particle_val, mutable=False)
    model.V_solid_per_particle = pyo.Param(initialize=V_solid_per_particle_val, mutable=False)
    model.mass_particle = pyo.Param(initialize=mass_particle_val, mutable=False)
    model.total_mass_kg = pyo.Param(initialize=total_mass_kg_val, mutable=False)
    model.dry_mass_kg = pyo.Param(initialize=dry_mass_kg_val, mutable=False)
    model.N_particles = pyo.Param(initialize=N_particles_val, mutable=False)

    # Time-varying input (T and P) as mutable Params over time domain
    model.T = pyo.Param(model.t, mutable=True, initialize=1.0)
    model.P = pyo.Param(model.t, mutable=True, initialize=1.0)

    # State variables
    variable_domains = {
        'positive': pyo.PositiveReals,
        'nonneg': pyo.NonNegativeReals,
        'unitint': pyo.UnitInterval,
    }

    # Declare variables with bounds and domains
    model.C_CH_void = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.C_CH_bulk = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.M_CH_solid = pyo.Var(model.t, domain=variable_domains['nonneg'])

    model.C_OH_void = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.C_OH_bulk = pyo.Var(model.t, domain=variable_domains['nonneg'])

    model.C_bicarbonate = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.C_proton = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.C_carbonate = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.C_CaCO3_precipitated = pyo.Var(model.t, domain=variable_domains['nonneg'])

    model.M_CSH_solid = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.C_CaCO3_shrinked = pyo.Var(model.t, domain=variable_domains['nonneg'])

    model.C_CO2_aq = pyo.Var(model.t, domain=variable_domains['nonneg'])
    model.alpha = pyo.Var(model.t, domain=variable_domains['unitint'])
    model.particle_diameter = pyo.Var(model.t, domain=variable_domains['positive'])
    model.core_diameter = pyo.Var(model.t, domain=variable_domains['positive'])

    # Derivatives
    model.dC_CH_void_dt = DerivativeVar(model.C_CH_void, wrt=model.t)
    model.dC_CH_bulk_dt = DerivativeVar(model.C_CH_bulk, wrt=model.t)
    model.dM_CH_solid_dt = DerivativeVar(model.M_CH_solid, wrt=model.t)

    model.dC_OH_void_dt = DerivativeVar(model.C_OH_void, wrt=model.t)
    model.dC_OH_bulk_dt = DerivativeVar(model.C_OH_bulk, wrt=model.t)

    model.dC_bicarbonate_dt = DerivativeVar(model.C_bicarbonate, wrt=model.t)
    model.dC_proton_dt = DerivativeVar(model.C_proton, wrt=model.t)
    model.dC_carbonate_dt = DerivativeVar(model.C_carbonate, wrt=model.t)
    model.dC_CaCO3_precipitated_dt = DerivativeVar(model.C_CaCO3_precipitated, wrt=model.t)

    model.dM_CSH_solid_dt = DerivativeVar(model.M_CSH_solid, wrt=model.t)
    model.dC_CaCO3_shrinked_dt = DerivativeVar(model.C_CaCO3_shrinked, wrt=model.t)

    model.dC_CO2_aq_dt = DerivativeVar(model.C_CO2_aq, wrt=model.t)
    model.dalpha_dt = DerivativeVar(model.alpha, wrt=model.t)
    model.dparticle_diameter_dt = DerivativeVar(model.particle_diameter, wrt=model.t)
    model.dcore_diameter_dt = DerivativeVar(model.core_diameter, wrt=model.t)

    # Helper Pyomo-only functions
    def vanthoff(K_eq_ref, deltaH, T):
        return K_eq_ref * pyo.exp(-deltaH / model.R_gas * (1 / T - 1 / model.T_ref))

    def arrhenius(A, Ea, T):
        return A * pyo.exp(-Ea / (model.R_gas * T))

    def gamma_calc(I, z, a_i):
        Ip = I * 1e-3
        lnGamma = (-model.A_DH * z ** 2 * pyo.sqrt(Ip) / (1 + a_i * model.B_DH * pyo.sqrt(Ip)) +
                   (0.2 - 4.17e-5 * Ip) * model.A_DH * z ** 2 * Ip / pyo.sqrt(1000))
        return pyo.exp(lnGamma)

    def kspCaCO3(T):
        return 10 ** (-(0.01183 * (T - 273) + 8.03))

    def kspCaOH2(T):
        S_CH = -0.01087 * T + 1.7465
        Mw = 74.093
        S_CH_molL = (S_CH / 1000) / (Mw / 1000)
        return 4 * S_CH_molL ** 3

    # ODE constraint rule collection
    def ode_rule(m, time):
        T_loc = m.T[time]
        P_loc = m.P[time]

        k_R1f = arrhenius(m.A_arrh[0], m.Ea_arrh[0], T_loc)
        k_R2f = arrhenius(m.A_arrh[1], m.Ea_arrh[1], T_loc) * 1e-3
        k_R5f = arrhenius(m.A_arrh[2], m.Ea_arrh[2], T_loc)
        k_R3f = arrhenius(m.A_arrh[3], m.Ea_arrh[3], T_loc) * 1e-3
        kL = arrhenius(m.A_arrh[4], m.Ea_arrh[4], T_loc)
        kO = arrhenius(m.A_arrh[5], m.Ea_arrh[5], T_loc) * 1e-3
        D = arrhenius(m.A_arrh[6], m.Ea_arrh[6], T_loc)
        kLa = arrhenius(m.A_arrh[7], m.Ea_arrh[7], T_loc)

        Keq_R1 = vanthoff(m.Keq_R1_ref, m.deltaH_R1, T_loc)
        Keq_R2 = vanthoff(m.Keq_R2_ref, m.deltaH_R2, T_loc)
        Keq_R3 = vanthoff(m.Keq_R3_ref, m.deltaH_R3, T_loc)
        Keq_R4 = vanthoff(m.Keq_R4_ref, m.deltaH_R4, T_loc)
        Keq_R5 = vanthoff(m.Keq_R5_ref, m.deltaH_R5, T_loc)

        Ksp_CaOH2 = kspCaOH2(T_loc) * 1e-9
        Ksp_CaCO3 = kspCaCO3(T_loc) * 1e-6

        particle_diameter = m.particle_diameter[time]
        core_diameter = m.core_diameter[time]

        solid_liquid_total_surface = m.N_particles * 4 * 3.141592653589793 * (particle_diameter / 2) ** 2

        I_expr = 2 * m.C_CH_bulk[time] + 0.5 * m.C_OH_bulk[time] + 2 * m.C_bicarbonate[time] + 0.5 * m.C_proton[time] + 0.5 * m.C_carbonate[time] + 1e-12

        gamma_Ca = gamma_calc(I_expr, 2, m.aCa)
        gamma_OH = gamma_calc(I_expr, 1, m.aOH)
        gamma_CO3 = gamma_calc(I_expr, 2, m.aCO3)
        gamma_H = gamma_calc(I_expr, 1, m.aH)
        gamma_HCO3 = gamma_calc(I_expr, 1, m.aHCO3)

        c_std = 1000.0

        # Reaction kinetics
        r_R1 = k_R1f * m.C_CO2_aq[time] - (k_R1f / Keq_R1) * ((m.C_carbonate[time] / c_std) * gamma_HCO3) * ((m.C_proton[time] / c_std) * gamma_H)
        r_R2 = k_R2f * ((m.C_carbonate[time] / c_std) * gamma_HCO3) - (k_R2f / Keq_R2) * ((m.C_bicarbonate[time] / c_std) * gamma_CO3) * ((m.C_proton[time] / c_std) * gamma_H)
        r_R5 = k_R5f * m.C_CO2_aq[time] * ((m.C_OH_bulk[time] / c_std) * gamma_OH) - (k_R5f / Keq_R5) * ((m.C_carbonate[time] / c_std) * gamma_HCO3)
        r_R3 = k_R3f * (((m.C_CH_bulk[time] / c_std) * gamma_Ca) * ((m.C_bicarbonate[time] / c_std) * gamma_CO3) / Ksp_CaCO3 - 1)
        r_recomb = m.k5f_fixed * ((m.C_proton[time] / c_std) * gamma_H) * ((m.C_OH_bulk[time] / c_std) * gamma_OH) - m.k5f_fixed / Keq_R4

        S = ((m.C_CH_void[time] / 1000) * gamma_Ca) * (((m.C_OH_void[time] / 1000) * gamma_OH) ** 2) / Ksp_CaOH2

        CO2_sat = P_loc * 35 * pyo.exp(2400 * ((1 / T_loc) - (1 / m.T_ref)))
        dCO2_aq_dt_expr = kLa * (CO2_sat - m.C_CO2_aq[time]) - r_R1 - r_R5

        # Use if_then_else for min condition
        dM_CH_solid_dt_expr = pyo.if_then_else(
            (-kL * (1 - S) * m.M_CH_solid[time]) < 0,
            -kL * (1 - S) * m.M_CH_solid[time],
            0.0)

        delta = (particle_diameter / 2) - (core_diameter / 2) + 1e-12
        D_e = D * pyo.exp(-m.etha * time)
        dM_CSH_solid_dt_expr = -(4 * m.N_particles * 3.141592653589793 * (particle_diameter / 2) * (core_diameter / 2) * m.C_CO2_aq[time] * D_e) / delta

        term = 1 - (m.M_CSH_solid[time] * (m.Mw_CO2 / m.Mw_CaO)) / m.dry_mass_kg + 1e-12
        dcore_dt_expr = (2 / 3) * m.R_particle0 * term ** (2 / 3) * ((dM_CSH_solid_dt_expr * (m.Mw_CO2 / m.Mw_CaO)) / m.dry_mass_kg)

        dM_CSH_dt = dM_CSH_solid_dt_expr
        dcore_dt = dcore_dt_expr

        V_particle_new = (4 / 3) * 3.141592653589793 * (particle_diameter / 2) ** 3
        V_particles_total = V_particle_new * m.N_particles
        V_void_total = m.porosity * V_particles_total
        V_slurry = m.slurry_volume
        factor_void = 1 / V_void_total
        factor_bulk = 1 / V_slurry

        dC_CH_void_dt_expr = -factor_void * dM_CH_solid_dt_expr - factor_void * kO * solid_liquid_total_surface * (m.C_CH_void[time] - m.C_CH_bulk[time])
        dC_CH_bulk_dt_expr = factor_bulk * kO * solid_liquid_total_surface * (m.C_CH_void[time] - m.C_CH_bulk[time]) - r_R3

        dC_OH_void_dt_expr = -2 * factor_void * dM_CH_solid_dt_expr - factor_void * kO * solid_liquid_total_surface * (m.C_OH_void[time] - m.C_OH_bulk[time])
        dC_OH_bulk_dt_expr = factor_bulk * kO * solid_liquid_total_surface * (m.C_OH_void[time] - m.C_OH_bulk[time]) - r_recomb - r_R5

        dC_carbonate_dt_expr = r_R1 - r_R2 + r_R5
        dC_bicarbonate_dt_expr = r_R2 - r_R3
        dC_proton_dt_expr = r_R1 + r_R2 - r_recomb
        dC_CaCO3_shrinked_dt_expr = dM_CSH_dt / m.slurry_volume

        dC_CaCO3_precipitated_dt_expr = r_R3

        dV_deposit_dt_expr = dC_CaCO3_precipitated_dt_expr * m.slurry_volume * (m.Mw_CaCO3 / m.rho_solid_CaCO3) / ((1 - m.CaCO3_porosity) * m.N_particles)

        dR_dt = dV_deposit_dt_expr / (4 * 3.141592653589793 * (particle_diameter / 2) ** 2)
        dparticle_dt_expr = 2 * dR_dt

        dalpha_dt_expr = (((4 / 3) * 3.141592653589793 * m.N_particles * (particle_diameter / 2) ** 3) - ((4 / 3) * 3.141592653589793 * m.N_particles * (core_diameter / 2) ** 3)) / ((4 / 3) * 3.141592653589793 * m.N_particles * (m.dp / 2) ** 3)

        return [
            m.dC_CH_void_dt[time] == dC_CH_void_dt_expr,
            m.dC_CH_bulk_dt[time] == dC_CH_bulk_dt_expr,
            m.dM_CH_solid_dt[time] == dM_CH_solid_dt_expr,
            m.dC_OH_void_dt[time] == dC_OH_void_dt_expr,
            m.dC_OH_bulk_dt[time] == dC_OH_bulk_dt_expr,
            m.dC_bicarbonate_dt[time] == dC_bicarbonate_dt_expr,
            m.dC_proton_dt[time] == dC_proton_dt_expr,
            m.dC_carbonate_dt[time] == dC_carbonate_dt_expr,
            m.dC_CaCO3_precipitated_dt[time] == dC_CaCO3_precipitated_dt_expr,
            m.dM_CSH_solid_dt[time] == dM_CSH_dt,
            m.dC_CaCO3_shrinked_dt[time] == dC_CaCO3_shrinked_dt_expr,
            m.dC_CO2_aq_dt[time] == dCO2_aq_dt_expr,
            m.dalpha_dt[time] == dalpha_dt_expr,
            m.dparticle_diameter_dt[time] == dparticle_dt_expr,
            m.dcore_diameter_dt[time] == dcore_dt,
        ]

    model.ode_con = pyo.Constraint(model.t, rule=ode_rule)

    # Initial conditions
    if y0 is not None:
        for var_name, val in y0.items():
            if hasattr(model, var_name):
                getattr(model, var_name)[model.t.first()].fix(val)

    return model


def run_carbonation_simulation(t, tii, tvi, theta, y0=None):
    model = create_carbonation_pyomo_model(t, tii, theta, y0)

    # Assign mutable time-varying Params
    for time_val, T_val, P_val in zip(t, tvi['T'], tvi['P']):
        model.T[time_val] = T_val
        model.P[time_val] = P_val

    # Discretize
    TransformationFactory('dae.finite_difference').apply_to(model, nfe=100, scheme='BACKWARD')

    solver = pyo.SolverFactory('ipopt')
    result = solver.solve(model, tee=True)

    times_sorted = sorted(model.t)
    C_CH_bulk_sol = np.array([pyo.value(model.C_CH_bulk[tau]) for tau in times_sorted])

    return {
        'time': times_sorted,
        'C_CH_bulk': C_CH_bulk_sol,
        # Add more extracted vars as needed
    }


# Example usage
if __name__ == '__main__':
    import numpy as np

    t = np.linspace(0, 3600, 300)

    tii = {'dp': 10e-6, 'SLR': 0.1}
    T_profile = np.full_like(t, 298.15)
    P_profile = np.full_like(t, 1.01325)

    tvi = {'T': T_profile, 'P': P_profile}

    theta = np.array([
        1.205e4, 4.813e7, 1.904e8, 3.340e5,
        3.458e-3, 4.694e2, 7.059e4, 9.585e1,
        36715, 15807, 39536, 47039,
        68801, 32132, 87550, 20037,
        2.0
    ])

    y0 = {
        'C_CH_void': 1e-4 * 1000,
        'C_CH_bulk': 1e-4 * 1000,
        'M_CH_solid': 0.26061 * 0.1 * 5e-4 / 56e-3,
        'C_OH_void': 1e-4 * 1000,
        'C_OH_bulk': 3.33e3,
        'C_bicarbonate': 1e-4 * 1000,
        'C_proton': 10 ** (-11.53) * 1000,
        'C_carbonate': 1e-4 * 1000,
        'C_CaCO3_precipitated': 1e-4 * 1000,
        'M_CSH_solid': 0.4034 * 0.1 * 5e-4 / 56e-3,
        'C_CaCO3_shrinked': 1e-4 * 1000,
        'C_CO2_aq': 1e-4 * 1000,
        'alpha': 0.0,
        'particle_diameter': 10e-6,
        'core_diameter': 10e-6,
    }

    results = run_carbonation_simulation(t, tii, tvi, theta, y0)

    print("C_CH_bulk over time:", results['C_CH_bulk'])
