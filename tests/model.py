import numpy as np
from scipy.integrate import solve_ivp

def solve_model(t, y0, phi, phit, theta):
    """
    Solve carbonation model based on CaO holdup and CO₂ concentration.
    Returns only carbonation_conversion and CO₂ concentration.
    """

    # Species definitions
    liquid_species = ['Water', 'Carbon Dioxide']
    solid_species = ['Calcium Carbonate', 'Silicon Oxide', 'Aluminium Oxide', 'Ferric Oxide', 'Calcium Oxide']
    MW_liquid = {'Water': 18.015, 'Carbon Dioxide': 44.01}
    MW_solid = {'Calcium Carbonate': 100.09, 'Silicon Oxide': 60.08, 'Aluminium Oxide': 101.96,
                'Ferric Oxide': 159.69, 'Calcium Oxide': 56.08}

    # Initial solid and liquid holdups
    solid_mass_frac_0 = np.array([0.1265, 0.6821, 0.0495, 0.0409, 0.1010])
    liquid_mass_frac_0 = np.array([1.0, 0.0])

    def f20_model(t, y, phi, phit, theta, te):
        n_solid = len(solid_species)
        n_liquid = len(liquid_species)
        solid_holdup = y[:n_solid]
        liquid_holdup = y[n_solid:n_solid + n_liquid]

        # Interpolation
        T = np.interp(t, te, phit['T'])
        P_CO2 = np.interp(t, te, phit['P'])

        # Constants
        R = 8.314
        ref_T = 298.15
        particle_size = phi['aps']
        slurry_density = phi['slr']
        liquid_phase_volume = 3.5
        true_density = 2400
        porosity = 0.0
        solid_density = true_density * (1 - porosity)
        particle_vol_surf = 6e6 / particle_size
        pre_exp_factor, activ_energy, decay_constant, kla = theta

        CaO0 = solid_mass_frac_0[-1] * slurry_density * liquid_phase_volume
        CaO = solid_holdup[-1]
        carbonation_conversion = (CaO0 - CaO) / CaO0
        carbonation_conversion = np.clip(carbonation_conversion, 0.0, 1.0)

        # Diffusion and reaction
        product_layer_thickness = max(particle_size * (1 - (1 - carbonation_conversion) ** 0.5) / 2, 1e-12)
        diff_coeff = pre_exp_factor * np.exp(-activ_energy / (R * T))
        eff_diff = diff_coeff * np.exp(-decay_constant * t)
        product_layer_coef = eff_diff / product_layer_thickness
        product_layer_resistance = 1 / (product_layer_coef * 1e-6 * particle_vol_surf)
        carbonation_constant = 1 / product_layer_resistance
        conc_CO2 = liquid_holdup[1] / liquid_phase_volume
        carbonation_kinetics = carbonation_constant * conc_CO2

        # Stoichiometry and reaction rates
        stoich = np.array([1, 0, 0, 0, -1, 0, -1])
        r_solid = np.zeros(n_solid)
        for i in range(n_solid):
            r_solid[i] = carbonation_kinetics * (MW_solid[solid_species[i]] / MW_liquid['Carbon Dioxide']) * stoich[i]

        r_liquid = np.zeros(n_liquid)
        for i in range(n_liquid):
            r_liquid[i] = carbonation_kinetics * (MW_liquid[liquid_species[i]] / MW_liquid['Carbon Dioxide']) * stoich[
                n_solid + i]

        # CO2 dissolution
        H_CO2 = 35 * MW_liquid['Carbon Dioxide'] * np.exp(2400 * ((1 / T) - (1 / ref_T)))
        eq_CO2 = P_CO2 * H_CO2
        dissol_CO2 = kla * (eq_CO2 - conc_CO2)

        # d/dt terms
        d_solid = r_solid * slurry_density * liquid_phase_volume / solid_density
        d_liquid = r_liquid.copy()
        d_liquid[1] += dissol_CO2

        return list(d_solid) + list(d_liquid)

    y0_full = list(solid_mass_frac_0 * phi['slr'] * 3.5) + list(liquid_mass_frac_0 * 1 * 3.5)
    result = solve_ivp(
        fun=f20_model,
        t_span=(t[0], t[-1]),
        y0=y0_full,
        args=(phi, phit, theta, t),
        t_eval=t,
        method='Radau',
        rtol=1e-6,
        atol=1e-9
    )

    CaO0 = solid_mass_frac_0[-1] * phi['slr'] * 3.5
    CaO = result.y[4, :]
    carbonation_conversion = (CaO0 - CaO) / CaO0
    concentration_CO2 = result.y[6, :] / 3.5

    return {
        'tv_ophi': {
            'y1': carbonation_conversion.tolist(),
            'y2': concentration_CO2.tolist()
        },
        'ti_ophi': {}
    }