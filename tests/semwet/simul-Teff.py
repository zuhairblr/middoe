import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_model(t, y0, tii, tvi, theta):
    """
    Carbonation of a Ca(OH)₂ slurry with explicit gas–liquid transfer.

    Parameters
    ----------
    t     : 1-D ndarray (s) – evaluation time grid
    y0    : length-2 ndarray
            y0[0] = (unused) initial uptake fraction
            y0[1] = initial dissolved-phase CO₂ concentration [kg L⁻¹]
    tii   : {'slr': solids loading}
            tii['slr'] = kg dry solids per litre of liquid
    tvi   : {'T': ndarray[K], 'P': ndarray[bar]} – same length as t
    theta : (A, Ea, n1, n2, kla)

    Returns
    -------
    dict with
      tvo['y1'] – CO₂ uptake fraction vs. t  (list)
      tvo['y2'] – dissolved CO₂ concentration [kg L⁻¹] vs. t  (list)
      tvo['T']  – slurry temperature vs. t (list)
    """
    V_liq = 3.5
    rho_liq = 0.997
    slr = tii['slr']
    T0 = tii['T']
    M_solids = slr * V_liq
    M_liquid = rho_liq * V_liq
    M_slurry = M_solids + M_liquid

    z_solid = np.array([0.97, 0, 0.03])
    z_solid /= z_solid.sum()
    m_CaOH2_0 = M_solids * z_solid[0]
    m_CaCO3_0 = M_solids * z_solid[1]
    m_dummy_0 = M_solids * z_solid[2]

    MW = dict(CaOH2=0.07409, CaCO3=0.10009, CO2=0.04401, H2O=0.018015)
    R = 8.314
    T_ref = 298.15
    m_H2O_0 = rho_liq * V_liq
    m_CO2_0 = y0[1] * V_liq

    y0_full = [m_CaOH2_0, m_CaCO3_0, m_dummy_0, m_H2O_0, m_CO2_0, T0]
    A, Ea, phi, n1, n2, n3, n4, npo, nco = theta

    def rhs(t_now, y):
        m_CaOH2, m_CaCO3, m_dummy, m_H2O, m_CO2, T = y
        P = np.interp(t_now, t, tvi['P'])
        C_CO2 = m_CO2 / V_liq
        H_CO2 = 0.035 * MW['CO2'] * np.exp(2400*(1/T - 1/T_ref))
        C_eq = P * H_CO2

        viscosity_gas = 1.837E-6
        density_gas = 1.184
        diff_C02 = 2.35E-6 * np.exp(-2064/T)
        slpm = 6
        vol_flow_gas = (slpm/60) * 1E-3 * (1.184 / density_gas)
        cross_section = 2.835E-2
        dbubble = 0.005
        velocity_gas = vol_flow_gas / cross_section

        a = 6 * phi / dbubble
        schmidt = viscosity_gas / (density_gas * diff_C02)
        reynolds = density_gas * velocity_gas * dbubble / viscosity_gas
        sherwood = n1 + n2 * reynolds**n3 * schmidt**n4
        kl = sherwood * diff_C02 / dbubble
        J_CO2 = kl * a * (C_eq - C_CO2)

        if m_CaOH2 > 0:
            C_CaOH2 = m_CaOH2 / V_liq
            r = A * np.exp(-Ea/(R*T)) * (C_CaOH2**npo) * (C_CO2 ** nco)
        else:
            r = 0.0

        r_mol = r * V_liq / MW['CaOH2']
        dm_CaOH2 = -r_mol * MW['CaOH2']
        dm_CaCO3 =  r_mol * MW['CaCO3']
        dm_H2O   =  r_mol * MW['H2O']
        dm_CO2   = -r_mol * MW['CO2'] + J_CO2 * V_liq

        reaction_heat = -113000
        Cp_slurry = 4185
        dT = (-reaction_heat * r_mol) / (M_slurry * Cp_slurry)

        return [dm_CaOH2, dm_CaCO3, 0.0, dm_H2O, dm_CO2, dT]

    sol = solve_ivp(rhs, (t[0], t[-1]), y0_full, t_eval=t, method='Radau', atol=1e-9, rtol=1e-6)
    m_CaCO3_t = sol.y[1, :]
    m_CO2_reacted = m_CaCO3_t * MW['CO2'] / MW['CaCO3']

    n_CaOH2_0 = m_CaOH2_0 / MW['CaOH2']
    n_CaCO3_0 = m_CaCO3_0 / MW['CaCO3']
    MW_CaO = 0.05608
    m_CaO_from_CaOH2 = n_CaOH2_0 * MW_CaO
    m_CaO_from_CaCO3 = n_CaCO3_0 * MW_CaO
    m_CaO_total = m_CaO_from_CaOH2 + m_CaO_from_CaCO3
    m_inert = m_dummy_0
    m_loifree = m_CaO_total + m_inert

    oxide_composition = {
        'CaO': 100 * m_CaO_total / m_loifree,
        'CaCO3': 100 * m_CaCO3_0 / m_loifree,
        'SO3': 0.0,
        'MgO': 0.0
    }

    CaO = oxide_composition['CaO']
    CaCO3 = oxide_composition['CaCO3']
    SO3 = oxide_composition['SO3']
    MgO = oxide_composition['MgO']

    Th_CO2 = (44 / 56) * (CaO - (56 / 100) * CaCO3 - (56 / 80) * SO3) + 1.091 * MgO
    alpha = m_CO2_reacted / m_loifree
    X_CO2 = alpha * 100 / Th_CO2

    C_CO2_liq = sol.y[4, :] / V_liq
    T_profile = sol.y[5, :]

    return {'tvo': {'y1': X_CO2.tolist(),
                    'y2': C_CO2_liq.tolist(),
                    'T': T_profile.tolist()},
            'tio': {}}

# === Run and plot ===
t = np.linspace(0, 10800, 500)
y0 = [0, 0.00]
tii = {'slr': 0.05, 'T': 298.15}  # solids loading in kg dry solids per litre of liquid
tvi = {'P': np.full_like(t, 1.0)}
theta = (700, 18000, 1, 1, 0.015, 0.89, 0.7, 1, 1)

result = solve_model(t, y0, tii, tvi, theta)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(t, result['tvo']['y1'], label='CO₂ Uptake Fraction', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Uptake Fraction (%)')
plt.title('Sorbacal Carbonation efficiency')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t, result['tvo']['y2'], label='Dissolved CO₂ Concentration', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('CO₂ Concentration (kg/L)')
plt.title('Dissolved CO₂ Concentration')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t, result['tvo']['T'], label='Temperature', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Slurry Temperature Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
