import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_model(t, y0, tii, tvi, theta):
    V_liq = 3.5
    rho_liq = 0.997

    slr = tii['slr']
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

    y0_full = [m_CaOH2_0, m_CaCO3_0, m_dummy_0, m_H2O_0, m_CO2_0]
    A, Ea, n1, n2, kla = theta

    def rhs(t_now, y):
        m_CaOH2, m_CaCO3, m_dummy, m_H2O, m_CO2 = y
        T = np.interp(t_now, t, tvi['T'])
        P = np.interp(t_now, t, tvi['P'])

        C_CO2 = m_CO2 / V_liq
        H_CO2 = 0.035 * MW['CO2'] * np.exp(2400*(1/T - 1/T_ref))
        C_eq = P * H_CO2
        J_CO2 = kla * (C_eq - C_CO2)

        if m_CaOH2 > 0:
            C_CaOH2 = m_CaOH2 / V_liq
            r = A * np.exp(-Ea / (R * T)) * C_CaOH2**n1 * C_CO2**n2
        else:
            r = 0.0

        r_mol = r * V_liq / MW['CaOH2']
        dm_CaOH2 = -r_mol * MW['CaOH2']
        dm_CaCO3 =  r_mol * MW['CaCO3']
        dm_H2O =    r_mol * MW['H2O']
        dm_CO2 =   -r_mol * MW['CO2'] + J_CO2 * V_liq

        return [dm_CaOH2, dm_CaCO3, 0.0, dm_H2O, dm_CO2]

    sol = solve_ivp(rhs, (t[0], t[-1]), y0_full, t_eval=t,
                    method='Radau', atol=1e-9, rtol=1e-6)

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

    C_CO2_liq = sol.y[-1, :] / V_liq

    return {'tvo': {'y1': X_CO2.tolist(),
                    'y2': C_CO2_liq.tolist()},
            'tio': {}}

# === Example usage and plotting ===
t = np.linspace(0, 7200, 100)  # 2 hours in seconds
y0 = [0, 0.00]  # initial uptake, initial CO₂ conc. in liquid
tii = {'slr': 0.05}
tvi = {'T': np.full_like(t, 298.15), 'P': np.full_like(t, 1.0)}
theta = (700, 18000, 1, 1, 0.013805002369112569)

result = solve_model(t, y0, tii, tvi, theta)

# Plot results
plt.figure(figsize=(12, 5))

# CO₂ uptake
plt.subplot(1, 2, 1)
plt.plot(t, result['tvo']['y1'], label='CO₂ Uptake Fraction (%)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Uptake Fraction (%)')
plt.title('Sorbacal Carbonation efficiency')
plt.grid(True)
plt.legend()

# Dissolved CO₂ concentration
plt.subplot(1, 2, 2)
plt.plot(t, result['tvo']['y2'], label='Dissolved CO₂ (kg/L)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('CO₂ Concentration (kg/L)')
plt.title('Dissolved CO₂ Concentration Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
