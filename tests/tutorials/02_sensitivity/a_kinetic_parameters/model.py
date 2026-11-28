"""
Model M of MIDDoE paper Case Study 2 (eq. 20): batch synthesis of the quaternary
chloride salt QS1Cl (HIV-drug intermediate; Moshiritabrizi et al., 2024).

Reactions
    SM + TMA  <=>  QS1Cl           forward kf, reverse kr = kf / K
    QS1Cl      ->  ClDMI + MeCl    side reaction kfs (irreversible)

States (all mol/L)                       measured as ('tvo')
    C_SM     starting material           y1
    C_TMA    trimethylamine              (not returned)
    C_QS1Cl  main product                y2
    C_ClDMI  impurity                    y3
    C_MeCl   volatile by-product         (not returned)

Controls
    u1  : reactor temperature T(t) ('tvi')                 [K]
    y10 : initial SM concentration  w1 = C_SM(0)  ('tii')  [mol/L]
    y20 : initial TMA concentration w2 = C_TMA(0) ('tii')  [mol/L]

Parameters theta (physical units, models['theta'] order), Arrhenius/van 't Hoff form
around T_ref = 296.15 K:
    theta1 kf_ref   forward rate constant at T_ref          [L/(mol h)]
    theta2 Ef       forward activation energy               [J/mol]
    theta3 kfs_ref  side-reaction rate constant at T_ref    [1/h]
    theta4 Efs      side-reaction activation energy         [J/mol]
    theta5 K_ref    equilibrium constant at T_ref           [L/mol]
    theta6 dH       reaction enthalpy term of K             [J/mol]
    kf  = theta1 * exp(-theta2/R * (1/T - 1/T_ref))
    kfs = theta3 * exp(-theta4/R * (1/T - 1/T_ref))
    K   = theta5 * exp( theta6/R * (1/T - 1/T_ref)),   kr = kf / K
True values used for the in-silico data: [50000, 75000, 0.4116, 111900, 9905, 30000].

Provenance: identical (apart from comments) to
'tests/paper/case study 2 - scenario 2/model.py' at commit 9b1612c, the last edit of
the paper-era runners. Note two numerical safeguards that are NOT in paper eq. 20:
kf is clipped to <= 1e5 and a non-finite forward rate is replaced by 1e6.
(The July 2025 version, commit 1ae3a30, used looser tolerances rtol=1e-2, atol=1e-3.)
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):
    """MIDDoE 'pys' model entry point (see krnl_simula): returns {'tvo': {...}, 'tio': {}}."""
    t = np.array(t, dtype=np.float64)                 # physical time grid [h]
    u1 = np.array(tvi['u1'], dtype=np.float64)        # temperature sampled on t [K]
    te = t
    theta = np.array(theta, dtype=np.float64)
    tii_vals = {k: np.float64(v) for k, v in tii.items()}

    R = np.float64(8.314)        # universal gas constant [J/(mol K)]
    T_ref = np.float64(296.15)   # reference temperature [K]

    kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta

    def model_rhs(t_local, y, tii, tvi, theta, te):
        y = np.asarray(y, dtype=np.float64)
        SM, TMA, QS1Cl, ClDMI, MeCl = y
        T = np.interp(np.float64(t_local), te, tvi['u1']).astype(np.float64)

        kf = kf_ref * np.exp(-Ef / R * (1 / T - 1 / T_ref))
        kf = np.clip(kf, 0, 1e5)                      # safeguard (not in eq. 20)
        K = Kref * np.exp(dH / R * (1 / T - 1 / T_ref))
        kr = kf / K
        kfs = kfs_ref * np.exp(-Efs / R * (1 / T - 1 / T_ref))

        r1f = kf * SM * TMA
        if not np.isfinite(r1f):
            r1f = 1e6                                 # safeguard (not in eq. 20)
        r1r = kr * QS1Cl
        r2 = kfs * QS1Cl

        dSM = np.float64(-r1f + r1r)
        dTMA = np.float64(-r1f + r1r)
        dQS1Cl = np.float64(r1f - r1r - r2)
        dClDMI = np.float64(r2)
        dMeCl = np.float64(r2)
        return np.array([dSM, dTMA, dQS1Cl, dClDMI, dMeCl], dtype=np.float64)

    # Initial state: SM and TMA from the design decisions, products start at zero.
    y0_arr = np.array([tii_vals['y10'], tii_vals['y20'], 0.0, 0.0, 0.0], dtype=np.float64)

    sol = solve_ivp(
        fun=model_rhs,
        t_span=(t[0], t[-1]),
        y0=y0_arr,
        args=(tii_vals, {'u1': u1}, theta, te),
        t_eval=t,
        method='BDF',                 # stiff implicit solver
        rtol=np.float64(1e-4),
        atol=np.float64(1e-5),
    )

    return {
        'tvo': {
            'y1': sol.y[0].astype(np.float64).tolist(),   # C_SM
            'y2': sol.y[2].astype(np.float64).tolist(),   # C_QS1Cl
            'y3': sol.y[3].astype(np.float64).tolist(),   # C_ClDMI
        },
        'tio': {},
    }
