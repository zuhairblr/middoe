# import numpy as np
# from scipy.integrate import solve_ivp
#
# def solve_model(t, y0, tii, tvi, theta):
#     """
#     MIDDoE-style pharmaceutical batch model.
#     Reactions:
#         SM + TMA <-> QS1Cl
#         QS1Cl -> ClDMI + MeCl
#
#     Parameters
#     ----------
#     t : ndarray
#         Time vector [h]
#     y0 : list or None
#         If None, y0 will be constructed from tii: [SM0, TMA0, 0, 0, 0]
#     tii : dict
#         Time-invariant inputs: must include 'SM0' and 'TMA0'
#     tvi : dict
#         Time-variant inputs, requires 'T' [K]
#     theta : list
#         Model parameters:
#         [kf_ref, Ef, kfs_ref, Efs, Kref, dH]
#
#     Returns
#     -------
#     dict
#         'tvo': dictionary with outputs y1 (SM), y2 (QS1Cl), y3 (ClDMI)
#         'tio': empty dictionary for time-invariant outputs
#     """
#
#     R = 8.314  # J/mol/K
#     T_ref = 296.15  # K
#
#     # kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta
#
#     kfs_ref, Efs, Kref = theta
#     dH=3333.3520300505625
#     kf_ref=20006.84647638291
#     Ef=91.36789932617795
#
#     def model_rhs(t_local, y, tii, tvi, theta, te):
#         SM, TMA, QS1Cl, ClDMI, MeCl = y
#         T = np.interp(t_local, te, tvi['u1'])
#
#         kf = kf_ref * np.exp(-Ef / R * (1 / T - 1 / T_ref))
#         K = Kref * np.exp(dH / R * (1 / T - 1 / T_ref))
#         kr = kf / K
#         kfs = kfs_ref * np.exp(-Efs / R * (1 / T - 1 / T_ref))
#
#         r1f = kf * SM * TMA
#         r1r = kr * QS1Cl
#         r2 = kfs * QS1Cl
#
#         dSM = -r1f + r1r
#         dTMA = -r1f + r1r
#         dQS1Cl = r1f - r1r - r2
#         dClDMI = r2
#         dMeCl = r2
#
#         return [dSM, dTMA, dQS1Cl, dClDMI, dMeCl]
#
#     y0 = [tii['y10'], tii['y20'], 0.0, 0.0, 0.0]
#     sol = solve_ivp(
#         fun=model_rhs,
#         t_span=(t[0], t[-1]),
#         y0=y0,
#         args=(tii, tvi, theta, t),
#         t_eval=t,
#         method='LSODA', rtol=1e-6, atol=1e-9
#     )
#
#     return {
#         'tvo': {
#             'y1': sol.y[0].tolist(),  # SM
#             'y2': sol.y[2].tolist(),  # QS1Cl
#             'y3': sol.y[3].tolist()   # ClDMI
#         },
#         'tio': {}
#     }
#
#
#
# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
#
# def solve_model(t, y0, tii, tvi, theta):
#     """
#     MIDDoE-style pharmaceutical batch model.
#     Reactions:
#         SM + TMA <-> QS1Cl
#         QS1Cl -> ClDMI + MeCl
#
#     Parameters
#     ----------
#     t : ndarray
#         Time vector [h]
#     y0 : list or None
#         If None, y0 will be constructed from tii: [SM0, TMA0, 0, 0, 0]
#     tii : dict
#         Time-invariant inputs: must include 'SM0' and 'TMA0'
#     tvi : dict
#         Time-variant inputs, requires 'T' [K]
#     theta : list
#         Model parameters:
#         [kf_ref, Ef, kfs_ref, Efs, Kref, dH]
#
#     Returns
#     -------
#     dict
#         'tvo': dictionary with outputs y1 (SM), y2 (QS1Cl), y3 (ClDMI)
#         'tio': empty dictionary for time-invariant outputs
#     """
#
#     R = 8.314  # J/mol/K
#     T_ref = 296.15  # K
#
#     kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta
#
#     # If y0 is not provided, build from tii
#     if y0 is None:
#         y0 = [tii['SM0'], tii['TMA0'], 0.0, 0.0, 0.0]
#
#     def model_rhs(t_local, y, tii, tvi, theta, te):
#         SM, TMA, QS1Cl, ClDMI, MeCl = y
#         T = np.interp(t_local, te, tvi['T'])
#
#         kf = kf_ref * np.exp(-Ef / R * (1 / T - 1 / T_ref))
#         K = Kref * np.exp(dH / R * (1 / T - 1 / T_ref))
#         kr = kf / K
#         kfs = kfs_ref * np.exp(-Efs / R * (1 / T - 1 / T_ref))
#
#         r1f = kf * SM * TMA
#         r1r = kr * QS1Cl
#         r2 = kfs * QS1Cl
#
#         dSM = -r1f + r1r
#         dTMA = -r1f + r1r
#         dQS1Cl = r1f - r1r - r2
#         dClDMI = r2
#         dMeCl = r2
#
#         return [dSM, dTMA, dQS1Cl, dClDMI, dMeCl]
#
#     sol = solve_ivp(
#         fun=model_rhs,
#         t_span=(t[0], t[-1]),
#         y0=y0,
#         args=(tii, tvi, theta, t),
#         t_eval=t,
#         method='LSODA',
#         rtol=1e-6,
#         atol=1e-9
#     )
#
#     return {
#         'tvo': {
#             'y1': sol.y[0].tolist(),  # SM
#             'y2': sol.y[2].tolist(),  # QS1Cl
#             'y3': sol.y[3].tolist()   # ClDMI
#         },
#         'tio': {}
#     }
#
# # Define time vector
# t = np.linspace(0, 16, 200)  # 16 hours
#
# # Define initial inputs
# tii = {'SM0': 0.366, 'TMA0': 0.19}
# tvi_1 = {'T': np.full_like(t, 296.15)}
# tvi_2 = {'T': np.full_like(t, 306.15)}
#
# # Define true parameters
# theta = [10000, 75000, 0.4116, 111900, 9905, 30000]
# # theta =[50000, 75000, 0.4116, 111900, 9905, 30000]
#
# # Run simulations
# res1 = solve_model(t, None, tii, tvi_1, theta)
# res2 = solve_model(t, None, tii, tvi_2, theta)
#
# # Plotting
# plt.figure(figsize=(10, 6))
#
# plt.plot(t, res1['tvo']['y1'], label='SM (296.15K)', linestyle='-')
# plt.plot(t, res1['tvo']['y2'], label='QS1Cl (296.15K)', linestyle='-')
# plt.plot(t, res1['tvo']['y3'], label='ClDMI (296.15K)', linestyle='-')
#
# plt.plot(t, res2['tvo']['y1'], label='SM (396.15K)', linestyle='--')
# plt.plot(t, res2['tvo']['y2'], label='QS1Cl (396.15K)', linestyle='--')
# plt.plot(t, res2['tvo']['y3'], label='ClDMI (396.15K)', linestyle='--')
#
# plt.xlabel('Time (h)')
# plt.ylabel('Concentration (mol/L)')
# plt.title('Batch Pharmaceutical Reaction at Two Temperatures')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import numpy as np
from scipy.integrate import solve_ivp

def solve_model(t, y0, tii, tvi, theta):
    """
    MIDDoE-style pharmaceutical batch model.
    Reactions:
        SM + TMA <-> QS1Cl
        QS1Cl -> ClDMI + MeCl

    All internal arrays and scalars are enforced as np.float64.
    """
    # Cast inputs to float64
    t = np.array(t, dtype=np.float64)
    u1 = np.array(tvi['u1'], dtype=np.float64)
    te = t  # time evaluation points, already float64
    theta = np.array(theta, dtype=np.float64)
    tii_vals = {k: np.float64(v) for k, v in tii.items()}

    R = np.float64(8.314)        # J/mol/K
    T_ref = np.float64(296.15)   # K

    kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta

    def model_rhs(t_local, y, tii, tvi, theta, te):
        # ensure y is float64
        y = np.asarray(y, dtype=np.float64)
        SM, TMA, QS1Cl, ClDMI, MeCl = y

        # interpolate temperature
        T = np.interp(np.float64(t_local), te, tvi['u1']).astype(np.float64)

        # Arrhenius expressions (all float64)
        kf   = kf_ref  * np.exp(-Ef  / R * (1/T - 1/T_ref))
        kf = np.clip(kf, 0, 1e5)
        K    = Kref    * np.exp(dH   / R * (1/T - 1/T_ref))
        kr   = kf / K
        kfs  = kfs_ref * np.exp(-Efs / R * (1/T - 1/T_ref))

        # reaction rates
        r1f = kf * SM * TMA
        if not np.isfinite(r1f):
            r1f = 1e6  # or 0 or skip depending on context
        r1r = kr  * QS1Cl
        r2  = kfs * QS1Cl

        # derivatives
        dSM     = np.float64(-r1f + r1r)
        dTMA    = np.float64(-r1f + r1r)
        dQS1Cl  = np.float64( r1f - r1r - r2)
        dClDMI  = np.float64( r2)
        dMeCl   = np.float64( r2)

        return np.array([dSM, dTMA, dQS1Cl, dClDMI, dMeCl], dtype=np.float64)

    # initial conditions as float64
    y0_arr = np.array([
        tii_vals['y10'],
        tii_vals['y20'],
        np.float64(0.0),
        np.float64(0.0),
        np.float64(0.0)
    ], dtype=np.float64)

    sol = solve_ivp(
        fun=model_rhs,
        t_span=(t[0], t[-1]),
        y0=y0_arr,
        args=(tii_vals, {'u1': u1}, theta, te),
        t_eval=t,
        method='BDF',
        rtol=np.float64(1e-2),
        atol=np.float64(1e-3)
    )

    return {
        'tvo': {
            'y1': sol.y[0].astype(np.float64).tolist(),  # SM
            'y2': sol.y[2].astype(np.float64).tolist(),  # QS1Cl
            'y3': sol.y[3].astype(np.float64).tolist()   # ClDMI
        },
        'tio': {}
    }







# import numpy as np
# from scipy.integrate import solve_ivp
#
# def solve_model(t, y0, tii, tvi, theta):
#     """
#     Faster but equivalent version of your MIDDoE batch model.
#     Only minimal structural changes; same inputs/outputs.
#     """
#
#     # ---------- Precompute / cast once ----------
#     t = np.asarray(t, dtype=np.float64)
#     te = t                               # evaluation times
#     u1 = np.asarray(tvi['u1'], dtype=np.float64)
#
#     theta = np.asarray(theta, dtype=np.float64)
#     kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta
#
#     tii_vals = {k: np.float64(v) for k, v in tii.items()}
#
#     R      = np.float64(8.314)
#     T_ref  = np.float64(296.15)
#
#     # Precompute interp slopes for temperature (piecewise linear):
#     # T(t) = a[i]*t + b[i] for t in [te[i], te[i+1]]
#     dt   = np.diff(te)
#     dT   = np.diff(u1)
#     a    = dT / dt
#     b    = u1[:-1] - a * te[:-1]
#
#     # Precompute Arrhenius constants pieces that don't change with T
#     Af   = -Ef  / R
#     Afs  = -Efs / R
#     AdH  =  dH  / R
#     invTref = 1.0 / T_ref
#
#     # Small helper to evaluate T(t) quickly (branchless search via np.searchsorted)
#     def T_of_t(tloc):
#         i = np.searchsorted(te, tloc) - 1
#         if i < 0:
#             return u1[0]
#         if i >= len(a):
#             return u1[-1]
#         return a[i] * tloc + b[i]
#
#     # ---------- RHS ----------
#     def rhs(t_local, y):
#         # unpack (no dtype casts)
#         SM, TMA, QS1Cl, ClDMI, MeCl = y
#
#         T = T_of_t(t_local)
#         invT = 1.0 / T
#         # Arrhenius terms
#         kf  = kf_ref  * np.exp(Af  * (invT - invTref))
#         K   = Kref    * np.exp(AdH * (invT - invTref))
#         kr  = kf / K
#         kfs = kfs_ref * np.exp(Afs * (invT - invTref))
#
#         r1f = kf * SM * TMA
#         r1r = kr * QS1Cl
#         r2  = kfs * QS1Cl
#
#         dSM    = -r1f + r1r
#         dTMA   = -r1f + r1r
#         dQS1Cl =  r1f - r1r - r2
#         dClDMI =  r2
#         dMeCl  =  r2
#
#         return np.array([dSM, dTMA, dQS1Cl, dClDMI, dMeCl], dtype=np.float64)
#
#     # ---------- Analytic Jacobian (w.r.t. y only; T-dependence ignored, OK for stiff solvers) ----------
#     def jac(t_local, y):
#         SM, TMA, QS1Cl, _, _ = y
#         T = T_of_t(t_local)
#         invT = 1.0 / T
#
#         kf  = kf_ref  * np.exp(Af  * (invT - invTref))
#         K   = Kref    * np.exp(AdH * (invT - invTref))
#         kr  = kf / K
#         kfs = kfs_ref * np.exp(Afs * (invT - invTref))
#
#         # partials
#         j11 = -kf*TMA             # d(dSM)/dSM
#         j12 = -kf*SM              # d(dSM)/dTMA
#         j13 =  kr                 # d(dSM)/dQS1Cl
#
#         j21 = j11                 # symmetry
#         j22 = j12
#         j23 = j13
#
#         j31 =  kf*TMA
#         j32 =  kf*SM
#         j33 = -kr - kfs
#
#         j41 = 0.0
#         j42 = 0.0
#         j43 =  kfs
#
#         j51 = 0.0
#         j52 = 0.0
#         j53 =  kfs
#
#         return np.array([
#             [j11, j12, j13, 0.0, 0.0],
#             [j21, j22, j23, 0.0, 0.0],
#             [j31, j32, j33, 0.0, 0.0],
#             [0.0, 0.0,  j43, 0.0, 0.0],
#             [0.0, 0.0,  j53, 0.0, 0.0]
#         ], dtype=np.float64)
#
#     # ---------- ICs ----------
#     y0_arr = np.array([
#         tii_vals['y10'],
#         tii_vals['y20'],
#         0.0, 0.0, 0.0
#     ], dtype=np.float64)
#
#     sol = solve_ivp(
#         rhs,
#         (t[0], t[-1]),
#         y0_arr,
#         t_eval=t,
#         method='BDF',
#         jac=jac,                    # <-- big win
#         rtol=1e-5, atol=1e-8,       # slightly looser; adjust if needed
#     )
#
#     return {
#         'tvo': {
#             'y1': sol.y[0].astype(np.float64).tolist(),
#             'y2': sol.y[2].astype(np.float64).tolist(),
#             'y3': sol.y[3].astype(np.float64).tolist()
#         },
#         'tio': {}
#     }
