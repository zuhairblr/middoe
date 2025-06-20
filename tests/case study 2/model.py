import numpy as np
from scipy.integrate import solve_ivp

def solve_model(t, y0, tii, tvi, theta):
    """
    MIDDoE-style pharmaceutical batch model.
    Reactions:
        SM + TMA <-> QS1Cl
        QS1Cl -> ClDMI + MeCl

    Parameters
    ----------
    t : ndarray
        Time vector [h]
    y0 : list or None
        If None, y0 will be constructed from tii: [SM0, TMA0, 0, 0, 0]
    tii : dict
        Time-invariant inputs: must include 'SM0' and 'TMA0'
    tvi : dict
        Time-variant inputs, requires 'T' [K]
    theta : list
        Model parameters:
        [kf_ref, Ef, kfs_ref, Efs, Kref, dH]

    Returns
    -------
    dict
        'tvo': dictionary with outputs y1 (SM), y2 (QS1Cl), y3 (ClDMI)
        'tio': empty dictionary for time-invariant outputs
    """

    R = 8.314  # J/mol/K
    T_ref = 296.15  # K

    # kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta

    kfs_ref, Efs, Kref = theta
    dH=3333.3520300505625
    kf_ref=20006.84647638291
    Ef=91.36789932617795

    def model_rhs(t_local, y, tii, tvi, theta, te):
        SM, TMA, QS1Cl, ClDMI, MeCl = y
        T = np.interp(t_local, te, tvi['u1'])

        kf = kf_ref * np.exp(-Ef / R * (1 / T - 1 / T_ref))
        K = Kref * np.exp(dH / R * (1 / T - 1 / T_ref))
        kr = kf / K
        kfs = kfs_ref * np.exp(-Efs / R * (1 / T - 1 / T_ref))

        r1f = kf * SM * TMA
        r1r = kr * QS1Cl
        r2 = kfs * QS1Cl

        dSM = -r1f + r1r
        dTMA = -r1f + r1r
        dQS1Cl = r1f - r1r - r2
        dClDMI = r2
        dMeCl = r2

        return [dSM, dTMA, dQS1Cl, dClDMI, dMeCl]

    y0 = [tii['y10'], tii['y20'], 0.0, 0.0, 0.0]
    sol = solve_ivp(
        fun=model_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        args=(tii, tvi, theta, t),
        t_eval=t,
        method='LSODA', rtol=1e-6, atol=1e-9
    )

    return {
        'tvo': {
            'y1': sol.y[0].tolist(),  # SM
            'y2': sol.y[2].tolist(),  # QS1Cl
            'y3': sol.y[3].tolist()   # ClDMI
        },
        'tio': {}
    }



import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_model(t, y0, tii, tvi, theta):
    """
    MIDDoE-style pharmaceutical batch model.
    Reactions:
        SM + TMA <-> QS1Cl
        QS1Cl -> ClDMI + MeCl

    Parameters
    ----------
    t : ndarray
        Time vector [h]
    y0 : list or None
        If None, y0 will be constructed from tii: [SM0, TMA0, 0, 0, 0]
    tii : dict
        Time-invariant inputs: must include 'SM0' and 'TMA0'
    tvi : dict
        Time-variant inputs, requires 'T' [K]
    theta : list
        Model parameters:
        [kf_ref, Ef, kfs_ref, Efs, Kref, dH]

    Returns
    -------
    dict
        'tvo': dictionary with outputs y1 (SM), y2 (QS1Cl), y3 (ClDMI)
        'tio': empty dictionary for time-invariant outputs
    """

    R = 8.314  # J/mol/K
    T_ref = 296.15  # K

    kf_ref, Ef, kfs_ref, Efs, Kref, dH = theta

    # If y0 is not provided, build from tii
    if y0 is None:
        y0 = [tii['SM0'], tii['TMA0'], 0.0, 0.0, 0.0]

    def model_rhs(t_local, y, tii, tvi, theta, te):
        SM, TMA, QS1Cl, ClDMI, MeCl = y
        T = np.interp(t_local, te, tvi['T'])

        kf = kf_ref * np.exp(-Ef / R * (1 / T - 1 / T_ref))
        K = Kref * np.exp(dH / R * (1 / T - 1 / T_ref))
        kr = kf / K
        kfs = kfs_ref * np.exp(-Efs / R * (1 / T - 1 / T_ref))

        r1f = kf * SM * TMA
        r1r = kr * QS1Cl
        r2 = kfs * QS1Cl

        dSM = -r1f + r1r
        dTMA = -r1f + r1r
        dQS1Cl = r1f - r1r - r2
        dClDMI = r2
        dMeCl = r2

        return [dSM, dTMA, dQS1Cl, dClDMI, dMeCl]

    sol = solve_ivp(
        fun=model_rhs,
        t_span=(t[0], t[-1]),
        y0=y0,
        args=(tii, tvi, theta, t),
        t_eval=t,
        method='LSODA',
        rtol=1e-6,
        atol=1e-9
    )

    return {
        'tvo': {
            'y1': sol.y[0].tolist(),  # SM
            'y2': sol.y[2].tolist(),  # QS1Cl
            'y3': sol.y[3].tolist()   # ClDMI
        },
        'tio': {}
    }

# Define time vector
t = np.linspace(0, 16, 200)  # 16 hours

# Define initial inputs
tii = {'SM0': 0.366, 'TMA0': 0.19}
tvi_1 = {'T': np.full_like(t, 296.15)}
tvi_2 = {'T': np.full_like(t, 306.15)}

# Define true parameters
theta = [10000, 75000, 0.4116, 111900, 9905, 30000]
# theta =[50000, 75000, 0.4116, 111900, 9905, 30000]

# Run simulations
res1 = solve_model(t, None, tii, tvi_1, theta)
res2 = solve_model(t, None, tii, tvi_2, theta)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(t, res1['tvo']['y1'], label='SM (296.15K)', linestyle='-')
plt.plot(t, res1['tvo']['y2'], label='QS1Cl (296.15K)', linestyle='-')
plt.plot(t, res1['tvo']['y3'], label='ClDMI (296.15K)', linestyle='-')

plt.plot(t, res2['tvo']['y1'], label='SM (396.15K)', linestyle='--')
plt.plot(t, res2['tvo']['y2'], label='QS1Cl (396.15K)', linestyle='--')
plt.plot(t, res2['tvo']['y3'], label='ClDMI (396.15K)', linestyle='--')

plt.xlabel('Time (h)')
plt.ylabel('Concentration (mol/L)')
plt.title('Batch Pharmaceutical Reaction at Two Temperatures')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
