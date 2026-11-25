import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):
    """Batch reactor A -> B with an Arrhenius rate constant."""
    k_ref, Ea = theta                      # the two unknown parameters
    t = np.asarray(t, dtype=float)         # time grid [h]
    T = np.asarray(tvi['u1'], dtype=float) # temperature profile on that grid [K]

    def rhs(time, c):
        Tnow = np.interp(time, t, T)
        k = k_ref * np.exp(-Ea / 8.314 * (1 / Tnow - 1 / 323.15))
        rate = k * c[0]
        return [-rate, rate]               # dA/dt, dB/dt

    c0 = [float(tii['y10']), 0.0]          # start: A = y10, B = 0
    sol = solve_ivp(rhs, (t[0], t[-1]), c0, t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)
    return {'tvo': {'y1': sol.y[0], 'y2': sol.y[1]}, 'tio': {}}
