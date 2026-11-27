"""
Candidate model MIV of MIDDoE paper Case Study 1 (eq. 18d) -- Monod kinetics WITHOUT maintenance.

Fed-batch baker's-yeast fermentation (Chen & Asprey, 2003), unstructured/unsegregated.
States (time-variant outputs, 'tvo'):
    y1 : biomass concentration                    [g/L]
    y2 : substrate concentration                  [g/L]
Controls:
    u1 : dilution rate (time-variant, 'tvi')      [1/h]
    u2 : feed substrate concentration ('tvi')     [g/L]
    y10: initial biomass w1 = y1(0) ('tii')       [g/L]
Parameters (theta, physical units):
    theta1 mu_max  maximum specific growth rate        [1/h]
    theta2 Ks      Monod saturation constant           [g/L]
    theta3 Yxs     biomass yield on substrate          [g/g]
This file is identical (apart from comments) to the paper-era file
tests/paper/CS1/MIV.py at commit 1ae3a30.

MIDDoE model-file contract ('pys' kernel, see krnl_simula):
    solve_model(t, y0, tii, tvi, theta) -> {'tvo': {name: list(len(t))}, 'tio': {}}
    t     : simulation grid in physical time [h]
    y0    : initial values in system['tvo'] order (ignored here; y1(0) comes from tii)
    tii   : dict of time-invariant inputs  -> tii['y10']
    tvi   : dict of time-variant inputs already sampled on t -> tvi['u1'], tvi['u2']
    theta : parameter vector in physical units, models['theta'] order
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):

    def rhs(t_local, y, tii, tvi, theta, te):
        y1, y2 = y
        # Controls are given on the grid te; interpolate to the solver's current time.
        u1 = np.interp(t_local, te, tvi['u1'])
        u2 = np.interp(t_local, te, tvi['u2'])
        r = theta[0] * y2 / (theta[1] + y2)          # Monod specific growth rate [1/h]
        dy1dt = (r - u1) * y1                        # no maintenance term
        dy2dt = -(r * y1 / theta[2]) + u1 * (u2 - y2)
        return [dy1dt, dy2dt]

    # Initial conditions: y1(0) = w1 (design decision 'y10'), y2(0) = 0.01 g/L (fixed).
    y_init = [tii['y10'], 0.01]

    result = solve_ivp(
        fun=rhs,
        t_span=(t[0], t[-1]),
        y0=y_init,
        args=(tii, tvi, theta, t),
        t_eval=t,
        method='Radau',      # stiff-safe implicit solver (as in the paper-era file)
        rtol=1e-6,
        atol=1e-9,
    )

    return {
        'tvo': {
            'y1': result.y[0, :].tolist(),
            'y2': result.y[1, :].tolist(),
        },
        'tio': {},
    }
