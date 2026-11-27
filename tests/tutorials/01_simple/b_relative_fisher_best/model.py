"""
Fed-batch baker's-yeast fermentation: Monod growth with maintenance.

This is model MI of MIDDoE paper Case Study 1 (eq. 18a; Chen & Asprey, 2003).
It is used by the tutorial notebook in this folder.

How this file talks to MIDDoE
-----------------------------
MIDDoE treats the model as a black box. With ``models['krt'] = {'MI': 'pys'}`` and
``models['src'] = {'MI': '<path>/model.py'}`` it imports this file and calls

    solve_model(t, y0, tii, tvi, theta)

with positional arguments:

    t      list of time points [h]: the simulation grid from system['t_s'] and system['t_r']
    y0     list of initial values, one per system['tvo'] output, in that order
           (here [0, 0.01] from the 'init' entries; this model ignores it and builds its own)
    tii    dict of time-invariant inputs, keys = system['tii'] keys   -> tii['y10']
    tvi    dict of time-variant inputs, keys = system['tvi'] keys, each an array of
           the input value at every point of t                        -> tvi['u1'], tvi['u2']
    theta  list of parameter values in physical units, in the order of models['theta']

and expects back

    {'tvo': {'y1': [...], 'y2': [...]},   one list of len(t) per MEASURED system['tvo'] output
     'tio': {}}                           time-invariant outputs (none here)

The names are the contract: 'y1', 'y2' must be the keys of system['tvo'], 'u1', 'u2' the
keys of system['tvi'] and 'y10' a key of system['tii'].
If this function raises an error or returns something else, MIDDoE stops with
middoe.krnl_simula.SimulationError and tells you why.

The model
---------
States (measured outputs):
    y1  biomass concentration     [g/L]
    y2  substrate concentration   [g/L]
Inputs:
    u1  dilution rate             [1/h]   changes during the run  (system['tvi'])
    u2  feed substrate conc.      [g/L]   changes during the run  (system['tvi'])
    y10 initial biomass y1(0)     [g/L]   fixed per run           (system['tii'])
Parameters (theta, in this order):
    theta[0] mu_max  maximum specific growth rate   [1/h]
    theta[1] Ks      Monod saturation constant      [g/L]
    theta[2] Yxs     biomass yield on substrate     [g/g]
    theta[3] m       maintenance coefficient        [1/h]
Equations:
    r      = mu_max * y2 / (Ks + y2)
    dy1/dt = (r - u1 - m) * y1
    dy2/dt = -r * y1 / Yxs + u1 * (u2 - y2)
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):
    mu_max, Ks, Yxs, m = theta              # unpack in the order of models['theta']
    t = np.asarray(t, dtype=float)

    def rhs(time, y):
        y1, y2 = y
        u1 = np.interp(time, t, tvi['u1'])   # input value at this time (tvi is given on t)
        u2 = np.interp(time, t, tvi['u2'])
        r = mu_max * y2 / (Ks + y2)
        dy1 = (r - u1 - m) * y1
        dy2 = -r * y1 / Yxs + u1 * (u2 - y2)
        return [dy1, dy2]

    start = [tii['y10'], 0.01]               # y1(0) is a design input; y2(0) = 0.01 g/L
    sol = solve_ivp(rhs, (t[0], t[-1]), start, t_eval=t,
                    method='Radau', rtol=1e-6, atol=1e-9)   # stiff-safe solver

    return {'tvo': {'y1': sol.y[0].tolist(), 'y2': sol.y[1].tolist()},
            'tio': {}}
