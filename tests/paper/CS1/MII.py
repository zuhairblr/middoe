import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):
    def M(t, y, tii, tvi, theta, te):
        """
        Model II: Canoid kinetics.
        Incorporates dependence of growth on both biomass and substrate concentrations.
        """
        y1, y2 = y
        u1 = np.interp(t, te, tvi['u1'])
        u2 = np.interp(t, te, tvi['u2'])

        mu_max = theta[0]  # maximum specific growth rate (1/h)
        Ks_biomass = theta[1]  # saturation constant influenced by biomass (L/g)
        Yxs = theta[2]  # biomass yield coefficient
        m = theta[3]  # maintenance coefficient

        r = mu_max * y2 / (Ks_biomass * y1 + y2)  # Canoid rate
        dy1dt = (r - u1 - m) * y1
        dy2dt = -(r * y1 / Yxs) + u1 * (u2 - y2)
        return [dy1dt, dy2dt]

    # Initial conditions from tii
    y0 = [tii['y10'], 0.01]

    result = solve_ivp(
        fun=M,
        t_span=(t[0], t[-1]),
        y0=y0,
        args=(tii, tvi, theta, t),
        t_eval=t,
        method='Radau',
        rtol=1e-6,
        atol=1e-9
    )

    return {
        'tvo': {
            'y1': result.y[0, :].tolist(),
            'y2': result.y[1, :].tolist()
        },
        'tio': {}
    }