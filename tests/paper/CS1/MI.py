import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):

    def M(t, y, tii, tvi, theta, te):
        """
        Model I: Monod kinetics with maintenance.
        Assumes specific growth rate saturates with substrate and includes maintenance loss.
        """
        y1, y2 = y
        u1 = np.interp(t, te, tvi['u1'])
        u2 = np.interp(t, te, tvi['u2'])

        mu_max = theta[0]  # maximum specific growth rate (1/h)
        Ks = theta[1]  # Monod saturation constant (g/L)
        Yxs = theta[2]  # biomass yield coefficient (g biomass / g substrate)
        m = theta[3]  # maintenance coefficient (1/h)

        r = mu_max * y2 / (Ks + y2)  # Monod rate
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