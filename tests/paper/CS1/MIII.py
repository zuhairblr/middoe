import numpy as np
from scipy.integrate import solve_ivp


def solve_model(t, y0, tii, tvi, theta):
    def M(t, y, tii, tvi, theta, te):
        """
        Model III: Linear growth kinetics.
        Assumes specific growth rate is linear with substrate.
        """
        y1, y2 = y
        u1 = np.interp(t, te, tvi['u1'])
        u2 = np.interp(t, te, tvi['u2'])

        a = theta[0]  # linear rate coefficient (1/h·(g/L))
        Yxs = theta[1]  # biomass yield
        m = theta[2]  # maintenance coefficient

        r = a * y2
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