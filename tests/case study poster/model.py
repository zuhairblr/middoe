import numpy as np
from scipy.integrate import solve_ivp

def solve_model(t, y0, tii, tvi, theta):

    def M(t, y, tii, tvi, theta, te):
        te_array = np.array(te) if not isinstance(te, np.ndarray) else te
        y1, y2 = y

        # Interpolate Temperature (T) and Heart Rate (HR) for the current time t
        u1 = np.interp(t, te_array, tvi['u1'])  # Interpolated temperature effect
        u2 = np.interp(t, te_array, tvi['u2'])  # Interpolated heart rate effect

        # Parameters
        theta1 = theta[0]  # Maximum specific growth rate
        theta2 = theta[1] # Michaelis constant
        theta3 = theta[2]   # Yield coefficient
        theta4 = theta[3]  # Biomass loss rate due to non-modeled factors
        theta5 = theta[4]  # Substrate inhibition constant

        # Differential equations for the pharmacokinetics and pharmacodynamics
        r=((theta1*y2)/(theta2+y2))* np.exp(-(y2/theta5))   #specific growth rate
        dy1dt = (r-u1-theta4)*y1       #biomass concentrations
        dy2dt = -((r*y1)/(theta3))*u1*(u2-y2)   #substrate concentrations
        return [dy1dt, dy2dt]

    # Initial conditions from tii
    y0 = [tii['y10'], tii['y20']]

    result = solve_ivp(
        fun=M,
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
            'y1': result.y[0, :].tolist(),
            'y2': result.y[1, :].tolist()
        },
        'tio': {}
    }