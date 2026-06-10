import numpy as np
from scipy.integrate import solve_ivp





def solve_model(t, y0, tii, tvi, theta):
    """
    Solve the 5-parameter, 3-response dynamic model.

    Parameters
    ----------
    t : array_like
        Time evaluation points
    y0 : array_like, shape (3,)
        Initial conditions [y1_0, y2_0, y3_0]
    tii : dict
        Time-invariant inputs {'w1': float}
    tvi : dict
        Time-varying inputs {'u1': array matching t}
    theta : array_like, shape (5,)
        Parameters [theta1, theta2, theta3, theta4, theta5]

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'tvo': {'y1': list, 'y2': list, 'y3': list}
        - 'tio': {} (empty dict)
    """
    # Convert inputs to float64
    t = np.array(t, dtype=np.float64)
    u1 = np.array(tvi['u1'], dtype=np.float64)
    theta = np.array(theta, dtype=np.float64)
    y0 = [0.5, 1.0, 0.3]
    y0_arr = np.array(y0, dtype=np.float64)

    # Extract parameters
    theta1, theta2, theta3, theta4, theta5 = theta

    # Extract time-invariant input
    w1 = tii['w1']

    def model_rhs(t_local, y):
        """ODE right-hand side function."""
        y1, y2, y3 = y

        # Interpolate u1 at current time point
        u1_t = np.interp(t_local, t, u1)

        # Model equations
        dy1dt = theta1 * (u1_t - y1) + 0.5 * np.sin(theta2 * y2) - 0.2 * y1 * y3

        dy2dt = (theta2 * (w1 * y1 - y2) + theta1 * np.cos(y3)
                 - 0.3 * y2 ** 2 / (1 + y2 ** 2) + theta4 * 0.1 * np.sin(t_local))

        dy3dt = (theta3 * (u1_t * w1 - y3) + 0.4 * np.sin(theta1 * t_local)
                 - 0.15 * y1 * y2 + theta5 * 0.05 * (y1 + y2)
                 - theta4 * 0.08 * y3)

        return [dy1dt, dy2dt, dy3dt]

    def model_jac(t_local, y):
        """Analytical Jacobian of the ODE system."""
        y1, y2, y3 = y

        jac = np.zeros((3, 3))

        # df1/dy
        jac[0, 0] = -theta1 - 0.2 * y3
        jac[0, 1] = 0.5 * theta2 * np.cos(theta2 * y2)
        jac[0, 2] = -0.2 * y1

        # df2/dy
        jac[1, 0] = theta2 * w1
        jac[1, 1] = -theta2 - 0.3 * 2 * y2 / (1 + y2 ** 2) ** 2
        jac[1, 2] = -theta1 * np.sin(y3)

        # df3/dy
        jac[2, 0] = -0.15 * y2 + theta5 * 0.05
        jac[2, 1] = -0.15 * y1 + theta5 * 0.05
        jac[2, 2] = -theta3 - theta4 * 0.08

        return jac

    # Solve ODE using LSODA method with relaxed tolerances and Jacobian
    sol = solve_ivp(
        fun=model_rhs,
        t_span=(t[0], t[-1]),
        y0=y0_arr,
        t_eval=t,
        method='LSODA',
        jac=model_jac,
        rtol=1e-2,
        atol=1e-3
    )

    # Check if integration was successful
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Return results in MBDoE format
    return {
        'tvo': {
            'y2': sol.y[1].tolist(),
        },
        'tio': {}
    }

#
# import matplotlib.pyplot as plt
# def solve_model1(t, y0, tii, tvi, theta):
#     """
#     Solve the 5-parameter, 3-response dynamic model.
#
#     Parameters
#     ----------
#     t : array_like
#         Time evaluation points
#     y0 : array_like, shape (3,)
#         Initial conditions [y1_0, y2_0, y3_0]
#     tii : dict
#         Time-invariant inputs {'w1': float}
#     tvi : dict
#         Time-varying inputs {'u1': array matching t}
#     theta : array_like, shape (5,)
#         Parameters [theta1, theta2, theta3, theta4, theta5]
#
#     Returns
#     -------
#     result : dict
#         Dictionary containing:
#         - 'tvo': {'y1': list, 'y2': list, 'y3': list}
#         - 'tio': {} (empty dict)
#     """
#     # Convert inputs to float64
#     t = np.array(t, dtype=np.float64)
#     u1 = np.array(tvi['u1'], dtype=np.float64)
#     theta = np.array(theta, dtype=np.float64)
#     y0 = [0.5, 1.0, 0.3]
#     y0_arr = np.array(y0, dtype=np.float64)
#
#     # Extract parameters
#     theta1, theta2, theta3, theta4, theta5 = theta
#
#     # Extract time-invariant input
#     w1 = tii['w1']
#
#
#     def model_rhs(t_local, y):
#         """ODE right-hand side function."""
#         y1, y2, y3 = y
#
#         # Interpolate u1 at current time point
#         u1_t = np.interp(t_local, t, u1)
#
#         # Model equations
#         dy1dt = theta1 * (u1_t - y1) + 0.5 * np.sin(theta2 * y2) - 0.2 * y1 * y3
#
#         dy2dt = (theta2 * (w1 * y1 - y2) + theta1 * np.cos(y3)
#                  - 0.3 * y2 ** 2 / (1 + y2 ** 2) + theta4 * 0.1 * np.sin(t_local))
#
#         dy3dt = (theta3 * (u1_t * w1 - y3) + 0.4 * np.sin(theta1 * t_local)
#                  - 0.15 * y1 * y2 + theta5 * 0.05 * (y1 + y2)
#                  - theta4 * 0.08 * y3)
#
#         return [dy1dt, dy2dt, dy3dt]
#
#     # Solve ODE using BDF method
#     sol = solve_ivp(
#         fun=model_rhs,
#         t_span=(t[0], t[-1]),
#         y0=y0_arr,
#         t_eval=t,
#         method='BDF',
#         rtol=1e-6,
#         atol=1e-8
#     )
#
#     # Check if integration was successful
#     if not sol.success:
#         raise RuntimeError(f"Integration failed: {sol.message}")
#
#     # Return results in MBDoE format
#     return {
#         'tvo': {
#             'y1': sol.y[0].tolist(),
#             'y2': sol.y[1].tolist(),
#             'y3': sol.y[2].tolist()
#         },
#         'tio': {}
#     }
#
# def run_and_plot(t_final=15, n_points=300, u1_value=2.5, w1_value=2.0,
#                  theta_true=None, y0=None, save_path='model_responses.png'):
#     """
#     Run the model and plot responses vs time with constant controls.
#
#     Parameters
#     ----------
#     t_final : float, optional
#         Final simulation time (default: 15)
#     n_points : int, optional
#         Number of time points (default: 300)
#     u1_value : float, optional
#         Constant u1 control value (default: 2.5)
#     w1_value : float, optional
#         Constant w1 control value (default: 2.0)
#     theta_true : array_like, optional
#         True parameter values (default: [2.5, 1.5, 1.2, 0.5, 0.3])
#     y0 : array_like, optional
#         Initial conditions (default: [0.5, 1.0, 0.3])
#     save_path : str, optional
#         Path to save plot (default: 'model_responses.png')
#
#     Returns
#     -------
#     result : dict
#         Simulation results with 'tvo' and 'tio' keys
#     """
#     # Default values
#     if theta_true is None:
#         theta_true = [2.5, 1.5, 1.2, 0.5, 0.3]
#
#     if y0 is None:
#         y0 = [0.5, 1.0, 0.3]
#
#     # Setup
#     t = np.linspace(0, t_final, n_points)
#     tii = {'w1': w1_value}
#     u1 = np.ones_like(t) * u1_value
#     tvi = {'u1': u1}
#
#     # Run simulation
#     result = solve_model(t, y0, tii, tvi, theta_true)
#     tvo = result['tvo']
#
#     # Convert to arrays
#     y1 = np.array(tvo['y1'])
#     y2 = np.array(tvo['y2'])
#     y3 = np.array(tvo['y3'])
#
#     # Create plot
#     fig, axes = plt.subplots(3, 1, figsize=(12, 10))
#
#     axes[0].plot(t, y1, 'b-', linewidth=2.5)
#     axes[0].set_ylabel('y₁', fontsize=14, fontweight='bold')
#     axes[0].grid(True, alpha=0.3)
#     axes[0].set_title(f'Response y₁ (mean={y1.mean():.2f}, std={y1.std():.2f})')
#
#     axes[1].plot(t, y2, 'r-', linewidth=2.5)
#     axes[1].set_ylabel('y₂', fontsize=14, fontweight='bold')
#     axes[1].grid(True, alpha=0.3)
#     axes[1].set_title(f'Response y₂ (mean={y2.mean():.2f}, std={y2.std():.2f})')
#
#     axes[2].plot(t, y3, 'g-', linewidth=2.5)
#     axes[2].set_ylabel('y₃', fontsize=14, fontweight='bold')
#     axes[2].set_xlabel('Time', fontsize=13, fontweight='bold')
#     axes[2].grid(True, alpha=0.3)
#     axes[2].set_title(f'Response y₃ (mean={y3.mean():.2f}, std={y3.std():.2f})')
#
#     theta_str = f"θ = {theta_true}"
#     plt.suptitle(f'Model Responses (u₁={u1_value}, w₁={w1_value}, {theta_str})',
#                  fontsize=13, fontweight='bold')
#
#     plt.tight_layout()
#     plt.show()
#
#     return result
#
#
# # Default settings (u1=2.5, w1=2.0, true theta values)
# result = run_and_plot()
#
# # Custom settings
# result = run_and_plot(
#     t_final=20,
#     n_points=400,
#     u1_value=3.0,
#     w1_value=2.5,
#     theta_true=[2.5, 1.5, 1.2, 0.5, 0.3],
#     y0=[1.0, 2.0, 0.5],
#     save_path='my_custom_plot.png'
# )
