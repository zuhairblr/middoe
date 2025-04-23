from pygpas.evaluate import evaluate, evaluate_trajectories
from pygpas.server import StartedConnected
from pygpas.special_variables import ExecutionOutcome

with StartedConnected() as client:
    # Open the gPAS model using the hardcoded path and credentials
    client.open(
        'C:/Users/Tadmin/Desktop/f11/model4.zip',
        '@@TTmnoa698'
    )

    # Set time-invariant inputs
    client.set_input_value('aps', 5.500000000000001e-05)
    client.set_input_value('cac', 44.93)
    client.set_input_value('mld', 36000.0)
    client.set_input_value('rho', 3191.0)

    # Set time-variant inputs
    P = [1.0] * 101
    T = [293.15] * 101
    client.set_input_value('P', P)
    client.set_input_value('T', T)

    # Set initial condition and parameter values
    client.set_input_value('y0', [0.001])
    client.set_input_value('theta', [3.1650075431484574e-07, 20239.398294836665, 1.2133668813455163])

    # Run model evaluation
    result = evaluate(client)
    if result.outcome != ExecutionOutcome.success:
        raise RuntimeError(f"Simulation failed'")
