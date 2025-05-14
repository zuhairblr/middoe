from middoe.log_utils import load_from_jac, save_sobol_results_to_excel
from middoe.iden_utils import run_postprocessing

results = load_from_jac()
iden = results['iden']
sensa = results['sensa']

save_sobol_results_to_excel(sensa)

run_postprocessing(
    round_data=results['iden'],
    solvers=['f20', 'f21'],
    selected_rounds=[ 2, 3, 4],
    plot_global_p_and_t=False,
    plot_confidence_spaces=False,
    plot_p_and_t_tests=True,
    export_excel_reports=False,
    plot_estimability=False
)

