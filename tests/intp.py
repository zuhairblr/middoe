from middoe.log_utils import load_from_jac, save_to_xlsx
from middoe.iden_utils import run_postprocessing

results = load_from_jac()
iden = results['iden']
sensa = results['sensa']

save_to_xlsx(sensa)

run_postprocessing(
    round_data=results['iden'],
    solvers=['f20'],
    selected_rounds=[ 2, 3],
    plot_global_p_and_t=False,
    plot_confidence_spaces=True,
    plot_p_and_t_tests=True,
    export_excel_reports=True,
    plot_estimability=False
)

