from middoe.log_utils import load_from_jac, save_sobol_results_to_excel
from middoe.iden_utils import run_postprocessing

results = load_from_jac()
iden = results['iden']
sensa = results['sensa']

save_sobol_results_to_excel(sensa)

run_postprocessing(round_data=iden, solvers=['f20', 'f21'], selected_rounds=[ 2, 3, 4])

