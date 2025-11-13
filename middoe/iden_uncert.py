import matplotlib.pyplot as plt
from pathlib import Path
from middoe.krnl_simula import simula
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def uncert(data, resultpr, system, models, iden_opt, case=None):
    """
    Perform uncertainty analysis on the optimization results.

    This function evaluates the uncertainty in the optimization results by analyzing
    the variance-covariance structure of the estimated parameters. It supports multiple
    methods for sensitivity analysis and variance-covariance computation.

    Parameters
    ----------
    data : dict
        Experimental data used for the analysis.
    resultpr : dict
        Dictionary containing the optimization results for each solver.
    system : dict
        System configuration, including variable definitions and constraints.
    models : dict
        Model definitions and settings, including mutation masks and parameter bounds.
    iden_opt : dict
        Identification options, including sensitivity method and variance-covariance type.
        - 'sens_m': str, optional
            Sensitivity method ('central' or 'forward'). Default is 'central'.
        - 'varcov': str, optional
            Variance-covariance type ('M', 'H', or 'B'). Default is 'H'.
    case : str, optional
        Specifies the analysis case. Default is None.

    Returns
    -------
    dict
        A dictionary containing the uncertainty analysis results, including:
        - 'results': dict
            Detailed results for each solver.
        - 'obs': Any
            Observed values from the analysis.

    Notes
    -----
    If `iden_opt['varcov'] == 'B'`, the function uses the bootstrap variance-covariance
    matrix stored in `resultpr[solver]['varcov']`.
    """
    # Sensitivity method and varcov selection
    sens_method = iden_opt.get('sens_m', 'central')
    varcov_key  = iden_opt.get('var-cov', 'H')  # 'M', 'H', or 'B'

    # Identify measurement variables
    tv_iphi_vars = list(system.get('tvi', {}))
    ti_iphi_vars = list(system.get('tii', {}))
    tv_ophi_vars = [v for v,c in system.get('tvo', {}).items() if c.get('meas', True)]
    ti_ophi_vars = [v for v,c in system.get('tio', {}).items() if c.get('meas', True)]

    # Collect measurement uncertainties (not directly used here)
    std_dev = {}
    for grp in ('tvo','tio'):
        for v,cfg in system.get(grp, {}).items():
            if cfg.get('meas', True):
                s = cfg.get('unc', 1.0)
                std_dev[v] = 1.0 if s is None or np.isnan(s) else s

    # Prepare epsilon settings
    eps_in = iden_opt.get('eps', None)
    if not isinstance(eps_in, dict):
        epsf = {sv: eps_in for sv in resultpr}
    else:
        epsf = eps_in.copy()

    resultun = {}
    observed_values = None

    # Loop over each solver's optimization result
    for solver, solver_results in resultpr.items():
        thetac = solver_results['scpr']
        thetas = models.get('mutation', {}).get(solver, [])
        # If epsilon not provided, run mesh-independency test
        if epsf.get(solver) is None:
            epsf[solver] = _fdm_mesh_independency(
                theta=np.ones_like(thetac),
                thetac=thetac,
                solver=solver,
                system=system,
                models=models,
                tv_iphi_vars=tv_iphi_vars,
                ti_iphi_vars=ti_iphi_vars,
                tv_ophi_vars=tv_ophi_vars,
                ti_ophi_vars=ti_ophi_vars,
                data=data,
                sens_method=sens_method,
                varcov=varcov_key,
                resultpr=resultpr
            )

        # Compute uncertainty metrics
        (
            optimized_data,
            LS, MLE, Chi,
            LSA, Z, V_matrix,
            CI, t_values,
            t_m, tv_input_m, ti_input_m,
            tv_output_m, ti_output_m,
            WLS, obs, MSE,
            R2_responses, R2_total, M
        ) = _uncert_metrics(
            theta=np.ones_like(thetac),
            data=data,
            active_solvers=[solver],
            thetac=thetac,
            eps=epsf,
            thetas=thetas,
            ti_iphi_vars=ti_iphi_vars,
            tv_iphi_vars=tv_iphi_vars,
            tv_ophi_vars=tv_ophi_vars,
            ti_ophi_vars=ti_ophi_vars,
            system=system,
            models=models,
            varcov=varcov_key,
            resultpr=resultpr,
            sens_method=sens_method
        )
        observed_values = obs

        resultun[solver] = {
            'optimization_result': solver_results,
            'data': optimized_data,
            'LS': LS,
            'MLE': MLE,
            'MSE': MSE,
            'Chi': Chi,
            'LSA': LSA,
            'Z': Z,
            'WLS': WLS,
            'R2_responses': R2_responses,
            'R2_total': R2_total,
            'V_matrix': V_matrix,
            'CI': CI,
            't_values': t_values,
            't_m': t_m,
            'tv_input_m': tv_input_m,
            'ti_input_m': ti_input_m,
            'tv_output_m': tv_output_m,
            'ti_output_m': ti_output_m,
            'estimations': thetac,
            'found_eps': epsf[solver],
            'M': M
        }

    # Store back eps
    iden_opt['eps'] = epsf

    # Summarize and report
    (
        scaled_params,
        solver_parameters,
        solver_cov_matrices,
        solver_confidence_intervals,
        resultun2
    ) = _report(resultun, models.get('mutation', {}), models, iden_opt.get('log', False))

    # Optionally update models in‐place
    if case is None:
        for sv in models['can_m']:
            if 'thetastart' not in models:
                models['thetastart'] = {}
                for sv in models['can_m']:
                    if sv not in models['thetastart']:
                        models['thetastart'][sv] = models['theta'][sv]

            models['theta'][sv]    = scaled_params[sv]
            models['V_matrix'][sv] = resultun[sv]['V_matrix']
            # Initialize 'LSA' as a dict if it does not exist; otherwise keep existing
            if 'LSA' not in models or not isinstance(models['LSA'], dict):
                models['LSA'] = {}

            # Update (or add) the LSA values for each solver key
            for sv in models['can_m']:
                models['theta'][sv] = scaled_params.get(sv)
                models['V_matrix'][sv] = resultun.get(sv, {}).get('V_matrix')

                lsa_value = resultun.get(sv, {}).get('LSA')
                if lsa_value is not None:
                    models['LSA'][sv] = lsa_value

    return {'results': resultun2, 'obs': observed_values}


def _uncert_metrics(
    theta, data, active_solvers, thetac, eps, thetas,
    ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
    system, models, varcov, resultpr, sens_method='forward'
):
    """
    Calculate detailed uncertainty metrics. Honors varcov=='B' to pull
    bootstrap covariance from resultpr.

    Assumes heteroskedastic (relative) measurement errors stored in MES_E:*.
    Each MES_E value is interpreted as a *fractional* standard deviation and
    is converted to an absolute sigma by multiplying with the signal magnitude
    (|y_pred|). A tiny floor avoids zero weights.

    Adds ill-conditioning diagnostics (rank/condition numbers/eigenvalues)
    without changing the original signature or return tuple. The diagnostic
    summary is logged (INFO to avoid colored 'red' outputs) and stored in
    resultpr[solver]['ill_report'] when possible.
    """
    # ---- diagnostic thresholds (adjust here if needed) -----------------
    _SLOPPY_THRESH = 1e3     # 'sloppy' if cond(M) >= this
    _COND_THRESH   = 1e8     # ill-conditioned if cond(M) >= this
    _TINY_EIG      = 1e-12   # ill if min eigenvalue < this
    _REPORT_LEVEL  = logging.INFO  # keep it white in most color schemes
    _SIG_FLOOR     = 1e-12   # floor for absolute sigmas

    # print(f'theta: {theta}')
    # print(f'thetac: {thetac}')

    theta_full = np.array(theta, dtype=float)
    active_idx = np.where(thetas)[0]
    print(f'active_idx: {active_idx}')
    n_active   = len(active_idx)
    thetac_arr = np.array(thetac, dtype=float)

    # Containers
    var_measurements = {}                     # {var: [(y_true, y_pred, sigma_abs), ...]}
    Q_accum = {v: [] for v in (tv_ophi_vars + ti_ophi_vars)}
    t_m = {}
    tv_input_m, ti_input_m = {}, {}
    tv_output_m, ti_output_m = {}, {}

    solver = active_solvers[0]
    h = eps[solver]

    # Loop over data sheets
    for sheet_name, sheet in data.items():
        # Time grid
        t_all = np.unique(sheet.get("X:all", pd.Series()).dropna().values)

        # Switch-pressure inputs
        swps = {}
        for v in tv_iphi_vars:
            tkey, lkey = f"{v}t", f"{v}l"
            if tkey in sheet and lkey in sheet:
                ta = sheet[tkey].dropna().values
                la = sheet[lkey].dropna().values
                if ta.size and la.size:
                    swps[tkey], swps[lkey] = ta, la

        # Input data
        ti_in = {v: sheet.get(v, pd.Series([np.nan])).iloc[0] for v in ti_iphi_vars}
        tv_in = {v: sheet.get(v, pd.Series()).dropna().values for v in tv_iphi_vars}
        cvp   = {v: sheet[f"CVP:{v}"].iloc[0] for v in system.get('tvi', {})}

        # Reference simulation
        tv_ref, ti_ref, _ = simula(
            t_all, swps, ti_in,
            {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
            theta_full, thetac_arr, cvp, tv_in, solver, system, models
        )

        # Store simulation outputs
        t_m[sheet_name]         = t_all.tolist()
        tv_input_m[sheet_name]  = tv_in
        ti_input_m[sheet_name]  = ti_in
        tv_output_m[sheet_name] = tv_ref
        ti_output_m[sheet_name] = ti_ref

        # Build perturbed sims for FD sensitivities
        perturbed = {}
        for p in active_idx:
            if sens_method == 'forward':
                thp = theta_full.copy(); thp[p] += h
                tvp, tip, _ = simula(
                    t_all, swps, ti_in,
                    {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
                    thp, thetac_arr, cvp, tv_in, solver, system, models
                )
                perturbed[p] = ('forward', (tvp, tip))
            else:
                thp = theta_full.copy(); thp[p] += h
                thm = theta_full.copy(); thm[p] -= h
                tvp, tip, _ = simula(
                    t_all, swps, ti_in,
                    {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
                    thp, thetac_arr, cvp, tv_in, solver, system, models
                )
                tvm, tim, _ = simula(
                    t_all, swps, ti_in,
                    {v:1 for v in ti_iphi_vars}, {v:1 for v in tv_iphi_vars}, 1,
                    thm, thetac_arr, cvp, tv_in, solver, system, models
                )
                perturbed[p] = ('central', (tvp, tip, tvm, tim))

        # Collect residuals and sensitivities
        for var in tv_ophi_vars + ti_ophi_vars:
            if var in tv_ophi_vars:
                # time-varying measurement
                xcol = f"MES_X:{var}"
                ycol = f"MES_Y:{var}"
                ecol = f"MES_E:{var}"
                mask = ~sheet[xcol].isna().values
                times = sheet[xcol][mask].values
                y_t   = sheet[ycol][mask].values
                idx   = np.isin(t_all, times)
                y_p   = np.array(tv_ref[var])[idx]

                # relative sigma from sheet, convert to absolute using |y_p|
                if ecol in sheet:
                    rel = sheet[ecol][mask].values.astype(float)
                    y_e = rel
                else:
                    y_e = np.full_like(y_t, 1.0, dtype=float)
            else:
                # time-independent measurement
                ycol = f"MES_Y:{var}"
                ecol = f"MES_E:{var}"
                y_t  = np.array([sheet[ycol].iloc[0]])
                y_p  = np.array([ti_ref[var]])
                if ecol in sheet:
                    rel = float(sheet[ecol].iloc[0])
                    # y_e = np.array([rel * np.abs(y_p[0]) + _SIG_FLOOR], dtype=float)
                    y_e = np.array([rel], dtype=float)
                else:
                    y_e = np.array([1.0], dtype=float)

            var_measurements.setdefault(var, []).extend(zip(y_t, y_p, y_e))

            # Sensitivity block
            cols = []
            for p in active_idx:
                mode, sims = perturbed[p]
                if mode == 'forward':
                    tvp, tip = sims
                    if var in tv_ophi_vars:
                        y2 = np.array([tvp[var][np.where(t_all==t)[0][0]] for t in times])
                    else:
                        y2 = np.array([tip[var]])
                    d = (y2 - y_p) / (h * thetac_arr[p])
                else:
                    tvp, tip, tvm, tim = sims
                    if var in tv_ophi_vars:
                        y_p_p = np.array([tvp[var][np.where(t_all==t)[0][0]] for t in times])
                        y_p_m = np.array([tvm[var][np.where(t_all==t)[0][0]] for t in times])
                    else:
                        y_p_p = np.array([tip[var]])
                        y_p_m = np.array([tim[var]])
                    d = (y_p_p - y_p_m) / (2*h*thetac_arr[p])
                cols.append(d)
            if len(cols):
                Q_accum[var].append(np.stack(cols, axis=1))

    # Flatten Q into LSA
    LSA = np.vstack([np.vstack(Q_accum[v]) for v in Q_accum])
    obs = sum(len(vals) for vals in var_measurements.values())
    dof = max(obs - n_active, 1)
    tcrit = stats.t.ppf(0.975, dof)

    # --- Variance–Covariance Selection ---
    if varcov == 'M':
        # Using absolute sigmas we just constructed
        sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]])
                               for v in Q_accum])
        # Avoid forming giant diagonal: scale rows instead
        scaled_J = LSA / sigs[:, None]
        M = scaled_J.T @ scaled_J
        V = np.linalg.pinv(M)

    elif varcov == 'H':
        hess_inv = resultpr[solver].get('hess_inv')
        err = resultpr[solver].get('fun', 1.0) / dof
        V = err * np.array(hess_inv)
        M = np.linalg.pinv(V)

    elif varcov == 'B':
        V = resultpr[solver].get('v')
        if V is None:
            raise ValueError(f"No bootstrap v matrix found for model '{solver}'")
        M = np.linalg.pinv(V)
        logger.info(f"Using bootstrap var–cov for {solver}")

    else:
        raise ValueError(f"Unknown varcov option '{varcov}'")

    # --- Ill-conditioning diagnostics ----------------------------------
    ill_report = {}
    try:
        s = np.linalg.svd(LSA, compute_uv=False)
        ill_report['Q_singular_values'] = s
        ill_report['Q_rank'] = int(np.sum(s > s[0] * np.finfo(float).eps))
        ill_report['Q_cond'] = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    except Exception as e:
        ill_report['Q_error'] = str(e)

    try:
        eig = np.linalg.eigvalsh(M)
        ill_report['M_eigenvalues'] = eig
        ill_report['M_min_eig'] = float(eig.min())
        ill_report['M_max_eig'] = float(eig.max())
        ill_report['M_cond'] = float(eig.max() / eig.min()) if eig.min() > 0 else np.inf
    except Exception as e:
        ill_report['M_error'] = str(e)

    ill_report['is_rank_deficient'] = ill_report.get('Q_rank', n_active) < n_active
    ill_report['is_sloppy'] = ill_report.get('M_cond', np.inf) >= _SLOPPY_THRESH
    ill_report['is_ill_conditioned'] = (
        ill_report['is_rank_deficient'] or
        ill_report.get('M_cond', 0) >= _COND_THRESH or
        ill_report.get('M_min_eig', _TINY_EIG) < _TINY_EIG
    )
    ill_report['summary'] = (
        f"Rank(Q)={ill_report.get('Q_rank','?')}/{n_active}, "
        f"cond(Q)≈{ill_report.get('Q_cond',np.nan):.2e}, "
        f"cond(M)≈{ill_report.get('M_cond',np.nan):.2e}, "
        f"λ_min(M)≈{ill_report.get('M_min_eig',np.nan):.2e}; "
        f"{'ILL-CONDITIONED' if ill_report['is_ill_conditioned'] else 'OK'}"
    )

    logger.log(_REPORT_LEVEL, "Conditioning diagnostics: %s", ill_report['summary'])

    # Stash report into resultpr without changing the API
    try:
        if isinstance(resultpr.get(solver, None), dict):
            resultpr[solver]['ill_report'] = ill_report
    except Exception:
        pass

    # Confidence intervals and t-values
    if varcov == 'B':
        CI = tcrit * np.sqrt(np.diag(V))[active_idx]
        theta_std = np.sqrt(np.diag(V))[active_idx]
    else:
        CI = tcrit * np.sqrt(np.diag(V))
        theta_std = np.sqrt(np.diag(V))
    print(f'CI: {CI}')

    # t_values = (theta_full[active_idx] * thetac_arr[active_idx]) / CI
    with np.errstate(divide='ignore', invalid='ignore'):
        numer = theta_full[active_idx] * thetac_arr[active_idx]
        t_values = np.where(CI != 0, numer / np.sqrt(np.diag(V))[active_idx], np.inf)

    # Z-scores (normalized sensitivities)
    resp_sigs = np.concatenate([np.array([s for _, _, s in var_measurements[v]]) for v in Q_accum])
    Z = LSA * theta_std / resp_sigs[:, None]

    # Compute LS, WLS, MLE, Chi, MSE, R2
    LS = WLS = MLE = Chi = 0.0
    total_err = 0.0
    R2_data = {v: {'y':[], 'yp':[]} for v in Q_accum}
    for v, vals in var_measurements.items():
        for y, y_pred, s in vals:
            r = y - y_pred
            LS  += r**2
            WLS += (r/s)**2
            MLE += 0.5*(np.log(2*np.pi*s**2)+(r/s)**2)
            Chi += r**2/s**2
            total_err += r**2
            R2_data[v]['y'].append(y)
            R2_data[v]['yp'].append(y_pred)

    R2_responses = {
        v: 1 - np.sum((np.array(d['y'])-np.array(d['yp']))**2) /
              np.sum((np.array(d['y'])-np.mean(d['y']))**2)
        if len(d['y']) > 1 else np.nan
        for v, d in R2_data.items()
    }
    MSE = total_err/obs if obs else np.nan

    # Aggregate all y and y_pred across variables for total R2
    all_y, all_yp = [], []
    for v_vals in var_measurements.values():
        for y, y_pred, _ in v_vals:
            all_y.append(y)
            all_yp.append(y_pred)

    if len(all_y) > 1:
        y_arr = np.array(all_y)
        yp_arr = np.array(all_yp)
        R2_total = 1 - np.sum((y_arr - yp_arr) ** 2) / np.sum((y_arr - np.mean(y_arr)) ** 2)
    else:
        R2_total = np.nan

    return (
        data, LS, MLE, Chi,
        LSA, Z, V, CI, t_values,
        t_m, tv_input_m, ti_input_m,
        tv_output_m, ti_output_m,
        WLS, obs, MSE, R2_responses, R2_total, M
    )



def _report(result, mutation, models, logging_flag):
    """
    Summarize and optionally log the uncertainty results.
    """
    scaled_params = {}
    solver_parameters = {}
    solver_cov_matrices = {}
    solver_confidence_intervals = {}
    sum_of_chi_squared = sum(1 / (res['Chi']) ** 2 for res in result.values())
    for solver, res in result.items():
        opt = res['optimization_result']
        scpr = opt['scpr']
        scaled_params[solver] = scpr.tolist()
        solver_parameters[solver] = opt['x']
        solver_cov_matrices[solver] = res['V_matrix']
        solver_confidence_intervals[solver] = {
            i: ci for i, ci in enumerate(res['CI'])
        }
        res['P'] = ((1 / (res['Chi']) ** 2) / sum_of_chi_squared) * 100
        if logging_flag:
            logger.info(f"Solver {solver}: Estimated θ = {scpr}")
            logger.info(f"Solver {solver}: t-val = {res['t_values']}")
            # logger.info(f"Solver {solver}: active = {res['optimization_result'][solver]['activeparams']}")
    return (
        scaled_params,
        solver_parameters,
        solver_cov_matrices,
        solver_confidence_intervals,
        result
    )


def _fdm_mesh_independency(
    theta, thetac, solver, system, models,
    tv_iphi_vars, ti_iphi_vars, tv_ophi_vars, ti_ophi_vars,
    data, sens_method, varcov, resultpr
):
    """
    FDM mesh-dependency test to choose epsilon for sensitivities.
    """
    logger.info(f"Performing mesh-independency test for {solver}")
    eps_values = np.logspace(-12, -1, 50)
    eigs = []
    varcov = 'M'
    for eps in eps_values:
        try:
            out = _uncert_metrics(
                theta, data, [solver], thetac, {solver:eps}, [True]*len(theta),
                ti_iphi_vars, tv_iphi_vars, tv_ophi_vars, ti_ophi_vars,
                system, models, varcov, resultpr, sens_method=sens_method
            )
            V = out[6]  # returned V_matrix
            eigs.append(np.linalg.eigvalsh(V))
        except Exception:
            eigs.append([np.nan]*len(theta))
    eigs = np.array(eigs)

    # Plot and save
    folder = Path.cwd() / "meshindep_plots"
    folder.mkdir(exist_ok=True)
    base = folder / f"{solver}_eps_test"
    plt.figure(figsize=(8,5))
    for i in range(eigs.shape[1]):
        plt.plot(eps_values, eigs[:,i], marker='o', label=f'eig{i+1}')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Epsilon'); plt.ylabel('Eigenvalues')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{base}.png"); plt.close()
    pd.DataFrame(eigs, index=eps_values).to_excel(f"{base}.xlsx")

    # Choose epsilon minimizing variation
    valid = np.isfinite(eigs).all(axis=1)
    idx = np.where(valid)[0]
    win = 5
    half = win//2
    best, best_var = eps_values[0], np.inf
    for i in idx[half:-half]:
        var = eigs[i-half:i+half+1].std(axis=0).sum()
        if var<best_var:
            best_var, best = var, eps_values[i]
    logger.info(f"Selected epsilon={best:.1e} for {solver}")
    return best

