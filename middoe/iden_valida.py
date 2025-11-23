# # iden_valida.py
#
# import copy
# from middoe.iden_utils import validation_R2
# from middoe.iden_parmest import parmest
# from middoe.iden_uncert import uncert
# import numpy as np
# from middoe.log_utils import  read_excel
#
# def validation(system, models, iden_opt, round_data):
#     """
#     Perform leave-one-out validation to evaluate the generalization of the parameter estimation.
#
#     This function splits the experimental data into training and validation sets for each fold,
#     performs parameter estimation on the training set, and evaluates the model's performance
#     on the validation set. It computes metrics such as R2 and MSE for both prediction and validation.
#
#     Parameters
#     ----------
#     system : dict
#         Structure of the model, including variables and their properties.
#     models : dict
#         Settings related to the modelling process, including theta parameters and mutation masks.
#     iden_opt : dict
#         Settings for the estimation process, including active solvers and plotting options.
#     round_data : dict
#         Data from previous rounds of the estimation process, including optimization results.
#
#     Returns
#     -------
#     dict
#         A dictionary containing the validation results, including:
#         - R2_prd : dict
#             R2 values for predictions in each fold.
#         - R2_val : dict
#             R2 values for validations in each fold.
#         - R2_stats : dict
#             Mean and standard deviation of R2 values for predictions and validations.
#         - MSE_stats : dict
#             Mean and standard deviation of MSE values for predictions and validations.
#     """
#     data = read_excel()
#     iden_optc = copy.deepcopy(iden_opt)
#     iden_optc['var-cov'] = 'M'
#     # Initialize cross-validation results
#     sheet_names = list(data.keys())
#     # sheet_names = list(data.keys())[1:-1]
#     n_sheets = len(sheet_names)
#     R2_prd, R2_val, parameters, MSE_pred, MSE_val = {}, {}, {}, {}, {}
#     models['mutation'] = {
#         solver: round_data[list(round_data.keys())[-1]]['result'][solver]['optimization_result']['activeparams']
#         for solver in models['can_m']
#     }
#
#     for i in range(n_sheets):
#         # Split data into training and validation sets
#         validation_sheet = sheet_names[i]
#         training_sheets = [sheet for sheet in sheet_names if sheet != validation_sheet]
#
#         training_data = {sheet: data[sheet] for sheet in training_sheets}
#         validation_data = {validation_sheet: data[validation_sheet]}
#
#         print(f"Running validation fold {i + 1}/{n_sheets}...")
#         print(f"Validation sheet: {validation_sheet}")
#
#         resultpr = parmest(system, models, iden_opt, training_data, case='strov')
#         uncert_results_pred = uncert(training_data, resultpr, system, models, iden_optc)
#
#         resultun_pred = uncert_results_pred['results']
#         scaled_params_pred = {solver: resultun_pred[solver]['optimization_result']['scpr'] for solver in resultun_pred}
#
#         uncert_results_vals = uncert(validation_data, resultpr, system, models, iden_optc)
#         resultun_val = uncert_results_vals['results']
#
#         # plotting1 = Plotting_Results(models, f'val{i+1}')  # Instantiate Plotting class
#         # plotting1.fit_plot(validation_data, resultun_val, system)
#         R2_prd[i + 1], R2_val[i + 1], R2_ref, MSE_pred[i + 1], MSE_val[i + 1], MSE_ref = compute_validation_error(
#             resultun_pred, resultun_val, scaled_params_pred, round_data)
#
#     # Ensure the 'modelling' directory exists
#     validation_R2(R2_prd, R2_val, R2_ref, case='R2')
#     validation_R2(MSE_pred, MSE_val, MSE_ref, case='MSE')
#     # validation_params(parameters, ref_params)
#
#     # Compute average and std dev for each solver
#     R2_stats, MSE_stats = {}, {}
#     solvers = models['can_m']
#
#     for solver in solvers:
#         pred_r2_values = [R2_prd[i + 1][solver] for i in range(n_sheets)]
#         val_r2_values = [R2_val[i + 1][solver] for i in range(n_sheets)]
#         pred_mse_values = [MSE_pred[i + 1][solver] for i in range(n_sheets)]
#         val_mse_values = [MSE_val[i + 1][solver] for i in range(n_sheets)]
#
#         R2_stats[solver] = {
#             'prediction_mean': np.mean(pred_r2_values),
#             'prediction_std': np.std(pred_r2_values),
#             'validation_mean': np.mean(val_r2_values),
#             'validation_std': np.std(val_r2_values)
#         }
#
#         MSE_stats[solver] = {
#             'prediction_mean': np.mean(pred_mse_values),
#             'prediction_std': np.std(pred_mse_values),
#             'validation_mean': np.mean(val_mse_values),
#             'validation_std': np.std(val_mse_values)
#         }
#
#     validres = {'R2_prd': R2_prd,
#                 'R2_val': R2_val,
#                 'R2_stats': R2_stats,
#                 'MSE_stats': MSE_stats}
#
#     print(validres)
#
#     return validres
#
# def compute_validation_error(resultun_pred, resultun_val, scaled_params, round_data):
#     prediction_R2, validation_R2 = {}, {}
#     prediction_MSE, validation_MSE = {}, {}
#     R2_ref, MSE_ref = {}, {}
#
#     last_key = list(round_data.keys())[-1]
#
#     for solver in resultun_pred:
#         if solver in resultun_val:
#             prediction_R2[solver] = resultun_pred[solver]['R2_total']
#             prediction_MSE[solver] = resultun_pred[solver]['MSE']
#             validation_R2[solver] = resultun_val[solver]['R2_total']
#             validation_MSE[solver] = resultun_val[solver]['MSE']
#             R2_ref[solver] = round_data[last_key]['result'][solver]['R2_total']
#             MSE_ref[solver] = round_data[last_key]['result'][solver]['MSE']
#             # params = scaled_params
#             # ref_params[solver] = round_data[last_key]['scaled_params'][solver]
#
#     return prediction_R2, validation_R2, R2_ref, prediction_MSE, validation_MSE, MSE_ref


# iden_valida.py

import copy
from middoe.iden_utils import validation_R2
from middoe.iden_parmest import parmest
from middoe.iden_uncert import uncert
import numpy as np
from middoe.log_utils import read_excel


def validation(system, models, iden_opt, round_data):
    r"""
    Perform leave-one-out cross-validation to assess parameter estimation generalization.

    This function implements k-fold cross-validation where k equals the number of experimental
    sheets. For each fold, one sheet is held out as validation data while remaining sheets
    are used for training. Parameter estimation is performed on the training set, and model
    performance is evaluated on both training (prediction) and validation sets. This assesses
    whether the model generalizes beyond the data used for parameter fitting.

    Parameters
    ----------
    system : dict
        System configuration including:
            - 'tvi' : dict
                Time-variant input definitions.
            - 'tii' : dict
                Time-invariant input definitions.
            - 'tvo' : dict
                Time-variant output definitions with measurement flags.
            - 'tio' : dict
                Time-invariant output definitions with measurement flags.

    models : dict
        Model definitions:
            - 'can_m' : list[str]
                Active model/solver names.
            - 'theta' : dict[str, list[float]]
                Nominal parameter values.
            - 'mutation' : dict[str, list[bool]]
                Parameter activity masks (updated from round_data).
            - 't_u', 't_l' : dict
                Parameter bounds.

    iden_opt : dict
        Identification options:
            - 'meth' : str
                Optimization method (from paper Table S3):
                    * 'SLSQP', 'LMBFGS', 'TC', 'NMS', 'BFGS', 'DE'
            - 'ob' : str
                Objective function (from paper Table S2):
                    * 'LS', 'WLS', 'MLE', 'CS'
            - 'var-cov' : str
                Covariance method (internally overridden to 'J' for validation).
            - 'log' : bool
                Logging flag.

    round_data : dict
        Results from previous estimation rounds. Used to:
            - Extract final parameter activity masks from last round.
            - Compute reference R² and MSE for comparison.

    Returns
    -------
    validres : dict
        Comprehensive validation results:
            - 'R2_prd' : dict[int, dict[str, float]]
                R² values for predictions (training set) in each fold.
                Format: {fold_number: {solver: R2_value}}
            - 'R2_val' : dict[int, dict[str, float]]
                R² values for validation (held-out set) in each fold.
            - 'R2_stats' : dict[str, dict[str, float]]
                Summary statistics across folds for each solver:
                    * 'prediction_mean' : Mean R² on training sets
                    * 'prediction_std' : Std dev of R² on training sets
                    * 'validation_mean' : Mean R² on validation sets
                    * 'validation_std' : Std dev of R² on validation sets
            - 'MSE_stats' : dict[str, dict[str, float]]
                Summary statistics for Mean Squared Error (same structure as R2_stats).

    Notes
    -----
    **Leave-One-Out Cross-Validation (LOOCV)**:
    For n experimental sheets:
        1. **Fold i**: Hold out sheet i, train on remaining n-1 sheets
        2. **Estimate parameters** on training data using parmest()
        3. **Evaluate** on both training set (R²_pred, MSE_pred) and validation set (R²_val, MSE_val)
        4. **Repeat** for all n folds
        5. **Aggregate** statistics (mean ± std) across folds

    **Validation Workflow**:
        1. Extract final mutation masks from last estimation round
        2. For each fold:
            - Split data into training/validation
            - Run parameter estimation on training data (case='strov' uses previous round's parameters)
            - Compute uncertainty on training data
            - Compute uncertainty on validation data (using parameters from training)
            - Calculate R² and MSE for both sets
        3. Generate validation plots (R² envelope, MSE envelope)
        4. Compute summary statistics

    **R² Calculation** (from paper, referenced to Bard 1974):
        \[
        R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
        \]
    where \( y_i \) are observations, \( \hat{y}_i \) are predictions, and \( \bar{y} \)
    is the mean of observations.

    **Interpretation**:
        - **R²_val ≈ R²_pred**: Model generalizes well (no overfitting)
        - **R²_val << R²_pred**: Overfitting (model memorizes training data)
        - **Low R²_val std**: Consistent performance across folds (robust model)
        - **High R²_val std**: Performance depends strongly on specific data selection

    **Output Plots**:
        - R² envelope plot showing prediction vs validation ranges
        - MSE envelope plot showing error magnitude across folds
        - Both saved via validation_R2() function

    **Case Parameter**:
    'strov' (start from previous round optimal values) ensures parameter estimation
    starts from the best estimates obtained in previous rounds, improving convergence
    and reducing computational cost for each fold.

    **Best Practices**:
        - Use LOOCV after completing sequential MBDoE rounds
        - High validation R² (>0.95) indicates good model quality
        - Small std deviation (<0.05) indicates robust parameter estimates
        - Large difference between prediction and validation R² suggests need for more data

    References
    ----------
    .. [1] Tabrizi, Z., Barbera, E., Leal da Silva, W.R., & Bezzo, F. (2025).
       MIDDoE: An MBDoE Python package for model identification, discrimination,
       and calibration.
       *Digital Chemical Engineering*, 17, 100276.
       https://doi.org/10.1016/j.dche.2025.100276

    .. [2] Stone, M. (1974).
       Cross-validatory choice and assessment of statistical predictions.
       *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 111–133.
       https://doi.org/10.1111/j.2517-6161.1974.tb00994.x

    .. [3] Bard, Y. (1974).
       *Nonlinear Parameter Estimation*. Academic Press, New York.

    See Also
    --------
    parmest : Parameter estimation routine called for each fold.
    uncert : Uncertainty analysis routine for computing R² and MSE.
    validation_R2 : Plotting function for validation results.
    compute_validation_error : Helper function for metric extraction.

    Examples
    --------
    >>> import numpy as np
    >>> from middoe import iden_valida
    >>>
    >>> # After completing 3 rounds of sequential MBDoE-PP
    >>> round_data = {
    ...     'Round 1': {...},  # Preliminary experiments
    ...     'Round 2': {...},  # First MBDoE-PP design
    ...     'Round 3': {...}   # Second MBDoE-PP design (final)
    ... }
    >>>
    >>> # Configure validation
    >>> iden_opt = {
    ...     'meth': 'SLSQP',
    ...     'ob': 'WLS',
    ...     'log': True
    ... }
    >>>
    >>> # Run leave-one-out cross-validation
    >>> valid_results = iden_valida.validation(system, models, iden_opt, round_data)
    >>>
    >>> # Check generalization quality
    >>> for solver in models['can_m']:
    ...     stats = valid_results['R2_stats'][solver]
    ...     print(f"\nModel {solver}:")
    ...     print(f"  Training R²:   {stats['prediction_mean']:.4f} ± {stats['prediction_std']:.4f}")
    ...     print(f"  Validation R²: {stats['validation_mean']:.4f} ± {stats['validation_std']:.4f}")
    ...
    ...     # Check for overfitting
    ...     diff = stats['prediction_mean'] - stats['validation_mean']
    ...     if diff > 0.05:
    ...         print(f"  ⚠️  Warning: Possible overfitting (ΔR² = {diff:.4f})")
    ...     elif stats['validation_mean'] > 0.95:
    ...         print(f"  ✓ Excellent generalization!")
    ...     else:
    ...         print(f"  ✓ Good generalization")

    >>> # Example output:
    >>> # Model M1:
    >>> #   Training R²:   0.9987 ± 0.0008
    >>> #   Validation R²: 0.9985 ± 0.0015
    >>> #   ✓ Excellent generalization!
    """

    data = read_excel()
    iden_optc = copy.deepcopy(iden_opt)
    iden_optc['var-cov'] = 'M'

    # Initialize cross-validation results
    sheet_names = list(data.keys())
    n_sheets = len(sheet_names)
    R2_prd, R2_val, parameters, MSE_pred, MSE_val = {}, {}, {}, {}, {}

    # Extract mutation masks from final round
    models['mutation'] = {
        solver: round_data[list(round_data.keys())[-1]]['result'][solver]['optimization_result']['activeparams']
        for solver in models['can_m']
    }

    # Leave-one-out cross-validation loop
    for i in range(n_sheets):
        # Split data into training and validation sets
        validation_sheet = sheet_names[i]
        training_sheets = [sheet for sheet in sheet_names if sheet != validation_sheet]

        training_data = {sheet: data[sheet] for sheet in training_sheets}
        validation_data = {validation_sheet: data[validation_sheet]}

        print(f"Running validation fold {i + 1}/{n_sheets}...")
        print(f"Validation sheet: {validation_sheet}")

        # Estimate parameters on training data
        resultpr = parmest(system, models, iden_opt, training_data, case='strov')

        # Compute metrics on training set (prediction)
        uncert_results_pred = uncert(training_data, resultpr, system, models, iden_optc)
        resultun_pred = uncert_results_pred['results']
        scaled_params_pred = {
            solver: resultun_pred[solver]['optimization_result']['scpr']
            for solver in resultun_pred
        }

        # Compute metrics on validation set
        uncert_results_vals = uncert(validation_data, resultpr, system, models, iden_optc)
        resultun_val = uncert_results_vals['results']

        # Extract R2 and MSE for this fold
        R2_prd[i + 1], R2_val[i + 1], R2_ref, MSE_pred[i + 1], MSE_val[i + 1], MSE_ref = compute_validation_error(
            resultun_pred, resultun_val, scaled_params_pred, round_data
        )

    # Generate validation plots
    validation_R2(R2_prd, R2_val, R2_ref, case='R2')
    validation_R2(MSE_pred, MSE_val, MSE_ref, case='MSE')

    # Compute summary statistics across folds
    R2_stats, MSE_stats = {}, {}
    solvers = models['can_m']

    for solver in solvers:
        pred_r2_values = [R2_prd[i + 1][solver] for i in range(n_sheets)]
        val_r2_values = [R2_val[i + 1][solver] for i in range(n_sheets)]
        pred_mse_values = [MSE_pred[i + 1][solver] for i in range(n_sheets)]
        val_mse_values = [MSE_val[i + 1][solver] for i in range(n_sheets)]

        R2_stats[solver] = {
            'prediction_mean': np.mean(pred_r2_values),
            'prediction_std': np.std(pred_r2_values),
            'validation_mean': np.mean(val_r2_values),
            'validation_std': np.std(val_r2_values)
        }

        MSE_stats[solver] = {
            'prediction_mean': np.mean(pred_mse_values),
            'prediction_std': np.std(pred_mse_values),
            'validation_mean': np.mean(val_mse_values),
            'validation_std': np.std(val_mse_values)
        }

    validres = {
        'R2_prd': R2_prd,
        'R2_val': R2_val,
        'R2_stats': R2_stats,
        'MSE_stats': MSE_stats
    }

    print(validres)

    return validres


def compute_validation_error(resultun_pred, resultun_val, scaled_params, round_data):
    r"""
    Extract and organize validation metrics from uncertainty analysis results.

    This helper function processes uncertainty analysis outputs to extract R² and MSE
    metrics for both prediction (training set) and validation (held-out set). It also
    retrieves reference metrics from the final estimation round for comparison.

    Parameters
    ----------
    resultun_pred : dict[str, dict]
        Uncertainty analysis results for predictions (training set).
        Each solver dict contains:
            - 'R2_total' : float
                Overall R² across all response variables.
            - 'MSE' : float
                Mean Squared Error.
    resultun_val : dict[str, dict]
        Uncertainty analysis results for validation (held-out set).
        Same structure as resultun_pred.
    scaled_params : dict[str, np.ndarray]
        Estimated parameters for each solver (currently unused but passed for extensibility).
    round_data : dict
        Results from all previous estimation rounds. Used to extract reference metrics
        from the final round.

    Returns
    -------
    prediction_R2 : dict[str, float]
        R² values for each solver on training set.
    validation_R2 : dict[str, float]
        R² values for each solver on validation set.
    R2_ref : dict[str, float]
        Reference R² values from final estimation round (full dataset).
    prediction_MSE : dict[str, float]
        MSE values for each solver on training set.
    validation_MSE : dict[str, float]
        MSE values for each solver on validation set.
    MSE_ref : dict[str, float]
        Reference MSE values from final estimation round (full dataset).

    Notes
    -----
    **Return Value Interpretation**:
        - **prediction_R2**: How well model fits training data used for parameter estimation.
        - **validation_R2**: How well model predicts held-out data (true generalization test).
        - **R2_ref**: Baseline R² when all data is used (no cross-validation).

    **Comparison Metrics**:
        - Compare validation_R2 to R2_ref to assess if cross-validation degrades performance
        - Compare validation_R2 to prediction_R2 to detect overfitting
        - Ideal: validation_R2 ≈ prediction_R2 ≈ R2_ref

    **Dictionary Keys**:
    All dictionaries are keyed by solver name (e.g., 'M1', 'M2'). Only solvers present
    in both resultun_pred and resultun_val are included in the output.

    **Reference Round**:
    The reference metrics are extracted from the last key in round_data, which should
    correspond to the final (most refined) estimation round using all available data.

    See Also
    --------
    validation : Main cross-validation routine that calls this function.
    uncert : Computes R2_total and MSE during uncertainty analysis.

    Examples
    --------
    >>> # After running uncertainty analysis on one fold
    >>> pred_r2, val_r2, ref_r2, pred_mse, val_mse, ref_mse = compute_validation_error(
    ...     resultun_pred, resultun_val, scaled_params, round_data
    ... )
    >>>
    >>> # Check generalization for each solver
    >>> for solver in pred_r2:
    ...     print(f"{solver}:")
    ...     print(f"  Training R²: {pred_r2[solver]:.3f}")
    ...     print(f"  Validation R²: {val_r2[solver]:.3f}")
    ...     print(f"  Reference R²: {ref_r2[solver]:.3f}")
    ...     if val_r2[solver] < pred_r2[solver] - 0.1:
    ...         print("  ⚠️ Possible overfitting detected!")
    """
    prediction_R2, validation_R2 = {}, {}
    prediction_MSE, validation_MSE = {}, {}
    R2_ref, MSE_ref = {}, {}

    # Extract reference metrics from last round
    last_key = list(round_data.keys())[-1]

    # Process each solver
    for solver in resultun_pred:
        if solver in resultun_val:
            # Extract prediction (training) metrics
            prediction_R2[solver] = resultun_pred[solver]['R2_total']
            prediction_MSE[solver] = resultun_pred[solver]['MSE']

            # Extract validation (held-out) metrics
            validation_R2[solver] = resultun_val[solver]['R2_total']
            validation_MSE[solver] = resultun_val[solver]['MSE']

            # Extract reference metrics from final round
            R2_ref[solver] = round_data[last_key]['result'][solver]['R2_total']
            MSE_ref[solver] = round_data[last_key]['result'][solver]['MSE']

    return prediction_R2, validation_R2, R2_ref, prediction_MSE, validation_MSE, MSE_ref