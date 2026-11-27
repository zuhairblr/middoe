#iden_utils.py

import os
import re
import logging
import functools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.patches import Ellipse, Patch, Rectangle
from scipy.stats import norm, chi2
from scipy.special import gamma

# Whether plotting helpers open figure windows (plt.show) or only save and close them.
# run_postprocessing(..., show=False) switches it off for its own duration.
_SHOW_PLOTS = True


_NON_INTERACTIVE = {'agg', 'pdf', 'ps', 'svg', 'pgf', 'cairo', 'template'}


def _can_show():
    """True when the matplotlib backend can display figures (GUI window or notebook inline)."""
    try:
        return mpl.get_backend().lower() not in _NON_INTERACTIVE
    except Exception:
        return False


def _show():
    """Show the current figure when display is on and possible, then close it."""
    if _SHOW_PLOTS and _can_show():
        plt.show()
    plt.close()

from scipy.interpolate import interp1d
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
#                                   Shared plotting helpers
# ---------------------------------------------------------------------------------------------
# One categorical palette (colour-blind checked), recessive ink and grid. Every plotting
# function runs inside ``_styled`` so the user's own matplotlib settings are never changed.
_PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
_INK, _INK2, _GRID = '#0b0b0b', '#52514e', '#e2e1dd'
_LINESTYLES = ['-', '--', '-.', ':']
_MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
_CORR_CMAP = LinearSegmentedColormap.from_list('middoe_corr', ['#2a78d6', '#efeeeb', '#e34948'])
_STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.edgecolor': _INK2, 'axes.labelcolor': _INK, 'text.color': _INK,
    'xtick.color': _INK2, 'ytick.color': _INK2, 'xtick.labelcolor': _INK2, 'ytick.labelcolor': _INK2,
    'axes.grid': True, 'grid.color': _GRID, 'grid.linestyle': '-', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': mpl.cycler(color=_PALETTE),
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'figure.titlesize': 11, 'legend.frameon': False, 'lines.linewidth': 1.8, 'lines.markersize': 5,
    'savefig.dpi': 200, 'figure.dpi': 100, 'axes.formatter.useoffset': False,
}

# Accepted spellings of save_rounds' ``design_type`` -> (legend label, shading colour)
_DESIGN_KINDS = {
    'preliminary': ('preliminary / available data', '#8f8e88'),
    'MD': ('MBDoE-MD (discrimination)', _PALETTE[1]),
    'PP': ('MBDoE-PP (precision)', _PALETTE[0]),
    'MBDoE': ('MBDoE', _PALETTE[6]),
}


def _color(i):
    """Categorical colour of slot ``i`` (slots repeat only past eight series)."""
    return _PALETTE[int(i) % len(_PALETTE)]


def _styled(func):
    """Run a plotting function inside the MIDDoE style; rcParams are restored afterwards."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(_STYLE):
            return func(*args, **kwargs)
    return wrapper


def _finish(fig, filename, show=True):
    """Save ``fig`` and show or close it."""
    fig.savefig(filename, bbox_inches='tight')
    if show and _SHOW_PLOTS and _can_show():
        plt.figure(fig.number)
        plt.show()
    plt.close(fig)


def _log_axis(ax):
    """Log y axis with plain-number labels (2, 5, 10, 20 ... instead of 2x10^1)."""
    ax.set_yscale('log')
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    fmt = FuncFormatter(lambda v, _: f'{v:g}')
    ax.yaxis.set_major_formatter(fmt)
    if lo > 0 and hi / lo < 300:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
        ax.yaxis.set_minor_formatter(fmt)
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


def _design_kind(label):
    """Map any design label used with save_rounds onto 'preliminary', 'MD', 'PP' or 'MBDoE'."""
    if label is None:
        return None
    tokens = set(re.split(r'[^a-z0-9]+', str(label).lower()))
    if tokens & {'md', 'discrimination', 'discriminating'}:
        return 'MD'
    if tokens & {'pp', 'precision'}:
        return 'PP'
    if tokens & {'preliminary', 'classic', 'available', 'historical', 'historic', 'initial',
                 'factorial', 'prior', 'lhs', 'sobol', 'random'}:
        return 'preliminary'
    if tokens & {'doe', 'mbdoe', 'design', 'optimal'}:
        return 'MBDoE'
    return None


def _param_names(models, solver, n):
    """Parameter labels: ``models['theta_names'][solver]`` when given, otherwise theta_1..theta_n."""
    names = None
    if isinstance(models, dict):
        names = (models.get('theta_names') or {}).get(solver)
    if names is not None and len(names) == n:
        return [str(s) for s in names]
    return [rf'$\theta_{{{i + 1}}}$' for i in range(n)]


def _get(obj, key, default=None):
    """dict-style or attribute access (OptimizeResult is both)."""
    if obj is None:
        return default
    try:
        return obj.get(key, default)
    except AttributeError:
        return getattr(obj, key, default)


def _active_idx(sol, mutation=None):
    """Indices of the parameters estimated in ``sol`` (the order of CI, t_values and V)."""
    n = len(np.ravel(sol['estimations']))
    act = _get(sol.get('optimization_result'), 'activeparams')
    if act is None:
        act = mutation
    mask = np.asarray(act, dtype=bool).ravel() if act is not None else np.ones(n, bool)
    if mask.size != n:
        mask = np.ones(n, bool)
    idx = np.where(mask)[0]
    ci = np.ravel(sol.get('CI', []))
    if ci.size and ci.size != idx.size and ci.size == n:   # results computed on all parameters
        idx = np.arange(n)
    return idx


def _param_stats(sol, mutation=None):
    """Physical estimates and the covariance, sd, CI and t of the active parameters."""
    theta = np.asarray(sol['estimations'], dtype=float).ravel()
    idx = _active_idx(sol, mutation)
    m = idx.size
    V = sol.get('V_matrix')
    if V is None:
        V = np.full((m, m), np.nan)
    else:
        V = np.asarray(V, dtype=float)
        if V.shape != (m, m):
            V = V[np.ix_(idx, idx)] if V.shape == (theta.size, theta.size) else np.full((m, m), np.nan)
    var = np.diag(V).copy()
    sd = np.where(np.isfinite(var) & (var > 0), np.sqrt(np.abs(var)), np.nan)
    ci = np.asarray(sol.get('CI', np.full(m, np.nan)), dtype=float).ravel()
    tv = np.asarray(sol.get('t_values', np.full(m, np.nan)), dtype=float).ravel()
    ci = ci if ci.size == m else np.full(m, np.nan)
    tv = tv if tv.size == m else np.full(m, np.nan)
    try:
        pd_ok = m > 0 and bool(np.all(np.isfinite(V))) and bool(np.all(np.linalg.eigvalsh((V + V.T) / 2) > 0))
    except np.linalg.LinAlgError:
        pd_ok = False
    X = _get(sol.get('optimization_result'), 'X')
    boot = None
    if X is not None:
        X = np.asarray(X, dtype=float)
        if X.ndim == 2 and X.shape[0] >= 3 and X.shape[1] == theta.size:
            boot = X[:, idx]
    return {'theta': theta, 'idx': idx, 'V': V, 'sd': sd, 'CI': ci, 't': tv, 'pd': pd_ok, 'boot': boot}


def _num(col):
    """Column of a data sheet as a float array ('none'/text -> NaN)."""
    return pd.to_numeric(pd.Series(list(col)), errors='coerce').to_numpy(dtype=float)


def _unc_of(system, grp, var):
    try:
        u = float(((system or {}).get(grp) or {}).get(var, {}).get('unc'))
    except (TypeError, ValueError, AttributeError):
        return np.nan
    return u if np.isfinite(u) and u > 0 else np.nan


def _measured(sheet, sol, sheet_name, system):
    """Measured outputs of one sheet: {var: {kind, t, y, sigma, yhat}}.

    sigma is the MES_E column (absolute), with the system 'unc' for missing columns or empty
    cells, exactly as used by parmest/uncert; NaN when neither is available. yhat is the
    model at the sampling times.
    """
    out = {}
    t_m = np.asarray((sol.get('t_m') or {}).get(sheet_name, []), dtype=float).ravel()
    for var, ymod in ((sol.get('tv_output_m') or {}).get(sheet_name) or {}).items():
        xcol, ycol, ecol = f'MES_X:{var}', f'MES_Y:{var}', f'MES_E:{var}'
        if xcol not in sheet or ycol not in sheet:
            continue
        t, y = _num(sheet[xcol]), _num(sheet[ycol])
        ok = np.isfinite(t) & np.isfinite(y)
        if not ok.any():
            continue
        unc = _unc_of(system, 'tvo', var)
        sig = _num(sheet[ecol])[ok] if ecol in sheet else np.full(ok.sum(), np.nan)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, unc)
        ymod = np.asarray(ymod, dtype=float).ravel()
        if t_m.size and t_m.size == ymod.size:
            yhat = np.interp(t[ok], t_m, ymod)
        else:
            yhat = np.full(ok.sum(), np.nan)
        out[var] = {'kind': 'tv', 't': t[ok], 'y': y[ok], 'sigma': sig, 'yhat': yhat}
    for var, ymod in ((sol.get('ti_output_m') or {}).get(sheet_name) or {}).items():
        ycol, ecol = f'MES_Y:{var}', f'MES_E:{var}'
        if ycol not in sheet:
            continue
        y = _num(sheet[ycol])[:1]
        if not (y.size and np.isfinite(y[0])):
            continue
        sig = _num(sheet[ecol])[:1] if ecol in sheet else np.array([np.nan])
        if not (np.isfinite(sig[0]) and sig[0] > 0):
            sig = np.array([_unc_of(system, 'tio', var)])
        out[var] = {'kind': 'ti', 't': None, 'y': y, 'sigma': sig,
                    'yhat': np.asarray(ymod, dtype=float).ravel()[:1]}
    return out


def _fit_stats(data, sol, system, n_active):
    """Sum of squared weighted residuals over all sheets, N, p and the 95 % range of chi2/(N-p)."""
    wls, n = 0.0, 0
    for sheet_name, sheet in data.items():
        for m in _measured(sheet, sol, sheet_name, system).values():
            r = (m['y'] - m['yhat']) / m['sigma']
            r = r[np.isfinite(r)]
            wls += float(np.sum(r ** 2))
            n += r.size
    dof = n - n_active
    if n == 0 or dof <= 0:
        return {'WLS': wls, 'N': n, 'p': n_active, 'ratio': np.nan, 'lo': np.nan, 'hi': np.nan}
    return {'WLS': wls, 'N': n, 'p': n_active, 'ratio': wls / dof,
            'lo': chi2.ppf(0.025, dof) / dof, 'hi': chi2.ppf(0.975, dof) / dof}


def _prediction_band(sol, sheet_name, var, mutation=None):
    """Half-width of the 95 % band of the model prediction from the parameter covariance."""
    S = ((sol.get('tv_sens_m') or {}).get(sheet_name) or {}).get(var)
    if S is None:
        return None
    st = _param_stats(sol, mutation)
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] != st['idx'].size or not np.all(np.isfinite(st['V'])):
        return None
    with np.errstate(invalid='ignore'):
        var_y = np.einsum('ij,jk,ik->i', S, st['V'], S)
    var_y = np.where(np.isfinite(var_y) & (var_y > 0), var_y, 0.0)
    return 1.96 * np.sqrt(var_y)


def _sheet_label(name):
    return f'experiment {name}'


def _conf_grid(entries, names, title, filename, show_boot=True, show=True, legend_extra=None):
    """Corner plot of 95 % confidence regions.

    Diagonal: normal density of each parameter with its 95 % CI (dashed) and, when available,
    the histogram of the bootstrap replicates. Lower triangle: 95 % joint confidence ellipses
    (radius sqrt(chi2_0.95,2) = 2.45 sd along the principal axes) with the bootstrap cloud.
    Upper triangle: correlation coefficient of the last entry.

    entries : list of dicts from _param_stats plus 'label', 'color', 'ls'.
    """
    params = sorted(set().union(*[set(e['idx'].tolist()) for e in entries])) if entries else []
    m = len(params)
    if m == 0:
        logger.warning(f"No estimated parameter to plot for '{title}'.")
        return False
    k95 = np.sqrt(chi2.ppf(0.95, 2))

    # axis range of each parameter: estimate +- 3.5 sd over all entries (and the bootstrap spread)
    lims = {}
    for p in params:
        lo, hi = [], []
        for e in entries:
            if p not in e['idx']:
                continue
            li = int(np.where(e['idx'] == p)[0][0])
            th, sd = e['theta'][p], e['sd'][li]
            if np.isfinite(sd):
                lo.append(th - 3.5 * sd)
                hi.append(th + 3.5 * sd)
            if show_boot and e.get('boot') is not None:
                q = np.nanpercentile(e['boot'][:, li], [0.5, 99.5])
                lo.append(q[0])
                hi.append(q[1])
            if not np.isfinite(sd) and e.get('boot') is None:
                d = 0.1 * abs(th) if th != 0 else 1.0
                lo.append(th - d)
                hi.append(th + d)
        lims[p] = (min(lo), max(hi)) if lo else (-1.0, 1.0)

    cell = min(2.4, max(1.6, 11.0 / m))
    fig, axs = plt.subplots(m, m, figsize=(cell * m + 0.8, cell * m + 1.0), squeeze=False)
    corr_e = entries[-1]
    any_boot = False

    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            ax = axs[i, j]
            if i == j:
                ax.set_yticks([])
                ax.grid(False)
                for e in entries:
                    if pi not in e['idx']:
                        continue
                    li = int(np.where(e['idx'] == pi)[0][0])
                    th, sd, ci = e['theta'][pi], e['sd'][li], e['CI'][li]
                    if show_boot and e.get('boot') is not None:
                        any_boot = True
                        b = e['boot'][:, li]
                        b = b[np.isfinite(b)]
                        if b.size:
                            ax.hist(b, bins=min(30, max(8, b.size // 5)), range=lims[pi], density=True,
                                    color=e['color'], alpha=0.22, edgecolor='none')
                    if np.isfinite(sd):
                        x = np.linspace(*lims[pi], 300)
                        ax.plot(x, norm.pdf(x, th, sd), color=e['color'], ls=e['ls'])
                    if np.isfinite(ci):
                        for c in (th - ci, th + ci):
                            ax.axvline(c, color=e['color'], ls='--', lw=1)
                    ax.axvline(th, color=e['color'], lw=0.8, alpha=0.6)
                ax.set_xlim(lims[pi])
            elif i > j:
                ax.set_xlim(lims[pj])
                ax.set_ylim(lims[pi])
                for e in entries:
                    if pi not in e['idx'] or pj not in e['idx']:
                        continue
                    li = int(np.where(e['idx'] == pi)[0][0])
                    lj = int(np.where(e['idx'] == pj)[0][0])
                    if show_boot and e.get('boot') is not None:
                        ax.scatter(e['boot'][:, lj], e['boot'][:, li], s=5, color=e['color'],
                                   alpha=0.35, linewidths=0, zorder=1)
                    C = e['V'][np.ix_([lj, li], [lj, li])]
                    if np.all(np.isfinite(C)):
                        vals, vecs = np.linalg.eigh((C + C.T) / 2)
                        if np.all(vals > 0):
                            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
                            ax.add_patch(Ellipse((e['theta'][pj], e['theta'][pi]),
                                                 width=2 * k95 * np.sqrt(vals[1]),
                                                 height=2 * k95 * np.sqrt(vals[0]), angle=ang,
                                                 fc='none', ec=e['color'], lw=1.6, ls=e['ls'], zorder=3))
                    ax.plot(e['theta'][pj], e['theta'][pi], 'o', color=e['color'], ms=3.5, zorder=4)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if pi in corr_e['idx'] and pj in corr_e['idx']:
                    li = int(np.where(corr_e['idx'] == pi)[0][0])
                    lj = int(np.where(corr_e['idx'] == pj)[0][0])
                    sdi, sdj = corr_e['sd'][li], corr_e['sd'][lj]
                    if np.isfinite(sdi) and np.isfinite(sdj):
                        r = float(np.clip(corr_e['V'][li, lj] / (sdi * sdj), -1, 1))
                        ax.add_patch(Rectangle((0.06, 0.06), 0.88, 0.88, transform=ax.transAxes,
                                               color=_CORR_CMAP((r + 1) / 2), alpha=0.55, lw=0))
                        ax.text(0.5, 0.5, f'{r:+.2f}', ha='center', va='center', transform=ax.transAxes,
                                fontsize=10 if m <= 5 else 8, color=_INK,
                                fontweight='bold' if abs(r) >= 0.9 else 'normal')
                        continue
                ax.axis('off')
            # labels only on the outer axes
            if i >= j:
                if i == m - 1:
                    ax.set_xlabel(names[pj])
                    ax.tick_params(axis='x', labelrotation=30)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0 and i > 0:
                    ax.set_ylabel(names[pi])
                elif i != j:
                    ax.tick_params(labelleft=False)
    if m == 1:
        axs[0, 0].set_ylabel('density')

    handles = [Line2D([], [], color=e['color'], ls=e['ls'], label=e['label']) for e in entries]
    handles.append(Line2D([], [], color=_INK2, ls='--', lw=1, label='95 % CI'))
    if m > 1:
        handles.append(Line2D([], [], color=_INK2, label='ellipse: 95 % joint region'))
    if any_boot:
        handles.append(Line2D([], [], color=_INK2, marker='o', ls='none', ms=3, alpha=0.5,
                              label='bootstrap replicates'))
    for h in (legend_extra or []):
        handles.append(h)
    H = fig.get_figheight()
    rows = int(np.ceil(len(handles) / 4))
    fig.suptitle(title, y=1 - 0.08 / H, va='top')
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, 1 - 0.38 / H), frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 1 - (0.3 + 0.2 * rows) / H])
    if m > 1:
        fig.text(0.99, 0.005, f'upper cells: correlation coefficients ({corr_e["label"]})',
                 ha='right', va='bottom', fontsize=7, color=_INK2)
    _finish(fig, filename, show)
    return True


# ---------------------------------------------------------------------------------------------
#                               Plots of one identification round
# ---------------------------------------------------------------------------------------------
class Plotting_Results:
    r"""
    Plots of one identification round, written by ``save_rounds``.

    * ``fit_plot``: per experiment, model against data (error bars +-2 sigma from MES_E),
      95 % prediction band from the parameter covariance, weighted residuals and the inputs.
    * ``parity_plot``: measured against predicted for every measured output.
    * ``conf_plot``: 95 % confidence regions of the estimated parameters.

    Parameters
    ----------
    models : dict
        Model settings (``mutation`` and, optionally, ``theta_names`` are used).
    pltshow : bool
        Display the figures (``plt.show``) in addition to saving them.
    round : int, optional
        Round number, used in the file names.
    """

    def __init__(self, models, pltshow, round=None):
        self.base_path = Path.cwd()
        self.round = str(round)
        self.models = models
        self.mutation_settings = models.get('mutation', {})
        self.modelling_folder = self.base_path / 'modelling'
        self.confidence_folder = self.base_path / 'confidence'
        self.modelling_folder.mkdir(parents=True, exist_ok=True)
        self.confidence_folder.mkdir(parents=True, exist_ok=True)
        self.pltshowing = bool(pltshow)

    @_styled
    def fit_plot(self, data, result, system):
        """
        Model against data for every experiment (sheet) of the round.

        One panel per time-varying output (data with +-2 sigma error bars, model curve per
        model, shaded 95 % band of the prediction due to the parameter uncertainty), under each
        measured output its weighted residuals (y - y_model)/sigma with the +-2 lines, and one
        panel per time-varying input with the sampling times marked. The title gives
        sum((y - y_model)/sigma)^2 / (N - p) over all experiments with its 95 % range for a
        correct model and correct sigma.

        Parameters
        ----------
        data : dict
            Experimental data (sheet name -> DataFrame), as read by ``read_excel``.
        result : dict
            ``uncert`` results per model.
        system : dict
            System settings (inputs, outputs and their ``unc``).
        """
        solvers = list(result)
        if not solvers:
            return
        stats = {}
        for sv in solvers:
            n_act = _param_stats(result[sv], self.mutation_settings.get(sv))['idx'].size
            stats[sv] = _fit_stats(data, result[sv], system, n_act)
        stat_txt = '\n'.join(
            (f'{sv}: ' if len(solvers) > 1 else '')
            + (rf'$\Sigma[(y-\hat y)/\sigma]^2/(N-p)$ = {s["ratio"]:.2f}'
               f'  (95 % range if model and σ are right: {s["lo"]:.2f}–{s["hi"]:.2f}; N={s["N"]}, p={s["p"]})'
               if np.isfinite(s['ratio']) else f'N={s["N"]}, p={s["p"]}')
            for sv, s in stats.items())

        for sheet_name, sheet in data.items():
            first = result[solvers[0]]
            if sheet_name not in (first.get('tv_output_m') or {}):
                continue
            out_vars = list(first['tv_output_m'][sheet_name].keys())
            meas = {sv: _measured(sheet, result[sv], sheet_name, system) for sv in solvers}
            in_vars = [v for v in (system.get('tvi') or {}) if v in sheet]
            ti_meas = {v: m for v, m in meas[solvers[0]].items() if m['kind'] == 'ti'}
            if not out_vars and not in_vars:
                continue

            ratios, kinds = [], []
            for v in out_vars:
                ratios.append(3.0)
                kinds.append(('out', v))
                if v in meas[solvers[0]]:
                    ratios.append(1.1)
                    kinds.append(('res', v))
            for v in in_vars:
                ratios.append(1.3)
                kinds.append(('in', v))
            rug_vars = [v for v, m in meas[solvers[0]].items() if m['kind'] == 'tv']
            if rug_vars:
                ratios.append(0.25 + 0.16 * len(rug_vars))
                kinds.append(('rug', None))
            fig, axes = plt.subplots(len(ratios), 1, sharex=True, squeeze=False,
                                     figsize=(7.5, 1.4 + 0.55 * sum(ratios)),
                                     gridspec_kw={'height_ratios': ratios})
            axes = axes[:, 0]
            t_x = _num(sheet['X:all']) if 'X:all' in sheet else None

            for ax, (kind, var) in zip(axes, kinds):
                if kind == 'out':
                    c = _color(out_vars.index(var))
                    for js, sv in enumerate(solvers):
                        sol = result[sv]
                        tm = np.asarray(sol['t_m'][sheet_name], dtype=float).ravel()
                        ym = np.asarray(sol['tv_output_m'][sheet_name][var], dtype=float).ravel()
                        if tm.size != ym.size:
                            continue
                        ax.plot(tm, ym, color=c, ls=_LINESTYLES[js % 4], zorder=2,
                                label=f'model {sv}' if len(solvers) > 1 else 'model')
                        band = _prediction_band(sol, sheet_name, var, self.mutation_settings.get(sv))
                        if band is not None and band.size == ym.size and np.any(band > 0):
                            ax.fill_between(tm, ym - band, ym + band, color=c, alpha=0.18, lw=0, zorder=1,
                                            label='95 % band (parameter uncertainty)' if js == 0 else None)
                    mv = meas[solvers[0]].get(var)
                    if mv is not None:
                        err = np.where(np.isfinite(mv['sigma']), 2 * mv['sigma'], 0.0)
                        ax.errorbar(mv['t'], mv['y'], yerr=err if np.any(err > 0) else None, fmt='o',
                                    ms=4.5, mfc='white', mec=c, mew=1.2, ecolor=c, elinewidth=1, capsize=2,
                                    zorder=3, label='data ± 2σ' if np.any(err > 0) else 'data')
                    ax.set_ylabel(var)
                    ax.legend(loc='best')
                elif kind == 'res':
                    c = _color(out_vars.index(var))
                    for js, sv in enumerate(solvers):
                        mv = meas[sv].get(var)
                        if mv is None:
                            continue
                        r = (mv['y'] - mv['yhat']) / mv['sigma']
                        ax.plot(mv['t'], r, ls='none', marker=_MARKERS[js % len(_MARKERS)], ms=5 if js == 0 else 3.5,
                                color=c, mfc=c if js == 0 else 'white',
                                label=sv if len(solvers) > 1 else None)
                    ax.axhline(0, color=_INK2, lw=0.8)
                    for s in (-2, 2):
                        ax.axhline(s, color=_INK2, lw=0.8, ls='--')
                    rr = np.concatenate([(meas[sv][var]['y'] - meas[sv][var]['yhat']) / meas[sv][var]['sigma']
                                         for sv in solvers if var in meas[sv]])
                    rr = rr[np.isfinite(rr)]
                    lim = max(3.0, 1.15 * float(np.max(np.abs(rr)))) if rr.size else 3.0
                    ax.set_ylim(-lim, lim)
                    ax.set_ylabel(r'$(y-\hat y)/\sigma$')
                    if len(solvers) > 1:
                        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), fontsize=7)
                elif kind == 'in':
                    c = _color(len(out_vars) + in_vars.index(var))
                    u = _num(sheet[var])
                    if t_x is not None and t_x.size == u.size:
                        ok = np.isfinite(t_x) & np.isfinite(u)
                        ax.plot(t_x[ok], u[ok], color=c)
                    ax.set_ylabel(f'{var}\n(input)')
                    ax.margins(y=0.15)
                else:   # sampling times of each measured output
                    for k, v in enumerate(rug_vars):
                        tv = meas[solvers[0]][v]['t']
                        c = _color(out_vars.index(v)) if v in out_vars else _INK2
                        ax.plot(tv, np.full(tv.size, k), ls='none', marker='|', ms=8, mew=1.4, color=c)
                    ax.set_yticks(range(len(rug_vars)))
                    ax.set_yticklabels(rug_vars, fontsize=7)
                    ax.set_ylim(-0.6, len(rug_vars) - 0.4)
                    ax.invert_yaxis()
                    ax.grid(axis='y', visible=False)
                    ax.set_ylabel('sampled', fontsize=8)
            axes[-1].set_xlabel('time')

            title = f'{_sheet_label(sheet_name)}, round {self.round}'
            if ti_meas:
                title += '\n' + ', '.join(
                    f'{v}: measured {m["y"][0]:.4g}'
                    + (f' ± {2 * m["sigma"][0]:.2g}' if np.isfinite(m['sigma'][0]) else '')
                    + f', model {m["yhat"][0]:.4g}' for v, m in ti_meas.items())
            fig.suptitle(title)
            fig.text(0.5, 0.003, stat_txt, ha='center', va='bottom', fontsize=7.5, color=_INK2)
            fig.align_ylabels(axes)
            fig.tight_layout(rect=[0, 0.012 + 0.018 * len(solvers), 1, 1])
            _finish(fig, self.modelling_folder / f"{self.round}_round_{sheet_name}.png", self.pltshowing)

    @_styled
    def parity_plot(self, data, result, system):
        """
        Measured against predicted values, one panel per measured output and one figure per
        model, with +-2 sigma error bars and the identity line. Saved as
        ``modelling/<round>_parity_<model>.png``.
        """
        for sv, sol in result.items():
            per_var = {}
            for sheet_name, sheet in data.items():
                for var, mv in _measured(sheet, sol, sheet_name, system).items():
                    d = per_var.setdefault(var, {'y': [], 'yhat': [], 'sigma': []})
                    d['y'].append(mv['y'])
                    d['yhat'].append(mv['yhat'])
                    d['sigma'].append(mv['sigma'])
            if not per_var:
                continue
            nv = len(per_var)
            nc = min(3, nv)
            nr = int(np.ceil(nv / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.1 * nr + 0.5), squeeze=False)
            for k, (var, d) in enumerate(per_var.items()):
                ax = axs[k // nc, k % nc]
                y, yh, s = (np.concatenate(d[key]) for key in ('y', 'yhat', 'sigma'))
                c = _color(k)
                err = np.where(np.isfinite(s), 2 * s, 0.0)
                ax.errorbar(yh, y, yerr=err if np.any(err > 0) else None, fmt='o', ms=4, mfc='white',
                            mec=c, ecolor=c, elinewidth=0.9, capsize=1.5)
                vals = np.concatenate([y, yh, y - err, y + err])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    lo, hi = vals.min(), vals.max()
                    pad = 0.05 * (hi - lo if hi > lo else (abs(hi) or 1.0))
                    lo, hi = lo - pad, hi + pad
                    ax.plot([lo, hi], [lo, hi], color=_INK2, lw=1)
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
                ok = np.isfinite(y) & np.isfinite(yh)
                if ok.sum() > 1 and np.var(y[ok]) > 0:
                    r2 = 1 - np.sum((y[ok] - yh[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
                    ax.set_title(f'{var}   (R² = {r2:.3f}, n = {ok.sum()})')
                else:
                    ax.set_title(f'{var}   (n = {ok.sum()})')
                ax.set_xlabel('model')
                ax.set_ylabel('measured (± 2σ)')
                ax.set_aspect('equal', adjustable='box')
            for k in range(nv, nr * nc):
                axs[k // nc, k % nc].axis('off')
            fig.suptitle(f'Measured vs model, {sv}, round {self.round}')
            fig.tight_layout()
            _finish(fig, self.modelling_folder / f"{self.round}_parity_{sv}.png", self.pltshowing)

    @_styled
    def conf_plot(self, parameters, cov_matrices, confidence_intervals, result=None):
        """
        95 % confidence regions of the estimated parameters of each model (corner plot):
        densities with the 95 % CI on the diagonal, 95 % joint ellipses (and the bootstrap
        replicates for var-cov='B') below, correlation coefficients above. Saved as
        ``confidence/<round>_th_round_<model>.png``.

        Parameters
        ----------
        parameters : dict
            Estimated (physical) parameters per model.
        cov_matrices : dict
            Covariance matrices per model (active block or full size).
        confidence_intervals : dict
            95 % CI half-widths of the active parameters per model.
        result : dict, optional
            ``uncert`` results per model; used for the active set and the bootstrap replicates.
        """
        for solver, theta in parameters.items():
            sol = dict((result or {}).get(solver) or {})
            sol['estimations'] = theta
            sol['V_matrix'] = cov_matrices[solver]
            ci = confidence_intervals[solver]
            sol['CI'] = list(ci.values()) if isinstance(ci, dict) else ci
            if 'optimization_result' not in sol:
                sol['optimization_result'] = {'activeparams': self.mutation_settings.get(solver)}
            st = _param_stats(sol, self.mutation_settings.get(solver))
            st.update(label=f'round {self.round}', color=_color(0), ls='-')
            names = _param_names(self.models, solver, st['theta'].size)
            _conf_grid([st], names, f'Confidence regions, {solver}, round {self.round}',
                       self.confidence_folder / f"{self.round}_th_round_{solver}.png",
                       show_boot=True, show=self.pltshowing)


# ---------------------------------------------------------------------------------------------
#                              Plots and reports across rounds
# ---------------------------------------------------------------------------------------------
class Plotting_FinalResults:
    r"""
    Plots and reports across identification rounds (``run_postprocessing``).

    Parameters
    ----------
    round_data : dict
        Rounds stored by ``save_rounds`` ('Round 1', 'Round 2', ...).
    winner_solver : str or None
        Model to plot (None for the plots comparing all models).
    selected_rounds : list of int
        Rounds to include; empty means all.
    true_theta : list or dict, optional
        True parameter values (in silico studies), drawn on the parameter-estimate plot;
        a list, or a dict {model: list}.
    """

    def __init__(self, round_data, winner_solver, selected_rounds, true_theta=None):
        self.round_data = round_data
        self.winner_solver = winner_solver
        self.selected_rounds = list(selected_rounds or [])
        if isinstance(true_theta, dict):
            true_theta = true_theta.get(winner_solver)
        self.true_theta = None if true_theta is None else np.asarray(true_theta, dtype=float).ravel()

    # ------------------------------------------------------------------ utilities
    def _get_round_labels(self):
        """Sorted labels 'Round 1', 'Round 2', ... present in round_data."""
        labels = [lbl for lbl in self.round_data if str(lbl).startswith('Round ')]
        labels.sort(key=lambda x: int(x.replace('Round ', '')))
        return labels

    def _get_round_number(self, round_label):
        """'Round 3' -> 3."""
        return int(round_label.split(' ')[-1])

    def _rounds_to_process(self, exclude_broad_cov=False):
        """
        (round number, label) of the selected rounds, sorted. With ``exclude_broad_cov`` the
        rounds whose covariance (of the estimated parameters) is not positive definite are left out.
        """
        out = []
        for lbl in self._get_round_labels():
            rnd = self._get_round_number(lbl)
            if self.selected_rounds and rnd not in self.selected_rounds:
                continue
            if exclude_broad_cov and self.winner_solver in self.round_data[lbl].get('result', {}):
                sol = self.round_data[lbl]['result'][self.winner_solver]
                mut = (self.round_data[lbl].get('mutation') or {}).get(self.winner_solver)
                if 'V_matrix' in sol and not _param_stats(sol, mut)['pd']:
                    continue
            out.append((rnd, lbl))
        missing = [r for r in self.selected_rounds if f'Round {r}' not in self.round_data]
        for r in missing:
            logger.warning(f"Selected round {r} is missing from round_data.")
        return out

    def _entries(self):
        """Per selected round: parameter statistics of the winner model plus round info."""
        entries = []
        for rnd, lbl in self._rounds_to_process():
            rd = self.round_data[lbl]
            sol = (rd.get('result') or {}).get(self.winner_solver)
            if not sol or 'estimations' not in sol:
                continue
            st = _param_stats(sol, (rd.get('mutation') or {}).get(self.winner_solver))
            st.update(round=rnd, label=f'round {rnd}', color=_color(rnd - 1),
                      ls=_LINESTYLES[(rnd - 1) // len(_PALETTE) % 4],
                      trv=(rd.get('trv') or {}).get(self.winner_solver, np.nan),
                      P=sol.get('P', np.nan), design=_design_kind(rd.get('design_type')))
            entries.append(st)
        return entries

    def _names(self, n):
        models = None
        for lbl in reversed(self._get_round_labels()):
            models = self.round_data[lbl].get('models')
            if models:
                break
        return _param_names(models, self.winner_solver, n)

    def _n_models(self):
        return max([len(rd.get('result') or {}) for rd in self.round_data.values()
                    if isinstance(rd, dict)] or [0])

    def _shade_designs(self, ax, rounds, legend=True):
        """Shade each round by its design type; legend only for the types present."""
        present = []
        for rnd in rounds:
            kind = _design_kind(self.round_data.get(f'Round {rnd}', {}).get('design_type'))
            if kind is None:
                continue
            ax.axvspan(rnd - 0.5, rnd + 0.5, color=_DESIGN_KINDS[kind][1], alpha=0.10, lw=0, zorder=0)
            if kind not in present:
                present.append(kind)
        return [Patch(color=_DESIGN_KINDS[k][1], alpha=0.25, label=_DESIGN_KINDS[k][0]) for k in present]

    # ------------------------------------------------------------------ confidence
    @_styled
    def conf_plot(self, filename,
                  ellipsoid_volume_filename='ellipsoid_volume_across_rounds.png',
                  std_devs_filename='parameter_std_devs_across_rounds.png',
                  parameter_estimates_filename='parameter_estimates_across_rounds.png'):
        """
        95 % confidence regions of the winner model for the selected rounds on one corner plot
        (densities and 95 % CI on the diagonal, 95 % joint ellipses below, correlation
        coefficients of the last round above); then ``conf_plot_metrics`` and ``accvsprec_plot``.

        Parameters
        ----------
        filename : str
            File of the confidence-region figure.
        ellipsoid_volume_filename, std_devs_filename, parameter_estimates_filename : str, optional
            Files of the auxiliary figures.
        """
        entries = self._entries()
        if not entries:
            logger.warning("No valid rounds to plot. Aborting conf_plot.")
            return
        good = [e for e in entries if e['pd']]
        for e in entries:
            if not e['pd']:
                logger.info(f"conf_plot: {e['label']} left out (covariance not positive definite).")
        if good:
            _conf_grid(good, self._names(good[0]['theta'].size),
                       f'Confidence regions across rounds, {self.winner_solver}', filename,
                       show_boot=False, show=_SHOW_PLOTS)
        self.conf_plot_metrics(ellipsoid_volume_filename, std_devs_filename)
        self.accvsprec_plot(parameter_estimates_filename)

    @_styled
    def conf_plot_metrics(self, ellipsoid_volume_filename, std_devs_filename):
        """
        Volume of the 95 % confidence ellipsoid and relative standard deviations (% of the
        estimate) across rounds. Volumes are only comparable between rounds with the same
        number of estimated parameters; a change is marked on the plot.

        Parameters
        ----------
        ellipsoid_volume_filename : str
            File of the volume figure.
        std_devs_filename : str
            File of the relative standard-deviation figure.
        """
        entries = [e for e in self._entries() if e['pd']]
        if not entries:
            logger.info("No round with a positive-definite covariance for conf_plot_metrics.")
            return
        rounds = np.array([e['round'] for e in entries])
        n_par = np.array([e['idx'].size for e in entries])
        vols = []
        for e in entries:
            p = e['idx'].size
            vols.append((np.pi ** (p / 2) / gamma(p / 2 + 1)) * chi2.ppf(0.95, p) ** (p / 2)
                        * np.sqrt(np.linalg.det(e['V'])))
        vols = np.array(vols)

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for k in range(len(rounds)):
            if k > 0:
                same = n_par[k] == n_par[k - 1]
                ax.plot(rounds[k - 1:k + 1], vols[k - 1:k + 1], color=_PALETTE[0],
                        ls='-' if same else ':', lw=1.8 if same else 1.0)
                if not same:
                    ax.axvline((rounds[k] + rounds[k - 1]) / 2, color=_INK2, ls=':', lw=0.8)
            ax.plot(rounds[k], vols[k], 'o', color=_PALETTE[0], ms=6)
            ax.annotate(f'p = {n_par[k]}', (rounds[k], vols[k]), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=7.5, color=_INK2)
        ax.set_yscale('log')
        ax.set_xticks(rounds)
        ax.set_xlabel('round')
        ax.set_ylabel('volume of the 95 % ellipsoid')
        ax.set_title(f'Joint confidence region, {self.winner_solver}')
        if len(set(n_par.tolist())) > 1:
            fig.text(0.5, 0.0, 'dotted: the number of estimated parameters p changes; '
                               'volumes in different dimensions are not comparable',
                     ha='center', va='bottom', fontsize=7, color=_INK2)
        ax.margins(y=0.12)
        fig.tight_layout(rect=[0, 0.04 if len(set(n_par.tolist())) > 1 else 0, 1, 1])
        _finish(fig, ellipsoid_volume_filename, _SHOW_PLOTS)

        n = entries[0]['theta'].size
        names = self._names(n)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for p in range(n):
            rel = []
            for e in entries:
                if p in e['idx']:
                    li = int(np.where(e['idx'] == p)[0][0])
                    th = e['theta'][p]
                    rel.append(100 * e['sd'][li] / abs(th) if th != 0 else np.nan)
                else:
                    rel.append(np.nan)
            rel = np.array(rel)
            if not np.any(np.isfinite(rel)):
                continue
            ax.plot(rounds, rel, marker=_MARKERS[p % len(_MARKERS)], color=_color(p),
                    ls=_LINESTYLES[p // len(_PALETTE) % 4], label=names[p])
        _log_axis(ax)
        ax.set_xticks(rounds)
        ax.set_xlabel('round')
        ax.set_ylabel('standard deviation (% of estimate)')
        ax.set_title(f'Relative parameter precision, {self.winner_solver} (gaps: parameter fixed)')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        _finish(fig, std_devs_filename, _SHOW_PLOTS)

    @_styled
    def accvsprec_plot(self, filename):
        """
        Physical estimate +- 95 % CI of every parameter per round (one panel per parameter),
        with the true values when ``true_theta`` is given. Parameters fixed in a round are
        drawn as open markers without an interval.

        Parameters
        ----------
        filename : str
            Output file.
        """
        entries = self._entries()
        if not entries:
            logger.info("No valid rounds for accvsprec_plot.")
            return
        n = entries[0]['theta'].size
        names = self._names(n)
        nc = min(3, n)
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.6 * nr + 0.6), squeeze=False)
        rounds = [e['round'] for e in entries]
        for p in range(n):
            ax = axs[p // nc, p % nc]
            c = _color(p)
            for e in entries:
                th = e['theta'][p] if p < e['theta'].size else np.nan
                if p in e['idx']:
                    li = int(np.where(e['idx'] == p)[0][0])
                    ci = e['CI'][li]
                    ax.errorbar(e['round'], th, yerr=ci if np.isfinite(ci) else None, fmt='o', color=c,
                                ms=5, capsize=3, elinewidth=1.2)
                else:
                    ax.plot(e['round'], th, 'o', mfc='white', mec=c, ms=5)
            ax.plot(rounds, [e['theta'][p] for e in entries], color=c, lw=0.8, alpha=0.6)
            if self.true_theta is not None and p < self.true_theta.size:
                ax.axhline(self.true_theta[p], color=_INK2, ls='--', lw=1)
            ax.set_xticks(rounds)
            ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
            ax.set_title(names[p])
            if p // nc == nr - 1 or p + nc >= n:
                ax.set_xlabel('round')
        for k in range(n, nr * nc):
            axs[k // nc, k % nc].axis('off')
        handles = [Line2D([], [], color=_INK2, marker='o', ls='none', label='estimate ± 95 % CI'),
                   Line2D([], [], color=_INK2, marker='o', mfc='white', ls='none', label='fixed (not estimated)')]
        if self.true_theta is not None:
            handles.append(Line2D([], [], color=_INK2, ls='--', label='true value'))
        fig.suptitle(f'Parameter estimates per round, {self.winner_solver}')
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        _finish(fig, filename, _SHOW_PLOTS)

    # ------------------------------------------------------------------ tests
    @_styled
    def pt_plot(self, filename, roundc):
        """
        Model probability P (only when several models are compared) and the absolute t-value of every parameter
        against the reference t per round, with the rounds shaded by design type.

        Parameters
        ----------
        filename : str
            Output file.
        roundc : dict
            Not used; kept for compatibility. The shading by design type ('preliminary',
            'classic', 'MBDOE_MD', 'MBDoE-PP ...', 'DOE', ...) is taken from the
            ``design_type`` of each round in ``self.round_data``.
        """
        entries = self._entries()
        if not entries:
            logger.info("No valid rounds for pt_plot.")
            return
        rounds = [e['round'] for e in entries]
        show_p = self._n_models() > 1
        n = entries[0]['theta'].size
        names = self._names(n)
        fig, axs = plt.subplots(1, 2 if show_p else 1, figsize=(11, 4.2) if show_p else (6.8, 4.2),
                                squeeze=False)
        axs = axs[0]
        if show_p:
            ax = axs[0]
            ax.plot(rounds, [e['P'] for e in entries], 'o-', color=_PALETTE[0])
            ax.set_ylim(-3, 103)
            ax.set_ylabel('P (%)')
            ax.set_title(f'Model probability, {self.winner_solver}')
        ax = axs[-1]
        for p in range(n):
            tv = []
            for e in entries:
                if p in e['idx']:
                    tv.append(abs(e['t'][int(np.where(e['idx'] == p)[0][0])]))
                else:
                    tv.append(np.nan)
            tv = np.array(tv)
            if not np.any(np.isfinite(tv)):
                continue
            trv = np.array([e['trv'] for e in entries], dtype=float)
            c = _color(p)
            ax.plot(rounds, tv, color=c, ls=_LINESTYLES[p // len(_PALETTE) % 4], lw=1.2, label=names[p])
            ok = tv > trv
            ax.plot(np.array(rounds)[ok], tv[ok], 'o', color=c, ms=5)
            ax.plot(np.array(rounds)[~ok & np.isfinite(tv)], tv[~ok & np.isfinite(tv)], 'o', mfc='white',
                    mec=c, ms=5)
        ax.plot(rounds, [e['trv'] for e in entries], color=_INK, ls='--', lw=1.2, label='reference t')
        _log_axis(ax)
        ax.set_ylabel('|t|')
        ax.set_title(f't-test, {self.winner_solver} (open marker: |t| ≤ reference t)')
        for a in axs:
            patches = self._shade_designs(a, rounds)
            a.set_xticks(rounds)
            a.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
            a.set_xlabel('round')
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h + patches, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        _finish(fig, filename, _SHOW_PLOTS)

    @_styled
    def pcomp_plot(self, save_path):
        """
        Model probabilities P per round (one figure per round). Skipped when only one model
        is identified (P is then always 100 %).
        """
        if self._n_models() <= 1:
            print("P-value plots skipped: only one model, so its probability is always 100 %.")
            return
        for rnd, lbl in self._rounds_to_process():
            res = self.round_data[lbl].get('result') or {}
            names = [m for m in res if res[m].get('P') is not None]
            if not names:
                logger.warning(f"No P values found for {lbl}. Skipping plot.")
                continue
            vals = [float(res[m]['P']) for m in names]
            fig, ax = plt.subplots(figsize=(max(3.5, 0.9 * len(names) + 1.5), 3.6))
            bars = ax.bar(names, vals, color=[_color(k) for k in range(len(names))], width=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f} %', ha='center', va='bottom',
                        fontsize=8)
            ax.set_ylim(0, 110)
            ax.set_ylabel('P (%)')
            ax.set_title(f'Model probabilities, round {rnd}')
            ax.grid(axis='x', visible=False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            fig.tight_layout()
            _finish(fig, os.path.join(save_path, f'p_values_{lbl.replace(" ", "_").lower()}.png'), _SHOW_PLOTS)

    @_styled
    def tcomp_plot(self, save_path):
        """
        Absolute t-value of every estimated parameter of every model, per round, against each model's own
        reference t (filled bar: above it).
        """
        for rnd, lbl in self._rounds_to_process():
            rd = self.round_data[lbl]
            res = rd.get('result') or {}
            trv = rd.get('trv') or {}
            models_stored = rd.get('models')
            groups = []
            for m in sorted(res):
                sol = res[m]
                if 'estimations' not in sol or len(np.ravel(sol.get('t_values', []))) == 0:
                    continue
                st = _param_stats(sol, (rd.get('mutation') or {}).get(m))
                names = _param_names(models_stored, m, st['theta'].size)
                groups.append((m, [names[p] for p in st['idx']], np.abs(st['t']), trv.get(m, np.nan)))
            if not groups:
                logger.warning(f"No t values found for {lbl}. Skipping plot.")
                continue
            width = max(4.0, 0.42 * sum(len(g[1]) for g in groups) + 0.8 * len(groups) + 1.5)
            fig, ax = plt.subplots(figsize=(min(width, 16), 4.0))
            x0, ticks, tick_labels, mids = 0, [], [], []
            for k, (m, pn, tv, tref) in enumerate(groups):
                xs = np.arange(x0, x0 + len(pn))
                passed = tv > tref
                ax.bar(xs, np.where(np.isfinite(tv), tv, 0), width=0.7,
                       color=[_PALETTE[0] if ok else '#c9c8c2' for ok in passed])
                if np.isfinite(tref):
                    ax.hlines(tref, xs[0] - 0.45, xs[-1] + 0.45, color=_INK, ls='--', lw=1.2)
                    ax.text(xs[-1] + 0.5, tref, f'{tref:.2f}', va='center', fontsize=7, color=_INK2)
                ticks += list(xs)
                tick_labels += pn
                mids.append((xs[0] + xs[-1]) / 2)
                x0 += len(pn) + 1.5
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, fontsize=8)
            for xm, (m, *_r) in zip(mids, groups):
                ax.text(xm, -0.16, m, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9)
            vals = np.concatenate([g[2] for g in groups])
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size and vals.max() / max(vals.min(), 1e-300) > 50:
                _log_axis(ax)
            ax.set_ylabel('|t|')
            ax.grid(axis='x', visible=False)
            ax.set_title(f't-values, round {rnd}')
            ax.legend(handles=[Patch(color=_PALETTE[0], label='|t| > reference t'),
                               Patch(color='#c9c8c2', label='|t| ≤ reference t'),
                               Line2D([], [], color=_INK, ls='--', label='reference t (per model)')],
                      loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
            fig.tight_layout()
            _finish(fig, os.path.join(save_path, f't_values_{lbl.replace(" ", "_").lower()}.png'), _SHOW_PLOTS)

    # ------------------------------------------------------------------ report
    def reporter(self, filename):
        """
        Excel report per round and a text summary of the last round.

        ``<filename>_<round>.xlsx``: one sheet per experiment with, per measured output, the
        sampling times, measurements, the sigma used (MES_E or 'unc'), the model at the sampling
        times, the residual and the weighted residual; then the model curves and inputs on the
        model time grid. Sheet ``_summary``: estimates, sd, 95 % CI, t against the reference t,
        and the fit statistics (N, p, sum of squared weighted residuals and its ratio to N - p
        with the 95 % range).

        ``<filename>_summary.txt``: the same summary for every model of the last selected round.

        Parameters
        ----------
        filename : str
            Base file name.
        """
        if not self.round_data:
            logger.warning("No round data found. Aborting reporter.")
            return
        used = [r for r, _ in self._rounds_to_process()]
        if not used:
            return
        for rnd in used:
            lbl = f'Round {rnd}'
            rd = self.round_data[lbl]
            sol = (rd.get('result') or {}).get(self.winner_solver)
            if not sol or 'data' not in sol or 'tv_output_m' not in sol:
                logger.info(f"Skipping {lbl}: no data/model output for {self.winner_solver}.")
                continue
            system = rd.get('system') or {}
            sheets = {}
            for exp_name, sheet in sol['data'].items():
                if exp_name not in sol['tv_output_m']:
                    continue
                cols = {}
                for var, mv in _measured(sheet, sol, exp_name, system).items():
                    if mv['kind'] == 'tv':
                        cols[f'MES_X:{var}'] = mv['t']
                    cols[f'MES_Y:{var}'] = mv['y']
                    cols[f'MES_E:{var} (sigma used)'] = mv['sigma']
                    cols[f'MOD:{var}'] = mv['yhat']
                    cols[f'RES:{var}'] = mv['y'] - mv['yhat']
                    cols[f'WRES:{var}'] = (mv['y'] - mv['yhat']) / mv['sigma']
                t_m = np.asarray((sol.get('t_m') or {}).get(exp_name, []), dtype=float).ravel()
                if t_m.size:
                    cols['SIM_t_model'] = t_m
                    for var, vals in sol['tv_output_m'][exp_name].items():
                        cols[f'SIM_{var}'] = np.asarray(vals, dtype=float).ravel()
                    for iname, ivals in ((sol.get('tv_input_m') or {}).get(exp_name) or {}).items():
                        iv = np.asarray(ivals, dtype=float).ravel()
                        if iv.size == t_m.size:
                            cols[f'INP_{iname}'] = iv
                        elif iv.size > 1:
                            t_in = np.linspace(t_m[0], t_m[-1], iv.size)
                            cols[f'INP_{iname}'] = interp1d(t_in, iv, bounds_error=False,
                                                            fill_value='extrapolate')(t_m)
                sheets[str(exp_name)[:31]] = pd.DataFrame({k: pd.Series(v) for k, v in cols.items()})
            sheets['_summary'] = self._summary_frame(lbl, self.winner_solver)
            with pd.ExcelWriter(f"{filename}_{rnd}.xlsx", engine='xlsxwriter') as writer:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=name, index=False)
            logger.info(f"Reporter wrote {lbl} to '{filename}_{rnd}.xlsx'.")

        last = f'Round {used[-1]}'
        with open(f"{filename}_summary.txt", "w", encoding="utf-8") as fh:
            rd = self.round_data[last]
            fh.write(f"{last}  (design: {rd.get('design_type')})\n\n")
            for model in (rd.get('result') or {}):
                df = self._summary_frame(last, model)
                fh.write(f"Model: {model}\n")
                fh.write(df.to_string(index=False, na_rep=''))
                fh.write("\n")
                sol = rd['result'][model]
                fh.write(f"Overall R²: {sol.get('R2_total', 'N/A')}\n")
                for resp, r2 in (sol.get('R2_responses') or {}).items():
                    fh.write(f"  R² {resp}: {r2}\n")
                fh.write(f"Chi (relative LS): {sol.get('Chi', 'N/A')}\n")
                fh.write(f"P (%): {sol.get('P', 'N/A')}\n")
                ranking = (rd.get('ranking') or {}).get(model) if isinstance(rd.get('ranking'), dict) else None
                if ranking is not None:
                    fh.write(f"Estimability ranking (most to least): {list(ranking)}\n")
                if rd.get('k_optimal_value') is not None:
                    fh.write(f"Optimal number of estimated parameters: {rd['k_optimal_value']}\n")
                fh.write("\n")
        logger.info(f"Summary written to '{filename}_summary.txt'.")

    def _summary_frame(self, lbl, model):
        """Parameter table and fit statistics of one model in one round."""
        rd = self.round_data[lbl]
        sol = rd['result'][model]
        st = _param_stats(sol, (rd.get('mutation') or {}).get(model))
        names = _param_names(rd.get('models'), model, st['theta'].size)
        names = [re.sub(r'[$\\{}]', '', s).replace('theta_', 'theta') for s in names]
        trv = (rd.get('trv') or {}).get(model, np.nan)
        rows = []
        for p, th in enumerate(st['theta']):
            row = {'parameter': names[p], 'estimate': th, 'estimated': p in st['idx']}
            if p in st['idx']:
                li = int(np.where(st['idx'] == p)[0][0])
                sd, ci, tv = st['sd'][li], st['CI'][li], st['t'][li]
                row.update({'sd': sd, 'CI95 (±)': ci,
                            'CI95 (% of estimate)': 100 * ci / abs(th) if th != 0 else np.nan,
                            't': tv, 't_ref': trv,
                            'significant': bool(abs(tv) > trv) if np.isfinite(tv) and np.isfinite(trv) else None})
            rows.append(row)
        df = pd.DataFrame(rows)
        if 'data' in sol:
            fs = _fit_stats(sol['data'], sol, rd.get('system') or {}, st['idx'].size)
            extra = pd.DataFrame([
                {'parameter': 'N (measurements)', 'estimate': fs['N']},
                {'parameter': 'p (estimated parameters)', 'estimate': fs['p']},
                {'parameter': 'sum((y - y_model)/sigma)^2', 'estimate': fs['WLS']},
                {'parameter': 'ratio to N - p (expected ~1)', 'estimate': fs['ratio']},
                {'parameter': '95 % range of the ratio', 'estimate': f"{fs['lo']:.3f} - {fs['hi']:.3f}"},
            ])
            df = pd.concat([df, pd.DataFrame([{}]), extra], ignore_index=True)
        return df


def _symlog_rcc(ax, values):
    """Symmetric-log y axis suited to rCC values (negative, zero and large positive)."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v) & (v > 0)]
    lin = max(float(np.min(v)) * 0.5, 1e-6) if v.size else 1e-3
    if v.size and v.max() / lin > 100:
        ax.set_yscale('symlog', linthresh=lin)
    ax.axhline(0, color=_INK2, lw=0.8)


@_styled
def Plot_estimability(round_data, path, solver):
    r"""
    Corrected critical ratio rCC against the number of estimated parameters k, one line per
    round, with the selected k (minimum rCC) of each round marked. Saved as
    ``<path>/estimability/rCC vs parameters in model_<solver>.png``.

    Parameters
    ----------
    round_data : dict
        Rounds stored by ``save_rounds``.
    path : str
        Base folder.
    solver : str
        Model name.
    """
    curves = []
    for lbl, info in round_data.items():
        rcc = (info.get('rCC_values') or {}) if isinstance(info, dict) else {}
        vals = rcc.get(solver) if isinstance(rcc, dict) else None
        if vals is None or not str(lbl).startswith('Round '):
            continue
        vals = np.asarray(vals, dtype=float).ravel()
        if vals.size:
            curves.append((int(str(lbl).split()[-1]), vals))
    if not curves:
        print("No rCC_values found for any round. Plotting is skipped.")
        return
    curves.sort(key=lambda c: c[0])
    full_path = os.path.join(path, 'estimability')
    os.makedirs(full_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for rnd, vals in curves:
        k = np.arange(1, vals.size + 1)
        c = _color(rnd - 1)
        ax.plot(k, vals, 'o-', color=c, ms=4.5, label=f'round {rnd}')
        if np.any(np.isfinite(vals)):
            kb = int(np.nanargmin(vals))
            ax.plot(k[kb], vals[kb], 'o', ms=10, mfc='none', mec=c, mew=1.6)
    _symlog_rcc(ax, np.concatenate([v for _, v in curves]))
    ax.set_xticks(np.arange(1, max(v.size for _, v in curves) + 1))
    ax.set_xlabel('number of estimated parameters k (in estimability order)')
    ax.set_ylabel('corrected critical ratio rCC')
    ax.set_title(f'Estimability, model {solver} (ring: selected k = minimum rCC)')
    ax.legend(loc='best')
    fig.tight_layout()
    filename = os.path.join(full_path, f'rCC vs parameters in model_{solver}.png')
    _finish(fig, filename, True)
    print(f"Plot saved in {full_path}")


@_styled
def plot_rCC_vs_k(x_values, rCC_values, round, solver, show=True):
    r"""
    rCC against k for one round and model (called by ``estima``), the selected k marked.
    Saved as ``./estimability/rCC vs k round_<round> in model_<solver>.png``.

    Parameters
    ----------
    x_values : list
        k values.
    rCC_values : list
        rCC for each k.
    round : int
        Round number.
    solver : str
        Model name.
    show : bool, optional
        Display the figure as well (default True); it is always saved.
    """
    estimability_dir = Path.cwd() / "estimability"
    estimability_dir.mkdir(parents=True, exist_ok=True)
    filename = estimability_dir / f"rCC vs k round_{str(round)} in model_{solver}.png"
    k = np.asarray(x_values, dtype=float)
    v = np.asarray(rCC_values, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(k, v, 'o-', color=_PALETTE[0], ms=5)
    if np.any(np.isfinite(v)):
        kb = int(np.nanargmin(v))
        ax.plot(k[kb], v[kb], 'o', ms=11, mfc='none', mec=_PALETTE[1], mew=1.8,
                label=f'selected: k = {int(k[kb])}')
        ax.legend(loc='best')
    _symlog_rcc(ax, v)
    ax.set_xticks(k)
    ax.set_xlabel('number of estimated parameters k (in estimability order)')
    ax.set_ylabel('rCC')
    ax.set_title(f'Estimability, model {solver}, round {round}')
    fig.tight_layout()
    _finish(fig, filename, show)


@_styled
def plot_sobol_results(time_samples, sobol_analysis_results, sobol_problem, solver, response_key, show=True):
    r"""
    First-order (S1) and total (ST) Sobol indices of every parameter over time for one
    response, with their confidence bands when available. Saved as
    ``./Sobol_SIs_response_<response> for model_<solver>.png``.

    Parameters
    ----------
    time_samples : list
        Times of the samples.
    sobol_analysis_results : list of dict
        SALib results per time ('S1', 'ST' and optionally 'S1_conf', 'ST_conf').
    sobol_problem : dict
        SALib problem ('num_vars', 'names').
    solver : str
        Model name.
    response_key : str
        Response plotted.
    show : bool, optional
        Display the figure as well (default True); it is always saved.
    """
    t = np.asarray(time_samples, dtype=float).ravel()
    names = list(sobol_problem['names'])
    n = int(sobol_problem['num_vars'])
    filename = os.path.join(Path.cwd(), f'Sobol_SIs_response_{response_key} for model_{solver}.png')

    def stack(key):
        rows = [np.asarray(r.get(key, np.full(n, np.nan)), dtype=float).ravel() for r in sobol_analysis_results]
        return np.vstack(rows) if rows else np.zeros((0, n))

    S1, ST, S1c, STc = stack('S1'), stack('ST'), stack('S1_conf'), stack('ST_conf')
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, S, Sc, title in ((axes[0], S1, S1c, 'first-order index S1'),
                             (axes[1], ST, STc, 'total index ST')):
        for i in range(n):
            c = _color(i)
            ax.plot(t, S[:, i], color=c, ls=_LINESTYLES[i // len(_PALETTE) % 4],
                    label=names[i] if i < len(names) else f'p{i + 1}')
            if np.any(np.isfinite(Sc[:, i]) & (Sc[:, i] > 0)):
                ax.fill_between(t, S[:, i] - Sc[:, i], S[:, i] + Sc[:, i], color=c, alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel('time')
        if t.size:
            ax.set_xlim(t[0], t[-1])
    axes[0].set_ylabel('Sobol index')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    has_band = bool(np.any(np.isfinite(S1c) & (S1c > 0)) or np.any(np.isfinite(STc) & (STc > 0)))
    fig.suptitle(f'Global sensitivity of {response_key}, model {solver}'
                 + (' (bands: 95 % confidence of the indices)' if has_band else ''))
    fig.tight_layout()
    _finish(fig, filename, show)


def _initialize_dictionaries(models, iden_opt):
    r"""
    Initialize dictionaries for modelling settings and estimation settings.

    Parameters:
    models (dict): Dictionary containing modelling settings, including 'theta_parameters', 'bound_min', and 'bound_max'.
    iden_opt (dict): Dictionary containing estimation settings.

    Returns:
    None

    Raises:
    KeyError: If required keys are missing in models or iden_opt.
    """
    # Check if keys are missing in models and iden_opt
    keys_to_check = ['original_positions', 'masked_positions']
    if any(key not in models for key in keys_to_check) or 'x0' not in iden_opt:
        # Generate x0_dict with random initialization within bounds
        # if iden_opt['init'] == 'rand':
        #     x0_dict = {
        #         solver: np.array([
        #             np.random.uniform(
        #                 models['t_l'][solver][i],
        #                 models['t_u'][solver][i]
        #             )
        #             for i in range(len(params))
        #         ])
        #         for solver, params in models['theta'].items()
        #         if solver in models['t_l'] and solver in models['t_u']
        #     }
        # else:
        #     x0_dict = {
        #         solver: [1] * len(params)  # Replace all elements in the list with 1
        #         for solver, params in models['theta'].items()
        #         if solver in models['t_l'] and solver in models['t_u']
        #     }

        # Populate original_positions and masked_positions
        original_positions = {
            solver: list(range(len(params)))
            for solver, params in models['theta'].items()
        }
        masked_positions = {
            solver: list(range(len(params)))
            for solver, params in models['theta'].items()
        }

        # Remove any existing empty placeholders and update with generated values
        models.pop('original_positions', None)
        models.pop('masked_positions', None)
        # iden_opt.pop('x0', None)

        models['original_positions'] = original_positions
        models['masked_positions'] = masked_positions
        # iden_opt['x0'] = x0_dict

    # Ensure V_matrix and mutation dictionaries are initialized
    v_matrix_dict = models.get('V_matrix', {})
    mutation_dict = models.get('mutation', {})

    # Loop through each theta in 'theta_parameters'
    for key, theta_values in models['theta'].items():
        length = len(theta_values)

        # No placeholder V_matrix: a missing V means "no prior information"
        # (a fake 1e-50*I would dominate the PP prior term).

        # Create mutation list if not available
        if key not in mutation_dict:
            mutation_dict[key] = [True] * length

    # Update models with the (possibly empty) V_matrix and mutation dictionaries
    models['V_matrix'] = v_matrix_dict
    models['mutation'] = mutation_dict


@_styled
def validation_R2(prediction_metric, validation_metric, reference_metric, case, show=True):
    r"""
    Leave-one-experiment-out validation plots per model (called by ``validation``).

    ``validation/<metric> of validation for <model>.png``: the metric on the training
    experiments (prediction) and on the held-out experiment (validation) for every fold, with
    the value of the fit to all data. ``validation/Average <metric> Values for <model>.png``:
    mean +- sd over the folds against the all-data value.

    Parameters
    ----------
    prediction_metric : dict
        {fold: {model: value}} on the training experiments.
    validation_metric : dict
        {fold: {model: value}} on the held-out experiment.
    reference_metric : dict
        {model: value} of the fit to all experiments.
    case : str
        'R2' or 'MSE'.
    show : bool, optional
        Display the figures as well (default True); they are always saved.
    """
    label = {'R2': 'R²', 'MSE': 'MSE'}.get(case, str(case))
    folds = sorted(prediction_metric)
    if not folds:
        return
    solvers = list(prediction_metric[folds[0]].keys())
    validation_dir = os.path.join(os.getcwd(), 'validation')
    os.makedirs(validation_dir, exist_ok=True)

    for solver in solvers:
        pred = np.array([prediction_metric[f].get(solver, np.nan) for f in folds], dtype=float)
        val = np.array([validation_metric[f].get(solver, np.nan) for f in folds], dtype=float)
        ref = float(reference_metric.get(solver, np.nan))

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.fill_between(folds, pred, val, color=_INK2, alpha=0.10, lw=0)
        ax.plot(folds, pred, 'o-', color=_PALETTE[0], label='prediction (training experiments)')
        ax.plot(folds, val, 's-', color=_PALETTE[1], label='validation (held-out experiment)')
        if np.isfinite(ref):
            ax.axhline(ref, color=_INK, ls='--', lw=1.2, label=f'all data = {ref:.4g}')
        ax.set_xticks(folds)
        ax.set_xlabel('fold (held-out experiment)')
        ax.set_ylabel(label)
        ax.set_title(f'Cross-validation, {label}, model {solver}')
        ax.legend(loc='best')
        fig.tight_layout()
        _finish(fig, os.path.join(validation_dir, f'{label} of validation for {solver}.png'), show)

        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        means = [np.nanmean(pred), np.nanmean(val), ref]
        sds = [np.nanstd(pred), np.nanstd(val), 0.0]
        names = ['prediction', 'validation', 'all data']
        ax.bar(names, means, yerr=sds, capsize=4, width=0.55,
               color=[_PALETTE[0], _PALETTE[1], '#8f8e88'], error_kw={'elinewidth': 1.2})
        for k, (m, s) in enumerate(zip(means, sds)):
            if np.isfinite(m):
                ax.annotate(f'{m:.3g}' + (f' ± {s:.2g}' if s > 0 else ''), (k, m + s),
                            textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8)
        finite = [v for m, s in zip(means, sds) for v in (m - s, m + s) if np.isfinite(v)]
        if finite and case == 'R2':
            lo, hi = min(finite), max(finite)
            pad = 0.15 * (hi - lo) if hi > lo else 0.05 * (abs(hi) or 1.0)
            ax.set_ylim(lo - pad, hi + 2 * pad)
        ax.grid(axis='x', visible=False)
        ax.set_ylabel(f'{label} (mean ± sd over folds)')
        ax.set_title(f'Average {label}, model {solver}')
        fig.tight_layout()
        _finish(fig, os.path.join(validation_dir, f'Average {label} Values for {solver}.png'), show)


def validation_params(parameters, ref_params):
    r"""
    Plot normalized parameters for each model in each fold, divided by the corresponding member in ref_params.

    Parameters
    ----------
    parameters : dict
        Dictionary containing parameter values for each fold and model. (one data out at a time based cross-validation)
    ref_params : dict
        Dictionary containing reference parameter values for each model. (all data used estimation)

    Returns
    -------
    None
    """
    solvers = next(iter(parameters.values())).keys()
    n_folds = len(parameters)

    for solver in solvers:
        param_trends = {}
        for i in range(n_folds):
            if isinstance(parameters[i + 1][solver], dict):
                for param_name, param_value in parameters[i + 1][solver].items():
                    if param_name not in param_trends:
                        param_trends[param_name] = []
                    normalized_value = param_value / ref_params[solver].get(param_name, 1)
                    param_trends[param_name].append(normalized_value)
            elif isinstance(parameters[i + 1][solver], list):
                for idx, param_value in enumerate(parameters[i + 1][solver]):
                    param_name = f'param_{idx}'
                    if param_name not in param_trends:
                        param_trends[param_name] = []
                    normalized_value = param_value / ref_params[solver][idx]
                    param_trends[param_name].append(normalized_value)
            else:
                raise TypeError(f"Unexpected type for parameters[{i + 1}][{solver}]: {type(parameters[i + 1][solver])}")

        # Plotting each parameter trend
        plt.figure(figsize=(10, 6))
        for param_name, values in param_trends.items():
            plt.plot(range(1, n_folds + 1), values, marker='o', linestyle='-', label=param_name)

        plt.title(f'Normalized Parameter Trends for Solver: {solver}')
        plt.xlabel('Fold')
        plt.ylabel('Normalized Parameter Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        validation_dir = os.path.join(os.getcwd(), 'validation')
        os.makedirs(validation_dir, exist_ok=True)
        full_path = validation_dir
        plot_filename = os.path.join(full_path, f'Normalized Parameter Trends for {solver}.png')
        plt.savefig(plot_filename, dpi=300)
        _show()


def run_postprocessing(round_data, solvers, selected_rounds,
                       plot_global_p_and_t=True,
                       plot_confidence_spaces=True,
                       plot_p_and_t_tests=True,
                       export_excel_reports=True,
                       plot_estimability=True,
                       show=True,
                       true_theta=None):
    r"""
    Run post-processing analysis for specified solvers and rounds with plot/excel export options.

    Parameters
    ----------
    round_data : dict
        Data containing results from parameter estimation rounds.
    solvers : list of str
        Solvers to include in the post-analysis (e.g. ['f20', 'f21']).
    selected_rounds : list of int
        Rounds to include in the per-model plots and reports (e.g. [1, 2, 3]). The global
        P/t comparison plots and the estimability plot always use all rounds.
    plot_global_p_and_t : bool
        If True, generates global P and t comparison plots across all models and rounds.
    plot_confidence_spaces : bool
        If True, generates confidence space, ellipsoid volume, std dev, and parameter estimate plots.
    plot_p_and_t_tests : bool
        If True, generates P-value and t-value evolution plots over rounds per model.
    export_excel_reports : bool
        If True, generates Excel files with experimental, simulation, and input data + summary.
    plot_estimability : bool
        If True, generates rCC vs k estimability plots per model.
    show : bool
        If True (default), each figure is also displayed (``plt.show()``). Set it to False in
        scripts so the run does not stop at every figure; the figures are still saved.
    true_theta : list or dict, optional
        True parameter values (in silico studies), drawn on the parameter-estimate plot: a list
        for a single model or {model: list}.
    """
    global _SHOW_PLOTS
    show_orig, _SHOW_PLOTS = _SHOW_PLOTS, bool(show)
    try:
        _run_postprocessing(round_data, solvers, selected_rounds, plot_global_p_and_t,
                            plot_confidence_spaces, plot_p_and_t_tests, export_excel_reports,
                            plot_estimability, true_theta)
    finally:
        _SHOW_PLOTS = show_orig


def _run_postprocessing(round_data, solvers, selected_rounds, plot_global_p_and_t,
                        plot_confidence_spaces, plot_p_and_t_tests, export_excel_reports,
                        plot_estimability, true_theta=None):

    postprocessing_dir = os.path.join(os.getcwd(), "post_processing")
    os.makedirs(postprocessing_dir, exist_ok=True)

    if plot_global_p_and_t:
        overall_plotter = Plotting_FinalResults(round_data, None, [])
        overall_plotter.pcomp_plot(postprocessing_dir)
        overall_plotter.tcomp_plot(postprocessing_dir)

    for solver_name in solvers:
        print(f"Post-processing model: {solver_name}")
        plotter = Plotting_FinalResults(round_data, solver_name, selected_rounds, true_theta=true_theta)

        if plot_confidence_spaces:
            plotter.conf_plot(
                filename=os.path.join(postprocessing_dir, f"{solver_name}_confidence_space.png"),
                ellipsoid_volume_filename=os.path.join(postprocessing_dir, f"{solver_name}_ellipsoid_volume.png"),
                std_devs_filename=os.path.join(postprocessing_dir, f"{solver_name}_parameter_std_devs.png"),
                parameter_estimates_filename=os.path.join(postprocessing_dir, f"{solver_name}_parameter_estimates.png"),
            )

        if plot_p_and_t_tests:
            plotter.pt_plot(
                filename=os.path.join(postprocessing_dir, f"{solver_name}_pt_tests.png"),
                roundc=round_data
            )

        if export_excel_reports:
            plotter.reporter(
                filename=os.path.join(postprocessing_dir, f"{solver_name}_report")
            )

        if plot_estimability:
            Plot_estimability(
                round_data=round_data,
                path=postprocessing_dir,
                solver=solver_name
            )

    print(f"Post-processing completed for: {', '.join(solvers)}")


