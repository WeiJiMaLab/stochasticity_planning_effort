import os
import sys
import json
import itertools
from collections import defaultdict

# Local package imports (``modeling`` etc. live alongside this file in ``analysis/src/``).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import xarray as xr
from scipy.stats import chi2
import statsmodels.formula.api as smf

from modeling import *
from modelchecking import *
from modelchecking import get_hierarchical_means, simulate_model, empirical, summary_statistics, trialwise_ignore
from utils import get_user_data, get_stochasticity_levels, strsimplify, get_colormap
from prodict import Prodict


def get_results(model):
    model.fit()
    if hasattr(model, 'convergence_status') and not "[1] TRUE" in model.convergence_status:
        return None, None, "Failed to converge"
    elif "singular" in " ".join(model.r_console).lower(): 
        return None, None, "Singular"
    elif "warning" in " ".join(model.r_console).lower(): 
        return None, None, "Convergence warnings"
    elif not model.fitted: 
        return None, None, "Not properly fitted"
    result = model.result_fit.to_pandas().set_index("term").transpose()
    loglik = model.result_fit_stats.to_pandas()["logLik"].iloc[0]
    return result, loglik, "Converged"

def lmm(df):
    """Fit y ~ conditions with the richest random-effects structure that converges.

    Tries, in order: correlated random slopes, uncorrelated slopes (||),
    random intercepts only, and finally fixed-effects OLS. The first structure that
    converges is returned, so the fitted model is data-dependent -- which rung was
    used is recorded in result["modeltype"] and the rejected ones in error_log
    (these become the Supplementary "LMM Robustness" table).

    Returns (result, error_log). CIs are Wald, from the t distribution on the
    reported dof. Raises RuntimeError if even OLS fails.
    """
    from pymer4.models import lmer, lm
    from scipy.stats import t as _t_dist
    df = pl.from_pandas(df)
    error_log = []
    for formula, label in [
        ("y ~ conditions + (1 + conditions|participants)", "Full"),
        ("y ~ conditions + (1 + conditions||participants)", "Uncorrelated Slopes"),
        ("y ~ conditions + (1|participants)", "Intercept-only")]:
        model = lmer(formula, data = df)
        result, _, message = get_results(model)
        error_log.append(f"{formula}:{message}")
        if result is not None:
            estimate = result["conditions"]["estimate"]
            tstat = result["conditions"]["t_stat"]
            dof = result["conditions"]["df"]
            pval = result["conditions"]["p_value"]
            _se = abs(estimate / tstat)
            _t_crit = _t_dist.ppf(0.975, dof)
            ci_lower = estimate - _t_crit * _se
            ci_upper = estimate + _t_crit * _se
            return {
                "modeltype": label,
                "beta": estimate,
                "tstat": tstat,
                "dof": dof,
                "pval": pval,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "residuals": model.data["resid"].to_list()
            }, error_log

    formula, label = "y ~ conditions + 1", "Fixed Effects"
    model = lm(formula, data = df)
    result, _, message = get_results(model)
    error_log.append(f"{formula}:{message}")
    if result is not None:
        estimate = result["conditions"]["estimate"]
        tstat = result["conditions"]["t_stat"]
        dof = result["conditions"]["df"]
        pval = result["conditions"]["p_value"]
        _se = abs(estimate / tstat)
        _t_crit = _t_dist.ppf(0.975, dof)
        ci_lower = estimate - _t_crit * _se
        ci_upper = estimate + _t_crit * _se
        return {
            "modeltype": label,
            "beta": estimate,
            "tstat": tstat,
            "dof": dof,
            "pval": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "residuals": model.data["resid"].to_list()
        }, error_log

    raise RuntimeError(message)

def glmm(df):
    """Binomial GLMM for y ~ x * conditions, with the same convergence ladder as lmm().

    At each rung three nested models are fitted, and effects are likelihood-ratio
    tests between them rather than Wald tests on a single fit: full vs
    no-interaction gives the interaction chi2, and no-interaction vs no-main gives
    the main effect, both on 1 df. Betas come from the model in which each term is
    the highest-order one (interaction from the full fit, main effect from the
    no-interaction fit); CIs are Wald, 1.96 * SE.

    Returns (result, error_log); result["modeltype"] records which rung converged.
    """
    from pymer4.models import glmer
    df = pl.from_pandas(df)
    error_log = []
    for full, no_interaction, no_main, label in [ 
        (
            "y ~ x * conditions + (1 + x + conditions|participants)", 
            "y ~ x + conditions + (1 + x + conditions|participants)", 
            "y ~ conditions + (1 + conditions|participants)", 
            "Full"
        ), 
        (
            "y ~ x * conditions + (1 + x + conditions||participants)", 
            "y ~ x + conditions + (1 + x + conditions||participants)", 
            "y ~ conditions + (1 + conditions||participants)", 
            "Uncorrelated Slopes"
        ), 
        (
            "y ~ x * conditions + (1|participants)", 
            "y ~ x + conditions + (1|participants)", 
            "y ~ conditions + (1|participants)", 
            "Intercept-only") 
        ]: 

        full_model = glmer(
            full,
            data=df,
            family="binomial",
            control="glmerControl(optimizer='bobyqa')"
        )

        no_interaction_model = glmer(
            no_interaction,
            data=df,
            family="binomial",
            control="glmerControl(optimizer='bobyqa')"
        )

        no_main_model = glmer(
            no_main,
            data=df,
            family="binomial",
            control="glmerControl(optimizer='bobyqa')"
        )

        full_result, full_loglik, full_message = get_results(full_model)
        no_interaction_result, no_interaction_loglik, no_interaction_message = get_results(no_interaction_model)
        no_main_result, no_main_loglik, no_main_message = get_results(no_main_model)

        error_log.append(f"{full}:{full_message}")

        if full_result is not None and no_interaction_result is not None and no_main_result is not None:
            chi2_inter = 2 * (full_loglik - no_interaction_loglik)
            pval_inter = chi2.sf(chi2_inter, 1)

            chi2_main = 2 * (no_interaction_loglik - no_main_loglik)
            pval_main = chi2.sf(chi2_main, 1)

            beta_inter = full_result["x:conditions"]["estimate"]
            beta_main = no_interaction_result["x"]["estimate"]

            se_main = no_interaction_result["x"]["std_error"]
            se_inter = full_result["x:conditions"]["std_error"]

            ci_lower_main = beta_main - 1.96 * se_main
            ci_upper_main = beta_main + 1.96 * se_main
            ci_lower_inter = beta_inter - 1.96 * se_inter
            ci_upper_inter = beta_inter + 1.96 * se_inter

            result = {
                "modeltype": label,
                "beta_main": beta_main,
                "chi2_main": chi2_main,
                "pval_main": pval_main,
                "ci_lower_main": ci_lower_main,
                "ci_upper_main": ci_upper_main,
                "beta_inter": beta_inter,
                "chi2_inter": chi2_inter,
                "pval_inter": pval_inter,
                "ci_lower_inter": ci_lower_inter,
                "ci_upper_inter": ci_upper_inter
            }

            return result, error_log
    raise RuntimeError(full_message, no_interaction_message, no_main_message)

def clean_model_name(model):
    # Replacement order matters: e.g. ``filter_adapt`` must become ``FA`` before ``filter_`` is stripped.
    return (
        model.replace("filter_adapt", "FA")
        .replace("policy_compress", "PC")
        .replace("filter_", "")
        .replace("value_", "")
        .replace(".", " ")
    )

# Display order for model-comparison panels, as "<effort> <filter> <value>" (the output of
# ``clean_model_name``). Groups run effort -> filter -> value function, most-structured first.
_EFFORT_ORDER = ["PC", "FA"]
_FILTER_ORDER = ["depth", "value", "rank"]
_VALUE_ORDER = ["EV", "path", "levelmean", "sum", "max"]


def draw_shared_xlabel(fig, ax_main, ax_zoom, color, y=0.055):
    """Centre one x title under the main+inset pair, with "baseline" styled.

    Both panels show the same quantity, so they get a single shared title rather than one
    label each. "baseline" is drawn bold and in the variant's colour to tie the axis
    definition to the highlighted baseline row; matplotlib cannot mix styles inside one
    text, so that middle line is assembled from three abutting pieces.
    """
    x0 = ax_main.get_position().x0
    x1 = (ax_zoom if ax_zoom is not None else ax_main).get_position().x1
    xc = (x0 + x1) / 2

    # Line 1, assembled in pieces so "Baseline" alone is bold and coloured.
    parts = [("Δ NLL (Model $-$ ", "normal", "black"),
             ("Baseline", "bold", color),
             (")", "normal", "black")]
    renderer = fig.canvas.get_renderer()
    widths = []
    for txt, weight, _c in parts:
        probe = fig.text(0, 0, txt, fontweight=weight, fontsize=20)
        widths.append(probe.get_window_extent(renderer).width / fig.bbox.width)
        probe.remove()
    x = xc - sum(widths) / 2
    for (txt, weight, col), w in zip(parts, widths):
        fig.text(x, y, txt, fontweight=weight, color=col,
                 ha="left", va="center", fontsize=20)
        x += w

    # Line 2.
    fig.text(xc, y - 0.020, "$\\leftarrow$ better fit", ha="center", va="top", fontsize=20)


def model_display_name(model_name):
    """Tick-label form of a model name. Display-only — never use as a key.

    ``clean_model_name`` output ("PC depth levelmean") is an identifier: it keys
    ``model_data`` / ``model_params``, selects baselines, and is written into the paper's
    ``\\pgfval`` macro names and figure folder paths. So the shortening happens here, at
    render time, and leaves those untouched.
    """
    return model_name.replace("levelmean", "level")


def model_sort_key(model_name):
    """Sort key placing models in the canonical PC/FA x depth/value/rank x EV..max order.

    Unknown tokens sort after the known ones (and then alphabetically) rather than raising,
    so adding a model family cannot silently drop it from a figure.
    """
    parts = model_name.split()
    effort, filt, value = (parts + [""] * 3)[:3]

    def rank(token, order):
        return order.index(token) if token in order else len(order)

    return (
        rank(effort, _EFFORT_ORDER),
        rank(filt, _FILTER_ORDER),
        rank(value, _VALUE_ORDER),
        model_name,
    )


def extract_reaction_time_data(user_dict):
    rows = [
        {
            "player": player,
            "game": game["name"],
            "log_first_rt": np.log(game["trials"][0]["rt"] / 1000),
            "log_total_rt": np.log(sum(trial["rt"] for trial in game["trials"]) / 1000),
            "condition": game["p"]
        }
        for player, games in user_dict.items()
        for game in games
    ]
    return pd.DataFrame(rows)


def get_fits_and_params(fit_folder):
    player_fits = [f for f in os.listdir(fit_folder) if f.endswith("_data.json")]

    params = defaultdict(lambda: defaultdict(lambda: {}))
    rows = []

    for fname in sorted(player_fits):
        player = fname.replace("_data.json", "")
        with open(os.path.join(fit_folder, fname), "r") as f:
            for model_name, model_info in json.load(f).items():
                full_params = model_info.get('full_params', {})
                row = {
                    'player': player,
                    'model': model_info.get('model'),
                    'nll': model_info.get('NLL_CV'),
                    **{k: v for k, v in full_params.items() if k != 'filter_params'},
                    **full_params.get('filter_params', {})
                }
                rows.append(pd.Series(row))
                params[clean_model_name(model_name)][player] = full_params

    df = pd.DataFrame(rows)
    fits = {clean_model_name(model): group.dropna(axis=1, how="all").sort_values(by="player").reset_index(drop=True) for model, group in df.groupby("model")}
    return fits, params

def bootstrap(x, n = 1e4): 
    n_samps = len(x)
    samp_indices = np.random.choice(np.arange(n_samps), (int(n), n_samps), replace = True)
    return x[samp_indices]

class Analyzer(): 
    def __init__(self, fit_folder, data_folder="../data/raw/"):
        """Load CV fits from ``fit_folder`` and participant JSON under ``data_folder`` (per ``variant``).

        ``data_folder`` defaults to ``../data/raw/`` (relative to cwd: use from ``analysis/workflows``
        or adjust for simulated trees when calling overlay-style plots).
        """
        self.fit_folder = fit_folder
        self.data_folder = data_folder
        self.data = {}
        self.variant = self.fit_folder.rstrip("/").split("/")[-1]
        for user in [f"user{i}" for i in range(100)]:
            self.data[user] = get_user_data(user, self.variant, self.data_folder)
        self.reaction_time_data = extract_reaction_time_data(self.data)
        self.model_data, self.model_params = get_fits_and_params(fit_folder)
        self.name_to_fns = {
            clean_model_name(f"{effort}.{filter_fn.__name__}.{value_fn.__name__}"): (effort, filter_fn, value_fn)
            for effort, filter_fn, value_fn in itertools.product(*get_effort_filter_value_options(self.variant))
        }
        self.colors = get_colormap(self.variant)
        self.conditions = get_stochasticity_levels(self.variant)

    def plot_model_comparison(self, baseline_name, n_bootstrap=1e6, verbose=False, ax=None, ax_zoom=None, zoom_range=None, main_xlim=None):
        '''Bootstrap ΔNLL violins vs ``baseline_name`` (must be a key of ``model_data``).'''
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        plot = defaultdict(lambda: [])
        keys = list(self.model_data.keys())
        assert baseline_name in keys, (
            f"baseline_name {baseline_name!r} not in model_data keys {sorted(keys)!r}"
        )
        keys = sorted(self.model_data.keys(), key=model_sort_key, reverse=True)

        for key in keys:
            baseline = self.model_data[baseline_name]
            model = self.model_data[key]
            diff = np.array(model["nll"] - baseline["nll"])

            # bootstrapped NLL model - NLL baseline
            bootstrap_diff = bootstrap(diff, n_bootstrap).sum(-1)

            mean = np.mean(bootstrap_diff)
            conf = np.quantile(bootstrap_diff, [0.025, 0.975])

            plot["mean"].append(mean)
            plot["conf"].append(conf - mean)
            plot["name"].append(key)
            plot["dist"].append(bootstrap_diff)

            if verbose: print(key, conf)
        
        plot = Prodict(plot)

        axes = [ax, ax_zoom] if ax_zoom is not None else [ax]
        for axis in axes:
            violin = axis.violinplot(np.array(plot.dist).T,
                            showmeans=False,
                            showextrema = False,
                            vert = False,
                            widths = 0.8,
                            quantiles = [[0.025, 0.975]] * len(plot.name)
                            )

            # plot formatting. Models sit in a fixed canonical order (see ``model_sort_key``),
            # so the baseline is identified by highlighting rather than by position: a tinted
            # band across its row, and a bold coloured tick label.
            color = self.colors(0.5)
            for part in violin['bodies']:
                part.set_facecolor(color)
                part.set_alpha(0.4)

            for partname in (['cquantiles']):
                vp = violin[partname]
                vp.set_edgecolor(color)
                vp.set_linewidth(1.5)

            # Band the baseline's row. The baseline's own violin collapses to a sliver at 0
            # (it is being differenced against itself), so a fill alone would be invisible;
            # the band is what actually marks the row now that position no longer does.
            baseline_row = plot.name.index(baseline_name) + 1
            axis.axhspan(baseline_row - 0.5, baseline_row + 0.5,
                         color=color, alpha=0.13, zorder=0)

            ticks = np.arange(len(plot.name)) + 1
            axis.set_yticks(ticks, labels=[model_display_name(nm) for nm in plot.name])
            for ylab, nm in zip(axis.get_yticklabels(), plot.name):
                if nm == baseline_name:
                    ylab.set_fontweight("bold")
                    ylab.set_color(color)
            axis.set_ylim([0.2, len(plot.name) + 0.5])
            axis.vlines(0, -1, len(plot.name) + 1, colors=self.colors(0.5), alpha=0.2, linestyles='dashed', linewidths=2.5, zorder=1)
            axis.grid(c = [0.95, 0.95, 0.95], axis = 'y', linewidth = 1)
            axis.spines[['top', 'right']].set_visible(False)
            axis.spines[['left', 'bottom']].set_linewidth(1.5)
            axis.set_axisbelow(True)
            axis.xaxis.set_tick_params(width=1.5, length = 10)
            axis.yaxis.set_tick_params(width=1.5, length = 10)
            # These panels are narrow (the inset especially), so cap the tick count: the default
            # locator packs in enough ticks that the labels overlap into an unreadable blob.
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            # Mathtext ($\leftarrow$) draws the arrow from matplotlib's own font rather
            # than the figure face, so it survives even when that face lacks U+2190
            # (Helvetica Neue, used previously, did). Keep the mathtext span on its own line:
            # matplotlib mis-orders a "$...$" run followed by more text when exporting to SVG
            # with svg.fonttype='none' (the span jumps to the end of the line).
            axis.set_xlabel("Δ CV NLL\n(model - baseline)\n$\\leftarrow$ better fit", labelpad=20)

        # Rescale the tick labels so they stay short: five-digit ticks are what collide at the
        # shared panel boundary. Both panels use 1e2 so they are read on one common scale.
        # ``ScalarFormatter`` in scientific mode draws the multiplier as the axis offset text
        # -- i.e. sitting on the axis line at its right end -- rather than as a separate line
        # in the xlabel.
        def _scale_axis(axis_, exponent, flush_right=False):
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            # Force this exact exponent for every panel; the default would pick per-axis.
            fmt.set_powerlimits((exponent, exponent))
            axis_.xaxis.set_major_formatter(fmt)
            offset = axis_.xaxis.get_offset_text()
            offset.set_fontsize(16)
            if flush_right:
                # Default offset text is left-aligned just past the axis end, which leaves it
                # hanging into the gap. Right-align it on the axis's right edge instead.
                offset.set_horizontalalignment("right")
                offset.set_position((1.0, 0.0))

        _scale_axis(ax, 2, flush_right=True)
        # No per-axis xlabel: the caller centres one shared title under the main/inset pair
        # (see ``draw_shared_xlabel``), so the two panels read as a single x axis.
        ax.set_xlabel("")
        # Fix the full-range limits rather than letting each variant autoscale: it puts all
        # three panels on one comparable scale, and stops the largest tick landing hard against
        # the panel edge (where it collided with the inset). Covers every variant's data, which
        # spans about -0.4e3 to 24.8e3 against this baseline.
        if main_xlim is not None:
            ax.set_xlim(*main_xlim)

        if ax_zoom is not None:
            ax_zoom.set_xlim(*zoom_range)
            _scale_axis(ax_zoom, 2, flush_right=True)
            ax_zoom.set_xlabel("")
            # "(Zoom)" sits above the inset rather than below it, so the only thing under the
            # panels is the shared x title.
            ax_zoom.set_title("(Zoom)", fontsize=18, pad=10)
            ax_zoom.set_yticklabels([])
            ax_zoom.tick_params(axis='y', length=0)
            ax_zoom.grid(True, axis='y')

        stats = [
            {"name": nm, "mean": float(m), "ci_low": float(m + c[0]), "ci_high": float(m + c[1])}
            for nm, m, c in zip(plot.name, plot.mean, plot.conf)
        ]
        return stats

    def plot_stochasticity_vs_conditional_inv_temp(self, baseline_name, ax=None):
        """Per-participant conditional log β vs stochasticity (for LMM inputs use returned columns).

        If *baseline_name* contains ``"EV"`` (e.g. ``PC depth EV``), only the first four
        ``condition_inv_temp_*`` columns are used: the highest-stochasticity index is omitted
        because softmax slope is poorly identified there (choices near chance; confounded with
        lapse or extreme β). That level is **100%** for variants **R/V** and **50%** for **T**
        (see ``get_stochasticity_levels``). Other model-check paths are unchanged.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        baseline = self.model_data[baseline_name]
        # EV baselines: omit condition_inv_temp_4 (see docstring).
        n_conditions = 5 if "EV" not in baseline_name else 4
        
        mean = np.mean(baseline[[f"condition_inv_temp_{i}" for i in range(n_conditions)]].values, axis = 0)
        std = np.std(baseline[[f"condition_inv_temp_{i}" for i in range(n_conditions)]].values, axis = 0, ddof = 1)/np.sqrt(len(baseline))
        ax.errorbar(np.array(self.conditions[:n_conditions]) * 100, mean, std, capsize = 3, elinewidth=2, markeredgewidth=2, linewidth=2, color = self.colors(0.5))
        ax.set_xlabel("Stochasticity Level (%)\n")
        ax.set_ylabel(r"Log $\beta$")
        ax.set_xticks(np.array(self.conditions[:n_conditions]) * 100)
        
        df = pd.melt(baseline, "player", value_vars = ["condition_inv_temp_0", "condition_inv_temp_1", "condition_inv_temp_2", "condition_inv_temp_3", "condition_inv_temp_4"][:n_conditions])
        df.variable = df.variable.apply(lambda x: get_stochasticity_levels(self.variant)[int(x.split("_")[-1])])
        df["condition"] = df["variable"].astype(float)
        df["inv_temp"] = df["value"]
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        out_df = df.copy()
        out_df = out_df.rename(columns = {"condition": "conditions", "inv_temp": "y", "player": "participants"})

        stats = {"x": np.array(self.conditions[:n_conditions]) * 100, "mean": mean, "sem": std}
        return out_df[["participants", "conditions", "y"]], stats


    def plot_stochasticity_vs_rt(self, yspace = np.linspace(0.8, 1.8, 6), ax=None, first_rt = True): 
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        self.reaction_time_data["y"] = self.reaction_time_data["log_first_rt"] if first_rt else self.reaction_time_data["log_total_rt"]

        mean_x, mean_rt, sem_rt = get_hierarchical_means(self.reaction_time_data["player"], 
                                                         self.reaction_time_data["condition"],
                                                         self.reaction_time_data["y"],
                                                         group = self.reaction_time_data["condition"])

        ax.errorbar(mean_x * 100, mean_rt, sem_rt, capsize = 3, elinewidth=2, markeredgewidth=2, linewidth=2, color = self.colors(0.5))
        ax.set_xticks(mean_x * 100)
        ax.set_xlabel("Stochasticity Level (%)\n")
        ax.set_ylabel("First Choice RT (s)" if first_rt else "Total RT (s)")

        ax.set_yticks(np.log(yspace))
        ax.set_yticklabels([strsimplify(float(z)) for z in np.round(yspace, 2)])

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        out_df = self.reaction_time_data.copy()
        out_df = out_df.rename(columns = {"condition": "conditions", "player": "participants"})
        stats = {"x": mean_x * 100, "mean": mean_rt, "sem": sem_rt}
        return out_df[["participants", "conditions", "y"]], stats

    def plot_checking_condition(self, y_fun, baseline_name = None, ax=None): 
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        sim_x, sim_y = None, None
        if baseline_name is not None:
            effort_version, filter_fn, value_fn = self.name_to_fns[baseline_name]
            fit_params = self.model_params[baseline_name]
            _, sim_y = simulate_model(data = self.data, 
                effort_version = effort_version, 
                filter_fn = filter_fn, 
                value_fn = value_fn, 
                variant = self.variant, 
                fitted_params = fit_params,
                x_fun = trialwise_ignore,
                y_fun = y_fun
                )

            sim_x, participants = (xr.ones_like(sim_y) * sim_y.conditions), (xr.ones_like(sim_y) * sim_y.participants)
            model_x, model_y, model_sem = get_hierarchical_means(participants.values.flatten(), sim_x.values.flatten(), sim_y.values.flatten())

            ax.fill_between(model_x * 100, model_y - model_sem, model_y + model_sem, alpha = 0.5, color = self.colors(0.5))

        _, y = empirical(
            data = self.data,
            x_fun = trialwise_ignore, 
            y_fun = y_fun
        )
        x, participants = (xr.ones_like(y) * y.conditions), (xr.ones_like(y) * y.participants)
        emp_x, emp_y, emp_sem = get_hierarchical_means(participants.values.flatten(), x.values.flatten(), y.values.flatten())

        ax.errorbar(emp_x * 100 , emp_y, yerr=emp_sem, elinewidth=2, markeredgewidth=2, capsize = 3, linestyle='None', color=self.colors(0.5))
        if baseline_name is None: ax.plot(emp_x * 100, emp_y, color=self.colors(0.5), linestyle='-', linewidth=2)  # Connect error bars with a line

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)

        df = y.to_dataframe(name="y").reset_index()
        stats = {
            "empirical": {"x": emp_x * 100, "y": emp_y, "sem": emp_sem},
            "model": ({"x": model_x * 100, "y": model_y, "sem": model_sem} if baseline_name is not None else None),
        }
        if baseline_name is not None:
            df_sim = sim_y.to_dataframe(name="y").reset_index()
            return df, df_sim, stats
        else:
            return df, None, stats

    def plot_checking(self, x_fun, y_fun, baseline_name = None, n_bins = None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        sim_x, sim_y = None, None
        if baseline_name is not None:
            effort_version, filter_fn, value_fn = self.name_to_fns[baseline_name]
            fit_params = self.model_params[baseline_name]

            sim_x, sim_y = simulate_model(data = self.data, 
                effort_version = effort_version, 
                filter_fn = filter_fn, 
                value_fn = value_fn, 
                variant = self.variant, 
                fitted_params = fit_params,
                x_fun = x_fun, 
                y_fun = y_fun
                )

            model_x, model_y, model_sem = summary_statistics(sim_x, sim_y, n_bins = n_bins)
            num_model_keys = len(model_x.keys())
            for i, k in enumerate(model_x.keys()): 
                # Get color from colormap based on index
                color = self.colors(i / (num_model_keys - 1)) if num_model_keys > 1 else self.colors(0.5)
                ax.fill_between(model_x[k], model_y[k] - model_sem[k], model_y[k] + model_sem[k], alpha = 0.5, color = color)

        x, y = empirical(
            data = self.data,
            x_fun = x_fun, 
            y_fun = y_fun
        )

        emp_x, emp_y, emp_sem = summary_statistics(x, y, n_bins = n_bins)
        num_emp_keys = len(emp_x.keys())

        colorlegend = {}

        for i, k in enumerate(emp_x.keys()): 
            # Get color from colormap based on index
            color = self.colors(i / (num_emp_keys - 1)) if num_emp_keys > 1 else self.colors(0.5)
            ax.errorbar(emp_x[k], emp_y[k], yerr = emp_sem[k], elinewidth=2, markeredgewidth=2, capsize = 3, linestyle = 'None', label = k, color = color)
            if baseline_name is None: ax.plot(emp_x[k], emp_y[k], color=color, linestyle='-', linewidth=2)  # Connect error bars with a line
            colorlegend[f"{strsimplify(k * 100)}%"] = color

        # After plotting, manually create a patch-based legend
        patches = [mpatches.Patch(color=color, label=label) for label, color in colorlegend.items()]

        ax.legend(handles=patches, loc='lower right', frameon=False, fontsize=13, title="$q$", title_fontsize = 13)

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_tick_params(width=1.5, length = 10)
        ax.yaxis.set_tick_params(width=1.5, length = 10)
        ax.grid(c = [0.95, 0.95, 0.95], axis = 'both', linewidth = 1)
        
        df = pd.concat([x.to_dataframe(name="x"), y.to_dataframe(name="y")], axis=1).reset_index()
        stats = {
            "empirical": {"x": dict(emp_x), "y": dict(emp_y), "sem": dict(emp_sem)},
            "model": ({"x": dict(model_x), "y": dict(model_y), "sem": dict(model_sem)} if baseline_name is not None else None),
        }
        if baseline_name is not None:
            df_sim = pd.concat([sim_x.to_dataframe(name="x"), sim_y.to_dataframe(name="y")], axis=1).reset_index()
            return df, df_sim, stats
        else:
            return df, None, stats

