"""Paper-style analysis grids (`figures/.../analysis_grid.pdf`). Run from
``analysis/workflows`` with ``Github/venv`` activated (see project README).

EV-named baselines (e.g. ``PC depth EV``): ``analysis_wrapper`` skips volatility variant **V**;
``Analyzer.plot_stochasticity_vs_conditional_inv_temp`` omits the highest-stochasticity inverse-
temperature level from plots/LMM inputs (README → Plotting → ``analysis_main.py``).
"""
import sys, os
import time

# First byte to the Slurm log before heavy imports (matplotlib, analysis/rpy2, etc.).
print(f"[{time.perf_counter():9.1f}s] analysis_main.py: starting imports (pid={os.getpid()})", flush=True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Make project source modules importable when running this script directly.
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from modelchecking import trialwise_rewards, trialwise_greedydiff, trialwise_chooseleft
from analysis import Analyzer, lmm, glmm
from plots import set_text_style
from utils import strsimplify, get_stochasticity_levels, alphabet, write_latex_results, write_latex_loggers, get_colormap
import glob
font_regular, font_bold = set_text_style()
import tqdm


_VARIANT_NAMES = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}
_VAR_LABELS = {"first_rt": "log(First RT)", "total_rt": "log(Total RT)",
               "points": "Points Earned", "invtemp": "log(β)"}

_current_residuals: list = []  # cleared at the start of each analysis_wrapper call


def _collect_residuals(residuals, variant: str, variable: str, label: str = None) -> None:
    """Accumulate LMM residuals so ``_cache_residuals`` can write them out per figure folder."""
    _current_residuals.append({
        "variant": variant, "variable": variable, "label": label,
        "residuals": np.asarray(residuals, dtype=float),
    })


def _cache_residuals(folder: str) -> None:
    """Cache each raw LMM residual vector as ``.npy`` for downstream diagnostics.

    ``diagnostics_assumptions.py`` globs ``figures/**/residuals_*.npy`` to build the
    Supplementary "LMM Robustness" table (skew / kurtosis / residual-variance ratio),
    so these files must keep being written even though no figure is produced from them.
    """
    if not _current_residuals:
        return

    os.makedirs(f"figures/{folder}", exist_ok=True)

    for rec in _current_residuals:
        tag = f"{rec['variant']}_{rec['variable']}"
        if rec["label"]:
            tag += f"_{rec['label'].replace(' ', '_')}"
        np.save(f"figures/{folder}/residuals_{tag}.npy", rec["residuals"])

# Monotonic progress logging for Slurm logs (`python -u` or PYTHONUNBUFFERED=1).
_T0: float | None = None

def _log(msg: str) -> None:
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    dt = time.perf_counter() - _T0
    job = os.environ.get("SLURM_JOB_ID", "")
    jid = f" job={job}" if job else ""
    print(f"[{dt:9.1f}s{jid}] {msg}", flush=True)

def analyze_responsetime(analyzer: Analyzer, ax: plt.Axes, label: str = None, first_rt = True, skip_lmm = False):
    # Use different y-axis ranges for first-response RT vs total RT.
    yspace = np.linspace(0.7, 1.9, 7) if first_rt else np.linspace(2.8, 6.8, 6)
    df, _ = analyzer.plot_stochasticity_vs_rt(ax=ax, yspace = yspace, first_rt=first_rt)
    condition_ticks = np.array(get_stochasticity_levels(analyzer.variant)) * 100
    ax.set(xlabel="Stochasticity Level (%)",
            ylabel="First Choice RT (s)" if first_rt else "Total RT (s)",
            xticks=condition_ticks,
            ylim=[np.log(yspace[0]), np.log(yspace[-1])],
            xticklabels=[strsimplify(x) for x in condition_ticks])
    if label: ax.text(-0.3, 1.15, label, transform=ax.transAxes, fontsize=28, fontproperties=font_bold, va='top', ha='left')
    if skip_lmm: return None, None

    # Run LMM
    _log(f"LMM start: variant={analyzer.variant} rt={'first' if first_rt else 'total'}")
    t_fit = time.perf_counter()
    result, logs_ = lmm(df)
    _log(f"LMM done:  variant={analyzer.variant} rt={'first' if first_rt else 'total'} ({time.perf_counter()-t_fit:.1f}s)")
    residuals = result.pop('residuals', None)
    info = {"Model Name": "Empirical", "Stochasticity Type": analyzer.variant, "Variable": "first_rt" if first_rt else "total_rt"}
    result.update(info)
    logs = [info | {"Formula / Status": f"\\texttt{{{log}}}"} for log in logs_]
    if residuals is not None:
        _collect_residuals(residuals, analyzer.variant, "first_rt" if first_rt else "total_rt")
    return result, logs

def analyze_greedydiff(analyzer: Analyzer, ax: plt.Axes, label: str = None, baseline_name = None, skip_lmm = False):
    # Plot psychometric curve against left-right label difference.
    df, _, _ = analyzer.plot_checking(trialwise_greedydiff, trialwise_chooseleft, n_bins=5, baseline_name=baseline_name, ax=ax)
    xticks = np.asarray([-6, -3, 0, 3, 6], dtype=float)
    yticks = np.linspace(0, 1, 6)
    ax.set(xlabel="Label Difference\n(Left - Right)", ylabel="P(Choice = Left)",
           xticks=xticks, xticklabels=[strsimplify(x) for x in xticks],
           yticks=yticks, yticklabels=[strsimplify(y) for y in yticks],
           ylim=[yticks[0], yticks[-1]])
    if label: ax.text(-0.3, 1.15, label, transform=ax.transAxes, fontsize=28, fontproperties=font_bold, va='top', ha='left')
    if baseline_name or skip_lmm: return None, None

    # Run GLMM
    _log(f"GLMM start: variant={analyzer.variant} greedydiff")
    t_fit = time.perf_counter()
    result, logs_ = glmm(df)
    _log(f"GLMM done:  variant={analyzer.variant} greedydiff ({time.perf_counter()-t_fit:.1f}s)")
    info = {"Model Name": "Empirical", "Stochasticity Type": analyzer.variant, "Variable": "greedydiff", "isGLMM": True}
    result.update(info)
    logs = [info | {"Formula / Status": f"\\texttt{{{log}}}"} for log in logs_]
    return result, logs

def analyze_rewards(analyzer: Analyzer, ax: plt.Axes, label: str = None, baseline_name = None, skip_lmm = False):
    # Compare achieved reward across stochasticity levels.
    df_reward, _, _ = analyzer.plot_checking_condition(trialwise_rewards, baseline_name=baseline_name, ax=ax)
    condition_ticks = np.array(get_stochasticity_levels(analyzer.variant)) * 100
    yticks = [4.8, 5.2, 5.6, 6, 6.4]
    ax.set(ylabel="Points Earned", xlabel="Stochasticity Level (%)",
            yticks=yticks, xticks=condition_ticks,
            ylim=[yticks[0], yticks[-1]],
            xticklabels=[strsimplify(x) for x in condition_ticks],
            yticklabels=[strsimplify(y) for y in yticks])
    if label: ax.text(-0.3, 1.15, label, transform=ax.transAxes, fontsize=28, fontproperties=font_bold, va='top', ha='left')
    if baseline_name or skip_lmm: return None, None

    # Run LMM
    _log(f"LMM start: variant={analyzer.variant} variable=points (rewards)")
    t_fit = time.perf_counter()
    result, logs_ = lmm(df_reward)
    _log(f"LMM done:  variant={analyzer.variant} variable=points ({time.perf_counter()-t_fit:.1f}s)")
    residuals = result.pop('residuals', None)
    info = {"Model Name": "Empirical", "Stochasticity Type": analyzer.variant, "Variable": "points"}
    result.update(info)
    logs = [info | {"Formula / Status": f"\\texttt{{{log}}}"} for log in logs_]
    if residuals is not None:
        _collect_residuals(residuals, analyzer.variant, "points")
    return result, logs

def analyze_invtemp(analyzer: Analyzer, ax: plt.Axes, label: str = None, baseline_name = None, skip_lmm = False):
    # Conditional log β by stochasticity; EV-named baselines omit the highest-stochasticity level
    # for identifiability (see Analyzer.plot_stochasticity_vs_conditional_inv_temp docstring).
    df, _ = analyzer.plot_stochasticity_vs_conditional_inv_temp(baseline_name = baseline_name, ax=ax)
    condition_ticks = np.array(get_stochasticity_levels(analyzer.variant)) * 100
    if baseline_name is not None and "EV" in baseline_name:
        condition_ticks = condition_ticks[:-1]
    yticks = ax.get_yticks()
    ax.set(xlabel="Stochasticity Level (%)",
            xticks=condition_ticks,
            yticks=yticks,
            ylim=[yticks[0], yticks[-1]],
            xticklabels=[strsimplify(x) for x in condition_ticks],
            yticklabels=[strsimplify(y) for y in yticks])
    if label: ax.text(-0.3, 1.15, label, transform=ax.transAxes, fontsize=28, fontproperties=font_bold, va='top', ha='left')
    if skip_lmm: return None, None
    
    # Run LMM
    _log(f"LMM start: variant={analyzer.variant} invtemp baseline={baseline_name!r}")
    t_fit = time.perf_counter()
    result, logs_= lmm(df)
    _log(f"LMM done:  variant={analyzer.variant} invtemp ({time.perf_counter()-t_fit:.1f}s)")
    residuals = result.pop('residuals', None)
    info = {"Model Name": baseline_name, "Stochasticity Type": analyzer.variant, "Variable": "invtemp"}
    result.update(info)
    logs = [info | {"Formula / Status": f"\\texttt{{{log}}}"} for log in logs_]
    if residuals is not None:
        _collect_residuals(residuals, analyzer.variant, "invtemp", label=baseline_name)
    return result, logs

def analysis_wrapper(analyzer_cache, folder="raw", plot_fns=["greedydiff", "rewards", "invtemp"], baseline_name = None, skip_lmm = False):
    global _current_residuals
    _current_residuals = []

    # EV value heads are only in ``get_effort_filter_value_options`` for R and T, not V.
    # Separately, EV baselines drop the 5th inv-temp level in plot_stochasticity_vs_conditional_inv_temp (README).
    variants = ["R", "V", "T"] if baseline_name is None or "EV" not in baseline_name else ["R", "T"]

    # Output folder for figures and optional model-fit logs.
    os.makedirs(f"figures/{folder}/", exist_ok=True)

    log_df, result_df, glmm_result_df = [], [], []
    t_wrap = time.perf_counter()
    _log(
        f"wrapper START folder={folder!r} baseline={baseline_name!r} "
        f"plot_fns={plot_fns} variants={variants} skip_lmm={skip_lmm}"
    )

    # Grid layout: columns are stochasticity variants (R/V/T), rows are analyses.
    fig, axs = plt.subplots(len(plot_fns), len(variants), figsize=(5 * len(variants), 5 * len(plot_fns)), gridspec_kw={'hspace': 0.5, 'wspace': 0.4})
    for col, variant in enumerate(variants):
        analyzer = analyzer_cache[variant]
        for row, plot_fn in enumerate(plot_fns):
            t_cell = time.perf_counter()
            ax = axs[col] if len(plot_fns) == 1 else axs[row, col]

            if row == 0:
                if col == 0:
                    ax.text(-0.3, 1.4, f"{'Empirical Data' if baseline_name is None else baseline_name}",
                            transform=ax.transAxes, fontsize=28, fontproperties=font_regular, va='top', ha='left')
                title = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[variant]
                ax.set_title(title, color=analyzer.colors(0.5), pad=15)

            if plot_fn == "greedydiff":
                glmmresult, logs = analyze_greedydiff(analyzer, ax, alphabet(row * 3 + col), baseline_name=baseline_name, skip_lmm=skip_lmm)
                if not skip_lmm and not baseline_name:
                    log_df.extend(logs)
                    glmm_result_df.append(glmmresult)
            elif plot_fn == "rewards":
                result, logs = analyze_rewards(analyzer, ax, alphabet(row * 3 + col), baseline_name=baseline_name, skip_lmm=skip_lmm)
                if not skip_lmm and not baseline_name:
                    log_df.extend(logs)
                    result_df.append(result)
            elif plot_fn == "total_rt":
                result, logs = analyze_responsetime(analyzer, ax, alphabet(row * 3 + col), skip_lmm=skip_lmm, first_rt=False)
                if not skip_lmm and not baseline_name:
                    log_df.extend(logs)
                    result_df.append(result)
            elif plot_fn == "rt":
                result, logs = analyze_responsetime(analyzer, ax, alphabet(row * 3 + col), skip_lmm=skip_lmm, first_rt=True)
                if not skip_lmm and not baseline_name:
                    log_df.extend(logs)
                    result_df.append(result)
            elif plot_fn == "invtemp":
                result, logs = analyze_invtemp(analyzer, ax, alphabet(row * 3 + col), baseline_name = baseline_name,skip_lmm=skip_lmm)
                if not skip_lmm:
                    log_df.extend(logs)
                    result_df.append(result)

            _log(
                f"wrapper cell DONE plot_fn={plot_fn!r} variant={variant!r} "
                f"({time.perf_counter()-t_cell:.1f}s this cell)"
            )

    # Save panel figure and, when enabled, tabular outputs for downstream reporting.
    os.makedirs(f"figures/{folder}/", exist_ok=True)
    fig.savefig(f"figures/{folder}/analysis_grid.pdf", bbox_inches='tight')

    if not skip_lmm:
        _cache_residuals(folder)

    if not skip_lmm:
        if len(log_df) > 0:
            pd.DataFrame(log_df).to_csv(f"figures/{folder}/logger.csv", index=False)
        if len(result_df) > 0:
            pd.DataFrame(result_df).to_csv(f"figures/{folder}/result_lmm.csv", index=False)
        if len(glmm_result_df) > 0:
            pd.DataFrame(glmm_result_df).to_csv(f"figures/{folder}/result_glmm.csv", index=False)

    _log(f"wrapper END folder={folder!r} (total {time.perf_counter()-t_wrap:.1f}s for this figure)")

if __name__ == "__main__":
    _log(
        f"analysis_main.py: pid={os.getpid()} "
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '')!r} "
        f"SLURM_JOB_NODELIST={os.environ.get('SLURM_JOB_NODELIST', '')!r}"
    )

    t_load = time.perf_counter()
    _log("Loading Analyzer objects (3× fit folders + data)...")
    analyzers = {
        "R": Analyzer("../fit/raw/R"),
        "V": Analyzer("../fit/raw/V"),
        "T": Analyzer("../fit/raw/T")
    }
    _log(f"Analyzers ready ({time.perf_counter()-t_load:.1f}s)")

    analysis_wrapper(analyzers, folder = "empirical", plot_fns = ["greedydiff", "rewards", "rt"], skip_lmm=False)
    analysis_wrapper(analyzers, folder = "empirical_totalrt", plot_fns = ["total_rt"], skip_lmm=False)
    baselines = ["PC depth EV", "PC depth path", "PC depth max", "PC depth sum", "PC depth levelmean"]
    for bi, baseline_name in enumerate(tqdm.tqdm(baselines, file=sys.stdout, mininterval=10.0), start=1):
        _log(f"BASELINE {bi}/{len(baselines)}: {baseline_name!r}")
        if baseline_name == "PC depth EV":
            analysis_wrapper(analyzers, folder = baseline_name, skip_lmm=False, baseline_name = baseline_name, plot_fns = ["greedydiff"])
        else: 
            analysis_wrapper(analyzers, folder = baseline_name, skip_lmm=False, baseline_name = baseline_name)

    t_tex = time.perf_counter()
    _log("Writing LaTeX tables from aggregated CSVs...")
    df_log = pd.concat([pd.read_csv(logger) for logger in glob.glob("figures/*/logger.csv")])
    write_latex_loggers(df_log, output_file = "figures/loggers.tex")

    df_results = pd.concat([pd.read_csv(logger) for logger in glob.glob("figures/*/*_lmm.csv")]).set_index(["Model Name", "Stochasticity Type", "Variable"])
    df_glmm_results = pd.concat([pd.read_csv(logger) for logger in glob.glob("figures/*/*_glmm.csv")]).set_index(["Model Name", "Stochasticity Type", "Variable"])
    write_latex_results(df_results, df_glmm_results, output_file = "figures/results.tex")
    _log(f"LaTeX done ({time.perf_counter()-t_tex:.1f}s). analysis_main.py finished.")