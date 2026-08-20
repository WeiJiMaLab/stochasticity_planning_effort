"""Numbers behind the TEXT-3 (assumptions) paragraph in the Methods.

Scope: only the LMMs assume normally distributed residuals. The GLMMs are
binomial and assume no such thing, so nothing here applies to them.

Run from ``analysis/workflows``; writes ``figures/diagnostics.tex`` for the
Supplementary "LMM Robustness" table and a summary CSV. Internal only.
"""
import csv
import glob
import re
import os
import sys
import warnings

import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.getcwd()) + "/src")
import utils

def get_ols_slope(y, q):
    """OLS slope of ``y`` on ``q``; ``y`` is per-level, or participants x levels."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    if y.shape[-1] != len(q):
        raise ValueError(
            f"y has {y.shape[-1]} levels but q has {len(q)}")
    qc = q - q.mean()
    return (y * qc).sum(axis=-1) / (qc ** 2).sum()


def residual_shape():
    """Skew/kurtosis/tail mass for every cached LMM residual vector."""
    rows = []
    for path in sorted(glob.glob("figures/**/residuals_*.npy", recursive=True)):
        r = np.load(path).astype(float).ravel()
        r = r[np.isfinite(r)]
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        if sd == 0:
            continue
        z = (r - r.mean()) / sd
        rows.append({
            "file": os.path.relpath(path, "figures"),
            "n": len(r),
            "skew": stats.skew(r),
            "excess_kurtosis": stats.kurtosis(r),
            "frac_absz_gt2": float(np.mean(np.abs(z) > 2)),
            "frac_absz_gt3": float(np.mean(np.abs(z) > 3)),
        })
    return rows


def load_outcomes(
    raw_dir="../data/raw",
    fit_dir="../fit/raw",
    variants=("R", "V", "T"),
    baselines=("PC depth path", "PC depth levelmean",
               "PC depth max", "PC depth sum"),
):
    """Every observation feeding every LMM, as one xarray Dataset per variant.

    Each variable is indexed by ``(participant, condition)`` plus whatever
    trailing dimension it naturally has, because the outcomes are genuinely
    ragged: ``points`` is per decision (games x trials), ``first_rt_log`` is per
    game, and log beta is a single fitted value per cell. A flat table would
    force either a meaningless alignment between them or NaN padding.

    Variables: ``points`` (participant, condition, games, trials),
    ``first_rt_log`` / ``total_rt_log`` / ``first_rt_raw`` (participant,
    condition, games), and one per baseline (participant, condition).

    Values must match what ``analysis_main.py`` feeds ``lmm()``. Two traps:
    ``points`` is the reward on each of the 7 modelled decisions
    (``trialwise_rewards``), not ``game["total"]`` -- the per-game total is ~7x
    larger and misstates beta while leaving the sign tests unchanged, so it
    fails silently. ``first_rt_log`` comes from ``extract_reaction_time_data``
    rather than a re-derived log.
    """
    import pandas as pd
    import xarray as xr
    from analysis import extract_reaction_time_data, get_fits_and_params
    from modelchecking import trialwise_rewards

    datasets = {}
    for variant in variants:
        levels = utils.get_stochasticity_levels(variant)
        points, participants = [], []
        rt_cols = {"log_first_rt": [], "log_total_rt": [], "raw": []}

        for path in sorted(glob.glob(f"{raw_dir}/{variant}/user*_data.json")):
            participant = os.path.basename(path).split("_data")[0]
            games = utils.get_user_data(participant, variant, raw_dir)
            participants.append(participant)

            points.append(trialwise_rewards(utils.preprocess_data(games)))

            rt = extract_reaction_time_data({participant: games})
            rt["raw"] = [g["trials"][0]["rt"] / 1000.0 for g in games]
            # One row per game; reshape to (conditions, games) in level order.
            grid = rt.pivot_table(index="condition", values=list(rt_cols),
                                  aggfunc=list).reindex(levels)
            for col, acc in rt_cols.items():
                acc.append(np.array(grid[col].tolist(), dtype=float))

        coords = {"participant": participants, "condition": levels}
        per_game = ("participant", "condition", "games")
        ds = xr.Dataset(
            {
                "points": xr.concat(points, dim="participant").rename(
                    conditions="condition").assign_coords(
                        participant=participants),
                "first_rt_log": (per_game, np.stack(rt_cols["log_first_rt"])),
                "total_rt_log": (per_game, np.stack(rt_cols["log_total_rt"])),
                "first_rt_raw": (per_game, np.stack(rt_cols["raw"])),
            },
            coords=coords,
        )

        fits, _ = get_fits_and_params(f"{fit_dir}/{variant}")
        cols = [f"condition_inv_temp_{i}" for i in range(len(levels))]
        for baseline in baselines:
            table = fits.get(baseline)
            if table is None or not all(c in table.columns for c in cols):
                continue
            table = table.set_index("player").reindex(participants)
            ds[baseline] = (("participant", "condition"),
                            table[cols].to_numpy(dtype=float))

        datasets[variant] = ds
    return datasets


def outcome_names(ds, extra=("first_rt_raw",)):
    """Analysed outcomes in a Dataset: everything but the helper scales."""
    return [name for name in ds.data_vars if name not in extra]


def robustness_row(variant, outcome, da):
    """Distribution-free tests of one outcome's effect.

    Per-participant slopes make the participant the unit of analysis, so
    Wilcoxon and the sign test assume nothing about the residuals. For log-beta
    outcomes these are fitted parameters, so the two-stage assumption remains.
    """
    # Average away any trailing dims (games, trials) to one mean per cell.
    extra = [d for d in da.dims if d not in ("participant", "condition")]
    means = da.mean(dim=extra) if extra else da
    assert not means.isnull().any(), f"{variant}/{outcome}: missing cells"
    s = get_ols_slope(means.to_numpy(), means["condition"].to_numpy())
    assert np.isfinite(s).all(), (
        f"{variant}/{outcome}: {(~np.isfinite(s)).sum()} non-finite slopes")
    t, p_t = stats.ttest_1samp(s, 0)
    _, p_w = stats.wilcoxon(s, zero_method="zsplit")  # split ties, not drop
    n_neg = int((s < 0).sum())  # ties count against (negative only)
    p_sign = stats.binomtest(n_neg, len(s), 0.5).pvalue
    return {
        "variant": variant, "outcome": outcome, "n": len(s),
        "mean_slope": s.mean(), "t": t, "p_t": p_t,
        "p_wilcoxon": p_w, "p_sign": p_sign,
        "n_negative": n_neg,
    }


def rt_transform_check(ds, variant):
    """Skew/kurtosis of first-choice RT before and after the log transform."""
    raw = ds["first_rt_raw"].to_numpy().ravel()
    lg = ds["first_rt_log"].to_numpy().ravel()
    assert len(raw) == len(lg), (
        f"{variant}: {len(raw)} raw RTs but {len(lg)} log RTs")
    return {
        "variant": variant, "n": len(raw),
        "raw_skew": stats.skew(raw), "raw_kurtosis": stats.kurtosis(raw),
        "log_skew": stats.skew(lg), "log_kurtosis": stats.kurtosis(lg),
    }


def fitted_lmm():
    """Fitted LMM estimates from the pipeline's ``result_lmm.csv`` files.

    Full precision from the data the pipeline wrote, not re-parsed from
    formatted LaTeX. Keyed ``(variant, outcome)``; {} if absent.
    """
    out = {}
    for path in sorted(glob.glob("figures/*/result_lmm.csv")):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                variant = row["Stochasticity Type"]
                # "Empirical" for trial-level LMMs, else the baseline name.
                outcome = (row["Variable"] if row["Model Name"] == "Empirical"
                           else row["Model Name"])
                outcome = {"first_rt": "first_rt_log",
                           "total_rt": "total_rt_log"}.get(outcome, outcome)
                out[(variant, outcome)] = {
                    "beta": float(row["beta"]),
                    "tstat": float(row["tstat"]),
                    "dof": float(row["dof"]),
                }
    return out

def write_latex_keys(shape, transforms, sd_ratios, output_file):
    """Emit prose-embeddable values as pgfkeys, mirroring ``results.tex``.

    The SI already inputs ``figures/results`` and defines ``\\pgfval``, so these
    are used the same way. Emitting them keeps quoted numbers in the manuscript
    from drifting when 3a/3b are re-run.
    """
    worst = max(sd_ratios, key=lambda r: r[1])
    rt_raw = [t["raw_skew"] for t in transforms]
    rt_log = [t["log_skew"] for t in transforms]
    beta = [r for r in shape if "invtemp" in r["file"]]
    points = [r for r in shape if "points" in r["file"]]

    keys = {
        "Diag.sd.ratio.max": f"{worst[1]:.2f}",
        "Diag.sd.ratio.max.model": worst[0],
        "Diag.rt.skew.raw": f"{min(rt_raw):.1f}--{max(rt_raw):.1f}",
        "Diag.rt.skew.log": f"{min(rt_log):.2f}--{max(rt_log):.2f}",
        "Diag.points.skew": f"{min(r['skew'] for r in points):.2f}",
        "Diag.points.kurtosis": f"{max(r['excess_kurtosis'] for r in points):.2f}",
        "Diag.beta.skew": (f"{min(r['skew'] for r in beta):.2f}--"
                           f"{max(r['skew'] for r in beta):.2f}"),
    }
    with open(output_file, "w") as fh:
        for key, value in keys.items():
            fh.write(f"\\pgfkeyssetvalue{{{key}}}{{{value}}}\n")
    return keys


def write_latex_diagnostics(rob, output_file, only=None):
    """Supplementary LMM-robustness table as a bare ``tabular``.

    Follows ``write_latex_loggers``: no float wrapper, so the SI supplies
    ``\\captionof``/``\\label``. ``only`` restricts rows to a set of
    ``(variant, outcome)`` pairs.
    """
    variant_full = {"R": "Reliability", "V": "Volatility",
                    "T": "Controllability"}
    outcome_label = {"points": r"$y$ = points",
                     "first_rt_log": r"$y$ = log(firstRT)",
                     "total_rt_log": r"$y$ = log(totalRT)"}
    fitted = fitted_lmm()

    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Stochasticity & Model & Variable & $\beta$ (LMM) & $\beta$ (slope) "
        r"& $P$ (Wilcoxon) & $P$ (sign) & Same sign \\",
        r"\midrule",
    ]

    def _p(value):
        """Format a p-value the way the manuscript does elsewhere."""
        return r"$<0.001$" if value < 1e-3 else f"${value:.3f}$"

    for row in rob:
        if only is not None and (row["variant"], row["outcome"]) not in only:
            continue
        variant, outcome = row["variant"], row["outcome"]
        model = "Empirical" if outcome in outcome_label else outcome
        variable = outcome_label.get(outcome, r"$y$ = log($\beta$)")
        ref = fitted.get((variant, outcome))
        lmm_beta = f"${ref['beta']:.2f}$" if ref else "--"
        lines.append(
            f"{variant_full[variant]} & {model} & {variable} & "
            f"{lmm_beta} & ${row['mean_slope']:.2f}$ & "
            f"{_p(row['p_wilcoxon'])} & {_p(row['p_sign'])} & "
            f"{row['n_negative']}/{row['n']} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(output_file, "w") as fh:
        fh.write("\n".join(lines))


def main():
    RULE = "=" * 78
    out_dir = "figures/_diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    # Read once; every section below is a pure function of these Datasets.
    print("Loading outcomes ...", flush=True)
    data = load_outcomes()
    assert data, "no observations loaded; check the raw data and fit folders"
    n_obs = sum(v.size for ds in data.values() for v in ds.data_vars.values())
    print(f"  {n_obs:,} observations | {len(data)} variants | "
          f"{len(outcome_names(next(iter(data.values()))))} outcomes\n")

    # 1. Shape without a p-value: skew/kurtosis are sample-size independent.
    print(f"{RULE}\n1. RESIDUAL SHAPE  (cached LMM residuals)\n{RULE}\n"
          f"{'file':<58}{'n':>7}{'skew':>7}{'exkurt':>8}{'|z|>3':>8}")
    shape = residual_shape()
    for row in shape:
        print(f"{row['file']:<58}{row['n']:>7}{row['skew']:>7.2f}"
              f"{row['excess_kurtosis']:>8.2f}{row['frac_absz_gt3']:>8.4f}")
    print("  (normal reference: skew 0, excess kurtosis 0, |z|>3 = 0.0027)")

    # 2. What the log transform actually buys, as a number.
    print(f"\n{RULE}\n2. LOG TRANSFORM  (first-choice RT, raw seconds vs log)\n"
          f"{RULE}\n{'variant':<9}{'n':>7}{'raw skew':>11}{'raw kurt':>12}"
          f"{'log skew':>11}{'log kurt':>11}")
    transforms = []
    for variant, ds in data.items():
        c = rt_transform_check(ds, variant)
        transforms.append(c)
        print(f"{c['variant']:<9}{c['n']:>7}{c['raw_skew']:>11.1f}"
              f"{c['raw_kurtosis']:>12.1f}{c['log_skew']:>11.2f}"
              f"{c['log_kurtosis']:>11.2f}")

    # 3. The half of the checklist question that asks about equal variances.
    print(f"\n{RULE}\n3. EQUAL VARIANCES  (within-condition SD across "
          f"stochasticity levels)\n{RULE}")
    sd_ratios = []
    for variant, ds in data.items():
        for outcome in outcome_names(ds):
            da = ds[outcome]
            # SD of the observations themselves, not of participant means.
            extra = [d for d in da.dims if d != "condition"]
            sds = da.std(dim=extra, ddof=1).to_numpy()
            ratio = sds.max() / sds.min()
            sd_ratios.append((f"{variant}/{outcome}", ratio))
            spread = ", ".join(f"{x:.3f}" for x in sds)
            print(f"  {variant}/{outcome:<20} SD = [{spread}]  "
                  f"max/min = {ratio:.2f}")
    print("  (a ratio below ~2 is unremarkable for a mixed model)")

    # 4. Load-bearing: conclusions do not depend on normality at all.
    print(f"\n{RULE}\n4. DISTRIBUTION-FREE ROBUSTNESS  (participant as unit "
          f"of analysis)\n{RULE}\n"
          f"{'variant':<8}{'outcome':<20}{'n':>4}{'slope':>11}{'t':>9}"
          f"{'p(t)':>11}{'p(Wilcox)':>12}{'p(sign)':>11}{'neg':>6}")
    rob = []
    for variant, ds in data.items():
        for outcome in outcome_names(ds):
            rob.append(robustness_row(variant, outcome, ds[outcome]))
    for row in rob:
        print(f"{row['variant']:<8}{row['outcome']:<20}{row['n']:>4}"
              f"{row['mean_slope']:>11.3f}{row['t']:>9.2f}{row['p_t']:>11.1e}"
              f"{row['p_wilcoxon']:>12.1e}{row['p_sign']:>11.1e}"
              f"{row['n_negative']:>4}/{row['n']}")
    print("\n  The paired t reproduces the LMM t-statistic; Wilcoxon and sign "
          "assume nothing\n  about the residuals. For the log-beta rows the "
          "values are fitted parameters,\n  so those tests drop the normality "
          "assumption but not the two-stage one.")

    # Guard: a slope missing the LMM coefficient means a bad reconstruction.
    fitted = fitted_lmm()
    compared = [(r, fitted[(r["variant"], r["outcome"])]) for r in rob
                if (r["variant"], r["outcome"]) in fitted]
    if not compared:
        warnings.warn("no result_lmm.csv found; slopes were not checked against "
                      "the fitted models and the table's beta column will be '--'")
    drift = [f"{r['variant']}/{r['outcome']}: LMM {ref['beta']:+.3f} vs slope "
             f"{r['mean_slope']:+.3f}" for r, ref in compared
             if abs(r["mean_slope"] - ref["beta"]) > 0.02 * max(1.0, abs(ref["beta"]))]
    assert not drift, ("slope disagrees with the fitted LMM coefficient; suspect "
                       "the outcome reconstruction, not the model:\n  "
                       + "\n  ".join(drift))
    if compared:
        print(f"\n  All {len(compared)} slopes agree with the fitted LMM "
              "coefficients.")

    # Summary CSV for inspection; the .tex below is what the SI consumes.
    csv_path = f"{out_dir}/assumptions_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rob[0].keys()))
        writer.writeheader()
        writer.writerows(rob)

    # Generated beside loggers.tex, so the SI never \input{}s a stale copy.
    tex_path = "figures/diagnostics.tex"
    write_latex_diagnostics(rob, tex_path)
    keys_path = "figures/diagnostics_keys.tex"
    keys = write_latex_keys(shape, transforms, sd_ratios, keys_path)
    print("  keys: " + ", ".join(f"{k.split('.',1)[1]}={v}" for k, v in keys.items()))
    print(f"\nWrote {csv_path}, {tex_path}, {keys_path}")


if __name__ == "__main__":
    main()
