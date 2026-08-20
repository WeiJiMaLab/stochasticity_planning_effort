"""Guardrail tests for analysis utilities (audit-driven).

Covers: ``clean_model_name`` ordering, ``get_user_data`` raw-game counts,
``bootstrap`` shapes, ``get_fits_and_params`` assembly, ``plot_model_comparison``
baseline validation, inverse-temperature plot labeling, BMC-style matrix
preflight invariants, and ``strsimplify`` edge cases.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _SRC_DIR)

from analysis import Analyzer, bootstrap, clean_model_name, get_fits_and_params  # noqa: E402
from utils import get_stochasticity_levels, get_user_data, strsimplify  # noqa: E402


def _games_payload(n_nonpractice: int, n_practice: int = 0) -> list[dict]:
    games = [{"name": f"practice_{j}", "p": 0.0, "trials": [{"rt": 100}]} for j in range(n_practice)]
    games.extend(
        {"name": f"game_{i}", "p": 0.25, "trials": [{"rt": 500}]} for i in range(n_nonpractice)
    )
    return games


def test_clean_model_name_ordering_contract():
    """``filter_adapt`` must shorten before ``filter_`` stripping (comment in ``analysis.py``)."""
    assert clean_model_name("policy_compress.filter_depth.value_sum") == "PC depth sum"
    assert clean_model_name("filter_adapt.filter_rank.value_EV") == "FA rank EV"


def test_bootstrap_shape_and_sampling():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(7)
    out = bootstrap(x, n=200)
    assert out.shape == (200, 7)
    assert np.all(np.isin(out.ravel(), x))


def test_strsimplify_near_integer_and_float_noise():
    assert strsimplify(0.5499999999999999) == "0.55"
    assert strsimplify(3.0) == "3"
    assert strsimplify(np.nan) == "nan"


def test_get_user_data_raw_requires_150_nonpractice_games(tmp_path):
    raw_root = tmp_path / "raw"
    variant_dir = raw_root / "R"
    variant_dir.mkdir(parents=True)
    user_file = variant_dir / "user0_data.json"
    user_file.write_text(json.dumps(_games_payload(149)), encoding="utf-8")

    with pytest.raises(AssertionError, match="150"):
        get_user_data("user0", "R", str(raw_root))


def test_get_user_data_raw_accepts_150_after_practice_filtered(tmp_path):
    raw_root = tmp_path / "raw"
    variant_dir = raw_root / "R"
    variant_dir.mkdir(parents=True)
    payload = _games_payload(150, n_practice=2)
    (variant_dir / "user0_data.json").write_text(json.dumps(payload), encoding="utf-8")

    games = get_user_data("user0", "R", str(raw_root))
    assert len(games) == 150
    assert all("practice" not in g["name"] for g in games)


def test_get_user_data_non_raw_skips_game_count_assert(tmp_path):
    """Folders whose basename is not ``raw`` do not enforce the 150-game rule."""
    other = tmp_path / "simulated.foo"
    variant_dir = other / "R"
    variant_dir.mkdir(parents=True)
    (variant_dir / "user0_data.json").write_text(json.dumps(_games_payload(3)), encoding="utf-8")

    games = get_user_data("user0", "R", str(other))
    assert len(games) == 3


def test_get_fits_and_params_groups_by_clean_model_field(tmp_path):
    fit_dir = tmp_path
    inner_key = "policy_compress.filter_depth.value_sum"
    cleaned = clean_model_name(inner_key)
    payload = {
        inner_key: {
            "model": cleaned,
            "NLL_CV": 42.5,
            "full_params": {"lapse": 0.05},
        }
    }
    (fit_dir / "sim_user0_data.json").write_text(json.dumps(payload), encoding="utf-8")

    fits, params = get_fits_and_params(str(fit_dir))
    assert cleaned in fits
    assert fits[cleaned]["nll"].iloc[0] == 42.5
    assert fits[cleaned]["model"].iloc[0] == cleaned
    assert params[cleaned]["sim_user0"]["lapse"] == 0.05


def _bare_analyzer(model_data: dict, variant: str = "R") -> Analyzer:
    a = object.__new__(Analyzer)
    a.model_data = model_data
    a.variant = variant
    a.conditions = get_stochasticity_levels(variant)
    from utils import get_colormap

    a.colors = get_colormap(variant)
    return a


def test_plot_model_comparison_asserts_unknown_baseline():
    fig, ax = plt.subplots(figsize=(4, 3))
    a = _bare_analyzer(
        {
            "m_a": pd.DataFrame({"nll": [0.0, 0.1]}),
            "m_b": pd.DataFrame({"nll": [1.0, 1.5]}),
        }
    )

    with pytest.raises(AssertionError, match="not in model_data"):
        a.plot_model_comparison("missing_model", n_bootstrap=30, ax=ax)

    plt.close(fig)


def test_plot_model_comparison_runs_with_valid_baseline():
    fig, ax = plt.subplots(figsize=(4, 3))
    a = _bare_analyzer(
        {
            "baseline_model": pd.DataFrame({"nll": [0.0, 0.0]}),
            "other_model": pd.DataFrame({"nll": [1.0, 2.0]}),
        }
    )
    a.plot_model_comparison("baseline_model", n_bootstrap=40, ax=ax)
    plt.close(fig)


def test_plot_stochasticity_vs_conditional_inv_temp_default_ylabel():
    rows = []
    for player in ("u0", "u1"):
        row = {"player": player, "nll": 0.0}
        for i in range(5):
            row[f"condition_inv_temp_{i}"] = 0.5 + 0.1 * i
        rows.append(row)
    df_b = pd.DataFrame(rows)

    a = _bare_analyzer({"fit_model": df_b}, variant="R")
    fig, ax = plt.subplots(figsize=(4, 3))
    a.plot_stochasticity_vs_conditional_inv_temp("fit_model", ax=ax)
    ylab = ax.get_ylabel()
    assert r"\beta" in ylab or "β" in ylab
    plt.close(fig)


def test_random_effects_bmc_preflight_invariants():
    """Same logical checks as ``analysis/workflows/analysis_randeffects_bmc.py`` before ``GroupBMC``."""
    ok = pd.DataFrame({"m1": [1.0, 2.0], "m2": [3.0, 4.0]}, index=["u0", "u1"])
    assert not ok.isna().any().any()
    assert ok.index.is_unique

    bad_nan = pd.DataFrame({"m1": [1.0, np.nan]}, index=["u0", "u1"])
    with pytest.raises(AssertionError):
        assert not bad_nan.isna().any().any()

    bad_dup = pd.DataFrame({"m1": [1.0, 2.0]}, index=["u0", "u0"])
    with pytest.raises(AssertionError):
        assert bad_dup.index.is_unique
