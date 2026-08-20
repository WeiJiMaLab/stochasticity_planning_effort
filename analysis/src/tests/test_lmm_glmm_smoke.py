"""Integration smoke tests for lmm() and glmm() with synthetic data.

These tests exercise the actual pymer4/R code paths end-to-end with small
datasets, catching issues like wrong column names or missing attributes before
a full pipeline run. Each test must complete in under 3 minutes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _SRC_DIR)

from analysis import lmm, glmm  # noqa: E402
from utils import format_lmm_result, format_glmm_main_effect, format_glmm_interaction  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_lmm_df(n_participants: int = 30, n_conditions: int = 5, seed: int = 0) -> pd.DataFrame:
    """LMM schema: y (continuous), conditions (numeric), participants (int)."""
    rng = np.random.default_rng(seed)
    cond_vals = np.linspace(0, 1, n_conditions)
    rows = []
    for pid in range(n_participants):
        intercept = rng.normal(0, 0.3)
        slope = rng.normal(-0.5, 0.1)
        for c in cond_vals:
            y = intercept + slope * c + rng.normal(0, 0.2)
            rows.append({"y": y, "conditions": c, "participants": pid})
    return pd.DataFrame(rows)


def _make_glmm_df(n_participants: int = 30, n_conditions: int = 5, seed: int = 1) -> pd.DataFrame:
    """GLMM schema: y (0/1), x (continuous), conditions (numeric), participants (int)."""
    rng = np.random.default_rng(seed)
    cond_vals = np.linspace(0, 1, n_conditions)
    rows = []
    for pid in range(n_participants):
        intercept = rng.normal(0, 0.5)
        for c in cond_vals:
            for _ in range(6):
                x = rng.normal(0, 2)
                logit = intercept + 0.4 * x - 0.3 * c
                p = 1 / (1 + np.exp(-logit))
                y = int(rng.binomial(1, p))
                rows.append({"y": y, "x": x, "conditions": c, "participants": pid})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LMM smoke tests
# ---------------------------------------------------------------------------

class TestLMMSmoke:
    @pytest.fixture(scope="class")
    def lmm_result(self):
        df = _make_lmm_df()
        result, logs = lmm(df)
        return result, logs

    def test_returns_two_values(self, lmm_result):
        result, logs = lmm_result
        assert isinstance(result, dict)
        assert isinstance(logs, list)

    def test_required_keys_present(self, lmm_result):
        result, _ = lmm_result
        for key in ("beta", "tstat", "dof", "pval", "ci_lower", "ci_upper", "residuals", "modeltype"):
            assert key in result, f"Missing key: {key}"

    def test_ci_brackets_beta(self, lmm_result):
        result, _ = lmm_result
        assert result["ci_lower"] < result["beta"] < result["ci_upper"], (
            f"beta {result['beta']:.3f} not inside CI "
            f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        )

    def test_ci_symmetric(self, lmm_result):
        result, _ = lmm_result
        half_width_lo = result["beta"] - result["ci_lower"]
        half_width_hi = result["ci_upper"] - result["beta"]
        assert abs(half_width_lo - half_width_hi) < 1e-8, "CI should be symmetric around beta"

    def test_residuals_is_nonempty_list_of_floats(self, lmm_result):
        result, _ = lmm_result
        resids = result["residuals"]
        assert isinstance(resids, list) and len(resids) > 0
        assert all(isinstance(r, (int, float)) for r in resids), "residuals must be numeric"

    def test_residuals_sum_near_zero(self, lmm_result):
        result, _ = lmm_result
        assert abs(np.sum(result["residuals"])) < 1.0, "LMM residuals should sum near zero"

    def test_format_lmm_result(self, lmm_result):
        result, _ = lmm_result
        s = format_lmm_result(result)
        assert "95\\% CI" in s
        assert "P" in s
        assert "beta" not in s.lower() or "$\\beta" in s


# ---------------------------------------------------------------------------
# GLMM smoke tests
# ---------------------------------------------------------------------------

class TestGLMMSmoke:
    @pytest.fixture(scope="class")
    def glmm_result(self):
        df = _make_glmm_df()
        result, logs = glmm(df)
        return result, logs

    def test_returns_two_values(self, glmm_result):
        result, logs = glmm_result
        assert isinstance(result, dict)
        assert isinstance(logs, list)

    def test_required_keys_present(self, glmm_result):
        result, _ = glmm_result
        for key in (
            "beta_main", "chi2_main", "pval_main",
            "beta_inter", "chi2_inter", "pval_inter",
            "ci_lower_main", "ci_upper_main",
            "ci_lower_inter", "ci_upper_inter",
            "modeltype",
        ):
            assert key in result, f"Missing key: {key}"

    def test_ci_main_brackets_beta(self, glmm_result):
        result, _ = glmm_result
        assert result["ci_lower_main"] < result["beta_main"] < result["ci_upper_main"]

    def test_ci_inter_brackets_beta(self, glmm_result):
        result, _ = glmm_result
        assert result["ci_lower_inter"] < result["beta_inter"] < result["ci_upper_inter"]

    def test_format_glmm_main(self, glmm_result):
        result, _ = glmm_result
        s = format_glmm_main_effect(result)
        assert "95\\% CI" in s
        assert "P" in s

    def test_format_glmm_interaction(self, glmm_result):
        result, _ = glmm_result
        s = format_glmm_interaction(result)
        assert "95\\% CI" in s
        assert "P" in s
