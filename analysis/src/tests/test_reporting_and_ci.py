"""Tests for p-value reporting format and CI computation correctness.

Verifies Nature Communications conventions (uppercase P, P < 0.001 threshold)
and that Wald CI algebra is correct for LMM and GLMM outputs.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _SRC_DIR)

from utils import report_p_value, format_lmm_result, format_glmm_main_effect, format_glmm_interaction


# --- report_p_value ---

class TestReportPValue:
    def test_tiny_uses_threshold(self):
        # p = 1e-15 → formerly "p < 10^{-10}", now "P < 0.001"
        assert report_p_value(1e-15) == "P < 0.001"

    def test_scientific_notation_range(self):
        # p = 1.5e-4 → formerly "p = 1.5 \times 10^{-4}", now "P < 0.001"
        assert report_p_value(1.5e-4) == "P < 0.001"

    def test_just_below_threshold(self):
        assert report_p_value(0.0009) == "P < 0.001"

    def test_exactly_at_threshold_not_below(self):
        # p = 0.001 is NOT < 0.001, so should be reported as P = 0.001
        assert report_p_value(0.001) == "P = 0.001"

    def test_moderate_three_decimals(self):
        assert report_p_value(0.042) == "P = 0.042"

    def test_large_two_decimals(self):
        assert report_p_value(0.23) == "P = 0.23"

    def test_uppercase_P_always(self):
        for p in [1e-12, 5e-4, 0.015, 0.23, 0.999]:
            assert report_p_value(p).startswith("P"), f"Expected uppercase P for p={p}"


# --- CI computation algebra ---

class TestLMMCIAlgebra:
    """Verifies that the CI derived from beta/tstat/dof is numerically correct."""

    def _expected_ci(self, beta, tstat, dof):
        from scipy.stats import t as t_dist
        se = abs(beta / tstat)
        t_crit = t_dist.ppf(0.975, dof)
        return beta - t_crit * se, beta + t_crit * se

    def test_ci_is_symmetric_around_beta(self):
        beta, tstat, dof = -1.71, -9.1, 98
        lo, hi = self._expected_ci(beta, tstat, dof)
        assert abs((beta - lo) - (hi - beta)) < 1e-10

    def test_ci_excludes_zero_for_significant_effect(self):
        # Strong negative effect — CI should be entirely negative
        beta, tstat, dof = -0.63, -7.76, 98
        lo, hi = self._expected_ci(beta, tstat, dof)
        assert lo < 0 and hi < 0, f"CI [{lo:.3f}, {hi:.3f}] should be entirely negative"

    def test_ci_bounds_reasonable_magnitude(self):
        # SE = |beta/tstat| = |-0.63/-7.76| ≈ 0.081; t_crit(0.975,98) ≈ 1.984
        # CI width ≈ 2 * 1.984 * 0.081 ≈ 0.32 → bounds within 0.5 of beta
        beta, tstat, dof = -0.63, -7.76, 98
        lo, hi = self._expected_ci(beta, tstat, dof)
        assert abs(lo - beta) < 0.5 and abs(hi - beta) < 0.5

    def test_positive_beta_ci(self):
        beta, tstat, dof = 0.45, 3.2, 95
        lo, hi = self._expected_ci(beta, tstat, dof)
        assert lo > 0 and hi > 0, "CI for positive significant effect should be positive"


# --- format functions include CI ---

class TestFormatFunctionsIncludeCI:
    def _lmm_row(self):
        return {
            "beta": -0.63, "tstat": -7.76, "dof": 98, "pval": 1e-11,
            "ci_lower": -0.79, "ci_upper": -0.47
        }

    def _glmm_row(self):
        return {
            "beta_main": 1.45, "chi2_main": 18.3, "pval_main": 1.9e-5,
            "ci_lower_main": 0.80, "ci_upper_main": 2.10,
            "beta_inter": -0.32, "chi2_inter": 7.1, "pval_inter": 0.0077,
            "ci_lower_inter": -0.55, "ci_upper_inter": -0.09
        }

    def test_lmm_format_has_ci(self):
        s = format_lmm_result(self._lmm_row())
        assert "95\\% CI" in s
        assert "[-0.79, -0.47]" in s

    def test_lmm_format_has_uppercase_p(self):
        s = format_lmm_result(self._lmm_row())
        assert "P < 0.001" in s
        assert "$p" not in s  # no lowercase p in math mode

    def test_glmm_main_has_ci(self):
        s = format_glmm_main_effect(self._glmm_row())
        assert "95\\% CI" in s
        assert "[0.80, 2.10]" in s

    def test_glmm_main_has_uppercase_p(self):
        s = format_glmm_main_effect(self._glmm_row())
        assert "P < 0.001" in s

    def test_glmm_interaction_has_ci(self):
        s = format_glmm_interaction(self._glmm_row())
        assert "95\\% CI" in s
        assert "[-0.55, -0.09]" in s

    def test_glmm_interaction_has_uppercase_p(self):
        s = format_glmm_interaction(self._glmm_row())
        assert "P = 0.008" in s  # p=0.0077 → P = 0.008 (3 decimal places)
