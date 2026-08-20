"""Reference tests for ``modelvalues.value_EV`` (reliability vs controllability semantics).

Run with the project convention (``cd analysis/src && pytest tests/``).
"""
import numpy as np
import pytest
import xarray as xr

from modelvalues import value_EV, value_path


def _tiny_triangle_pov(condition_key: float, bottom_row: tuple):
    """Single games/trials slice, 3×3 lower-triangular POV; decision at ``rows == 1``."""
    d0, d1, d2 = bottom_row
    base = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [d0, d1, d2]],
        dtype=float,
    )
    return xr.DataArray(
        base.reshape(1, 1, 1, 3, 3),
        dims=["conditions", "games", "trials", "rows", "cols"],
        coords={
            "conditions": [condition_key],
            "games": [0],
            "trials": [0],
            "rows": np.arange(3),
            "cols": np.arange(3),
        },
    )


def test_value_ev_reliability_two_step_reference():
    """R: mystery mean *E*, deterministic moves; first-split contrast = subtree total diff."""
    arr = _tiny_triangle_pov(0.25, (10.0, 0.0, 0.0))
    p_mystery = 0.2
    E = 5.0
    q = 1 - p_mystery
    vd = value_EV(arr, variant="R", value_params={"exp_value": E, 0.25: p_mystery})
    t20 = q * 10 + (1 - q) * E
    t21 = (1 - q) * E
    t22 = (1 - q) * E
    t10 = q * 0 + (1 - q) * E + max(t20, t21)
    t11 = q * 0 + (1 - q) * E + max(t21, t22)
    expected = t10 - t11
    assert np.isclose(float(vd.squeeze().values), expected)


def test_value_ev_controllability_two_step_reference():
    """T: move flips; reported logit contrast = (2q - 1) * (A - B) for row-1 subtree totals A, B."""
    arr = _tiny_triangle_pov(0.25, (10.0, 0.0, 0.0))
    p_flip = 0.2
    q = 1 - p_flip
    vd = value_EV(arr, variant="T", value_params={0.25: p_flip})
    b20, b21, b22 = 10.0, 0.0, 0.0
    vl0 = q * b20 + (1 - q) * b21
    vr0 = (1 - q) * b20 + q * b21
    A = max(vl0, vr0)
    vl1 = q * b21 + (1 - q) * b22
    vr1 = (1 - q) * b21 + q * b22
    B = max(vl1, vr1)
    expected = (2 * q - 1) * (A - B)
    assert np.isclose(float(vd.squeeze().values), expected)


def test_value_ev_reliability_zero_mystery_matches_value_path():
    arr = _tiny_triangle_pov(0.25, (10.0, 0.0, 0.0))
    vp = value_path(arr, variant="R")
    ve = value_EV(arr, variant="R", value_params={"exp_value": 5.0, 0.25: 0.0})
    assert np.isclose(float(vp.squeeze().values), float(ve.squeeze().values))


def test_value_ev_controllability_zero_flip_matches_value_path():
    arr = _tiny_triangle_pov(0.25, (10.0, 0.0, 0.0))
    vp = value_path(arr, variant="T")
    ve = value_EV(arr, variant="T", value_params={0.25: 0.0})
    assert np.isclose(float(vp.squeeze().values), float(ve.squeeze().values))


def test_value_ev_rejects_volatility_variant():
    arr = _tiny_triangle_pov(0.25, (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="value_EV supports"):
        value_EV(arr, variant="V")
