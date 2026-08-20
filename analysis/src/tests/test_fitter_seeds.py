"""RNG seeds for ``fitter.py``: uniqueness, deterministic assignment, reproducible draws.

Parallelism uses ``loky`` (separate processes). Seed integers depend only on
``base_seed``, model tuple, CV fold index, and ``cv``/``full`` purpose — **`n_jobs`** is
never passed into `_multistart_rng_seed`; it affects **scheduling** only. We prove
distinct seeds per combo, deterministic assignment, reproducible NumPy streams, and that
changing ``Parallel(n_jobs=...)`` leaves the parallel seed-vector identical.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from joblib import Parallel, delayed

_ANALYSIS_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _ANALYSIS_DIR / "scripts"
_SRC_DIR = _ANALYSIS_DIR / "src"

# Golden outputs for ``_GOLDEN_POSIX_PATH`` below (macOS / Linux ``abspath``).
# If the seed recipe intentionally changes, update these and note it in the changelog.
_GOLDEN_POSIX_PATH = "/unit/contract/foo_data.json"
_GOLDEN_BASE_SEED_LOCAL_OR_TASK0 = 946065185
_GOLDEN_BASE_SEED_ARRAY7 = 953065206
_GOLDEN_MULTISTART_CV = 3504553148
_GOLDEN_MULTISTART_FULL = 3323216802


def _load_fitter():
    """Load ``fitter`` as when run from ``analysis/scripts`` (its ``sys.path`` logic)."""
    for p in (_SRC_DIR, _SCRIPTS_DIR):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    old_cwd = os.getcwd()
    os.chdir(_SCRIPTS_DIR)
    try:
        spec = importlib.util.spec_from_file_location("fitter", _SCRIPTS_DIR / "fitter.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(old_cwd)


@pytest.fixture(scope="module")
def fitter_mod():
    return _load_fitter()


def _all_multistart_seeds(fitter_mod, base_seed: int, variant: str, n_folds: int):
    """Same seeds as produced along ``model_cross_validation`` for one (effort, filter, value)."""
    opts = fitter_mod.get_effort_filter_value_options(variant)
    effort_versions, filter_fns, value_fns = opts
    seeds: list[int] = []
    keys: list[tuple] = []
    for effort_version, filter_fn, value_fn in product(effort_versions, filter_fns, value_fns):
        fn = filter_fn.__name__
        vn = value_fn.__name__
        for fold_idx in range(n_folds):
            s = fitter_mod._multistart_rng_seed(
                base_seed,
                effort_version,
                fn,
                vn,
                purpose=b"cv|",
                purpose_idx=fold_idx,
            )
            seeds.append(s)
            keys.append((variant, effort_version, fn, vn, "cv", fold_idx))
        s = fitter_mod._multistart_rng_seed(
            base_seed,
            effort_version,
            fn,
            vn,
            purpose=b"full|",
            purpose_idx=0,
        )
        seeds.append(s)
        keys.append((variant, effort_version, fn, vn, "full", 0))
    return seeds, keys


def test_all_multistart_seeds_unique_per_cv_job(fitter_mod):
    """No duplicate ``rng_seed`` across parallel model tasks in one ``cv_all_models`` run.

    Parallelism is over ``(effort_version, filter_fn, value_fn)``; workers must each get a
    distinct multistart stream (no correlated restarts via identical ``default_rng`` state).
    """
    base = fitter_mod._data_file_base_seed("/fake/analysis/data_split/raw/R/user0_data.json", -1)

    for variant in ("R", "V", "T"):
        seeds, keys = _all_multistart_seeds(fitter_mod, base, variant, n_folds=5)
        if len(set(seeds)) != len(seeds):
            by_seed: dict[int, list[tuple]] = {}
            for s, k in zip(seeds, keys):
                by_seed.setdefault(s, []).append(k)
            dups = {hex(s): ks for s, ks in by_seed.items() if len(ks) > 1}
            raise AssertionError(
                "Duplicate multistart seeds — parallel workers would share RNG streams. "
                f"variant={variant}, collisions={dups}"
            )


def test_seed_functions_repeatable_on_identical_inputs(fitter_mod):
    """Pure functions: same inputs always yield the same seeds (no hidden state)."""
    p = "/consistent/repro/user0_data.json"
    for ai in (-1, 0, 3, 99):
        vals = [fitter_mod._data_file_base_seed(p, ai) for _ in range(50)]
        assert len(set(vals)) == 1
    base = 42
    vals = [
        fitter_mod._multistart_rng_seed(
            base,
            "policy_compress",
            "filter_depth",
            "value_max",
            purpose=b"cv|",
            purpose_idx=3,
        )
        for _ in range(50)
    ]
    assert len(set(vals)) == 1


def test_parallel_n_jobs_preserves_ordered_multistart_seeds(fitter_mod):
    """``Parallel(n_jobs=...)`` only schedules work — seed integers are untouched.

    ``cv_all_models`` never passes worker count into ``_multistart_rng_seed``.
    Dispatching pure seed recomputation via ``Parallel`` with several ``n_jobs`` must yield
    the **same ordered vector** as ``_all_multistart_seeds`` (covers every CV fold + full-refit seed).

    **Threading backend:** avoids ``loky`` pickle issues with the dynamically loaded ``fitter``
    module in pytest; numerical seed derivation is unchanged in production worker processes.

    Numerical optimizer output can still differ slightly with ``backend="loky"`` vs threading
    (BLAS races, sparse ties on ``numpy.random.choice`` elsewhere in ``BehaviorModel``);
    assigned multistart **seeds** are independent of parallelism degree regardless.
    """
    variant = "R"
    base = fitter_mod._data_file_base_seed("/parallel/n_jobs/invariant_check.json", -1)
    sequential, meta = _all_multistart_seeds(fitter_mod, base, variant, n_folds=5)

    def reconstruct_seed(ix: int) -> int:
        _, effort, fn, vn, tag, fold_or_unused = meta[ix]
        if tag == "cv":
            return fitter_mod._multistart_rng_seed(
                base, effort, fn, vn, purpose=b"cv|", purpose_idx=fold_or_unused
            )
        return fitter_mod._multistart_rng_seed(
            base, effort, fn, vn, purpose=b"full|", purpose_idx=0
        )

    index_range = range(len(sequential))
    cpus = os.cpu_count() or 4
    for n_jobs in sorted({1, 2, min(8, max(2, cpus))}):
        par = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(reconstruct_seed)(i) for i in index_range
        )
        assert par == sequential, (
            f"Parallel(backend=threading, n_jobs={n_jobs}) differs from sequential seeds "
            f"(unexpected worker coupling to seed assignment)."
        )


def test_worker_count_absent_from_seed_helpers_source(fitter_mod):
    """Static guardrail: regressions that tie seeds to worker id should jump out."""
    import inspect

    combined = "".join(
        (
            inspect.getsource(fitter_mod._data_file_base_seed),
            inspect.getsource(fitter_mod._multistart_rng_seed),
        )
    )
    assert "n_jobs" not in combined
    assert "Parallel" not in combined


def test_seed_assignment_stable_across_fitter_reload():
    """Fresh ``exec_module`` still assigns the same integers (no import-time randomness)."""
    a = _load_fitter()
    b = _load_fitter()
    path = "/reload/stability/check_data.json"
    for ai in (-1, 0):
        assert a._data_file_base_seed(path, ai) == b._data_file_base_seed(path, ai)
    base = a._data_file_base_seed(path, 0)
    expect = a._multistart_rng_seed(
        base, "filter_adapt", "filter_rank", "value_sum", purpose=b"cv|", purpose_idx=2
    )
    assert (
        b._multistart_rng_seed(
            base, "filter_adapt", "filter_rank", "value_sum", purpose=b"cv|", purpose_idx=2
        )
        == expect
    )


@pytest.mark.skipif(sys.platform == "win32", reason="``abspath`` + golden integers are POSIX-oriented")
def test_golden_base_and_multistart_integers_contract(fitter_mod):
    """Pin exact 32‑bit seeds so accidental recipe changes fail CI."""
    assert fitter_mod._data_file_base_seed(_GOLDEN_POSIX_PATH, -1) == _GOLDEN_BASE_SEED_LOCAL_OR_TASK0
    assert fitter_mod._data_file_base_seed(_GOLDEN_POSIX_PATH, 0) == _GOLDEN_BASE_SEED_LOCAL_OR_TASK0
    assert fitter_mod._data_file_base_seed(_GOLDEN_POSIX_PATH, 7) == _GOLDEN_BASE_SEED_ARRAY7
    b = _GOLDEN_BASE_SEED_LOCAL_OR_TASK0
    assert (
        fitter_mod._multistart_rng_seed(
            b,
            "filter_adapt",
            "filter_depth",
            "value_path",
            purpose=b"cv|",
            purpose_idx=0,
        )
        == _GOLDEN_MULTISTART_CV
    )
    assert (
        fitter_mod._multistart_rng_seed(
            b,
            "filter_adapt",
            "filter_depth",
            "value_path",
            purpose=b"full|",
            purpose_idx=0,
        )
        == _GOLDEN_MULTISTART_FULL
    )


def test_default_rng_stream_reproduces_from_same_rng_seed(fitter_mod):
    """``fit_model`` uses ``default_rng(rng_seed)``; same seed ⇒ same float stream."""
    rng_seed = fitter_mod._multistart_rng_seed(
        _GOLDEN_BASE_SEED_LOCAL_OR_TASK0,
        "filter_adapt",
        "filter_depth",
        "value_path",
        purpose=b"cv|",
        purpose_idx=1,
    )
    n_draws = 500
    draws_a = np.empty(n_draws)
    draws_b = np.empty(n_draws)
    rng_a = np.random.default_rng(rng_seed)
    rng_b = np.random.default_rng(rng_seed)
    for i in range(n_draws):
        draws_a[i] = rng_a.uniform(0.0, 1.0)
        draws_b[i] = rng_b.uniform(0.0, 1.0)
    np.testing.assert_array_equal(draws_a, draws_b)


def test_different_models_get_different_rng_streams(fitter_mod):
    """Distinct seeds ⇒ first multistart draw vectors differ (not just cosmetic uniqueness)."""
    base = 12345
    s1 = fitter_mod._multistart_rng_seed(
        base,
        "filter_adapt",
        "filter_depth",
        "value_path",
        purpose=b"cv|",
        purpose_idx=0,
    )
    s2 = fitter_mod._multistart_rng_seed(
        base,
        "policy_compress",
        "filter_depth",
        "value_path",
        purpose=b"cv|",
        purpose_idx=0,
    )
    assert s1 != s2
    names = ["lapse", "inv_temp"]
    g1 = np.random.default_rng(s1).uniform(
        low=[fitter_mod.DEFAULT_BOUNDS[n][0] for n in names],
        high=[fitter_mod.DEFAULT_BOUNDS[n][1] for n in names],
    )
    g2 = np.random.default_rng(s2).uniform(
        low=[fitter_mod.DEFAULT_BOUNDS[n][0] for n in names],
        high=[fitter_mod.DEFAULT_BOUNDS[n][1] for n in names],
    )
    assert not np.allclose(g1, g2)


def test_base_seed_and_path_affect_streams(fitter_mod):
    """Different files / array shards should not silently reuse the same multistart seed."""
    b0 = fitter_mod._data_file_base_seed("/a/x_data.json", 0)
    b1 = fitter_mod._data_file_base_seed("/a/y_data.json", 0)
    b2 = fitter_mod._data_file_base_seed("/a/x_data.json", 1)
    assert len({b0, b1, b2}) == 3

    s0 = fitter_mod._multistart_rng_seed(
        b0, "filter_adapt", "filter_depth", "value_EV", purpose=b"full|", purpose_idx=0
    )
    s1 = fitter_mod._multistart_rng_seed(
        b1, "filter_adapt", "filter_depth", "value_EV", purpose=b"full|", purpose_idx=0
    )
    s2 = fitter_mod._multistart_rng_seed(
        b2, "filter_adapt", "filter_depth", "value_EV", purpose=b"full|", purpose_idx=0
    )
    assert len({s0, s1, s2}) == 3
