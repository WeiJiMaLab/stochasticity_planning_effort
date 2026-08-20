import glob
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
from prodict import Prodict
from itertools import product
import warnings
from joblib import Parallel, delayed
from tqdm.auto import tqdm
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 
from utils import NpEncoder
from modeling import BehaviorModel, DEFAULT_BOUNDS, get_effort_filter_value_options


def _data_file_base_seed(data_file: str, array_idx: int) -> int:
    """32-bit seed derived from absolute path and Slurm array index (0 if not in an array)."""
    path_key = os.path.abspath(data_file).encode()
    h = hashlib.sha256(path_key).digest()[:8]
    file_part = int.from_bytes(h, "big") & 0xFFFFFFFF
    shard = max(0, array_idx)
    return (file_part + shard * 1_000_003) % (2**32)


def _multistart_rng_seed(
    base_seed: int,
    effort_version: str,
    filter_fn_name: str,
    value_fn_name: str,
    *,
    purpose: bytes,
    purpose_idx: int,
) -> int:
    """Deterministic 32-bit seed for one ``fit_model`` multistart block (joblib-safe)."""
    payload = (
        base_seed.to_bytes(4, "big")
        + effort_version.encode()
        + b"|"
        + filter_fn_name.encode()
        + b"|"
        + value_fn_name.encode()
        + purpose
        + purpose_idx.to_bytes(4, "big")
    )
    return zlib.crc32(payload) & 0xFFFFFFFF


def _check_train_test_overlap(train_games, test_games):
    """Raise an error if any game appears in both train and test."""
    train_game_names = {game["name"] for game in train_games}
    test_game_names = {game["name"] for game in test_games}
    overlap = train_game_names.intersection(test_game_names)
    assert len(overlap) == 0, "Error: train and test data have overlapping names"

def get_train_test_data(user_file_path):
    """Return round-robin train/test splits (one pair of lists per fold)."""
    with open(user_file_path, "r") as f:
        data = json.load(f)

    n_folds = len(data.keys())
    test_splits = []
    train_splits = []

    for fold_idx in range(n_folds):
        test_games = data[f"fold_{fold_idx}"]
        train_games = [
            game
            for other_fold_idx in range(n_folds)
            if fold_idx != other_fold_idx
            for game in data[f"fold_{other_fold_idx}"]
        ]
        _check_train_test_overlap(train_games, test_games)
        test_splits.append(test_games)
        train_splits.append(train_games)

    # Reconstruct the full user dataset from all folds in explicit index order.
    full_data = []
    for fold_idx in range(n_folds):
        full_data.extend(data[f"fold_{fold_idx}"])

    return train_splits, test_splits, full_data

def fit_model(games, effort_version, filter_fn, value_fn, variant, n_multistarts, rng_seed: int):
    # Fit one model with multi-start random initializations and keep the best run.
    # ``rng_seed`` seeds a Generator in-process so loky/joblib workers are reproducible
    # without relying on inherited global RNG state.
    rng = np.random.default_rng(rng_seed)
    train_model = BehaviorModel(
        effort_version=effort_version,
        filter_fn=filter_fn,
        value_fn=value_fn,
        variant=variant,
        games=games,
    )

    if train_model.conditional_inv_temp:
        inv_temp_param_names = [f"condition_inv_temp_{i}" for i in range(5)]
        trainable_param_names = ["lapse", *inv_temp_param_names]
    else:
        trainable_param_names = ["lapse", "inv_temp"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit_results = []
        for _ in range(n_multistarts):
            try:
                fit_results.append(
                    Prodict(
                        train_model.fit(
                            {
                                name: float(rng.uniform(*DEFAULT_BOUNDS[name]))
                                for name in trainable_param_names
                            }
                        )
                    )
                )
            except (ValueError, RuntimeError):
                fit_results.append(Prodict({"nll": np.inf}))

    # Keep only the best finite solution across random restarts.
    best_params = min(fit_results, key=lambda result: result.nll)
    if not np.isfinite(best_params.nll):
        raise RuntimeError(
            f"All {n_multistarts} restarts failed for {train_model.name} ({variant})."
        )

    del best_params["nll"]
    return train_model, dict(best_params)


def train_test_model(
    train_games,
    test_games,
    effort_version,
    filter_fn,
    value_fn,
    variant,
    n_multistarts,
    rng_seed: int,
):
    """Fit a model on train games and evaluate NLL on train and test."""
    train_model, best_params = fit_model(
        train_games, effort_version, filter_fn, value_fn, variant, n_multistarts, rng_seed
    )

    # Evaluate on an independently-instantiated model bound to held-out games.
    test_model = BehaviorModel(
        effort_version=effort_version,
        filter_fn=filter_fn,
        value_fn=value_fn,
        variant=variant,
        games=test_games,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_params["NLL_train"] = train_model.evaluate_NLL(best_params)
        best_params["NLL_val"] = test_model.evaluate_NLL(best_params)
    return dict(best_params)


def model_cross_validation(
    train_splits,
    test_splits,
    full_data,
    effort_version,
    filter_fn,
    value_fn,
    variant,
    n_multistarts,
    base_seed: int,
):
    """Run cross-validation for a single (effort, filter, value) model."""
    filter_name = filter_fn.__name__
    value_name = value_fn.__name__
    fold_results = []
    for fold_idx, (train_games, test_games) in enumerate(zip(train_splits, test_splits)):
        fold_seed = _multistart_rng_seed(
            base_seed,
            effort_version,
            filter_name,
            value_name,
            purpose=b"cv|",
            purpose_idx=fold_idx,
        )
        fold_result = train_test_model(
            train_games,
            test_games,
            effort_version,
            filter_fn,
            value_fn,
            variant,
            n_multistarts,
            fold_seed,
        )
        fold_results.append(fold_result)

    # Also fit once on the entire dataset to export final parameters.
    full_seed = _multistart_rng_seed(
        base_seed,
        effort_version,
        filter_name,
        value_name,
        purpose=b"full|",
        purpose_idx=0,
    )
    _, full_params = fit_model(
        full_data, effort_version, filter_fn, value_fn, variant, n_multistarts, full_seed
    )

    fold_results.sort(key=lambda result: result["NLL_val"])
    cv_summary = {
        "model": fold_results[0]["model"],
        "NLL_CV": sum(result["NLL_val"] for result in fold_results),
        "best_fold": fold_results[0],
        "fold_data": fold_results,
        "full_params": full_params,
    }
    return cv_summary


def cv_all_models(
    train_splits, test_splits, full_data, variant, n_multistarts, base_seed: int, n_jobs=-1
):
    """Run cross-validation for all (effort, filter, value) model combinations."""

    # retrieve all possible options for effort_versions, filter_fns, and value_fns
    effort_versions, filter_fns, value_fns = get_effort_filter_value_options(variant)
    all_model_options = list(product(effort_versions, filter_fns, value_fns))

    cv_results = Parallel(n_jobs=n_jobs)(
        delayed(model_cross_validation)(
            train_splits,
            test_splits,
            full_data,
            effort_version,
            filter_fn,
            value_fn,
            variant,
            n_multistarts,
            base_seed,
        )
        for effort_version, filter_fn, value_fn in all_model_options
    )
    return {result["model"]: result for result in cv_results}


if __name__ == "__main__":
    n_multistarts = 50
    
    # SLURM configuration (falls back to local defaults when not running under SLURM)
    array_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    n_parallel = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    all_data = sorted(glob.glob("../data_split/*/*/*"))
    np.random.RandomState(42).shuffle(all_data)
    
    if array_idx == -1: 
        data_to_process = all_data
    else: 
        data_to_process = all_data[array_idx::100]

    for data_file in tqdm(data_to_process):
        fit_path = data_file.replace("data_split/", "fit/")

        # skip if the fit file already exists
        if os.path.isfile(fit_path):
            continue
        
        _, variant, _ = data_file.removeprefix("../data_split/").removesuffix(f"_data.json").split("/")
        train_splits, test_splits, full_data = get_train_test_data(data_file)
        base_seed = _data_file_base_seed(data_file, array_idx)
        cv_result = cv_all_models(
            train_splits,
            test_splits,
            full_data,
            variant,
            n_multistarts,
            base_seed,
            n_jobs=n_parallel,
        )

        os.makedirs(str(Path(data_file).parent).replace("data_split/", "fit/"), exist_ok=True)
        with open(fit_path, "w") as f:
            json.dump(cv_result, f, cls=NpEncoder)