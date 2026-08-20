import numpy as np
import xarray as xr
import pytest 
import os
import sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from modeling import (
    BehaviorModel, 
    filter_depth,
    filter_rank,
    filter_value,
    value_EV,
    value_path, 
    value_max,
    value_sum,
    value_levelmean,
)
from utils import make_pov_array, get_stochasticity_levels
from prodict import Prodict

def boards_paths_to_games(boards: xr.DataArray, paths: xr.DataArray, type_: str):
    games = []
    stochasticity_levels = get_stochasticity_levels(type_)
    for cond_idx, condition in enumerate(stochasticity_levels):
        for game_idx in range(boards.sizes["games"]):
            game_boards = boards.isel(conditions=cond_idx, games=game_idx).values
            game_paths = paths.isel(conditions=cond_idx, games=game_idx).values
            n_trials = len(game_paths)
            # actions and is_transition should match pov_array trials (len-1)
            actions = [False] * (n_trials - 1)
            is_transition = [False] * (n_trials - 1)
            game = Prodict({
                "p": condition,
                "boards": game_boards.tolist(),
                "tuplepath": game_paths.tolist(),
                "actions": actions,
                "is_transition": is_transition,
                "oracle": game_boards[-1].tolist(),
                "trials": [{"rt": 0} for _ in range(n_trials - 1)],
            })
            games.append(game)
    return games


def get_symmetric_board_pov(type_, paths = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]):
    # this board should always have a 50% chance of moving left no matter what the model is 
    board = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1, 0, 0], 
            [1, 1, 1, 1, 1, 1, 1, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    )
    n_games = 5
    n_conditions = 5
    boards = [board]*len(board - 1)
    boards = [boards]*n_games
    boards = [boards]*n_conditions
    boards = xr.DataArray(boards, dims = ["conditions","games", "trials", "rows", "cols"])
    paths = paths
    paths = [paths]*n_games
    paths = [paths]*n_conditions
    paths = xr.DataArray(paths, dims = ["conditions","games", "trials", "coords"])
    boards["conditions"] = get_stochasticity_levels(type_)
    paths["conditions"] = get_stochasticity_levels(type_)
    pov_array = make_pov_array(boards, paths)
    games = boards_paths_to_games(boards, paths, type_)
    return games

def get_asymmetric_board_pov(type_, paths = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]):
    # this board should always have a 100% chance of moving left
    # for all nonrandom filter functions and value functions 
    # exclusions: value_EV when stoch = 100%; filter functions when threshold = 0; when lapse = 1; inv_temp = -100
    board = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0], 
            [9, 1, 0, 0, 0, 0, 0, 0], 
            [9, 1, 1, 0, 0, 0, 0, 0], 
            [9, 1, 1, 1, 0, 0, 0, 0], 
            [9, 1, 1, 1, 1, 0, 0, 0], 
            [9, 1, 1, 1, 1, 1, 0, 0], 
            [9, 1, 1, 1, 1, 1, 1, 0], 
            [9, 1, 1, 1, 1, 1, 1, 1]
        ]
    )

    n_games = 5
    n_conditions = 5
    boards = [board]*len(board - 1)
    boards = [boards]*n_games
    boards = [boards]*n_conditions
    boards = xr.DataArray(boards, dims = ["conditions","games", "trials", "rows", "cols"])
    paths = paths
    paths = [paths]*n_games
    paths = [paths]*n_conditions
    paths = xr.DataArray(paths, dims = ["conditions","games", "trials", "coords"])
    boards["conditions"] = get_stochasticity_levels(type_)
    paths["conditions"] = get_stochasticity_levels(type_)
    pov_array = make_pov_array(boards, paths)
    games = boards_paths_to_games(boards, paths, type_)
    return games


@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_symmetric_equalprob(type_,effort_version, filter_fn, value_fn):
    games = get_symmetric_board_pov(type_)
    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }
    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }
    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )
    p_left = model.get_prob_left(params)
    assert np.allclose(p_left, 0.5)


@pytest.mark.parametrize("type_", ["R", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_EV])
def test_symmetric_equalprob_EV(type_,effort_version, filter_fn, value_fn):
    games = get_symmetric_board_pov(type_)
    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }
    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }
    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )
    p_left = model.get_prob_left(params)
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_asymmetric_always_left(type_,effort_version, filter_fn, value_fn):
    games = get_asymmetric_board_pov(type_)

    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }

    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }

    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left, 1)

@pytest.mark.parametrize("type_", ["R", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_EV])
def test_asymmetric_always_left_EV(type_,effort_version, filter_fn, value_fn):
    games = get_asymmetric_board_pov(type_)

    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 0
        }

    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }

    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[0]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[1]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[2]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[3]), 1)
    # in EV, we should get a 50% chance of moving left for maximum stochasticity
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[4]), 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_lapse(type_,effort_version, filter_fn, value_fn):
    games = get_asymmetric_board_pov(type_)

    # if we set the lapse to 1, we should *undo* the asymmetric effect
    # and we should get a 50% chance of moving left again
    if effort_version == "filter_adapt":
        params = {
            "inv_temp": 10,
            "lapse": 1
        }

    elif effort_version == "policy_compress":
        params = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 1
        }

    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_inv_temp(type_, filter_fn, value_fn):
    games = get_asymmetric_board_pov(type_)

    effort_version = "filter_adapt"
    # if we set the inv_temp to -100, we should get a 50% chance of moving left
    # where we would otherwise have a 100% chance of moving left
    params = {
        "inv_temp": -100,
        "lapse": 0
    }
    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    # we choose the highest filters so we don't actually filter anything out
    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_conditional_inv_temp(type_,filter_fn, value_fn): 
    games = get_asymmetric_board_pov(type_)

    effort_version = "policy_compress"
    # if we set the inv_temp to -100 for the middle condition, we should get a 50% chance of moving left
    # for the middle condition, where we would otherwise have a 100% chance of moving left
    # we should still have a 100% chance of moving left for the other conditions
    params = {
        "condition_inv_temp_0": 10,
        "condition_inv_temp_1": 10,
        "condition_inv_temp_2": -100,
        "condition_inv_temp_3": 10,
        "condition_inv_temp_4": 10,
        "lapse": 0
    }


    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    # we choose the highest filters so we don't actually filter anything out
    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[0]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[1]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[2]), 0.5)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[3]), 1)
    assert np.allclose(p_left.sel(conditions = get_stochasticity_levels(type_)[4]), 1)


@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_conditional_inv_temp_does_not_affect_filter_adapt(type_,filter_fn, value_fn):
    games = get_asymmetric_board_pov(type_)
    
    effort_version = "filter_adapt"
    params = {
        "inv_temp": 10,
        "lapse": 0,
        "condition_inv_temp_0": -100,
        "condition_inv_temp_1": -100,
        "condition_inv_temp_2": -100,
        "condition_inv_temp_3": -100,
        "condition_inv_temp_4": -100,
    }
    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )
    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left, 1)
    
    params = {
        "inv_temp": -100,
        "lapse": 0,
        "condition_inv_temp_0": 10,
        "condition_inv_temp_1": 10,
        "condition_inv_temp_2": 10,
        "condition_inv_temp_3": 10,
        "condition_inv_temp_4": 10,
    }
    p_left = model.get_prob_left(params)
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_inv_temp_does_not_affect_policy_compress(type_,filter_fn, value_fn):
    games = get_asymmetric_board_pov(type_)
    
    effort_version = "policy_compress"
    # if we set the inv_temp to -100, we should get a 50% chance of moving left
    # where we would otherwise have a 100% chance of moving left
    params = {
        "inv_temp": -100,
        "lapse": 0,
        "condition_inv_temp_0": 10,
        "condition_inv_temp_1": 10,
        "condition_inv_temp_2": 10,
        "condition_inv_temp_3": 10,
        "condition_inv_temp_4": 10,
    }
    
    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    p_left = model.get_prob_left(params)
    p_left = p_left.sel(filter_params = p_left.filter_params.max())
    assert np.allclose(p_left, 1)

    params = {
        "inv_temp": 10,
        "lapse": 0,
        "condition_inv_temp_0": -100,
        "condition_inv_temp_1": -100,
        "condition_inv_temp_2": -100,
        "condition_inv_temp_3": -100,
        "condition_inv_temp_4": -100,
    }
    p_left = model.get_prob_left(params)
    assert np.allclose(p_left, 0.5)

@pytest.mark.parametrize("type_", ["R", "V", "T"])
@pytest.mark.parametrize("effort_version", ["filter_adapt", "policy_compress"])
@pytest.mark.parametrize("filter_fn", [filter_depth, filter_rank, filter_value])
@pytest.mark.parametrize("value_fn", [value_path, value_max, value_sum, value_levelmean])
def test_asymmetric_NLL(type_,effort_version,filter_fn,value_fn): 
    games = get_asymmetric_board_pov(type_)
    if effort_version == "filter_adapt":
        params_low_inv_temp = {
            "inv_temp": -100,
            "lapse": 0
        }
        params_high_inv_temp = {
            "inv_temp": 10,
            "lapse": 0
        }
    elif effort_version == "policy_compress":
        params_low_inv_temp = {
            "condition_inv_temp_0": -100,
            "condition_inv_temp_1": -100,
            "condition_inv_temp_2": -100,
            "condition_inv_temp_3": -100,
            "condition_inv_temp_4": -100,
            "lapse": 0
        }
        params_high_inv_temp = {
            "condition_inv_temp_0": 10,
            "condition_inv_temp_1": 10,
            "condition_inv_temp_2": 10,
            "condition_inv_temp_3": 10,
            "condition_inv_temp_4": 10,
            "lapse": 0
        }

    model = BehaviorModel(
        effort_version = effort_version,
        filter_fn = filter_fn,
        value_fn = value_fn,
        variant = type_,
        games = games,
    )

    # Override choices to all-left for this test
    cond_n = model.pov_value_cache.sizes["conditions"]
    games_n = model.pov_value_cache.sizes["games"]
    trials_n = model.pov_value_cache.sizes["trials"]
    choose_left = xr.DataArray(np.ones((cond_n, games_n, trials_n)), dims=["conditions", "games", "trials"])
    choose_left["conditions"] = model.pov_value_cache["conditions"]
    model.choose_left = choose_left

    nll_low = model.optimize_filter_params(params_low_inv_temp.values(), params_low_inv_temp.keys())
    nll_high = model.optimize_filter_params(params_high_inv_temp.values(), params_high_inv_temp.keys())
    assert nll_low > nll_high
