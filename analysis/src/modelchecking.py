import xarray as xr
import numpy as np
import pandas as pd
from utils import make_pov_array, preprocess_data
from collections import defaultdict
import tqdm as tqdm
from modeling import sample_actions

#functions for model checking
def trialwise_rewards(game_data, baseline = "none"): 
    oracles_pov = make_pov_array(game_data.oracles, game_data.paths)
    boards_pov = make_pov_array(game_data.boards, game_data.paths)

    # if there was a transition, return 1 - choose_left, else return choose_left
    move_left = xr.where(game_data.is_transition, 1 - game_data.choose_left, game_data.choose_left)
    rewards = xr.where(move_left, oracles_pov.isel(rows = 1, cols = 0), oracles_pov.isel(rows = 1, cols = 1))
    
    if baseline == "none":
        return rewards
    
    elif baseline == "random":
        random_baseline = 0.5 * (oracles_pov.isel(rows = 1, cols = 0) + oracles_pov.isel(rows = 1, cols = 1))
        return rewards - random_baseline
    
    else: 
        assert False, "InputError: Baseline must be one of: none, random"


def trialwise_greedydiff(game_data): 
    boards_pov = make_pov_array(game_data.boards, game_data.paths)
    
    # left minus right choice
    greedy_diff = boards_pov.isel(rows = 1, cols = 0) - boards_pov.isel(rows = 1, cols = 1)
    return greedy_diff

def trialwise_chooseleft(game_data): 
    return game_data.choose_left

def trialwise_ignore(game_data): 
    return xr.zeros_like(game_data.choose_left)

def empirical(data, x_fun, y_fun): 
    x_ = []
    y_ = []

    for player in data.keys():
        game_data = preprocess_data(data[player])
        x_.append(x_fun(game_data))
        y_.append(y_fun(game_data))

    x = xr.concat(x_, dim = "participants")
    y = xr.concat(y_, dim = "participants")

    return x, y
     

def simulate_model(data, effort_version, filter_fn, value_fn, variant, fitted_params, x_fun, y_fun, iters = 1):
    x_ = []
    y_ = []
    
    for player in data.keys(): 
        x__ = []
        y__ = []

        for _ in range(iters):
            model_data = sample_actions(effort_version, filter_fn, value_fn, variant, fitted_params[player], data[player])
            x__.append(x_fun(model_data))
            y__.append(y_fun(model_data))
        
        x_.append(xr.concat(x__, dim = "trials"))
        y_.append(xr.concat(y__, dim = "trials"))

    x = xr.concat(x_, dim = "participants")
    y = xr.concat(y_, dim = "participants")
    return x, y

## helpers for model checking
def summary_statistics(x, y, n_bins=None):
    """Aggregate ``x``/``y`` per condition, returning ``(condition_x, condition_y, condition_sem)``:
    three dicts keyed by condition, holding mean x, mean y, and the SEM of the mean y.

    If ``n_bins`` is given, x is binned into per-participant quantiles and points are aggregated
    within bins (bin midpoints are used as the x values). Aggregation and SEM come from
    ``get_hierarchical_means``, whose SEM adds a within-participant correction term accounting for
    the uncertainty in each participant's own mean.
    """

    condition_x, condition_y, condition_sem = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])

    for condition in y.conditions: 
        y_ = y.sel(conditions = condition).values.flatten()
        x_ = x.sel(conditions = condition).values.flatten()
        participant_ = (xr.ones_like(y.sel(conditions = condition)) * y.participants).values.flatten()
        x_quantiles = None

        if n_bins is not None: 
            x_ = []
            x_quantiles = []

            # for each participant, bin the x values into quantiles
            for participant in x.participants: 
                x_quantile_, bins = pd.qcut(x.sel(conditions = condition, participants = participant).values.flatten(), n_bins, labels = np.arange(n_bins), retbins = True)
                bin_midpoints = (bins[1:] + bins[:-1])/2
                x_ += list(bin_midpoints[x_quantile_])
                x_quantiles += list(x_quantile_)

        condition_x[condition.item()], condition_y[condition.item()], condition_sem[condition.item()] = get_hierarchical_means(participant_, x_, y_, group = x_quantiles)

    return condition_x, condition_y, condition_sem


def get_hierarchical_means(participant, x, y, group = None, use_correction_term = True):
    """Two-stage mean and SEM: average within (participant, group), then across participants.

    This is the estimator behind the error bars in the paper's figures. Averaging
    within participant first keeps participants who contributed more trials from
    dominating the group mean.

    With use_correction_term, the SEM is sqrt(sem_of_means^2 + sum(var_y / n_i) / k^2),
    adding each participant's own sampling error to the spread between participants.
    Without it, per-participant means are treated as if measured exactly, which
    understates the error for groups with few trials per participant.

    `group` defaults to `x`. Returns (mean_x, mean_y, sem_y), one entry per group.
    """
    if group is None: 
        group = x
        
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "participant": participant,
            "group": group,
        }
    )

    # first, group by participant and group, computing the summary stats
    grouped = df.groupby(["participant", "group"]).agg(
        mean_x = ("x", "mean"),
        mean_y = ("y", "mean"),
        var_y = ("y", "var"),
        count = ("participant", "count")
    )

    # then, group by group, computing the summary stats
    hierarchical_grouped = grouped.groupby(["group"]).agg(
        mean_of_means_x = ("mean_x", "mean"),
        mean_of_means_y = ("mean_y", "mean"),
        sem_of_means_y = ("mean_y", "sem"),
    )

    if use_correction_term:
        # correction term adjusts for the within-participant uncertainty in the mean
        correction_term = grouped.groupby(["group"]).apply(
            lambda g: np.sum(g["var_y"] / g["count"]) / (len(g) ** 2)
        )

        hierarchical_grouped["sem_of_means_y"] = np.sqrt(hierarchical_grouped["sem_of_means_y"] ** 2 + correction_term)

    return hierarchical_grouped.mean_of_means_x.values, hierarchical_grouped.mean_of_means_y.values, hierarchical_grouped.sem_of_means_y.values


