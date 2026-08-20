import numpy as np 
from scipy.optimize import Bounds, minimize
from prodict import Prodict
import xarray as xr
from utils import *
from modelfilters import filter_depth, filter_rank, filter_value
from modelvalues import value_path, value_max, value_sum, value_levelmean, value_EV
import copy

# Default bounds for the parameters to be fitted.
# log-beta [-3, 3] is about the range this task identifies, not an arbitrary
# box: at |dV|=1 log-beta=3 already gives P=0.95, and at |dV|=7 log-beta=-3 is
# ~2.5 binomial SE from chance (n~210/cell). Widening only adds flat likelihood
# and degrades recovery; cells pinned at a bound are ones where beta is not
# identified.
DEFAULT_BOUNDS = {
    "inv_temp": (-3, 3),
    "condition_inv_temp_0": (-3, 3),
    "condition_inv_temp_1": (-3, 3),
    "condition_inv_temp_2": (-3, 3),
    "condition_inv_temp_3": (-3, 3),
    "condition_inv_temp_4": (-3, 3),
    "lapse": (1e-3, 1),
}

def get_effort_filter_value_options(type_):
    # return the possible effort versions, filter functions, and value functions for a given variant
    effort_versions = ["filter_adapt", "policy_compress"]
    filter_fns = [filter_depth, filter_rank, filter_value]
    value_fns = [value_path, value_max, value_sum, value_levelmean]
    if type_ == "R" or type_ == "T": value_fns.append(value_EV)
    return effort_versions, filter_fns, value_fns

def get_effort_version_info(effort_version: str):
    """Return the (conditional_filter, conditional_inv_temp) flags for an effort version.

    "filter_adapt" varies the filter parameter by condition; "policy_compress" varies the
    inverse temperature. "hybrid" varies both, but is never fitted: it is not returned by
    ``get_effort_filter_value_options``, so nothing in the pipeline reaches this branch.
    """
    if effort_version == "filter_adapt":
        conditional_filter = True
        conditional_inv_temp = False
    elif effort_version == "policy_compress":
        conditional_filter = False
        conditional_inv_temp = True
    elif effort_version == "hybrid":
        conditional_filter = True
        conditional_inv_temp = True
    else:
        raise ValueError(f"Invalid effort version: {effort_version}")
    return conditional_filter, conditional_inv_temp


class BehaviorModel():
    """A softmax choice model over the planning task: the player filters the visible tree
    (``filter_fn``), scores the resulting left/right subtrees (``value_fn``), and chooses via a
    softmax on that contrast. ``effort_version`` sets which parameter is free to vary by
    stochasticity condition.
    """

    def __init__(self, effort_version: str, filter_fn: callable, value_fn: callable, variant: str, games:list):
        """Build the model for one stochasticity ``variant`` ("R", "V", or "T").

        Precomputes ``self.pov_value_cache`` -- the value contrast for EVERY discrete
        filter-parameter setting at once (``filter_fn`` returns a ``filter_params`` dimension) --
        so ``fit()`` can grid-search filter parameters by indexing this cache instead of
        re-running the value recursion for each candidate.
        """
        assert variant in ["R", "V", "T"], "Invalid variant"
        self.effort_version, self.filter_fn, self.value_fn, self.variant = effort_version, filter_fn, value_fn, variant
        self.name = self.effort_version + "." + self.filter_fn.__name__ + "." + self.value_fn.__name__
        self.conditional_filter, self.conditional_inv_temp = get_effort_version_info(self.effort_version)

        game_data = preprocess_data(games)
        self.choose_left = game_data.choose_left
        self.pov_array = game_data.pov_array
        self.pov_value_cache = self.value_fn(self.filter_fn(self.pov_array), variant=self.variant)
        
    def fit(self, initial_params : dict):
        '''Fit the model by maximum likelihood.

        The continuous parameters (lapse, inverse temperature) are optimized with
        L-BFGS-B under DEFAULT_BOUNDS; the discrete filter parameter is not optimized
        but maximized over its grid at every objective evaluation, using the cache
        built in __init__. `initial_params` supplies the starting point and, by its
        keys, which parameters are free.

        Returns the fitted parameter dict plus "filter_params" (the winning grid
        point), "nll", and "model". Asserts that re-evaluating the returned
        parameters reproduces the reported nll.
        '''
        #split the initial parameter values and the names
        params_init, params_names = list(initial_params.values()), list(initial_params.keys())
        
        # clip the initial parameter values to the bounds and fit within the bounds
        pairs = [DEFAULT_BOUNDS[k] for k in params_names]
        lb = np.array([p[0] for p in pairs])
        ub = np.array([p[1] for p in pairs])
        params_init = np.clip(np.asarray(params_init, dtype=float), lb, ub)
        bounds = Bounds(lb, ub, keep_feasible=True)

        # minimize the negative log likelihood
        res = minimize(self.optimize_filter_params, params_init, 
                       args=(params_names,), 
                       bounds=bounds)
        
        #now we have selected the optimal set of parameters - we now pass these back into 
        #our maximum likelihood estimator to get the remaining best parameters
        params = dict(zip(params_names, res.x))
        best_ll, best_filter_params = self.select_best_filter_params(params)

        # this is the best NLL that we can get from the model
        nll = -np.asarray(best_ll).item()
        fitted_params = copy_and_update(params, {"filter_params": best_filter_params, "nll": nll, "model": self.name})

        assert np.isclose(self.evaluate_NLL(fitted_params), fitted_params["nll"], rtol=1e-4)
        return fitted_params

    def evaluate_NLL(self, params):
        # evaluates the NLL of the model given the parameters, including specified filter parameters
        assert "filter_params" in params, "Error: Filter parameters MUST be specified in params"

        p_left = self.get_prob_left(params)
        log_likelihood = xr.where(self.choose_left, np.log(p_left), np.log(1 - p_left))
        marginal_log_likelihood = log_likelihood.sum(["games", "trials"], keep_attrs=True)

        filter_params = params["filter_params"]
        if self.conditional_filter: 
            filter_params_da = xr.DataArray(
                list(filter_params.values()), 
                coords={"conditions": list(filter_params.keys())}, 
                dims="conditions")
            result = marginal_log_likelihood.sel(filter_params=filter_params_da)
        else: 
            result = marginal_log_likelihood.sel(filter_params=filter_params["global"])
        return -result.sum().item()

    def optimize_filter_params(self, params_values, params_names):
        return -self.select_best_filter_params(dict(zip(params_names, params_values)))[0]


    def select_best_filter_params(self, params:dict): 
        # finds the best set of filter parameters and returns the maximum likelihood and the parameters that achieved it
        p_left = self.get_prob_left(params)
        log_likelihood = xr.where(self.choose_left, np.log(p_left), np.log(1 - p_left))
        assert not log_likelihood.isnull().any(), "Error: Log likelihood is nan"

        # for cases where we want ONE filter parameter FOR EACH CONDITION
        if self.conditional_filter:
            # we marginalize over games and trials to get the marginal log likelihood for conditions x filter_params
            marginal_log_likelihood = log_likelihood.sum(["games", "trials"], keep_attrs = True)
            best_log_likelihood = marginal_log_likelihood.max("filter_params")

            # get the best filter parameter for each condition
            best_indices = [argmax_random_tiebreaker(marginal_log_likelihood.sel(conditions=c)) for c in marginal_log_likelihood.conditions]
            best_filter_params = xr.DataArray(
                [marginal_log_likelihood.filter_params[best_index].item() for best_index in best_indices],
                coords={"conditions": marginal_log_likelihood.conditions},
                dims="conditions"
            )

            assert not best_log_likelihood.isnull().any(), "Error: Best log likelihood is nan"
            return best_log_likelihood.sum(), dict(zip(best_filter_params.conditions.values, best_filter_params.values))
        
        # for cases where we want a GLOBAL filter parameter for ALL CONDITIONS
        else:
            # we marginalize over games, trials, AND conditions to get the marginal log likelihood for filter_params
            # and get the filter parameter that maximizes the log likelihood
            marginal_log_likelihood = log_likelihood.sum(["games", "trials", "conditions"], keep_attrs = True)
            best_log_likelihood = marginal_log_likelihood.max("filter_params")

            best_index = argmax_random_tiebreaker(marginal_log_likelihood)
            best_filter_param = marginal_log_likelihood.filter_params[best_index].item()

            assert not best_log_likelihood.isnull().any(), "Error: Best log likelihood is nan"
            return best_log_likelihood.item(), {"global": best_filter_param}
        

    def get_prob_left(self, params:dict):
        """Softmax P(choose left) from the cached left-minus-right value contrast.

        `params` must contain "lapse", plus either "inv_temp" or one
        "condition_inv_temp_<i>" per stochasticity level when the model varies
        sensitivity by condition. Inverse temperatures are in log space, so they may
        be negative; they are exponentiated here. Lapse mixes the result toward 0.5.
        """
        lapse = params["lapse"]
        if self.conditional_inv_temp:
            inv_temp = xr.DataArray([params[f"condition_inv_temp_{i}"] for i in range(len(get_stochasticity_levels(self.variant)))], dims="conditions")
        else:
            inv_temp = params["inv_temp"]

        # Compute the probability of choosing the left option using the sigmoid function and apply lapse rate
        p_left = sigmoid(self.pov_value_cache, b_1=np.exp(inv_temp))
        p_left = (1 - lapse) * p_left + lapse * 0.5
        return p_left

    
def sample_actions(effort_version: str, filter_fn: callable, value_fn: callable, variant: str, params:dict, games:list, rng:np.random.Generator = None):
    """Simulate choices from the model on real game data (for parameter recovery).

    ``params`` holds "lapse", "filter_params", and either "inv_temp" or the per-condition
    "condition_inv_temp_*", depending on ``effort_version``. Returns a Prodict mirroring the
    preprocessed game data (boards, oracles, paths, is_transition) with ``choose_left``
    replaced by the model's sampled choices, encoded left=1 / right=0.
    """
    conditional_filter, conditional_inv_temp = get_effort_version_info(effort_version)

    params = copy.deepcopy(params)
    #we process the game by taking the "point of view" of the player
    game_data = preprocess_data(games)

    values = value_fn(filter_fn(game_data.pov_array), variant=variant)
    if conditional_inv_temp:
        inv_temps = xr.DataArray([params[f"condition_inv_temp_{i}"] for i in range(len(get_stochasticity_levels(variant)))], dims="conditions")
    else:
        inv_temps = params["inv_temp"]

    p_left_ = sigmoid(values, b_1=np.exp(inv_temps))
    p_left_ = (1 - params["lapse"]) * p_left_ + params["lapse"] * 0.5

    if conditional_filter:
        p_left = xr.concat([p_left_.sel(
            conditions = condition, 
            filter_params = params["filter_params"][condition]) 
            for condition in p_left_.conditions.values], 
            dim = "conditions",
            coords = "different", 
            compat = "equals"
        )
    else:
        p_left = xr.concat([p_left_.sel(conditions = condition, 
            filter_params = params["filter_params"]["global"]) 
            for condition in p_left_.conditions.values], 
            dim = "conditions"
        )
    
    p_left['conditions'] = p_left_.conditions
    rng = np.random.default_rng() if rng is None else rng
    choose_left = (rng.random(p_left.shape) < p_left).astype(int)

    model_data = {
                    "boards": game_data.boards,
                    "oracles": game_data.oracles,
                    "choose_left": choose_left,
                    "paths": game_data.paths,
                    "is_transition": game_data.is_transition,
                }

    model_data = Prodict(model_data)
    return model_data
