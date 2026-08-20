import sys, os
import json
import numpy as np
import tqdm
# Set up path and imports
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)
from modeling import sample_actions, get_effort_filter_value_options
from modelfilters import get_filter_param_options
from itertools import product
from utils import (
    NpEncoder, 
    get_stochasticity_levels, 
    get_user_data
)
from joblib import Parallel, delayed
import hashlib

def simulate_model(user_id, games, effort_version, filter_fn, value_fn, type_, n_repeats, simulation_name):
    """Run simulation for a single user with specified model type"""
    sim_user = f"sim_{user_id}"

    # generate a seed for the random number generator based on the user_id, effort_version, filter_fn, value_fn, and type_
    seed = int(hashlib.md5(f"{user_id}{effort_version}{filter_fn.__name__}{value_fn.__name__}{type_}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    # create a model name for the simulation
    model_name = f"{effort_version}.{filter_fn.__name__}.{value_fn.__name__}.{type_}"

    if effort_version == "policy_compress":
        # typical bounds fall between -2 and 0.5
        # if we use policy_compress, we have a global filter parameter and variable inverse temperatures for each condition
        params = {
            "model": model_name,
            "filter_params": {"global": rng.choice(get_filter_param_options(filter_fn))},
            "lapse": rng.uniform(1e-3, 0.5),
            **{f"condition_inv_temp_{i}": rng.uniform(-2, 0.5) for i in range(5)}
        }
    elif effort_version == "filter_adapt":
        # if we use filter_adapt, we have variable filter parameters for each condition but a global inverse temperature
        params = {
            "model": model_name, 
            "filter_params": {i: rng.choice(get_filter_param_options(filter_fn)) for i in get_stochasticity_levels(type_)},
            "lapse": rng.uniform(1e-3, 0.5),
            "inv_temp": rng.uniform(-2, 0.5),
        }
    else: 
        raise NotImplementedError(f"Effort version {effort_version} not implemented")

    simulated_data = []
    # Run simulations
    for i in range(n_repeats):
        prediction = sample_actions(effort_version, filter_fn, value_fn, type_, params, games, rng)
        simulated_data.extend([
            {
                "name": f"game_{user_id}_c{condition}_g{game}_r{i}",
                "p": get_stochasticity_levels(type_)[condition],
                "boards": prediction.boards.isel(conditions=condition, games=game).values,
                "oracle": prediction.oracles.isel(conditions=condition, games=game, trials=0).values,
                "tuplepath": prediction.paths.isel(conditions=condition, games=game).values,
                "path": [f'{a},{b}' for a, b in prediction.paths.isel(conditions=condition, games=game).values],
                "actions": prediction.choose_left.isel(conditions=condition, games=game).astype(bool).values,
                "is_transition": prediction.is_transition.isel(conditions=condition, games=game).astype(bool).values,
                "trials": [{"rt": 0} for _ in range(7)],
            }
            for condition in range(5) for game in range(30)
        ])

    # Save results
    filedir = f"../data/{simulation_name}.{effort_version}.{filter_fn.__name__}.{value_fn.__name__}/{type_}"
    os.makedirs(filedir, exist_ok=True)
    with open(f"{filedir}/{sim_user}_data.json", "w") as f:
        json.dump(simulated_data, f, cls=NpEncoder)

    # Save the original parameters
    with open(f"{filedir}/{sim_user}_params.json", "w") as f:
        json.dump(params, f, cls=NpEncoder)

def model_simulation(n_users=100):
    user_ids = [f"user{i}" for i in range(n_users)]
    variants = ["R", "V", "T"]
    simulation_combos = [
        (user_id, variant, effort_version, filter_fn, value_fn)
        for variant in variants
        for effort_version, filter_fn, value_fn in product(*get_effort_filter_value_options(variant))
        for user_id in user_ids
    ]

    data = {}
    for user_id, variant in product(user_ids, variants):
        data[(user_id, variant)] = get_user_data(user_id, variant, data_folder="../data/raw")

    list(tqdm.tqdm(
        Parallel(n_jobs=-1, return_as="generator")(
            delayed(simulate_model)(
                user_id, 
                data[(user_id, variant)].copy(),
                effort_version, 
                filter_fn, 
                value_fn, 
                variant, 
                1, 
                "simulated"
            )
            for user_id, variant, effort_version, filter_fn, value_fn in simulation_combos
        ),
        total=len(simulation_combos),
        desc="🧠 Simulating users",
        unit="job"
    ))

if __name__ == "__main__":
    print("🚀 Starting model simulations...")
    model_simulation()
    print("✅ Done!")
