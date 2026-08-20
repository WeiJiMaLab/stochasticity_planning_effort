import os
import sys
import glob
import json
import itertools
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import hashlib
# Add src directory to path for imports
SRC_DIR = os.path.join(os.path.dirname(os.getcwd()), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from utils import get_user_data, NpEncoder
import numpy as np

def stable_seed(user, seed):
    key = f"{user}_{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

def data_split(user, variant, input_folder, seed=10, n_folds=5):
    """
    Splits a user's games into cross-validation folds, grouped by condition.
    Saves the split data as JSON in the output folder under the variant name.
    """
    rng = np.random.default_rng(stable_seed(user, seed))

    games = get_user_data(user, variant, input_folder)

    # Group by stochasticity level (game.p) so every fold is balanced across conditions.
    by_condition = defaultdict(list)
    for game in games:
        by_condition[game.p].append(game)

    folds = defaultdict(list)
    for condition, condition_games in by_condition.items():
        shuffled_games = list(condition_games)  # Copy list to avoid mutating original
        rng.shuffle(shuffled_games)
        for fold_num in range(n_folds):
            folds[f"fold_{fold_num}"].extend(shuffled_games[fold_num::n_folds])

    output_path = os.path.join(input_folder.replace("data/", "data_split/"), variant)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{user}_data.json"), "w") as f:
        json.dump(folds, f, cls=NpEncoder)

def split_data_in_folder(folder, variant, n_folds=5):
    """
    Splits all user data files in a given folder and variant into cross-validation folds.
    """
    # Seed depends on the top-level data folder (e.g. raw vs raw copy 1) -- default seed is 10
    folder_name = Path(folder).name
    seed = 10
    if folder_name == "raw copy 1":
        seed = 11
    elif folder_name == "raw copy 2":
        seed = 12
    elif folder_name == "raw copy 3":
        seed = 13
    elif folder_name == "raw copy 4":
        seed = 14

    data_files = glob.glob(os.path.join(folder, variant, "*_data.json"))
    for file in tqdm(data_files):
        user = Path(file).stem.split("_data")[0]
        data_split(user, variant, folder, seed=seed, n_folds=n_folds)

def call_split(args):
    """
    Helper function for multiprocessing: splits data for all users in (folder, variant).
    """
    folder, variant = args
    data_files = glob.glob(os.path.join(folder, variant, "*_data.json"))
    if not data_files:
        return
    split_data_in_folder(folder, variant)

if __name__ == "__main__":
    print("🚀 Starting data split...")
    folders = glob.glob("../data/*")
    variants = ["R", "T", "V"]
    combos = list(itertools.product(folders, variants))
    with Pool() as pool:
        pool.map(call_split, combos)
    print("✅ Done!")