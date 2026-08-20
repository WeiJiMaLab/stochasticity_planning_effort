import numpy as np
import json
import copy
from pathlib import Path
import xarray as xr
from collections import defaultdict
from prodict import Prodict
import sigfig
from plots import colormaps

########################################################
# Helper & Pretty-print Functions
########################################################
def alphabet(n):
    """Return the nth letter of the alphabet (0-indexed)"""
    return chr(ord('A') + n)

def copy_and_update(dict_orig, dict_update): 
    dict_copy = copy.deepcopy(dict_orig)
    dict_copy.update(dict_update)
    return dict_copy

def sigmoid(d, b_0 = 0, b_1 = 1): 
    return 1/(1 + np.exp(-(b_0 + b_1 * d)))

def argmax_random_tiebreaker(a):
    return np.random.choice(np.flatnonzero(np.isclose(a, a.max())))

def strsimplify(num):
    """Pretty string for tick labels / legend text.

    Matplotlib ticks are often floats like ``0.5499999999999999`` — plain ``str`` keeps
    the noise. Near-integers are shown as ints; otherwise we use positional formatting
    with trimmed trailing zeros.
    """
    try:
        x = float(num)
    except (TypeError, ValueError):
        return str(num)
    if not np.isfinite(x):
        return str(x)
    xr = round(x)
    if np.isclose(x, xr, rtol=0.0, atol=1e-5):
        return str(int(xr))
    return np.format_float_positional(x, unique=False, precision=12, trim="-")

def report_p_value(p, threshold_power=-10):
    """Prettify p-values per Nature Communications style (uppercase P)."""
    if p < 0.001:
        return "P < 0.001"
    if p >= 0.1:
        return f"P = {p:.2f}"
    return f"P = {p:.3f}"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


########################################################
# Data Loading & Preprocessing
########################################################
def get_stochasticity_levels(variant):
    '''
    Returns stochasticity levels for given variant
    '''  
    if variant == "T": 
        return [0, 0.125, 0.25, 0.375, 0.5]
    else: 
        return [0, 0.25, 0.5, 0.75, 1]

def format_games(games): 
    '''
    removes the practice games from the list of games
    '''
    return [Prodict(g) for g in games if not "practice" in g["name"]]

def get_user_data(user, type_, data_folder):
    data = json.load(open(f"{data_folder}/{type_}/{user}_data.json"))
    # filter out practice games
    games = format_games(data)
    if Path(data_folder).name == "raw":
        # make sure there are 150 games
        assert len(games) == 150, f"Number of games is not 150 for {user}"
    return games


def preprocess_data(games: list):
    '''
    Creates xarrays for boards, paths, and actions taken (where true is left and false is right)
    for each game - used later in model fitting.
    '''
    boards, paths, actions, oracles, is_transition = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])
    conditions = []

    for game in games:
        game = Prodict(game)
        boards[game.p].append(game.boards)
        paths[game.p].append(game.tuplepath)
        actions[game.p].append(game.actions)
        is_transition[game.p].append(game.is_transition)
        oracles[game.p].append([game.oracle]*len(game.boards))
        conditions.append(game.p)

    ps = sorted(list(set(conditions)))
    game_data = {
        "boards": xr.DataArray([boards[p] for p in ps], dims = ["conditions", "games", "trials", "rows", "cols"]),
        "paths": xr.DataArray([paths[p] for p in ps], dims = ["conditions", "games", "trials", "coords"]),
        "choose_left": xr.DataArray([actions[p] for p in ps], dims = ["conditions", "games", "trials"]).astype(int),
        "oracles": xr.DataArray([oracles[p] for p in ps], dims = ["conditions", "games", "trials", "rows", "cols"]), 
        "is_transition": xr.DataArray([is_transition[p] for p in ps], dims = ["conditions", "games", "trials"]).astype(int),
    }
    game_data["pov_array"] = make_pov_array(game_data["boards"], game_data["paths"])
    
    for key in game_data.keys(): 
        game_data[key]["conditions"] = ps

    player_data = Prodict.from_dict(game_data)
    
    return player_data

def make_pov_array(boards: xr.DataArray, paths: xr.DataArray): 
    '''
    Given a set of boards: dims = (conditions, games, trials, rows, columns)
    and a set of paths: dims = (conditions, games, trials, coord)
    Returns a board from the point-of-view of the player
    such that the position of the board at (game, trial) is matched
    to (0, 0), the upper-left corner of the board
    '''

    boards_pov = np.zeros_like(boards).flatten()

    #get the row and column of the paths taken
    path_row, path_col = paths.sel(coords = 0), paths.sel(coords = 1)
    rows_, cols_ = np.meshgrid(boards.rows, boards.cols, indexing='ij')
    rows = xr.ones_like(boards) * rows_
    cols = xr.ones_like(boards) * cols_
    
    #get the indices of the pov_array and match them with the indices
    #of the original array
    pov_indices = np.where(np.logical_and(rows <= np.max(rows) - path_row, cols <= np.max(cols) - path_col), True, False).flatten()
    orig_indices = np.where(np.logical_and(rows >= path_row, cols >= path_col), True, False).flatten()
    
    boards_pov[pov_indices] = boards.values.flatten()[orig_indices]
    boards_pov = xr.DataArray(np.reshape(boards_pov, boards.shape), dims = boards.dims)
    boards_pov["conditions"] = boards.conditions

    #zero the above-diagonal parts of the board
    boards_pov = xr.where(rows >= cols, boards_pov, 0)
    boards_pov = boards_pov.sel(trials = boards_pov.trials[:-1])
    
    return boards_pov


########################################################
# Plotting & LMM/GLMM formatting
########################################################
def get_colormap(type_):
    return {"R": colormaps["arctic"], "T": colormaps["berry"], "V": colormaps["grass"]}[type_]

def format_lmm_result(row):
    """Format LMM results into LaTeX string"""
    return (
        f"LMM, "
        f"$\\beta = {row['beta']:.2f}$, "
        f"95\\% CI $[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]$, "
        f"$t_{{{int(row['dof'])}}} = {sigfig.round(row['tstat'], 3)}$, "
        f"${report_p_value(row['pval'])}$"
    )

def format_glmm_main_effect(row):
    """Format GLMM main effect results into LaTeX string"""
    return (
        f"GLMM, "
        f"main effect $\\beta = {row['beta_main']:.2f}$, "
        f"95\\% CI $[{row['ci_lower_main']:.2f}, {row['ci_upper_main']:.2f}]$, "
        f"$\\chi^2(1) = {sigfig.round(row['chi2_main'], sigfigs=3)}$, "
        f"${report_p_value(row['pval_main'])}$"
    )

def format_glmm_interaction(row):
    """Format GLMM interaction results into LaTeX string"""
    return (
        f"GLMM, "
        f"interaction $\\beta = {row['beta_inter']:.2f}$, "
        f"95\\% CI $[{row['ci_lower_inter']:.2f}, {row['ci_upper_inter']:.2f}]$, "
        f"$\\chi^2(1) = {sigfig.round(row['chi2_inter'], sigfigs=3)}$, "
        f"${report_p_value(row['pval_inter'])}$"
    )

def write_latex_results(df_results, glmm_results = None, output_file = None):
    """Write formatted results to LaTeX file"""
    with open(output_file, 'w') as f:
        for (model, stoch, var), row in df_results.iterrows():
            key = f"{model.replace(' ', '.')}.{stoch}.{var}"
            f.write(f"\\pgfkeyssetvalue{{{key}}}{{{format_lmm_result(row)}}}\n")

        if glmm_results is not None:
            for (model, stoch, var), row in glmm_results.iterrows():
                key = f"{model.replace(' ', '.')}.{stoch}.{var}"
                f.write(f"\\pgfkeyssetvalue{{{key}}}{{{format_glmm_main_effect(row)}}}\n")
                f.write(f"\\pgfkeyssetvalue{{{key + '.interaction'}}}{{{format_glmm_interaction(row)}}}\n")

def write_latex_loggers(df_logs, output_file): 
    df_logs = df_logs[["Model Name", "Stochasticity Type", "Variable", "Formula / Status"]]
    df_logs["Stochasticity Type"] = df_logs["Stochasticity Type"].map({"R": "Reliability", "V": "Volatility", "T": "Controllability"})
    df_logs["Variable"] = df_logs["Variable"].map({"greedydiff": "y = P(left) x = label diff", "points": "y = points", "first_rt": "y = log(firstRT)", "total_rt": "y = log(totalRT)", "invtemp": "y = log(beta)"})
    df_logs = df_logs.set_index(["Model Name", "Stochasticity Type", "Variable", "Formula / Status"])
    df_logs = df_logs.sort_index()
    df_logs.to_latex(output_file, index=True)
    
    # Replace \cline with \cmidrule for better formatting
    with open(output_file, 'r') as f:
        content = f.read().replace('\\cline', '\\cmidrule(lr)').replace(' ~ ', '} $\\sim$ \\texttt{')
    with open(output_file, 'w') as f:
        f.write(content)
    


