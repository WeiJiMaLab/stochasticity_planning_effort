import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir) 

from plots import set_text_style
from utils import alphabet, get_colormap
from pathlib import Path
import json

def recovery_axis_labels(column: str) -> tuple[str, str]:
    """(True, Fitted) axis labels for one parameter-recovery panel.

    Inverse temperatures are fitted in log space, so they render as log(beta_k) with
    a 1-based index. "depth" is really filter_params["global"], the model's single
    filter parameter -- the label is only accurate because every model plotted here
    uses filter_depth.
    """
    if column == "lapse":
        core = "lapse"
    elif column == "depth":
        core = "depth"
    elif column.startswith("condition_inv_temp_"):
        idx = int(column.removeprefix("condition_inv_temp_"))
        core = rf"$\log(\beta_{{{idx + 1}}})$"
    else:
        core = str(column)
    return (rf"True {core}", rf"Fitted {core}")

def plot_parameter_recovery(model):
    folder = f"simulated.{model}"
    font_regular, font_bold = set_text_style()
    variants = ["R", "V", "T"] if "EV" not in model else ["R", "T"]
    fig, axes = plt.subplots(7, len(variants), figsize = (5 * len(variants), 35), gridspec_kw={'hspace': 0.7})
    
    for col, type_ in enumerate(variants):
        # load actual parameters from simulated data
        true_params = {}
        for f in glob.glob(f"../data/{folder}/{type_}/*params.json"):
            user = Path(f).stem.split("_params")[0]
            data = json.load(open(f, "r"))
            true_params[user] = {
                "lapse": data["lapse"],
                "depth": data["filter_params"]["global"],
                "condition_inv_temp_0": data["condition_inv_temp_0"],
                "condition_inv_temp_1": data["condition_inv_temp_1"],
                "condition_inv_temp_2": data["condition_inv_temp_2"],
                "condition_inv_temp_3": data["condition_inv_temp_3"],
                "condition_inv_temp_4": data["condition_inv_temp_4"],
            }

        true_params = pd.DataFrame(true_params).T.sort_index()


        # load fitted parameters from re-fit simulated data
        fitted = {}
        for f in glob.glob(f"../fit/{folder}/{type_}/*data.json"):
            user = Path(f).stem.split("_data")[0]
            with open(f, "r") as file:
                fit = json.load(file)[model]["full_params"]
                fitted[user] = {
                    "lapse": fit["lapse"],
                    "depth": fit["filter_params"]["global"],
                    "condition_inv_temp_0": fit["condition_inv_temp_0"],
                    "condition_inv_temp_1": fit["condition_inv_temp_1"],
                    "condition_inv_temp_2": fit["condition_inv_temp_2"],
                    "condition_inv_temp_3": fit["condition_inv_temp_3"],
                    "condition_inv_temp_4": fit["condition_inv_temp_4"],
                }

        fitted = pd.DataFrame(fitted).T.sort_index()

        assert list(true_params.index) == list(fitted.index), \
            f"User key mismatch for {type_}: true_params has {true_params.index.tolist()}, fitted has {fitted.index.tolist()}"

        for row, column in enumerate(fitted.columns): 
            ax = axes[row, col]
            if row == 0:
                    title = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_]
                    ax.text(0.5, 1.4, title, transform=ax.transAxes,
                            fontsize=28, va='top', ha='center', color=get_colormap(type_)(0.5))
            ax.text(-0.3, 1.15, alphabet(row * 3 + col), transform=ax.transAxes,
                            fontsize=28, fontproperties=font_bold, va='top', ha='left')
            xlab, ylab = recovery_axis_labels(column)
            r = np.corrcoef(true_params[column].values.astype(float), fitted[column].values.astype(float))[0, 1]
            ax.scatter(true_params[column], fitted[column], alpha = 0.2, color = get_colormap(type_)(0.5))
            # Get combined axis limits for square reference line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])
            ]
            ax.plot(lims, lims, 'k--', linewidth=1)

            ax.set(title=f"(r = {r:.2f})", xlabel=xlab, ylabel=ylab,
                    xlim=lims, ylim=lims, aspect='equal')

    os.makedirs(f"figures/{folder}", exist_ok=True)
    fig.savefig(f"figures/{folder}/recovery.pdf", bbox_inches='tight')

if __name__ == "__main__":
    models = [
        "policy_compress.filter_depth.value_levelmean",
        "policy_compress.filter_depth.value_path",
        "policy_compress.filter_depth.value_max",
        "policy_compress.filter_depth.value_sum",
        "policy_compress.filter_depth.value_EV",
    ]
    for model in models:
        plot_parameter_recovery(model)