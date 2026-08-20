import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)
from analysis import Analyzer, clean_model_name
from plots import set_text_style
from groupBMC.groupBMC import GroupBMC
from utils import alphabet
from utils import get_colormap

font_regular, font_bold = set_text_style()

# note raw copy 1, 2, 3, 4 are robustness checks ONLY for group BMC -- other results
# MUST be deliberately ignored to avoid cherry-picking by seed

for folder in ["raw", "raw copy 1", "raw copy 2", "raw copy 3", "raw copy 4"]:
    fig, axes = plt.subplots(2, 3, figsize=(22.5, 25), gridspec_kw={'hspace': 0.15, 'wspace': 1.0})
    results = {}

    analyzers = {
        "R": Analyzer(f"../fit/{folder}/R"),
        "V": Analyzer(f"../fit/{folder}/V"),
        "T": Analyzer(f"../fit/{folder}/T")
    }

    for col, type_ in enumerate(["R", "V", "T"]):
        print(col, type_)
        analyzer = analyzers[type_]

        models = sorted(analyzer.model_data.keys())
        model_evidence = pd.DataFrame({
            key: analyzer.model_data[key].set_index("player")["nll"]
            for key in models
        })
        assert not model_evidence.isna().any().any(), f"NaN NLL values in {type_}"
        assert model_evidence.index.is_unique, f"Duplicate players in {type_}"

        L = -model_evidence.values.T  # (K models, N subjects) pseudo-log-evidence
        print(f"{type_}: L.shape={L.shape}, FFX winner = {models[np.argmin(model_evidence.sum())]}")

        result = GroupBMC(L).get_result()
        results[type_] = (models, result)

        ax = axes[0, col]
        y_pos = np.arange(len(L))
        # Horizontal bar plot with error bars
        ax.barh(y_pos, result.frequency_mean, 
                xerr=np.sqrt(result.frequency_var) * 1.96, 
                align='center', 
                color=get_colormap(type_)(0.5), 
                ecolor='black', 
                alpha=0.5, 
                error_kw={"capsize": 4, "ecolor": "black"})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)

        ax.set_xticks(np.arange(0, 1.2, 0.2))
        ax.set_xlabel('Posterior Model Freq.', labelpad=10, x=0, ha='left')
        ax.set_xlim(0, 1.1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', alpha=0.3)
        title = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[type_]
        ax.text(0.5, 1.05, title, transform=ax.transAxes,
                fontsize=28, va='top', ha='center', color=get_colormap(type_)(0.5))
        ax.text(-0.3, 1.15, alphabet(col), transform=ax.transAxes,
                    fontsize=28, fontproperties=font_bold, va='top', ha='left')

        ax = axes[1, col]
        y_pos = np.arange(len(L))
        # Horizontal bar plot with error bars
        ax.barh(y_pos, result.protected_exceedance_probability, align='center', color=get_colormap(type_)(0.5), ecolor='black', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)

        ax.set_xticks(np.arange(0, 1.2, 0.2))
        ax.set_xlabel('Protected Exceedance Prob.', labelpad=10, x=0, ha='left')
        ax.set_xlim(0, 1.1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', alpha=0.3)
        ax.text(-0.3, 1.05, alphabet(3 + col), transform=ax.transAxes,
                    fontsize=28, fontproperties=font_bold, va='top', ha='left')

    os.makedirs(f"figures/{folder}/rfx_bmc", exist_ok=True)
    fig.savefig(f"figures/{folder}/rfx_bmc/VBA_BMC_model_comparison.pdf", bbox_inches='tight')

    print("\n=== Summary ===")
    for type_, (models, result) in results.items():
        bmc_winner = models[np.argmax(result.protected_exceedance_probability)]
        pxp = np.max(result.protected_exceedance_probability)
        print(f"  {type_}: BMC winner = {bmc_winner} (PXP={pxp:.3f})")