import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir) + "/src/"
sys.path.insert(0, parentdir)
from analysis import Analyzer, clean_model_name, draw_shared_xlabel
from plots import set_text_style
from utils import alphabet

_, font_bold = set_text_style()


def model_comparison_analysis(
    analyzer_cache,
    folder,
    baseline_name,
    zoom_range=(-230, 230),
    main_xlim=(-1000, 25000),
    n_bootstrap=1e6,
):

    variants = ["R", "V", "T"] if "EV" not in baseline_name else ["R", "T"]

    # Two panels per variant: a full-range panel and its zoom inset. Deliberately NOT
    # constrained_layout -- it applies one uniform gap to every column and so cannot pack the
    # inset tight against its own full-range panel (inner ``wspace``) while still leaving the
    # outer gap wide. The outer ``wspace`` must clear two sets of labels drawn outside the axes:
    # the inset's x tick labels, and the next variant's long y tick labels ("PC depth levelmean").
    n = len(variants)
    fig = plt.figure(figsize=(6.0 * n, 12))
    outer = fig.add_gridspec(1, n, wspace=0.75, left=0.13, right=0.99, bottom=0.13, top=0.90)
    axes = []
    for cell in outer:
        inner = cell.subgridspec(1, 2, width_ratios=[0.6, 0.4], wspace=0.12)
        axes.append(fig.add_subplot(inner[0]))
        axes.append(fig.add_subplot(inner[1]))

    for i, variant in enumerate(variants):
        # Initialize analyzer
        analyzer = analyzer_cache[variant]

        # Main panel showing full range
        ax_main, ax_zoom = axes[i * 2], axes[i * 2 + 1]
        analyzer.plot_model_comparison(baseline_name, n_bootstrap, ax=ax_main, ax_zoom=ax_zoom,
                                       zoom_range=zoom_range, main_xlim=main_xlim)
        # Panel letter and variant title sit above the "(Zoom)" title on the inset, so both
        # are raised clear of it.
        ax_main.text(
            -0.5,
            1.13,
            alphabet(i),
            transform=ax_main.transAxes,
            fontsize=32,
            fontproperties=font_bold,
            va="top",
            ha="left",
        )

        title = {"R": "Reliability", "V": "Volatility", "T": "Controllability"}[variant]
        ax_main.text(
            0.8,
            1.09,
            title,
            transform=ax_main.transAxes,
            color = analyzer.colors(0.5),
            fontsize=28,
            va="top",
            ha='center'
        )

        # One shared x title under the main+inset pair.
        draw_shared_xlabel(fig, ax_main, ax_zoom, analyzer.colors(0.5))

    # Save figure
    save_dir = f"figures/{folder}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/modelcomparison.pdf", bbox_inches="tight")


def run_model_comparison_for_folder(folder: str, n_bootstrap: float, fit_root: str = "../fit") -> None:
    """Load fit outputs for one folder and write ``figures/<folder>/modelcomparison.pdf``."""
    baseline_name = "PC depth path" if "raw" in folder else clean_model_name(folder.split("simulated.")[-1])
    analyzer_cache = {
        "R": Analyzer(os.path.join(fit_root, folder, "R")),
        "V": Analyzer(os.path.join(fit_root, folder, "V")) if "EV" not in baseline_name else None,
        "T": Analyzer(os.path.join(fit_root, folder, "T")),
    }
    # The fixed x-range is tuned to the empirical fits against the ``PC depth path`` baseline.
    # The simulated-recovery folders use their own generating model as baseline and span
    # different ranges, so they keep autoscaling rather than risk clipping a violin.
    model_comparison_analysis(
        analyzer_cache=analyzer_cache,
        folder=folder,
        baseline_name=baseline_name,
        main_xlim=(-1000, 25000) if "raw" in folder else None,
        n_bootstrap=n_bootstrap,
    )


def _list_fit_folders(fit_root: str) -> list[str]:
    fit_abs = os.path.abspath(fit_root)
    paths = sorted(glob.glob(os.path.join(fit_abs, "*")))
    return [
        os.path.basename(p)
        for p in paths
        if os.path.isdir(p) and not os.path.basename(p).startswith(".")
    ]


def main(
    n_bootstrap: float,
    fit_root: str = "../fit",
    folder: str | None = None,
    process_all: bool = False,
) -> None:
    folders = _list_fit_folders(fit_root)
    task_raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_raw is not None:
        folder_index = int(task_raw)
        if folder_index < 0 or folder_index >= len(folders):
            print(
                f"[modelcomparison] SLURM_ARRAY_TASK_ID={folder_index} out of range "
                f"(0..{len(folders) - 1 if folders else 'n/a'}, have {len(folders)} folders); skipping.",
                flush=True,
            )
            return
        name = folders[folder_index]
        print(f"[modelcomparison] task {folder_index + 1}/{len(folders)} folder={name!r} n_bootstrap={n_bootstrap}", flush=True)
        run_model_comparison_for_folder(name, n_bootstrap, fit_root)
        return
    if process_all:
        if not folders:
            print("[modelcomparison] no folders under fit_root; nothing to do.", flush=True)
            return
        for idx, name in enumerate(folders):
            print(
                f"[modelcomparison] {idx + 1}/{len(folders)} folder={name!r} n_bootstrap={n_bootstrap}",
                flush=True,
            )
            run_model_comparison_for_folder(name, n_bootstrap, fit_root)
        return
    if folder is not None:
        if folder not in folders:
            raise SystemExit(
                f"Unknown folder {folder!r}; candidates include {folders[:12]}{'...' if len(folders) > 12 else ''}"
            )
        run_model_comparison_for_folder(folder, n_bootstrap, fit_root)
        return
    raise SystemExit(
        "Set SLURM_ARRAY_TASK_ID (Slurm array job), pass --folder <name>, or pass --all "
        "(sorted list from glob.glob(fit_root + \"/*\") directories)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison for one fit folder (Slurm array or --folder).")
    parser.add_argument(
        "--n-bootstrap",
        type=float,
        default=1e6,
        help="Bootstrap draws per comparison panel.",
    )
    parser.add_argument(
        "--fit-root",
        type=str,
        default="../fit",
        help="Directory containing per-folder fit outputs (default: ../fit from cwd).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Process only this fit subfolder name (for runs without SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every fit subfolder under --fit-root (sorted). Ignored if SLURM_ARRAY_TASK_ID is set.",
    )
    args = parser.parse_args()
    main(
        n_bootstrap=args.n_bootstrap,
        fit_root=args.fit_root,
        folder=args.folder,
        process_all=args.all,
    )
