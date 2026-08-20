# Human Planning in Stochastic Environments

## Introduction

The world is stochastic, making planning difficult. Despite the ubiquity of stochasticity in real-world environments, it remains an open question how people effectively balance the cognitive costs of planning against its potential benefits when faced with stochasticity. To study this, we designed a planning task where participants face one of three forms of stochasticity commonly encountered in the real world: reliability, volatility, and controllability.

We find a robust pattern across all three manipulations: as stochasticity increases, people reduce their planning effort as measured by first-choice response times. To understand the processes underlying this effect, we developed several computational cognitive models to account for participants' choices. We find that rather than calculating expected values optimally, people chose a simpler strategy, acting as if the world were deterministic, a phenomenon known as determinizing. Consistent with our response time findings, we found that people reduce their effort by decreasing their sensitivity to values with increasing stochasticity, a phenomenon known as policy compression.

Our work highlights the often overlooked role of stochasticity in human planning and its impact on planning strategy and effort. Moreover, it reveals the limitations of studying stochasticity solely through single-shot decisions.

**Preprint:** [Human Planning in Stochastic Environments](https://osf.io/preprints/psyarxiv/bh56p_v1)

## Demo

Try the task in your browser. Each page offers the full walkthrough (with instructions,
comprehension check, and practice games) or the games on their own:

| Condition | Demo |
| --- | --- |
| Reliability | <https://triangle-treasure-demo.web.app/demoR> |
| Volatility | <https://triangle-treasure-demo.web.app/demoV> |
| Controllability | <https://triangle-treasure-demo.web.app/demoC> |

The demo runs entirely in the browser: it plays one game per condition and **records no
data**. To run it locally instead, see [Experiment](#experiment) below.

To link straight into the task, skipping the landing page, use
`/?type=demoR` (or `demoV` / `demoC`), and add `&skip=1` to bypass the screen checks,
instructions, comprehension check, and practice block. Attention checks are omitted from
all demo runs, and `skip` is ignored outside demo mode. Each landing page is tinted with
its condition's figure colors (`analysis/src/utils.py`, `get_colormap`).

## Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Project structure](#project-structure)
- [Setup](#setup)
  - [Conda (recommended)](#conda-recommended)
  - [Pip (fallback)](#pip-fallback)
  - [Fonts (optional)](#fonts-optional)
- [Experiment](#experiment)
  - [Running the task locally](#running-the-task-locally)
  - [Hosting the demo](#hosting-the-demo)
  - [Data](#data)
- [Analysis](#analysis)
  - [Pipeline summary](#pipeline-summary)
  - [Cluster configuration](#cluster-configuration)
  - [Running things yourself](#running-things-yourself)
  - [Model space](#model-space)
- [License](#license)

## Project structure

```
.
├── experiment/
│   ├── src/              # Browser-based task + demo landing pages. Serve as a static site.
│   └── deploy_demo.sh    # Deploy the demo to Firebase Hosting
├── analysis/
│   ├── data/             # Raw + simulated datasets (download from OSF, gitignored)
│   ├── data_split/       # Per-participant CV folds (generated, gitignored)
│   ├── fit/              # Cross-validated fit outputs (generated, gitignored)
│   ├── src/              # Core library: models, filters, values, fitting, analysis
│   ├── scripts/          # Pipeline entry points (shell + Python)
│   └── workflows/        # Figure-generating scripts
├── firebase.json         # Hosting config for the demo
├── environment.yml       # Conda environment spec
└── requirements.txt      # Pip fallback
```


## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate pise_env
```

`pise_env` bundles Python, R, and `pymer4`/`rpy2` for the mixed-effects models used in `analysis_main.py`. R package versions matter here — `lme4` must stay on the 1.1.x line (not 2.x), which can silently change GLMM convergence behavior. The exact pins are in `environment.yml`.

### Pip (fallback)

```bash
pip install -r requirements.txt
```

Not recommended: `pymer4` and `rpy2` have R system dependencies that pip doesn't manage well.

### Fonts (optional)

Figures are typeset in [Arimo](https://fonts.google.com/specimen/Arimo). The font files are
not stored in this repository; fetch them with:

```bash
bash analysis/scripts/get_fonts.sh
```

This is optional. Without it, plotting falls back to any metric-compatible sans that is
installed (Nimbus Sans, Liberation Sans, Helvetica, Arial), and only warns if it has to
use matplotlib's wider DejaVu Sans default. Run it to reproduce the published figures exactly.

## Experiment

A web-based treasure-hunt game (`experiment/src/`) presenting one of three stochastic environments — Reliability (R), Volatility (V), or Controllability (C) — with practice rounds, comprehension checks, and Prolific integration.

### Running the task locally

```bash
cd experiment/src
python -m http.server 8000
```

Then open `http://localhost:8000/?type=demoR` (or `demoV` / `demoC`).

### Hosting the demo

The demo is a static site, deployable to Firebase Hosting:

```bash
firebase login                      # one-time
bash experiment/deploy_demo.sh
```

`firebase.json` deploys Hosting only. Because the demo never writes data, the hosting
project needs no Realtime Database and no credentials. Any static host (GitHub Pages,
Netlify) works equally well — serve `experiment/src/` as the site root.

### Data

Experimental data, cross-validation splits, and fit outputs are **not** in this
repository. They are archived on OSF (public, DOI:
[10.17605/OSF.IO/VYH8U](https://doi.org/10.17605/OSF.IO/VYH8U)) and extracted into the
analysis tree:

| Archive | Extracts to | Holds |
|---------|-------------|-------|
| `data.tar.gz` | `analysis/data/` | Raw participant data (`raw/`), demographics, simulated datasets |
| `data_split.tar.gz` | `analysis/data_split/` | Per-participant cross-validation folds |
| `fit.tar.gz` | `analysis/fit/` | Cross-validated fit outputs |

Fetch them with:

```bash
cd analysis/scripts
./0_downloaddata.sh              # all three
./0_downloaddata.sh data fit     # or a subset
```

The script resolves each archive's current OSF file ID by name through the OSF API, so
re-uploads do not break it. `_pipeline.sh` runs it automatically as its first step.

## Analysis

### Pipeline summary

0. **Fetch** the archived data from OSF (`analysis/scripts/0_downloaddata.sh`) — see
   [Data](#data). Skip this if you are generating simulated data instead.
1. **Simulate** synthetic participants for model recovery (`modelsimulation.py`), or use real data under `analysis/data/raw/`.
2. **Split** each participant's data into cross-validation folds (`split_data.py`).
3. **Fit** every (effort × filter × value) model combination per fold, then refit on the full dataset (`fitter.py`).
4. **Analyze**: load fits into an `Analyzer` (`analysis/src/analysis.py`) and generate figures (`analysis/workflows/*.py`).

On a Slurm cluster, `bash analysis/scripts/_pipeline.sh` downloads data from OSF, then runs the whole chain — preprocess → fit → figures — as dependent jobs. Locally, prefix any stage script with `FIT_LOCAL=1` to run it in your current shell instead of submitting to Slurm. Each stage script carries a header comment describing its inputs, outputs, and Slurm resources.

### Cluster configuration

Site-specific settings live in a gitignored config so nobody has to edit the scripts:

```bash
cp pipeline.sample.yaml analysis/scripts/pipeline.local.yaml
# then set `base` (absolute path to this repo's analysis/ directory), and, if your
# cluster needs them, slurm.account and slurm.mail_user
```

`analysis/scripts/load_config.sh` reads `pipeline.local.yaml` when present and otherwise falls back to the committed placeholder defaults in `analysis/scripts/_pipeline.yaml`. `account` and `mail_user` are both optional — leave them unset and the corresponding `sbatch` flags are simply omitted.

Per-stage resource requests (`-c`, `--mem`, `-t`, `--array`) stay as `#SBATCH` lines in each script, since Slurm reads those from the file before any shell code runs. Note that some sites route jobs to a QOS by requested walltime, which can cap how many array tasks run concurrently.

The stage scripts also source `analysis/scripts/activate_env.sh`, which puts the `pise_env` interpreter and its bundled R on `PATH` (set `PISE_ENV_PREFIX` if your env lives elsewhere).

### Running things yourself

All commands assume `conda activate pise_env` and start from `analysis/scripts/` unless noted.

**Fit models on existing data:**
```bash
python split_data.py
python fitter.py
```

**Generate simulated data for recovery analyses, then fit it:**
```bash
python modelsimulation.py
python split_data.py
python fitter.py
```

**Generate figures**, from `analysis/workflows/`:
```bash
python analysis_main.py               # empirical + model-overlay grids
python analysis_modelcomparison.py    # bootstrapped model comparison (needs --folder or SLURM_ARRAY_TASK_ID)
python analysis_parameterrecovery.py  # true-vs-fitted parameter recovery
python analysis_randeffects_bmc.py    # group-level Bayesian model comparison
```

**Run tests:**
```bash
cd analysis/src
pytest tests/
```


### Model space

Models combine three choices, implemented in `analysis/src/`:

- **Effort** (`modeling.py`): `policy_compress`, `filter_adapt`
- **Filter** (`modelfilters.py`): `filter_depth`, `filter_rank`, `filter_value`
- **Value** (`modelvalues.py`): `value_EV`, `value_path`, `value_max`, `value_sum`, `value_levelmean`

## License

Released under the MIT License — see [`LICENSE`](./LICENSE).
