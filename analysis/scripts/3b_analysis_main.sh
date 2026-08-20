#!/bin/bash
#SBATCH -J analysis_main
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH --array=0
#SBATCH -o log/analysis_main/%j_%a.out
# #SBATCH values are Slurm-parsed defaults; runtime overrides come from load_config.sh (see _pipeline.yaml).

source "$(pwd)/load_config.sh"
source "$(pwd)/activate_env.sh"

export PYTHONUNBUFFERED=1

if [[ "${FIT_LOCAL:-}" == "1" ]]; then
  mkdir -p "$BASE/scripts/log/analysis_main"
  cd "$BASE/workflows"
  set -euo pipefail
  python -u analysis_main.py
  # Assumption diagnostics for the Supplementary "LMM Robustness" table.
  # Must follow analysis_main.py: it consumes that run's cached residuals
  # (figures/*/residuals_*.npy) and fitted estimates (figures/*/result_lmm.csv).
  python -u diagnostics_assumptions.py
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --chdir="$BASE/scripts" "${SBATCH_EXTRA_ARGS[@]}" "$BASE/scripts/3b_analysis_main.sh"
  exit 0
fi

mkdir -p "$BASE/scripts/log/analysis_main"
cd "$BASE/workflows"
set -euo pipefail
python -u analysis_main.py
# See the FIT_LOCAL branch above for why this runs second.
python -u diagnostics_assumptions.py
