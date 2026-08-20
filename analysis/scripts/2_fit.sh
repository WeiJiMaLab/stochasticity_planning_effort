#!/bin/bash
#SBATCH -J fit
#SBATCH -c 4
#SBATCH -t 40:00:00
#SBATCH --mem=2G
#SBATCH --array=0-99
#SBATCH -o log/fit/%j_%a.out
# #SBATCH values are Slurm-parsed defaults; runtime overrides come from load_config.sh (see _pipeline.yaml).

source "$(pwd)/load_config.sh"
source "$(pwd)/activate_env.sh"

if [[ "${FIT_LOCAL:-}" == "1" ]]; then
  cd "$BASE/scripts"
  python fitter.py
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --chdir="$BASE/scripts" "${SBATCH_EXTRA_ARGS[@]}" "$BASE/scripts/2_fit.sh"
  exit 0
fi

mkdir -p "$BASE/scripts/log/fit"
cd "$BASE/scripts"
python fitter.py
