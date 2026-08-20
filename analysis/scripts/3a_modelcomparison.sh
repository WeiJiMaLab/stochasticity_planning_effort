#!/bin/bash
#SBATCH -J modelcomparison
#SBATCH -c 1
#SBATCH --mem=3G
#SBATCH -t 1:15:00
#SBATCH --array=0-34
#SBATCH -o log/modelcomparison/%j_%a.out
# #SBATCH values are Slurm-parsed defaults; runtime overrides come from load_config.sh (see _pipeline.yaml).

source "$(pwd)/load_config.sh"
source "$(pwd)/activate_env.sh"
FIT="$BASE/fit"

export PYTHONUNBUFFERED=1

if [[ "${FIT_LOCAL:-}" == "1" ]]; then
  cd "$BASE/workflows"
  # --all walks every folder under $FIT, so this cannot drift out of sync with the
  # #SBATCH --array range above (a hardcoded loop here previously stopped at 30,
  # silently skipping the last folders).
  python -u analysis_modelcomparison.py --all --n-bootstrap 1000000 --fit-root "$FIT"
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --chdir="$BASE/scripts" "${SBATCH_EXTRA_ARGS[@]}" "$BASE/scripts/3a_modelcomparison.sh"
  exit 0
fi

mkdir -p "$BASE/scripts/log/modelcomparison"
cd "$BASE/workflows"
python -u analysis_modelcomparison.py --n-bootstrap 1000000 --fit-root "$FIT"
