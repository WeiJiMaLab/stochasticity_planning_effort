#!/bin/bash
#SBATCH -J preprocess
#SBATCH -c 24
#SBATCH --mem=24G
#SBATCH -t 00:15:00
#SBATCH --array=0
#SBATCH -o log/preprocess/%j_%a.out
# #SBATCH values are Slurm-parsed defaults; runtime overrides come from load_config.sh (see _pipeline.yaml).

source "$(pwd)/load_config.sh"
source "$(pwd)/activate_env.sh"

run_preprocess_steps() {
  set -euo pipefail
  ts() { date -Is; }
  echo "================================================================"
  echo "[preprocess $(ts)] start"
  echo "[preprocess $(ts)] HOST=$(hostname) USER=${USER:-} PWD=$(pwd)"
  echo "[preprocess $(ts)] BASE=$BASE"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[preprocess $(ts)] SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
    echo "[preprocess $(ts)] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-} SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
  else
    echo "[preprocess $(ts)] (not under Slurm — FIT_LOCAL or interactive)"
  fi
  echo "[preprocess $(ts)] Python: $(command -v python) — $(python --version 2>&1)"
  echo "----------------------------------------------------------------"
  echo "[preprocess $(ts)] running modelsimulation.py ..."
  python -u modelsimulation.py
  echo "[preprocess $(ts)] modelsimulation.py finished"
  echo "----------------------------------------------------------------"
  echo "[preprocess $(ts)] running split_data.py ..."
  python -u split_data.py
  echo "[preprocess $(ts)] split_data.py finished"
  echo "================================================================"
  echo "[preprocess $(ts)] done (success)"
}

if [[ "${FIT_LOCAL:-}" == "1" ]]; then
  cd "$BASE/scripts"
  run_preprocess_steps
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --chdir="$BASE/scripts" "${SBATCH_EXTRA_ARGS[@]}" "$BASE/scripts/1_preprocess.sh"
  exit 0
fi

mkdir -p "$BASE/scripts/log/preprocess"
cd "$BASE/scripts"
run_preprocess_steps
