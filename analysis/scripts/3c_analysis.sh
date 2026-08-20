#!/bin/bash
#SBATCH -J analysis
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 1:15:00
#SBATCH --array=0
#SBATCH -o log/analysis/%j_%a.out
# #SBATCH values are Slurm-parsed defaults; runtime overrides come from load_config.sh (see _pipeline.yaml).

source "$(pwd)/load_config.sh"
source "$(pwd)/activate_env.sh"

export PYTHONUNBUFFERED=1

run_3c_analysis_steps() {
  set -euo pipefail
  ts() { date -Is; }
  echo "================================================================"
  echo "[analysis $(ts)] start (3c: parameter recovery, RFX BMC, resource rationality)"
  echo "[analysis $(ts)] HOST=$(hostname) USER=${USER:-} PWD=$(pwd)"
  echo "[analysis $(ts)] BASE=$BASE"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[analysis $(ts)] SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
    echo "[analysis $(ts)] SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-} SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
  else
    echo "[analysis $(ts)] (not under Slurm — FIT_LOCAL or interactive)"
  fi
  echo "[analysis $(ts)] Python: $(command -v python) — $(python --version 2>&1)"
  echo "----------------------------------------------------------------"
  echo "[analysis $(ts)] running analysis_parameterrecovery.py ..."
  step_start=$SECONDS
  python -u analysis_parameterrecovery.py
  echo "[analysis $(ts)] analysis_parameterrecovery.py finished ($((SECONDS - step_start))s wall)"
  echo "----------------------------------------------------------------"
  echo "[analysis $(ts)] running analysis_randeffects_bmc.py ..."
  step_start=$SECONDS
  python -u analysis_randeffects_bmc.py
  echo "[analysis $(ts)] analysis_randeffects_bmc.py finished ($((SECONDS - step_start))s wall)"
  echo "----------------------------------------------------------------"
  echo "[analysis $(ts)] running analysis_resource_rationality_sim.py ..."
  step_start=$SECONDS
  python -u analysis_resource_rationality_sim.py
  echo "[analysis $(ts)] analysis_resource_rationality_sim.py finished ($((SECONDS - step_start))s wall)"
  echo "================================================================"
  echo "[analysis $(ts)] done (success)"
}

if [[ "${FIT_LOCAL:-}" == "1" ]]; then
  mkdir -p "$BASE/scripts/log/analysis"
  cd "$BASE/workflows"
  run_3c_analysis_steps
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  sbatch --chdir="$BASE/scripts" "${SBATCH_EXTRA_ARGS[@]}" "$BASE/scripts/3c_analysis.sh"
  exit 0
fi

mkdir -p "$BASE/scripts/log/analysis"
cd "$BASE/workflows"
run_3c_analysis_steps
