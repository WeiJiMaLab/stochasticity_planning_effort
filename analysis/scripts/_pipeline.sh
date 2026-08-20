#!/usr/bin/env bash
# Local: 0_downloaddata.sh (data + data_split + fit from OSF, unless skipped).
# Slurm: preprocess → fit → (3a modelcomparison ∥ 3b analysis_main ∥ 3c analysis); all only afterok fit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_config.sh"
SCRIPTS="$BASE/scripts"

DOWNLOAD_SPLIT=1
DOWNLOAD_FIT=1

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash $0 [--ignore_split] [--ignore_fit]"
  echo "Runs 0_downloaddata.sh locally (data + data_split + fit from OSF), then submits:"
  echo "  1_preprocess → 2_fit → (3a_modelcomparison ∥ 3b_analysis_main ∥ 3c_analysis), each afterok fit."
  echo "  --ignore_split   skip downloading data_split.tar.gz (e.g. you'll re-split locally)"
  echo "  --ignore_fit     skip downloading fit.tar.gz (e.g. you'll refit from scratch)"
  echo "Logs: $SCRIPTS/log/{preprocess,fit,modelcomparison,analysis_main,analysis}/"
  exit 0
fi

for arg in "$@"; do
  case "$arg" in
    --ignore_split) DOWNLOAD_SPLIT=0 ;;
    --ignore_fit) DOWNLOAD_FIT=0 ;;
    *)
      echo "Unknown arg '$arg' (use -h)." >&2
      exit 1
      ;;
  esac
done

DOWNLOAD_TARGETS=(data)
[[ "$DOWNLOAD_SPLIT" == "1" ]] && DOWNLOAD_TARGETS+=(data_split)
[[ "$DOWNLOAD_FIT" == "1" ]] && DOWNLOAD_TARGETS+=(fit)

echo "==> Downloading: ${DOWNLOAD_TARGETS[*]}"
bash "$SCRIPT_DIR/0_downloaddata.sh" "${DOWNLOAD_TARGETS[@]}"

mkdir -p "$SCRIPTS/log/preprocess" "$SCRIPTS/log/fit" "$SCRIPTS/log/modelcomparison" \
  "$SCRIPTS/log/analysis_main" "$SCRIPTS/log/analysis"

j1=$(sbatch --parsable --chdir="$SCRIPTS" "${SBATCH_EXTRA_ARGS[@]}" "$SCRIPTS/1_preprocess.sh")
j2=$(sbatch --parsable --chdir="$SCRIPTS" "${SBATCH_EXTRA_ARGS[@]}" --dependency=afterok:"$j1" "$SCRIPTS/2_fit.sh")
j3a=$(sbatch --parsable --chdir="$SCRIPTS" "${SBATCH_EXTRA_ARGS[@]}" --dependency=afterok:"$j2" "$SCRIPTS/3a_modelcomparison.sh")
j3b=$(sbatch --parsable --chdir="$SCRIPTS" "${SBATCH_EXTRA_ARGS[@]}" --dependency=afterok:"$j2" "$SCRIPTS/3b_analysis_main.sh")
j3c=$(sbatch --parsable --chdir="$SCRIPTS" "${SBATCH_EXTRA_ARGS[@]}" --dependency=afterok:"$j2" "$SCRIPTS/3c_analysis.sh")

echo "Submitted preprocess=$j1  fit=$j2  3a_modelcomparison=$j3a  3b_analysis_main=$j3b  3c_analysis=$j3c"
echo "Monitor: squeue -u \$USER"
