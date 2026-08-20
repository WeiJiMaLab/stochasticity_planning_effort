#!/usr/bin/env bash
# Source this from a stage script: `source "$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"`
# Puts pise_env's python/R on PATH and sets R_HOME, without depending on `conda
# activate` / `conda.sh` (not reliably on PATH in a fresh Slurm job — see
# no stage script activates pise_env itself).
#
# PISE_ENV_PREFIX can override the env location (default: ~/.conda/envs/pise_env,
# this repo's documented layout).

PISE_ENV_PREFIX="${PISE_ENV_PREFIX:-$HOME/.conda/envs/pise_env}"

if [[ ! -d "$PISE_ENV_PREFIX" ]]; then
  echo "activate_env.sh: pise_env not found at $PISE_ENV_PREFIX (set PISE_ENV_PREFIX to override)" >&2
  exit 1
fi

export PATH="$PISE_ENV_PREFIX/bin:$PATH"
export R_HOME="$PISE_ENV_PREFIX/lib/R"
