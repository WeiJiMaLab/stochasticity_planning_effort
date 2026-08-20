#!/usr/bin/env bash
# Source this from a stage script: `source "$(dirname "${BASH_SOURCE[0]}")/load_config.sh"`
# Exports BASE, SLURM_ACCOUNT, SLURM_MAIL_USER, SLURM_MAIL_TYPE, and the
# SBATCH_EXTRA_ARGS array from _pipeline.yaml (or pipeline.local.yaml, if present,
# which wins — see _pipeline.yaml's header).
#
# slurm.account / slurm.mail_user are optional: leave them unset (or empty) in
# your config if your cluster doesn't use Slurm accounts or you don't want mail
# notifications — SBATCH_EXTRA_ARGS omits --account/--mail-user/--mail-type
# entirely rather than passing them empty (`sbatch --account=` is a hard error on
# clusters that do require a valid account).
#
# Requires pyyaml in the active Python (already a pipeline dependency; see
# environment.yml / requirements.txt).

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$CONFIG_DIR/pipeline.local.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$CONFIG_DIR/_pipeline.yaml"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "load_config.sh: no _pipeline.yaml or pipeline.local.yaml found in $CONFIG_DIR" >&2
  exit 1
fi

eval "$(python3 - "$CONFIG_FILE" <<'EOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

def shq(value):
    """Single-quote a value for safe eval in the calling shell."""
    return "'" + str(value).replace("'", "'\\''") + "'"

print(f"BASE={shq(config['base'])}")
slurm = config.get('slurm', {}) or {}
account = slurm.get('account') or ''
mail_user = slurm.get('mail_user') or ''
mail_type = slurm.get('mail_type') or 'END'
print(f"SLURM_ACCOUNT={shq(account)}")
print(f"SLURM_MAIL_USER={shq(mail_user)}")
print(f"SLURM_MAIL_TYPE={shq(mail_type)}")
EOF
)"

SBATCH_EXTRA_ARGS=()
if [[ -n "$SLURM_ACCOUNT" ]]; then
  SBATCH_EXTRA_ARGS+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "$SLURM_MAIL_USER" ]]; then
  SBATCH_EXTRA_ARGS+=(--mail-user="$SLURM_MAIL_USER" --mail-type="$SLURM_MAIL_TYPE")
fi

# Must end on a success: an `if` with no matching branch would otherwise leave
# `source load_config.sh` returning non-zero, aborting callers that use `set -e`.
:
