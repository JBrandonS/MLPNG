#!/usr/bin/env bash
set -euo pipefail

# Submit initializor_2 -> generator_2 job chains for one or more settings.
#
# Usage:
#   ./run.sh                # submits n64, n128, n256
#   ./run.sh n64            # submits only n64
#   ./run.sh n128 n256      # submits selected settings

cd "$(dirname "$0")"

TMP_SBATCH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/mlpng-run-XXXXXX")
trap 'rm -rf "$TMP_SBATCH_DIR"' EXIT

COMMON_ARGS=(
  --nsims 222
  --pols T
  --shapes all
  --lensing
  --phi_scale 1
  --narray 10
  --fnl_range -1000 1000
)

ARRAY_RANGE="1-10"
DEFAULT_SETTINGS=(n64 n128 n256)

validate_setting() {
  case "$1" in
    n64|n128|n256) ;;
    *)
      echo "Unsupported setting: $1"
      echo "Allowed values: n64 n128 n256"
      exit 1
      ;;
  esac
}

submit_for_setting() {
  local setting="$1"
  local settings_file="settings/${setting}.json"
  local init_sbatch="${TMP_SBATCH_DIR}/initializor_2_${setting}.sbatch"
  local gen_sbatch="${TMP_SBATCH_DIR}/generator_2_${setting}.sbatch"

  if [[ ! -f "$settings_file" ]]; then
    echo "Missing settings file: $settings_file"
    exit 1
  fi

  sed 's/mlpng.initializor/mlpng.initializor_2/' sbatch/initializor.sbatch > "$init_sbatch"
  sed 's/mlpng.generator/mlpng.generator_2/' sbatch/generator.sbatch > "$gen_sbatch"

  echo "Submitting initializor_2 for ${setting}..."
  local init_jid
  # init_jid=$(sbatch --parsable "$init_sbatch" "$settings_file" "${COMMON_ARGS[@]}")
  # init_jid=${init_jid%%;*}

  # echo "  initializor job id: ${init_jid}"
  # echo "Submitting generator_2 array for ${setting} with dependency afterok:${init_jid}..."

  local gen_jid
  gen_jid=$(sbatch --parsable --array="${ARRAY_RANGE}" "$gen_sbatch" "$settings_file" "${COMMON_ARGS[@]}")
  gen_jid=${gen_jid%%;*}

  echo "  generator job id: ${gen_jid}"
  echo
}

if [[ $# -gt 0 ]]; then
  SETTINGS=("$@")
else
  SETTINGS=("${DEFAULT_SETTINGS[@]}")
fi

for setting in "${SETTINGS[@]}"; do
  validate_setting "$setting"
  submit_for_setting "$setting"
done

echo "Done submitting requested jobs."
