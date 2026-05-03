#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Remote worker script
# Runs one Pin/Pcr chunk on one remote machine inside tmux.
#
# Usage on remote:
#   ./remote_pin_worker.sh 0.1 1.0
#
# Assumes THOMAS_PCR_absorption_sweep.py is in the same folder.
# ============================================================

START_FACTOR="${1:?Need start Pin/Pcr factor, e.g. 0.1}"
END_FACTOR="${2:?Need end Pin/Pcr factor, e.g. 1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTDIR="pin_sweep_${START_FACTOR}_to_${END_FACTOR}"
LOGDIR="logs"
mkdir -p "$OUTDIR" "$LOGDIR"

echo "Host: $(hostname)"
echo "Running Pin/Pcr from $START_FACTOR to $END_FACTOR"
echo "Output folder: $OUTDIR"

# If you need conda, uncomment and edit:
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate cupy-clean

export CUDA_HOME=/usr/local/cuda-12.6.3
export CUDA_PATH=/usr/local/cuda-12.6.3
export CPATH="/usr/local/cuda-12.6.3/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6.3/bin:${PATH:-}"

python THOMAS_PCR_absorption_sweep.py \
    --start "$START_FACTOR" \
    --end "$END_FACTOR" \
    --step 0.1 \
    --output "$OUTDIR" \
    > "$LOGDIR/sweep_${START_FACTOR}_to_${END_FACTOR}.log" 2>&1

echo "Finished chunk $START_FACTOR to $END_FACTOR on $(hostname)"
