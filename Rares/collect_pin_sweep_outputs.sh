#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Local collector
# Pulls all PNG and NPZ outputs from all machines into one folder.
#
# Usage:
#   chmod +x collect_pin_sweep_outputs.sh
#   ./collect_pin_sweep_outputs.sh
# ============================================================

REMOTE_DIR="~/pin_sweep"
LOCAL_OUT="all_pin_sweep_outputs"

MACHINES=(
machine01
machine02
machine03
machine04
machine05
machine06
machine07
machine08
machine09
machine10
machine11
machine12
machine13
machine14
machine15
machine16
machine17
machine18
machine19
machine20
)

mkdir -p "$LOCAL_OUT"

for m in "${MACHINES[@]}"; do
    echo "Collecting from $m"
    mkdir -p "$LOCAL_OUT/$m"

    rsync -av \
        "$m:$REMOTE_DIR/pin_sweep_*_to_*/*.png" \
        "$LOCAL_OUT/$m/" || true

    rsync -av \
        "$m:$REMOTE_DIR/pin_sweep_*_to_*/*.npz" \
        "$LOCAL_OUT/$m/" || true

    rsync -av \
        "$m:$REMOTE_DIR/logs/*.log" \
        "$LOCAL_OUT/$m/" || true
done

echo "Done. Outputs are in $LOCAL_OUT/"
