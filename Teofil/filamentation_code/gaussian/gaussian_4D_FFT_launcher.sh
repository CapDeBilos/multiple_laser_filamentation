#!/bin/bash
set -euo pipefail

# Force decimal point instead of decimal comma.
export LC_ALL=C
export LANG=C

USER_NAME="teofil.voicu"
REMOTE_DIR="~/Documents/S4_ELDYN/gaussian_4D_FFT_run"
PYTHON_FILE="gaussian_4D_FFT_artificial_plasma_simulation.py"
WORKER_FILE="gaussian_4D_FFT_worker.sh"
MACHINES_FILE="machines.txt"

if [ ! -f "$MACHINES_FILE" ]; then
    echo "ERROR: $MACHINES_FILE not found"
    exit 1
fi

if [ ! -f "$PYTHON_FILE" ]; then
    echo "ERROR: $PYTHON_FILE not found"
    exit 1
fi

if [ ! -f "$WORKER_FILE" ]; then
    echo "ERROR: $WORKER_FILE not found"
    exit 1
fi

mapfile -t MACHINES < "$MACHINES_FILE"

# Pin/Pcr values: 0.1, 0.2, ..., 1.0
PINS=()
for i in $(seq 1 10); do
    val=$(awk -v i="$i" 'BEGIN {printf "%.1f", 0.1*i}')
    PINS+=("$val")
done

if [ "${#MACHINES[@]}" -ne "${#PINS[@]}" ]; then
    echo "ERROR: mismatch: ${#MACHINES[@]} machines but ${#PINS[@]} simulations"
    echo "For this launcher, machines.txt must contain exactly 10 machines."
    exit 1
fi

sanitize_pin() {
    # Example: 0.1 -> 000p1, 1.0 -> 001p0
    local pin="$1"
    local whole="${pin%.*}"
    local tenth="${pin#*.}"
    printf "%03dp%s" "$whole" "$tenth"
}

for idx in "${!MACHINES[@]}"; do
    MACHINE="${MACHINES[$idx]}"
    PIN="${PINS[$idx]}"
    TAG=$(sanitize_pin "$PIN")
    SESSION="gaussian_Pin_${TAG}_Pcr"
    LOGFILE="logs/${SESSION}.log"

    echo "===================================================="
    echo "Launching on $MACHINE with Pin/Pcr=$PIN"
    echo "Session: $SESSION"
    echo "===================================================="

    ssh "$USER_NAME@$MACHINE" "mkdir -p $REMOTE_DIR/logs"
    scp "$PYTHON_FILE" "$USER_NAME@$MACHINE:$REMOTE_DIR/"
    scp "$WORKER_FILE" "$USER_NAME@$MACHINE:$REMOTE_DIR/"

    ssh "$USER_NAME@$MACHINE" "
        cd $REMOTE_DIR &&
        chmod +x $WORKER_FILE &&
        tmux kill-session -t $SESSION 2>/dev/null || true &&
        tmux new-session -d -s $SESSION './$WORKER_FILE $PIN > $LOGFILE 2>&1'
    "
done

echo "===================================================="
echo "All Gaussian simulations launched."
echo "===================================================="
