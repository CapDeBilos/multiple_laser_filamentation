#!/bin/bash
set -euo pipefail

#
export LC_ALL=C
export LANG=C

USER="teofil.voicu"
REMOTE_DIR="~/Documents/S4_ELDYN/gaussian_4D_FFT_run"
MACHINES_FILE="machines.txt"
PYTHON_FILE="gaussian_4D_FFT_artificial_plasma_simulation.py"
WORKER_FILE="gaussian_4D_FFT_worker.sh"

# Exact Pin/Pcr values to run, one per machine
PINS=(1.1 1.3 1.4 1.6 1.7 1.8 1.9)

if [ ! -f "$MACHINES_FILE" ]; then
    echo "ERROR: machines.txt not found in current directory"
    exit 1
fi

mapfile -t MACHINES < "$MACHINES_FILE"

if [ ${#MACHINES[@]} -lt ${#PINS[@]} ]; then
    echo "ERROR: Need at least ${#PINS[@]} machines, but machines.txt has only ${#MACHINES[@]}"
    exit 1
fi

sanitize_pin() {
    local pin="$1"
    local whole tenth
    whole=$(printf "%.0f" "$(awk -v p="$pin" 'BEGIN {print int(p)}')")
    tenth=$(awk -v p="$pin" 'BEGIN {printf "%d", int((p - int(p))*10 + 0.5)}')
    printf "%03dp%d" "$whole" "$tenth"
}

for idx in "${!PINS[@]}"; do
    MACHINE="${MACHINES[$idx]}"
    PIN="${PINS[$idx]}"
    TAG=$(sanitize_pin "$PIN")
    SESSION="gaussian_Pin_${TAG}_Pcr"
    LOG="logs/${SESSION}.log"

    echo "===================================================="
    echo "Launching Pin/Pcr=$PIN on $MACHINE"
    echo "Session: $SESSION"
    echo "===================================================="

    ssh "$USER@$MACHINE" "mkdir -p $REMOTE_DIR/logs"
    scp "$PYTHON_FILE" "$USER@$MACHINE:$REMOTE_DIR/"
    scp "$WORKER_FILE" "$USER@$MACHINE:$REMOTE_DIR/"

    ssh "$USER@$MACHINE" "
        cd $REMOTE_DIR &&
        chmod +x $WORKER_FILE &&
        tmux kill-session -t $SESSION 2>/dev/null || true &&
        tmux new-session -d -s $SESSION './$WORKER_FILE $PIN > $LOG 2>&1' &&
        echo Started $SESSION
    "
done

echo "===================================================="
echo "All requested Gaussian simulations launched."
echo "Values: ${PINS[*]}"
echo "===================================================="
