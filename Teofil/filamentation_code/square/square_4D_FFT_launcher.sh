#!/bin/bash
set -euo pipefail

export LC_ALL=C
export LANG=C

# Change this only if your university login is different.
USER_NAME="teofil.voicu"
REMOTE_DIR="~/Documents/S4_ELDYN/square_4D_FFT_run"
MACHINES_FILE="machines.txt"
PYTHON_FILE="square_4D_FFT_artificial_plasma_simulation.py"
WORKER_FILE="square_4D_FFT_worker.sh"

# Run 20 simulations: Pin/Pcr = 1, 2, ..., 20.
PINS=()
for i in $(seq 1 20); do
    PINS+=("$i")
done

if [ ! -f "$MACHINES_FILE" ]; then
    echo "ERROR: $MACHINES_FILE not found in the current folder."
    exit 1
fi
if [ ! -f "$PYTHON_FILE" ]; then
    echo "ERROR: $PYTHON_FILE not found in the current folder."
    exit 1
fi
if [ ! -f "$WORKER_FILE" ]; then
    echo "ERROR: $WORKER_FILE not found in the current folder."
    exit 1
fi

mapfile -t MACHINES < <(grep -v '^[[:space:]]*$' "$MACHINES_FILE")

if [ "${#MACHINES[@]}" -ne "${#PINS[@]}" ]; then
    echo "ERROR: mismatch: ${#MACHINES[@]} machines vs ${#PINS[@]} Pin/Pcr values."
    echo "For this launcher, machines.txt must contain exactly 20 machines."
    exit 1
fi

sanitize_pin_tag() {
    # Input examples: 1, 1.0, 10, 20
    # Output examples: 001p0, 010p0, 020p0
    local pin="$1"
    awk -v p="$pin" 'BEGIN {
        whole = int(p);
        tenth = int((p - whole) * 10 + 0.5);
        if (tenth >= 10) { whole += 1; tenth = 0; }
        printf "%03dp%d", whole, tenth;
    }'
}

for idx in "${!MACHINES[@]}"; do
    MACHINE="${MACHINES[$idx]}"
    PIN="${PINS[$idx]}"
    TAG="$(sanitize_pin_tag "$PIN")"
    SESSION="square_Pin_${TAG}_Pcr"
    LOG_FILE="logs/${SESSION}.log"
    REMOTE="${USER_NAME}@${MACHINE}"

    echo "===== $MACHINE : Square Pin/Pcr=$PIN ====="

    ssh "$REMOTE" "mkdir -p $REMOTE_DIR/logs"
    scp "$PYTHON_FILE" "$REMOTE:$REMOTE_DIR/"
    scp "$WORKER_FILE" "$REMOTE:$REMOTE_DIR/"

    ssh "$REMOTE" "
        cd $REMOTE_DIR
        chmod +x $WORKER_FILE
        tmux kill-session -t '$SESSION' 2>/dev/null || true
        rm -f '$LOG_FILE'
        tmux new-session -d -s '$SESSION' './$WORKER_FILE $PIN > $LOG_FILE 2>&1'
        echo 'Started $SESSION on $MACHINE'
    "
done

echo "All 20 square-beam jobs launched."
