#!/bin/bash

set -euo pipefail

export LC_ALL=C
export LANG=C

USER="teofil.voicu"
REMOTE_DIR="~/Documents/S4_ELDYN/Hermite_4D_FFT_run"
MACHINES_FILE="machines.txt"

PINS=()
for i in $(seq 1 20); do
    PINS+=("$i")
done

if [ ! -f "$MACHINES_FILE" ]; then
    echo "machines.txt not found"
    exit 1
fi

mapfile -t MACHINES < "$MACHINES_FILE"

if [ ${#MACHINES[@]} -ne ${#PINS[@]} ]; then
    echo "Mismatch: ${#MACHINES[@]} machines vs ${#PINS[@]} pins"
    exit 1
fi

for idx in "${!MACHINES[@]}"; do
    MACHINE="${MACHINES[$idx]}"
    PIN="${PINS[$idx]}"

    TAG=$(printf "%03dp%d" "${PIN%.*}" "${PIN#*.}")
    SESSION="hermite_Pin_${TAG}_Pcr"

    echo "Launching on $MACHINE with Pin/Pcr=$PIN"

    ssh "$USER@$MACHINE" "mkdir -p $REMOTE_DIR/logs"

    scp Hermite_4D_FFT_artificial_plasma_simulation.py "$USER@$MACHINE:$REMOTE_DIR/"
    scp Hermite_4D_FFT_worker.sh "$USER@$MACHINE:$REMOTE_DIR/"

    ssh "$USER@$MACHINE" "
        cd $REMOTE_DIR
        chmod +x Hermite_4D_FFT_worker.sh
        tmux kill-session -t $SESSION 2>/dev/null || true
        tmux new-session -d -s $SESSION './Hermite_4D_FFT_worker.sh $PIN > logs/${SESSION}.log 2>&1'
        echo Started $SESSION
    "
done

echo "All jobs launched."
