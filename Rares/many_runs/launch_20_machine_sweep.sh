#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# launch_20_machine_sweep.sh
#
# Cylindrical THOMAS/PCR 20-machine launcher.
#
# Local requirements:
#   This file must be in the same local directory as:
#       remote_pin_worker.sh
#       THOMAS_PCR_absorption_sweep.py
#
# Remote layout created on each machine:
#   ~/cylindrical_after_the_delete_plasma_on/
#       remote_pin_worker.sh
#       THOMAS_PCR_absorption_sweep.py
#       logs/
#       pin_sweep_X_to_Y/
#
# Usage from Git Bash / WSL / Linux / macOS:
#   chmod +x launch_20_machine_sweep.sh
#   ./launch_20_machine_sweep.sh
# ============================================================

USER_NAME="rares.marinescu"
REMOTE_DIR="~/cylindrical_after_the_delete_plasma_on"

LOCAL_WORKER="remote_pin_worker.sh"
LOCAL_PYTHON="THOMAS_PCR_absorption_sweep.py"

machines=(
"hollande.polytechnique.fr"
"monaco.polytechnique.fr"
"pologne.polytechnique.fr"
"jabiru.polytechnique.fr"
"mouette.polytechnique.fr"
"nandou.polytechnique.fr"
"aerides.polytechnique.fr"
"gennaria.polytechnique.fr"
"habenaria.polytechnique.fr"
"ain.polytechnique.fr"
"dordogne.polytechnique.fr"
"jura.polytechnique.fr"
"ardennes.polytechnique.fr"
"thon.polytechnique.fr"
"raie.polytechnique.fr"
"baudroie.polytechnique.fr"
"radius.polytechnique.fr"
"metacarpe.polytechnique.fr"
"frontal.polytechnique.fr"
"cadillac.polytechnique.fr"
)

if [[ ! -f "$LOCAL_WORKER" ]]; then
    echo "ERROR: Cannot find $LOCAL_WORKER in the current local directory."
    exit 1
fi

if [[ ! -f "$LOCAL_PYTHON" ]]; then
    echo "ERROR: Cannot find $LOCAL_PYTHON in the current local directory."
    exit 1
fi

start="0.1"

for machine in "${machines[@]}"; do

    end="$(awk -v s="$start" 'BEGIN { printf "%.1f", s + 0.9 }')"

    start_tag="${start//./_}"
    end_tag="${end//./_}"
    session="sweep_${start_tag}_to_${end_tag}"

    echo
    echo "============================================================"
    echo "Machine: $machine"
    echo "Range:   Pin/Pcr = $start to $end"
    echo "Session: $session"
    echo "Remote:  $REMOTE_DIR"
    echo "============================================================"

    echo "[1/4] Creating remote folder..."
    ssh "${USER_NAME}@${machine}" "
        mkdir -p ${REMOTE_DIR}/logs
    "

    echo "[2/4] Uploading worker and Python simulation file..."
    scp "$LOCAL_WORKER" "${USER_NAME}@${machine}:${REMOTE_DIR}/"
    scp "$LOCAL_PYTHON" "${USER_NAME}@${machine}:${REMOTE_DIR}/"

    echo "[3/4] Making worker executable..."
    ssh "${USER_NAME}@${machine}" "
        chmod +x ${REMOTE_DIR}/remote_pin_worker.sh
    "

    echo "[4/4] Launching tmux job..."
    ssh "${USER_NAME}@${machine}" "
        cd ${REMOTE_DIR} || exit 1

        if tmux has-session -t ${session} 2>/dev/null; then
            echo 'Session ${session} already exists on ${machine}; skipping launch.'
        else
            tmux new-session -d -s ${session} './remote_pin_worker.sh ${start} ${end}'
            echo 'Started ${session} on ${machine}'
        fi
    "

    start="$(awk -v e="$end" 'BEGIN { printf "%.1f", e + 0.1 }')"

done

echo
echo "All launch commands have been sent."
echo
echo "To check machine 1:"
echo "  ssh ${USER_NAME}@hollande.polytechnique.fr \"cd ~/cylindrical_after_the_delete_plasma_on; tmux ls; ls; ls logs\""
echo
echo "To follow machine 1 log:"
echo "  ssh ${USER_NAME}@hollande.polytechnique.fr \"tail -f ~/cylindrical_after_the_delete_plasma_on/logs/sweep_0.1_to_1.0.log\""
