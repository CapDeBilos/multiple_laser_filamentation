#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Local launcher
# Starts one detached tmux sweep on each remote machine.
#
# Usage:
#   1) Edit MACHINES below.
#   2) Edit REMOTE_DIR below.
#   3) Run:
#        chmod +x launch_20_machine_sweep.sh
#        ./launch_20_machine_sweep.sh
#
# Requirement:
#   THOMAS_PCR_absorption_sweep.py and remote_pin_worker.sh
#   must already exist in REMOTE_DIR on every machine.
# ============================================================

REMOTE_DIR="~/pin_sweep"

MACHINES=(
rares.marinescu@hollande.polytechnique.fr
rares.marinescu@monaco.polytechnique.fr
rares.marinescu@pologne.polytechnique.fr
rares.marinescu@jabiru.polytechnique.fr
rares.marinescu@mouette.polytechnique.fr
rares.marinescu@nandou.polytechnique.fr
rares.marinescu@aerides.polytechnique.fr
rares.marinescu@gennaria.polytechnique.fr
rares.marinescu@habenaria.polytechnique.fr
rares.marinescu@ain.polytechnique.fr
rares.marinescu@dordogne.polytechnique.fr
rares.marinescu@jura.polytechnique.fr
rares.marinescu@ardennes.polytechnique.fr
rares.marinescu@thon.polytechnique.fr
rares.marinescu@raie.polytechnique.fr
rares.marinescu@baudroie.polytechnique.fr
rares.marinescu@radius.polytechnique.fr
rares.marinescu@metacarpe.polytechnique.fr
rares.marinescu@frontal.polytechnique.fr
rares.marinescu@cadillac.polytechnique.fr
)

# 200 values total: 0.1, 0.2, ..., 20.0
# 20 machines -> 10 values per machine.
START=0.1
CHUNK_WIDTH=0.9

for m in "${MACHINES[@]}"; do
    END=$(python3 - <<PY
start = float("$START")
print(f"{start + float("$CHUNK_WIDTH"):.1f}")
PY
)

    SESSION="sweep_${START}_to_${END}"
    echo "Starting $SESSION on $m"

    ssh "$m" "
        cd $REMOTE_DIR &&
        chmod +x remote_pin_worker.sh &&
        tmux new-session -d -s '$SESSION' './remote_pin_worker.sh $START $END'
    "

    START=$(python3 - <<PY
end = float("$END")
print(f"{end + 0.1:.1f}")
PY
)
done

echo
echo "All chunks launched."
echo "Check status with:"
echo "  ssh <machine> 'tmux ls'"
echo "  ssh <machine> 'tail -f $REMOTE_DIR/logs/sweep_0.1_to_1.0.log'"
