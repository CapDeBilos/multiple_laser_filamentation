$machines = @(
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

$start = 0.1

foreach ($m in $machines) {

    $end = [math]::Round($start + 0.9, 1)

    $start_str = $start.ToString("0.0")
    $end_str   = $end.ToString("0.0")

    $session = "sweep_${start_str}_to_${end_str}"

    Write-Host "Setting up $m"

    ssh rares.marinescu@$m "
        mkdir -p ~/4D_FFT_plasma_on/logs
    "

    scp .\THOMAS_PCR_adaptive_FFT_sweep_xy_no_t_memory_optimized.py rares.marinescu@${m}:~/4D_FFT_plasma_on/
    scp .\4D_remote_pin_worker.sh rares.marinescu@${m}:~/4D_FFT_plasma_on/

    ssh rares.marinescu@$m "
        chmod +x ~/4D_FFT_plasma_on/4D_remote_pin_worker.sh;
        cd ~/4D_FFT_plasma_on || exit 1;

        if tmux has-session -t $session 2>/dev/null; then
            echo 'Session $session already exists on $m, skipping';
        else
            tmux new-session -d -s $session './4D_remote_pin_worker.sh $start_str $end_str > logs/$session.log 2>&1'
            echo 'Started $session on $m'
        fi
    "

    Write-Host "Range $start_str → $end_str assigned to $m"

    $start = [math]::Round($end + 0.1, 1)
}
