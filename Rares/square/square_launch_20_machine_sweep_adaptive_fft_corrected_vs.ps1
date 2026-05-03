$machines = @(
"finlande.polytechnique.fr",
"irlande.polytechnique.fr",
"islande.polytechnique.fr",
"albatros.polytechnique.fr",
"autruche.polytechnique.fr",
"bengali.polytechnique.fr",
"telipogon.polytechnique.fr",
"vanda.polytechnique.fr",
"vanilla.polytechnique.fr",
"marne.polytechnique.fr",
"mayenne.polytechnique.fr",
"morbihan.polytechnique.fr",
"labre.polytechnique.fr",
"lieu.polytechnique.fr",
"lotte.polytechnique.fr",
"cote.polytechnique.fr",
"cubitus.polytechnique.fr",
"cuboide.polytechnique.fr",
"corvette.polytechnique.fr",
"fiat.polytechnique.fr"
)

$start = 0.1

foreach ($m in $machines) {

    $end = [math]::Round($start + 0.9, 1)

    $start_str = $start.ToString("0.0")
    $end_str   = $end.ToString("0.0")

    $session = "sweep_${start_str}_to_${end_str}"

    Write-Host "Setting up $m"

    ssh rares.marinescu@$m "
        mkdir -p ~/square_FFT_plasma_on/logs
    "

    scp .\square_THOMAS_PCR_adaptive_FFT_sweep_center_tmax.py rares.marinescu@${m}:~/square_FFT_plasma_on/
    scp .\square_remote_pin_worker_adaptive_fft_corrected.sh rares.marinescu@${m}:~/square_FFT_plasma_on/

    ssh rares.marinescu@$m "
        chmod +x ~/square_FFT_plasma_on/square_remote_pin_worker_adaptive_fft_corrected.sh;
        cd ~/square_FFT_plasma_on || exit 1;

        if tmux has-session -t $session 2>/dev/null; then
            echo 'Session $session already exists on $m, skipping';
        else
            tmux new-session -d -s $session './square_remote_pin_worker_adaptive_fft_corrected.sh $start_str $end_str > logs/$session.log 2>&1'
            echo 'Started $session on $m'
        fi
    "

    Write-Host "Range $start_str → $end_str assigned to $m"

    $start = [math]::Round($end + 0.1, 1)
}
