$ErrorActionPreference = "Stop"

# 20 target machines
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

$user = "rares.marinescu"
$remoteDir = "~/4D_noise_FFT_plasma_on"

$pythonFile = "THOMAS_PCR_adaptive_FFT_sweep_xy_no_t_memory_optimized_noise.py"
$workerFile = "4D_noise_remote_pin_worker.sh"

# Check that the required local files exist before starting SSH/SCP work.
if (-not (Test-Path ".\$pythonFile")) {
    throw "Missing local file: $pythonFile"
}
if (-not (Test-Path ".\$workerFile")) {
    throw "Missing local file: $workerFile"
}

$start = 0.1

foreach ($machine in $machines) {
    $end = [math]::Round($start + 0.9, 1)

    $startStr = $start.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)
    $endStr   = $end.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)

    $session = "sweep_${startStr}_to_${endStr}"
    $logFile = "logs/${session}.log"
    $remote = "${user}@${machine}"

    Write-Host "Setting up $machine"

    # Create remote working directory and logs directory.
    ssh $remote "mkdir -p $remoteDir/logs"

    # Copy the simulation and worker files to the remote directory.
    scp ".\$pythonFile" "${remote}:$remoteDir/"
    scp ".\$workerFile" "${remote}:$remoteDir/"

    # Launch the run inside tmux. This is deliberately one line to avoid
    # fragile multiline PowerShell -> SSH -> bash quoting problems.
    $remoteCommand = "cd $remoteDir && chmod +x $workerFile && if tmux has-session -t '$session' 2>/dev/null; then echo 'Session $session already exists on $machine, skipping'; else tmux new-session -d -s '$session' './$workerFile $startStr $endStr > $logFile 2>&1' && echo 'Started $session on $machine'; fi"

    ssh $remote "$remoteCommand"

    Write-Host "Range $startStr to $endStr assigned to $machine"

    $start = [math]::Round($end + 0.1, 1)
}
