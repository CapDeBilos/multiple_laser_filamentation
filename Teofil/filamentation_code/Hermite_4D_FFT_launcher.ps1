$ErrorActionPreference = "Stop"

# One machine per power value.
# Replace every EDIT_ME_MACHINE_* entry with a real different PC-room machine before running.
$machines = @(
"aerides.polytechnique.fr",
"barlia.polytechnique.fr",
"calanthe.polytechnique.fr",
"diuris.polytechnique.fr",
"encyclia.polytechnique.fr",
"epipactis.polytechnique.fr",
"gennaria.polytechnique.fr",
"habenaria.polytechnique.fr",
"isotria.polytechnique.fr",
"ipsea.polytechnique.fr",
"liparis.polytechnique.fr",
"lycaste.polytechnique.fr",
"malaxis.polytechnique.fr",
"neotinea.polytechnique.fr",
"oncidium.polytechnique.fr",
"ophrys.polytechnique.fr",
"orchis.polytechnique.fr",
"pleione.polytechnique.fr",
"pogonia.polytechnique.fr",
"serapias.polytechnique.fr",
"telipogon.polytechnique.fr",
"vanda.polytechnique.fr",
"vanilla.polytechnique.fr",
"xylobium.polytechnique.fr",
"zeuxine.polytechnique.fr",
"ain.polytechnique.fr",
"allier.polytechnique.fr",
"ardennes.polytechnique.fr",
"carmor.polytechnique.fr",
"charente.polytechnique.fr",
"cher.polytechnique.fr",
"creuse.polytechnique.fr",
"dordogne.polytechnique.fr",
"doubs.polytechnique.fr",
"essonne.polytechnique.fr",
"finistere.polytechnique.fr",
"gironde.polytechnique.fr",
"indre.polytechnique.fr",
"jura.polytechnique.fr",
"landes.polytechnique.fr"
)

$user = "teofil.voicu"
$remoteDir = "~/Documents/S4_ELDYN/Hermite_4D_FFT_run"
$pythonFile = "Hermite_4D_FFT_artificial_plasma_simulation.py"
$workerFile = "Hermite_4D_FFT_worker.sh"

if ($machines.Count -ne 40) {
    throw "Expected exactly 40 machines, found $($machines.Count)."
}
if (($machines | Select-Object -Unique).Count -ne 40) {
    throw "Machine list contains duplicates. Use 40 different machines."
}
if ($machines -match "EDIT_ME_MACHINE") {
    throw "Replace all EDIT_ME_MACHINE_* placeholders with real machine names before launching."
}
if (-not (Test-Path ".\$pythonFile")) { throw "Missing local file: $pythonFile" }
if (-not (Test-Path ".\$workerFile")) { throw "Missing local file: $workerFile" }

# Sweep: 0.5 Pcr, 1.0 Pcr, ..., 20.0 Pcr. This is 40 simulations.
$pins = 1..40 | ForEach-Object { [math]::Round(0.5 * $_, 1) }

for ($i = 0; $i -lt 40; $i++) {
    $machine = $machines[$i]
    $pin = $pins[$i]
    $pinStr = $pin.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)
    $pinTag = $pinStr.Replace(".", "p")
    $session = "hermite_Pin_${pinTag}_Pcr"
    $logFile = "logs/${session}.log"
    $remote = "${user}@${machine}"

    Write-Host "Setting up $machine for Hermite Pin/Pcr=$pinStr"

    ssh $remote "mkdir -p $remoteDir/logs"
    scp ".\$pythonFile" "${remote}:$remoteDir/"
    scp ".\$workerFile" "${remote}:$remoteDir/"

    $remoteCommand = "cd $remoteDir && chmod +x $workerFile && if tmux has-session -t '$session' 2>/dev/null; then echo 'Session $session already exists on $machine, skipping'; else tmux new-session -d -s '$session' './$workerFile $pinStr > $logFile 2>&1' && echo 'Started $session on $machine'; fi"
    ssh $remote $remoteCommand

    Write-Host "Assigned Hermite Pin/Pcr=$pinStr to $machine"
}
