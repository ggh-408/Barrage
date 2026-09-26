param([switch]$PlanOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = 'D:\miniconda3\envs\barrage\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'The barrage Python environment is unavailable.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $taskRoot
try {
    if ($PlanOnly) {
        & $taskPython -B tools/train_targeted_dagger.py --plan
    } else {
        & $taskPython -B tools/train_targeted_dagger.py --train
    }
    if ($LASTEXITCODE -ne 0) { throw "DAgger entry exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
