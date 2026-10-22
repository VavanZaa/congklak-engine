$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskConfig = Join-Path $taskRoot 'build\training-runtime.json'
if (-not (Test-Path -LiteralPath $taskConfig)) { throw 'Run tools/setup-training.ps1 to select an existing CUDA PyTorch environment first.' }
$taskRuntime = Get-Content -LiteralPath $taskConfig -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $taskRuntime.python -PathType Leaf)) { throw 'Configured Python executable is missing; run tools/setup-training.ps1 again.' }
Push-Location -LiteralPath $taskRoot
try {
    & $taskRuntime.python -m congklak.training @args
    $taskExit = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExit
