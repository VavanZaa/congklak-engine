param([int]$Port = 18767, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskProductRoot = Split-Path -Parent $PSScriptRoot
$taskPython = (Get-Command python -ErrorAction Stop).Source
$taskData = Join-Path $taskProductRoot 'data\product-v1'
if (-not (Test-Path -LiteralPath (Join-Path $taskProductRoot 'build\congklak_native.dll'))) {
  throw 'The CPU engine is missing. Run tools\build.ps1 once before starting the app.'
}
Write-Host "Congklak local app: http://127.0.0.1:$Port/"
Write-Host 'This window runs your local engine. Ctrl+C stops it; saved analyses remain available.'
$taskBrowserArgs = @()
if (-not $NoBrowser) { $taskBrowserArgs = @('--open-browser') }
Push-Location -LiteralPath $taskProductRoot
try { & $taskPython -B -m congklak.product_v1 --port $Port --data-dir $taskData @taskBrowserArgs; exit $LASTEXITCODE }
finally { Pop-Location }
