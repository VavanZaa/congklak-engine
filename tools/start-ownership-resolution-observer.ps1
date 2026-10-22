$ErrorActionPreference = 'Stop'
$taskRoot = 'F:\My Engine\Congklak Stockfish'
$taskLog = Join-Path $taskRoot 'build\ownership-resolution-v1\observer-v1'
if (Test-Path -LiteralPath $taskLog) { throw 'Observer log namespace already exists' }
$taskPrior = Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'build\tracing-speed-v1\audit\observer-process-v3.json') | ConvertFrom-Json
$taskListener = @(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListener.Count -ne 1 -or $taskListener[0].OwningProcess -ne $taskPrior.process_id) { throw 'Prior listener identity differs' }
$taskOld = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPrior.process_id)"
$taskExpectedTime = [datetimeoffset]$taskPrior.creation_date
if ($taskOld.CommandLine -notlike '*-m congklak.tracing_speed_progress serve --port 18766*' -or
    $taskOld.ExecutablePath -ne $taskPrior.executable_path -or
    $taskOld.CreationDate.ToUniversalTime().Ticks -ne $taskExpectedTime.UtcDateTime.Ticks) { throw 'Prior process creation/image/command differs' }
Stop-Process -Id $taskPrior.process_id -Force
New-Item -ItemType Directory -Path $taskLog | Out-Null
$taskPython = (Get-Command python).Source
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-m','congklak.ownership_resolution_progress','serve','--port','18766') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLog 'stdout.log') -RedirectStandardError (Join-Path $taskLog 'stderr.log') -PassThru
$taskDeadline = [datetime]::UtcNow.AddSeconds(10)
do {
    Start-Sleep -Milliseconds 100
    $taskReady = Get-Content -Raw -LiteralPath (Join-Path $taskLog 'stdout.log') -ErrorAction SilentlyContinue
    if ([datetime]::UtcNow -gt $taskDeadline) { throw 'Observer startup timed out; logs retained' }
} while (-not $taskReady)
$taskRecord = $taskReady | ConvertFrom-Json
$taskActual = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)"
if ($taskActual.CommandLine -notlike '*-m congklak.ownership_resolution_progress serve --port 18766*') { throw 'New observer command differs' }
(Get-Process -Id $taskRecord.pid).ProcessorAffinity = [intptr]1
$taskNewListener = @(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskNewListener.Count -ne 1 -or $taskNewListener[0].OwningProcess -ne $taskRecord.pid) { throw 'New listener differs' }
[pscustomobject]@{ process_id=$taskRecord.pid; source_sha256=$taskRecord.source_sha256; parent_launch_id=$taskProcess.Id; command_line=$taskActual.CommandLine; address='127.0.0.1'; port=18766; affinity=1 } | ConvertTo-Json -Compress
