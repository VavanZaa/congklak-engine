$ErrorActionPreference = 'Stop'
$taskRoot = 'F:\My Engine\Congklak Stockfish'
$taskReady = Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'build\ownership-resolution-v1\observer-v2\stdout.log') | ConvertFrom-Json
$taskOld = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskReady.pid)"
$taskOldStamp = Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'build\ownership-resolution-v1\observer-startup-debug\process-v2.json') | ConvertFrom-Json
$taskLog = Join-Path $taskRoot 'build\ownership-resolution-v1\observer-v3'
if (Test-Path -LiteralPath $taskLog) { throw 'New observer namespace already exists' }
if ($taskOld.CommandLine -notlike '*-m congklak.ownership_resolution_progress serve --port 18766*' -or
    $taskOld.ExecutablePath -ne $taskOldStamp.ExecutablePath -or
    $taskOld.CreationDate.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskOldStamp.CreationDate).UtcDateTime.Ticks -or
    $taskReady.source_sha256 -ne '303bcdc98f154b9e90a480ac3264caf604ba80d64ad9fc2dbb678211f1d4bfdc') { throw 'Old observer identity differs' }
$taskListeners = @(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListeners.Count -ne 1 -or $taskListeners[0].OwningProcess -ne $taskReady.pid) { throw 'Old listener differs' }
Stop-Process -Id $taskReady.pid -Force
New-Item -ItemType Directory -Path $taskLog | Out-Null
$taskPython=(Get-Command python).Source
$taskChild=Start-Process -FilePath $taskPython -ArgumentList @('-m','congklak.ownership_resolution_progress','serve','--port','18766') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLog 'stdout.log') -RedirectStandardError (Join-Path $taskLog 'stderr.log') -PassThru
$taskDeadline=[datetime]::UtcNow.AddSeconds(10)
do { Start-Sleep -Milliseconds 100; $taskNew=Get-Content -Raw -LiteralPath (Join-Path $taskLog 'stdout.log') -ErrorAction SilentlyContinue; if ([datetime]::UtcNow -gt $taskDeadline) { throw 'Observer startup timed out' } } while (-not $taskNew)
$taskRecord=$taskNew | ConvertFrom-Json
$taskActual=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)"
if ($taskActual.CommandLine -notlike '*-m congklak.ownership_resolution_progress serve --port 18766*') { throw 'New command differs' }
(Get-Process -Id $taskRecord.pid).ProcessorAffinity=[intptr]1
$taskListeners=@(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListeners.Count -ne 1 -or $taskListeners[0].OwningProcess -ne $taskRecord.pid) { throw 'New listener differs' }
[pscustomobject]@{ process_id=$taskRecord.pid; source_sha256=$taskRecord.source_sha256; command_line=$taskActual.CommandLine; affinity=1; address='127.0.0.1'; port=18766 } | ConvertTo-Json -Compress
