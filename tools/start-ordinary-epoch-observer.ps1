$ErrorActionPreference='Stop'
$taskRoot='F:\My Engine\Congklak Stockfish'
$taskOld=Get-CimInstance Win32_Process -Filter 'ProcessId = 17316'
$taskReady=Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'build\ownership-resolution-v1\observer-v3\stdout.log') | ConvertFrom-Json
$taskExpected=[datetimeoffset]'2026-10-16T22:08:53.809945+07:00'
if ($taskOld.CreationDate.ToUniversalTime().Ticks -ne $taskExpected.UtcDateTime.Ticks -or
    $taskOld.CommandLine -notlike '*-m congklak.ownership_resolution_progress serve --port 18766*' -or
    $taskOld.ExecutablePath -ne 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe' -or
    $taskReady.pid -ne 17316 -or $taskReady.source_sha256 -ne '5623b4c881105f5aa0ed2b4ff7796b0177e66c06a55d801d021bdf2d714990c5') { throw 'Old observer identity differs' }
$taskListeners=@(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListeners.Count -ne 1 -or $taskListeners[0].OwningProcess -ne 17316) { throw 'Old listener differs' }
$taskLog=Join-Path $taskRoot 'build\ordinary-epoch-pilot-v1\observer'
if (Test-Path -LiteralPath $taskLog) { throw 'Observer namespace exists' }
Stop-Process -Id 17316 -Force
New-Item -ItemType Directory -Path $taskLog | Out-Null
$taskPython=(Get-Command python).Source
$taskChild=Start-Process -FilePath $taskPython -ArgumentList @('-m','congklak.ordinary_epoch_progress','serve','--port','18766') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLog 'stdout.log') -RedirectStandardError (Join-Path $taskLog 'stderr.log') -PassThru
$taskDeadline=[datetime]::UtcNow.AddSeconds(10)
do { Start-Sleep -Milliseconds 100; $taskNew=Get-Content -Raw -LiteralPath (Join-Path $taskLog 'stdout.log') -ErrorAction SilentlyContinue; if ([datetime]::UtcNow -gt $taskDeadline) { throw 'Observer startup timed out' } } while (-not $taskNew)
$taskRecord=$taskNew | ConvertFrom-Json
$taskActual=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)"
if ($taskActual.CommandLine -notlike '*-m congklak.ordinary_epoch_progress serve --port 18766*') { throw 'New command differs' }
(Get-Process -Id $taskRecord.pid).ProcessorAffinity=[intptr]1
$taskListeners=@(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListeners.Count -ne 1 -or $taskListeners[0].OwningProcess -ne $taskRecord.pid) { throw 'New listener differs' }
$taskActual | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskLog 'process.json')
[pscustomobject]@{process_id=$taskRecord.pid;source_sha256=$taskRecord.source_sha256;command_line=$taskActual.CommandLine;affinity=1;port=18766} | ConvertTo-Json -Compress
