param([Parameter(Mandatory=$true)][string]$PreviousLog,[Parameter(Mandatory=$true)][string]$NewLog)
$ErrorActionPreference='Stop'
$taskRoot='F:\My Engine\Congklak Stockfish'
$taskBase=[IO.Path]::GetFullPath((Join-Path $taskRoot 'build\ordinary-epoch-pilot-v1'))
$taskPrevious=[IO.Path]::GetFullPath((Join-Path $taskRoot $PreviousLog))
$taskNext=[IO.Path]::GetFullPath((Join-Path $taskRoot $NewLog))
if (-not $taskPrevious.StartsWith($taskBase+'\') -or -not $taskNext.StartsWith($taskBase+'\')) { throw 'Observer namespace differs' }
if (Test-Path -LiteralPath $taskNext) { throw 'New log namespace exists' }
$taskReady=Get-Content -Raw -LiteralPath (Join-Path $taskPrevious 'stdout.log') | ConvertFrom-Json
$taskStamp=Get-Content -Raw -LiteralPath (Join-Path $taskPrevious 'process.json') | ConvertFrom-Json
$taskOld=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskReady.pid)"
if ($taskOld.CreationDate.ToUniversalTime().Ticks -ne ([datetimeoffset]$taskStamp.CreationDate).UtcDateTime.Ticks -or
    $taskOld.CommandLine -ne $taskStamp.CommandLine -or $taskOld.ExecutablePath -ne $taskStamp.ExecutablePath -or
    $taskOld.CommandLine -notlike '*-m congklak.ordinary_epoch_progress serve --port 18766*') { throw 'Old observer identity differs' }
$taskListeners=@(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListeners.Count -ne 1 -or $taskListeners[0].OwningProcess -ne $taskReady.pid) { throw 'Old listener differs' }
Stop-Process -Id $taskReady.pid -Force
New-Item -ItemType Directory -Path $taskNext | Out-Null
$taskPython=(Get-Command python).Source
$taskChild=Start-Process -FilePath $taskPython -ArgumentList @('-m','congklak.ordinary_epoch_progress','serve','--port','18766') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskNext 'stdout.log') -RedirectStandardError (Join-Path $taskNext 'stderr.log') -PassThru
$taskDeadline=[datetime]::UtcNow.AddSeconds(10)
do { Start-Sleep -Milliseconds 100; $taskNew=Get-Content -Raw -LiteralPath (Join-Path $taskNext 'stdout.log') -ErrorAction SilentlyContinue; if ([datetime]::UtcNow -gt $taskDeadline) { throw 'Observer startup timed out' } } while (-not $taskNew)
$taskRecord=$taskNew | ConvertFrom-Json
$taskActual=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)"
if ($taskActual.CommandLine -notlike '*-m congklak.ordinary_epoch_progress serve --port 18766*') { throw 'New command differs' }
(Get-Process -Id $taskRecord.pid).ProcessorAffinity=[intptr]1
$taskListeners=@(Get-NetTCPConnection -State Listen -LocalPort 18766)
if ($taskListeners.Count -ne 1 -or $taskListeners[0].OwningProcess -ne $taskRecord.pid) { throw 'New listener differs' }
$taskActual | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskNext 'process.json')
[pscustomobject]@{process_id=$taskRecord.pid;source_sha256=$taskRecord.source_sha256;command_line=$taskActual.CommandLine;affinity=1;port=18766} | ConvertTo-Json -Compress
