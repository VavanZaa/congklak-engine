param([int]$Port = 18766)
$ErrorActionPreference = 'Stop'
if ($Port -ne 18766) { throw 'This observer is registered only for port 18766.' }
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$observerRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'build/prospective-benchmark-v5/observer'))
if (-not $observerRoot.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Observer output escaped the workspace.' }
$entryPath = Join-Path $workspaceRoot 'tools/progress-prospective-benchmark-v5-live.py'
$oldImage = 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe'
$oldCreated = [DateTimeOffset]::Parse('2026-10-17T14:40:03.6286630+07:00')
if (-not ('GuardedPilotObserverArgv' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class GuardedPilotObserverArgv {
 [DllImport("shell32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern IntPtr CommandLineToArgvW(string command, out int count);
 [DllImport("kernel32.dll")] public static extern IntPtr LocalFree(IntPtr value);
}
'@
}
function Read-Argv([string]$Line) {
    $argumentCount = 0
    $address = [GuardedPilotObserverArgv]::CommandLineToArgvW($Line, [ref]$argumentCount)
    if ($address -eq [IntPtr]::Zero) { throw 'Cannot inspect actual process arguments.' }
    try {
        $parts = @()
        for ($argumentIndex=0; $argumentIndex -lt $argumentCount; $argumentIndex++) {
            $pointer = [Runtime.InteropServices.Marshal]::ReadIntPtr($address, $argumentIndex*[IntPtr]::Size)
            $parts += [Runtime.InteropServices.Marshal]::PtrToStringUni($pointer)
        }
        return ,$parts
    } finally { [void][GuardedPilotObserverArgv]::LocalFree($address) }
}
function Read-Identity([int]$TargetId) {
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$TargetId" -ErrorAction SilentlyContinue
    if (-not $cim) { return $null }
    $process = Get-Process -Id $TargetId -ErrorAction Stop
    return [pscustomobject]@{ pid=$TargetId; creation=([DateTimeOffset]$cim.CreationDate).ToString('o'); image=$cim.ExecutablePath;
        argv=(Read-Argv $cim.CommandLine); command_line=$cim.CommandLine; affinity=$process.ProcessorAffinity.ToInt64() }
}
function Test-NewIdentity($Identity) {
    if (-not $Identity) { return $false }
    $parts = @($Identity.argv)
    return $parts.Count -eq 6 -and $parts[1] -ceq '-B' -and [IO.Path]::GetFullPath($parts[2]) -eq $entryPath -and
        $parts[3] -ceq 'serve' -and $parts[4] -ceq '--port' -and $parts[5] -ceq '18766' -and
        [IO.Path]::GetFileName($Identity.image) -match '^python(?:3(?:\.12)?)?\.exe$'
}
New-Item -ItemType Directory -Path $observerRoot -Force | Out-Null
$stopReceiptPath = Join-Path $observerRoot 'predecessor-stop.json'
$old = Read-Identity 60524
$oldStopped = $false
if ($old) {
    $parts = @($old.argv)
    $sameOld = ([DateTimeOffset]::Parse($old.creation)).UtcTicks -eq $oldCreated.UtcTicks -and
        $old.image -eq $oldImage -and $parts.Count -eq 6 -and $parts[1] -ceq '-B' -and
        [IO.Path]::GetFullPath($parts[2]) -eq (Join-Path $workspaceRoot 'tools/progress-prospective-benchmark-v4-live.py') -and
        $parts[3] -ceq 'serve' -and $parts[4] -ceq '--port' -and $parts[5] -ceq '18766' 
    if (-not $sameOld) { throw 'PID 60524 no longer matches the exact old observer identity; no process was terminated.' }
    $oldReceiptPath = Join-Path $workspaceRoot 'build/prospective-benchmark-v4/observer/process.json'
    $oldReceiptSha = (Get-FileHash -LiteralPath $oldReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($oldReceiptSha -ne '669720afc5a346273779a842d6d74dcdea193fdb41d4e4d8fe2abd8efe36b295') { throw 'The exact frozen v4 observer receipt changed.' }
    $oldReceipt = Get-Content -LiteralPath $oldReceiptPath -Raw | ConvertFrom-Json
    if ($oldReceipt.process.pid -ne $old.pid -or ([DateTimeOffset]$oldReceipt.process.creation).UtcTicks -ne ([DateTimeOffset]::Parse($old.creation)).UtcTicks -or
        $oldReceipt.process.image -ne $old.image -or $oldReceipt.process.command_line -ne $old.command_line) { throw 'Actual v4 observer no longer matches the exact frozen receipt.' }
    Stop-Process -Id 60524 -ErrorAction Stop
    for ($stopWait=0; $stopWait -lt 50; $stopWait++) {
        if (-not (Read-Identity 60524)) { break }
        Start-Sleep -Milliseconds 20
    }
    if (Read-Identity 60524) { throw 'Exact v4 observer remained alive after termination.' }
    $stopBody = [ordered]@{ schema='prospective-benchmark-v5-exact-observer-stop'; stopped=$true;
        checked_at=[DateTimeOffset]::Now.ToString('o'); actual_prior_process=$old;
        old_receipt_ref=@{ path='build/prospective-benchmark-v4/observer/process.json';sha256=$oldReceiptSha };
        source_inputs=@{ 'tools/start-prospective-benchmark-v5-observer-live.ps1'=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() };
        native_queries=0;model_forwards=0;http_requests=0 }
    if (Test-Path -LiteralPath $stopReceiptPath) { throw 'A v3 predecessor-stop receipt already exists; it cannot be overwritten.' }
    $stopText = $stopBody | ConvertTo-Json -Depth 10
    Set-Content -LiteralPath $stopReceiptPath -Value $stopText -Encoding utf8
    $oldStopped = $true
}
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
$reused = $false
$stdoutPath = $null
$stderrPath = $null
if ($listeners.Count) {
    if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1') { throw 'Port 18766 has an unexpected listener; no process was changed.' }
    $identity = Read-Identity $listeners[0].OwningProcess
    if (-not (Test-NewIdentity $identity)) { throw 'Port 18766 belongs to another process; no process was changed.' }
    $recordPath = Join-Path $observerRoot 'process.json'
    if (-not (Test-Path -LiteralPath $recordPath)) { throw 'Existing new observer has no source-bound launch receipt; inspect it before reuse.' }
    $previous = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if ($previous.process.pid -ne $identity.pid -or ([DateTimeOffset]$previous.process.creation).UtcTicks -ne ([DateTimeOffset]::Parse($identity.creation)).UtcTicks -or
        $previous.process.image -ne $identity.image -or $previous.process.command_line -ne $identity.command_line) { throw 'Existing observer receipt does not match its actual process.' }
    foreach ($property in $previous.source_inputs.PSObject.Properties) {
        $sourcePath = Join-Path $workspaceRoot $property.Name
        if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) { throw 'Existing observer loaded older source; do not reuse it.' }
    }
    $stdoutPath=$previous.stdout; $stderrPath=$previous.stderr; $reused=$true
} else {
    New-Item -ItemType Directory -Path $observerRoot -Force | Out-Null
    $runTag = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N')
    $stdoutPath = Join-Path $observerRoot ($runTag + '.stdout.log')
    $stderrPath = Join-Path $observerRoot ($runTag + '.stderr.log')
    if (-not (Test-Path -LiteralPath $oldImage)) { throw 'The explicitly registered Python image is unavailable.' }
    $launched = Start-Process -FilePath $oldImage -ArgumentList @('-B', ('"' + $entryPath + '"'), 'serve', '--port', '18766') -WorkingDirectory $workspaceRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    for ($attempt=0; $attempt -lt 100; $attempt++) {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
        if ($listeners.Count) { break }
        if ($launched.HasExited) { throw 'New observer exited before listening; retained its logs.' }
        Start-Sleep -Milliseconds 100
    }
    if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1' -or $listeners[0].OwningProcess -ne $launched.Id) { throw 'New observer listener/PID binding did not establish; retained evidence.' }
    $identity = Read-Identity $launched.Id
    if (-not (Test-NewIdentity $identity)) { throw 'New observer actual arguments/image differ from registered entry.' }
}
$actualProcess = Get-Process -Id $identity.pid -ErrorAction Stop
$actualProcess.ProcessorAffinity = [IntPtr]1
$identity = Read-Identity $identity.pid
if ($identity.affinity -ne 1) { throw 'Observer affinity was not restricted to logical CPU 0.' }
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
if ($listeners.Count -ne 1 -or $listeners[0].OwningProcess -ne $identity.pid -or $listeners[0].LocalAddress -ne '127.0.0.1') { throw 'Final observer listener binding differs.' }
$sourceInputs = [ordered]@{}
foreach ($sourceName in @('congklak/prospective_benchmark_v5_progress_live.py','tools/progress-prospective-benchmark-v5-live.py','congklak/guarded_value_pilot_v2_progress.py','congklak/config.py','congklak/neurodiag_common.py','tools/start-prospective-benchmark-v5-observer-live.ps1')) {
    $sourceInputs[$sourceName] = (Get-FileHash -LiteralPath (Join-Path $workspaceRoot $sourceName) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$startup = $null
for ($attempt=0; $attempt -lt 30; $attempt++) {
    if (Test-Path -LiteralPath $stdoutPath) {
        $line = Get-Content -LiteralPath $stdoutPath -TotalCount 1
        if ($line) { $startup = $line | ConvertFrom-Json; break }
    }
    Start-Sleep -Milliseconds 100
}
if (-not $startup -or $startup.event -cne 'prospective_benchmark_v5_live_observer_ready' -or
    $startup.pid -ne $identity.pid -or $startup.port -ne $Port) { throw 'Observer startup source proof does not bind the actual PID/listener.' }
foreach ($property in $startup.loaded_inputs.PSObject.Properties) {
    if (-not $sourceInputs.Contains($property.Name) -or $sourceInputs[$property.Name] -ne $property.Value) { throw 'Observer source changed during startup.' }
}
$receipt = [ordered]@{schema='prospective-benchmark-v5-observer-os-binding'; checked_at=[DateTimeOffset]::Now.ToString('o');
    process=$identity; listener=@{address='127.0.0.1';port=$Port;pid=$identity.pid}; old_observer_stopped=$oldStopped;
    predecessor_stop_ref=@{path='build/prospective-benchmark-v5/observer/predecessor-stop.json';sha256=(Get-FileHash -LiteralPath $stopReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()};
    reused=$reused; stdout=$stdoutPath; stderr=$stderrPath; source_inputs=$sourceInputs; loaded_inputs=$startup.loaded_inputs;
    http_requests=0; native_queries=0; model_forwards=0}
$receiptTag = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N')
$text = $receipt | ConvertTo-Json -Depth 10
Set-Content -LiteralPath (Join-Path $observerRoot ('process-' + $receiptTag + '.json')) -Value $text -Encoding utf8
Set-Content -LiteralPath (Join-Path $observerRoot 'process.json') -Value $text -Encoding utf8
$text
