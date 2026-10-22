$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskVswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVsPath) { throw 'C++ Build Tools are required' }
& "$PSScriptRoot\build-ownership-native-v1.cmd" $taskVsPath
if ($LASTEXITCODE -ne 0) { throw "Ownership native trace build failed: $LASTEXITCODE" }
$taskInputs = @{}
foreach ($taskRelative in @('native/ownership_v1/engine.hpp','native/ownership_v1/rules.cpp','native/ownership_v1/search.cpp','native/ownership_v1/neural_relay.cpp','native/ownership_v1/neural_puct_cpu.cpp','native/ownership_v1/neural_value_sidecar.cpp','native/ownership_v1/trace.hpp','native/ownership_v1/trace.cpp','tools/build-ownership-native-v1.cmd','tools/build-ownership-native-v1.ps1','tools/prepare-ownership-native-v1.py')) {
    $taskInputs[$taskRelative] = (Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskRecord = @{
    schema = 'complete-state-ownership-native-build-v1'
    visual_studio = $taskVsPath
    flags = @('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD')
    inputs = $taskInputs
    source_projection_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskRoot 'build/ownership-implementation-v1/native/source-projection.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    dll_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskRoot 'build/ownership-implementation-v1/native/congklak_ownership_trace_v1.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    event_bytes = 24
    capacity_events_max = 2000000
}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/ownership-implementation-v1/native/build.json'),($taskRecord | ConvertTo-Json -Depth 6),(New-Object System.Text.UTF8Encoding($false)))
