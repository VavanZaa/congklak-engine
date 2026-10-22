$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskVswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath=& $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(-not $taskVsPath){throw 'C++ Build Tools required'}
& "$PSScriptRoot\build-trace-fast-v1.cmd" $taskVsPath
if($LASTEXITCODE -ne 0){throw "Fast trace build failed: $LASTEXITCODE"}
$taskInputs=@{}
foreach($taskName in @('engine.hpp','rules.cpp','search.cpp','neural_relay.cpp','neural_puct_cpu.cpp','neural_value_sidecar.cpp','trace.hpp','trace.cpp')){
 $taskRelative='native/ownership_v2/'+$taskName
 $taskInputs[$taskRelative]=(Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach($taskRelative in @('tools/build-trace-fast-v1.cmd','tools/build-trace-fast-v1.ps1')){
 $taskInputs[$taskRelative]=(Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskRecord=@{schema='fast-complete-trace-build-v1';visual_studio=$taskVsPath;flags=@('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD');link_libraries=@('bcrypt.lib');inputs=$taskInputs;dll_sha256=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'build/tracing-speed-v1/native/congklak_trace_fast_v1.dll') -Algorithm SHA256).Hash.ToLowerInvariant();event_bytes=24;owned_bytes=40;violation_bytes=16;maximum_events=2000000}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/tracing-speed-v1/native/build.json'),($taskRecord|ConvertTo-Json -Depth 6),(New-Object System.Text.UTF8Encoding($false)))
