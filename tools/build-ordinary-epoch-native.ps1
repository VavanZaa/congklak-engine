$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskVswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath=& $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(-not $taskVsPath){throw 'C++ Build Tools required'}
& "$PSScriptRoot\build-ordinary-epoch-native.cmd" $taskVsPath
if($LASTEXITCODE -ne 0){throw "Fast trace build failed: $LASTEXITCODE"}
$taskInputs=@{}
foreach($taskName in @('engine.hpp','rules.cpp','search.cpp','neural_relay.cpp','neural_puct_cpu.cpp','neural_value_sidecar.cpp','trace.hpp','trace.cpp')){
 $taskRelative='native/ordinary_epoch_v1/'+$taskName
 $taskInputs[$taskRelative]=(Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach($taskRelative in @('tools/build-ordinary-epoch-native.cmd','tools/build-ordinary-epoch-native.ps1')){
 $taskInputs[$taskRelative]=(Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskRecord=@{schema='fast-complete-trace-build-v1';visual_studio=$taskVsPath;flags=@('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD');link_libraries=@('bcrypt.lib');inputs=$taskInputs;dll_sha256=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'build/ordinary-epoch-pilot-v1/native/congklak_ordinary_epoch_v1.dll') -Algorithm SHA256).Hash.ToLowerInvariant();event_bytes=24;owned_bytes=40;violation_bytes=16;maximum_events=16000000;maximum_claims=8000000;maximum_unique_owned=4000000;claim_slots=16777216}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/ordinary-epoch-pilot-v1/native/build.json'),($taskRecord|ConvertTo-Json -Depth 6),(New-Object System.Text.UTF8Encoding($false)))
