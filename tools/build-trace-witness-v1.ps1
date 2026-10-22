$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskVswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath=& $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(-not $taskVsPath){throw 'C++ Build Tools required'}
& "$PSScriptRoot\build-trace-witness-v1.cmd" $taskVsPath
if($LASTEXITCODE -ne 0){throw "Trace witness build failed: $LASTEXITCODE"}
$taskInputs=@{}
foreach($taskRelative in @('native/puct_witness_v1.cpp','tools/build-trace-witness-v1.cmd','tools/build-trace-witness-v1.ps1')){
 $taskInputs[$taskRelative]=(Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskRecord=@{schema='immutable-packed24-order-witness-build-v1';flags=@('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD');inputs=$taskInputs;dll_sha256=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'build/tracing-speed-v1/native/puct_witness_v1.dll') -Algorithm SHA256).Hash.ToLowerInvariant();search_calls=0;engine_dll_changed=$false}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/tracing-speed-v1/native/witness-build-v1.json'),($taskRecord|ConvertTo-Json -Depth 6),(New-Object System.Text.UTF8Encoding($false)))
