$ErrorActionPreference = 'Stop'
$taskVswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVsPath) { throw 'C++ Build Tools are required' }
& "$PSScriptRoot\build-value-sidecar.cmd" $taskVsPath
if ($LASTEXITCODE -ne 0) { throw "Value sidecar build failed: $LASTEXITCODE" }
$taskRoot = Split-Path -Parent $PSScriptRoot
function Get-TaskSha([string]$taskFile) { return (Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash.ToLowerInvariant() }
$taskInputs = @{}
foreach ($taskRelative in @('native/engine.hpp','native/rules.cpp','native/search.cpp','native/neural_relay.cpp','native/neural_puct_cpu.cpp','native/neural_value_sidecar.cpp','tools/build-value-sidecar.cmd','tools/build-value-sidecar.ps1')) {
    $taskInputs[$taskRelative] = Get-TaskSha (Join-Path $taskRoot $taskRelative)
}
$taskRecord = @{
    schema = 'value-sidecar-native-build-v1'
    visual_studio = $taskVsPath
    flags = @('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD')
    dispatch = 'Frozen CPUID/OSXSAVE/XGETBV/AVX2 packed-double dispatch; scalar fallback, separate mul/add'
    inputs = $taskInputs
    dll_sha256 = Get-TaskSha (Join-Path $taskRoot 'build/congklak_value_sidecar.dll')
}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/value-sidecar-build.json'),($taskRecord | ConvertTo-Json -Depth 5),(New-Object System.Text.UTF8Encoding($false)))
