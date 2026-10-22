$ErrorActionPreference = 'Stop'
$taskVswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVsPath) { throw 'C++ Build Tools are required' }
& "$PSScriptRoot\build-puct-cpu.cmd" $taskVsPath
if ($LASTEXITCODE -ne 0) { throw "PUCT CPU build failed: $LASTEXITCODE" }
$taskRoot = Split-Path -Parent $PSScriptRoot
function Get-TaskSha([string]$taskFile) {
    return (Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskInputs = @{}
foreach ($taskRelative in @('native/engine.hpp','native/rules.cpp','native/search.cpp','native/neural_relay.cpp','native/neural_puct_cpu.cpp','tools/build-puct-cpu.cmd','tools/build-puct-cpu.ps1')) {
    $taskInputs[$taskRelative] = Get-TaskSha (Join-Path $taskRoot $taskRelative)
}
$taskRecord = @{
    schema = 'puct-cpu-structured-build-v1'
    visual_studio = $taskVsPath
    flags = @('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD')
    dispatch = 'Frozen CPUID/OSXSAVE/XGETBV/AVX2 packed-double dispatch; scalar fallback, separate mul/add'
    inputs = $taskInputs
    dll_sha256 = Get-TaskSha (Join-Path $taskRoot 'build/congklak_puct_cpu.dll')
}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/puct-cpu-build.json'),($taskRecord | ConvertTo-Json -Depth 5),(New-Object System.Text.UTF8Encoding($false)))
