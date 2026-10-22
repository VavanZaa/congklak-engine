$ErrorActionPreference = 'Stop'
$taskVswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVsPath) { throw 'C++ Build Tools are required' }
& "$PSScriptRoot\build-neural-fast.cmd" $taskVsPath
if ($LASTEXITCODE -ne 0) { throw "Fast neural build failed: $LASTEXITCODE" }
$taskRoot = Split-Path -Parent $PSScriptRoot
function Get-TaskSha([string]$taskFile) {
    $taskHasher = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($taskHasher.ComputeHash([IO.File]::ReadAllBytes($taskFile)))).Replace('-','').ToLowerInvariant() }
    finally { $taskHasher.Dispose() }
}
$taskInputs = @{}
foreach ($taskRelative in @('native/engine.hpp','native/rules.cpp','native/search.cpp','native/neural_fast.cpp','tools/build-neural-fast.cmd','tools/build-neural-fast.ps1')) {
    $taskInputs[$taskRelative] = Get-TaskSha (Join-Path $taskRoot $taskRelative)
}
$taskRecord = @{
    schema = 'neural-fast-build-v1'
    visual_studio = $taskVsPath
    flags = @('/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD')
    dispatch = 'CPUID/OSXSAVE/XGETBV/AVX2; baseline compiler architecture; separate mul/add; scalar fallback'
    inputs = $taskInputs
    dll_sha256 = Get-TaskSha (Join-Path $taskRoot 'build/congklak_neural_fast.dll')
}
[IO.File]::WriteAllText((Join-Path $taskRoot 'build/neural-fast-build.json'),($taskRecord | ConvertTo-Json -Depth 5),(New-Object System.Text.UTF8Encoding($false)))
