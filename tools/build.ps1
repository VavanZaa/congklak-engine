$ErrorActionPreference = 'Stop'
$taskVswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $taskVswhere)) { throw 'Visual Studio Build Tools with C++ support are required.' }
$taskVsPath = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVsPath) { throw 'No installed C++ toolchain found.' }
& "$PSScriptRoot\build-native.cmd" $taskVsPath
if ($LASTEXITCODE -ne 0) { throw "Native build failed: $LASTEXITCODE" }
$taskToolset = Get-ChildItem -LiteralPath "$taskVsPath\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1
$taskCompiler = Join-Path $taskToolset.FullName 'bin\Hostx64\x64\cl.exe'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskBuildRecord = @{
  compiler = 'MSVC C++20 /O2 /W4 /EHsc /LD'
  compilerVersion = (Get-Item -LiteralPath $taskCompiler).VersionInfo.FileVersion
  toolsetVersion = $taskToolset.Name
  nativeBuildSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskRoot 'build\congklak_native.dll')).Hash.ToLowerInvariant()
}
$taskBuildRecord | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $taskRoot 'build\native-build.json')
