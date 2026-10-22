$ErrorActionPreference = 'Stop'
$taskVswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$taskVsPath = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVsPath) { throw 'C++ Build Tools are required' }
& "$PSScriptRoot\build-neural.cmd" $taskVsPath
if ($LASTEXITCODE -ne 0) { throw "Neural build failed: $LASTEXITCODE" }
