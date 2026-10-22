param([string]$Python = 'D:\3D Project\Ministral-3-14B-Reasoning-2512\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable is missing: $Python" }
$taskPython = (Resolve-Path -LiteralPath $Python).Path
$taskInfo = & $taskPython -c 'import json,sys,torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(16,16,device="cuda"); y=x@x; torch.cuda.synchronize(); print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))'
if ($LASTEXITCODE -ne 0) { throw 'Existing PyTorch environment failed CUDA verification' }
$taskRuntime = $taskInfo | ConvertFrom-Json
$taskBuild = Join-Path $taskRoot 'build'
New-Item -ItemType Directory -Force -Path $taskBuild | Out-Null
$taskRuntime | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBuild 'training-runtime.json') -Encoding UTF8
$taskRuntime | ConvertTo-Json
