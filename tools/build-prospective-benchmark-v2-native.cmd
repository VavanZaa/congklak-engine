@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\prospective-benchmark-v2\native\obj mkdir build\prospective-benchmark-v2\native\obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\prospective_benchmark_v2\rules.cpp native\prospective_benchmark_v2\search.cpp native\prospective_benchmark_v2\neural_value_sidecar.cpp native\prospective_benchmark_v2\trace.cpp /Fobuild\prospective-benchmark-v2\native\obj\ /Febuild\prospective-benchmark-v2\native\congklak_prospective_benchmark_v2.dll /link bcrypt.lib /OUT:build\prospective-benchmark-v2\native\congklak_prospective_benchmark_v2.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
