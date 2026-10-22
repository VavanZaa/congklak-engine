@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\tracing-speed-v1\native\obj mkdir build\tracing-speed-v1\native\obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\ownership_v2\rules.cpp native\ownership_v2\search.cpp native\ownership_v2\neural_value_sidecar.cpp native\ownership_v2\trace.cpp /Fobuild\tracing-speed-v1\native\obj\ /Febuild\tracing-speed-v1\native\congklak_trace_fast_v1.dll /link bcrypt.lib /OUT:build\tracing-speed-v1\native\congklak_trace_fast_v1.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
