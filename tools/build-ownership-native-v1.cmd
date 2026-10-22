@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\ownership-implementation-v1\native\obj mkdir build\ownership-implementation-v1\native\obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\ownership_v1\rules.cpp native\ownership_v1\search.cpp native\ownership_v1\neural_value_sidecar.cpp native\ownership_v1\trace.cpp /Fobuild\ownership-implementation-v1\native\obj\ /Febuild\ownership-implementation-v1\native\congklak_ownership_trace_v1.dll /link /OUT:build\ownership-implementation-v1\native\congklak_ownership_trace_v1.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
