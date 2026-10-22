@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\value-sidecar-obj mkdir build\value-sidecar-obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\rules.cpp native\search.cpp native\neural_value_sidecar.cpp /Fobuild\value-sidecar-obj\ /Febuild\congklak_value_sidecar.dll /link /OUT:build\congklak_value_sidecar.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
