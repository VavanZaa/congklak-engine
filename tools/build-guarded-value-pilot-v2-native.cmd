@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\guarded-value-pilot-v2\native\obj mkdir build\guarded-value-pilot-v2\native\obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\guarded_value_pilot_v2\rules.cpp native\guarded_value_pilot_v2\search.cpp native\guarded_value_pilot_v2\neural_value_sidecar.cpp native\guarded_value_pilot_v2\trace.cpp /Fobuild\guarded-value-pilot-v2\native\obj\ /Febuild\guarded-value-pilot-v2\native\congklak_guarded_value_pilot_v2.dll /link bcrypt.lib /OUT:build\guarded-value-pilot-v2\native\congklak_guarded_value_pilot_v2.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
