@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\ordinary-epoch-pilot-v1\native\obj mkdir build\ordinary-epoch-pilot-v1\native\obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\ordinary_epoch_v1\rules.cpp native\ordinary_epoch_v1\search.cpp native\ordinary_epoch_v1\neural_value_sidecar.cpp native\ordinary_epoch_v1\trace.cpp /Fobuild\ordinary-epoch-pilot-v1\native\obj\ /Febuild\ordinary-epoch-pilot-v1\native\congklak_ordinary_epoch_v1.dll /link bcrypt.lib /OUT:build\ordinary-epoch-pilot-v1\native\congklak_ordinary_epoch_v1.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
