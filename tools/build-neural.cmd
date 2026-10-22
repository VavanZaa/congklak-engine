@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\neural-obj mkdir build\neural-obj
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\rules.cpp native\search.cpp native\neural.cpp /Fobuild\neural-obj\ /Febuild\congklak_neural.dll /link /OUT:build\congklak_neural.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
