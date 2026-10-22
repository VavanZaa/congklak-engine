@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build mkdir build
cl /nologo /std:c++20 /EHsc /O2 /W4 /LD native\rules.cpp native\search.cpp /Fobuild\ /Febuild\congklak_native.dll /link /OUT:build\congklak_native.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
