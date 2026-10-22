@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\puct-cpu-obj mkdir build\puct-cpu-obj
rem Runtime AVX2 dispatch reuses frozen strict double math; no global /arch:AVX2.
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\rules.cpp native\search.cpp native\neural_puct_cpu.cpp /Fobuild\puct-cpu-obj\ /Febuild\congklak_puct_cpu.dll /link /OUT:build\congklak_puct_cpu.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
