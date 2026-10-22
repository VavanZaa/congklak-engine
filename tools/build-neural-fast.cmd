@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\neural-fast-obj mkdir build\neural-fast-obj
rem No global /arch:AVX2: scalar fallback and CPUID dispatch must run without AVX.
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\rules.cpp native\search.cpp native\neural_fast.cpp /Fobuild\neural-fast-obj\ /Febuild\congklak_neural_fast.dll /link /OUT:build\congklak_neural_fast.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
