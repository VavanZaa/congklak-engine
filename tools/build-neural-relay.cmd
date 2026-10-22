@echo off
setlocal
call "%~1\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
pushd "%~dp0.."
if not exist build\neural-relay-obj mkdir build\neural-relay-obj
rem No global /arch:AVX2: scalar fallback and CPUID dispatch must run without AVX.
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\rules.cpp native\search.cpp native\neural_relay.cpp /Fobuild\neural-relay-obj\ /Febuild\congklak_neural_relay.dll /link /OUT:build\congklak_neural_relay.dll
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
