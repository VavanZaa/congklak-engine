@echo off
call "%~1\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0.."
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /LD native\puct_witness_v1.cpp /Fo:"build\tracing-speed-v1\native\puct_witness_v1.obj" /link /OUT:"build\tracing-speed-v1\native\puct_witness_v1.dll" /IMPLIB:"build\tracing-speed-v1\native\puct_witness_v1.lib"
exit /b %errorlevel%
