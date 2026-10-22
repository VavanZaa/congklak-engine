@echo off
pushd "%~dp0"
py -3.12 -m congklak %*
set TASK_EXIT=%ERRORLEVEL%
popd
exit /b %TASK_EXIT%
