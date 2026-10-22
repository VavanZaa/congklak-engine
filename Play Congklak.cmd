@echo off
pushd "%~dp0"
py -3.12 -B -m congklak.product_v2 --open-browser %*
set TASK_PLAY_EXIT=%ERRORLEVEL%
popd
exit /b %TASK_PLAY_EXIT%
