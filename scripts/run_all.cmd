@echo off
setlocal

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set ts=%%i

call "%~dp0run_generation.cmd" > "logs\gen_%ts%.log" 2>&1
call "%~dp0analyze.cmd" > "logs\analyze_%ts%.log" 2>&1

endlocal
