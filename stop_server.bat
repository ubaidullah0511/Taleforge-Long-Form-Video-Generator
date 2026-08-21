@echo off
REM Double-click this file to cleanly stop the server started by
REM start_server.bat -- no Task Manager/process-ID knowledge needed.
setlocal enabledelayedexpansion

echo Looking for a server running on port 8001...
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    echo Stopping process %%p on port 8001...
    taskkill /F /T /PID %%p >nul 2>&1
    set FOUND=1
)

if "!FOUND!"=="1" (
    echo Server stopped.
) else (
    echo No server was running on port 8001.
)
pause
