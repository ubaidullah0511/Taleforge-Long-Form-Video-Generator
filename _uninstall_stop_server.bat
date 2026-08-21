@echo off
REM Used only by the NSIS uninstaller (installer.nsi) to stop a running
REM server before deleting files, so nothing is locked mid-uninstall.
REM Not meant to be double-clicked by users -- see stop_server.bat for that.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    taskkill /F /T /PID %%p >nul 2>&1
)
exit /b 0
