@echo off
REM Double-click this file to start the server and open the website.
REM Safe to double-click again while already running -- it stops the old
REM one first, so you never end up with two servers fighting over the
REM same port.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Could not find .venv\Scripts\activate.bat
    echo Make sure this file is in the clip_generation project folder and the venv is set up.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

REM Belt-and-suspenders: activate.bat only edits PATH, it doesn't verify
REM the venv is actually usable. If activation didn't take (VIRTUAL_ENV
REM not pointing at this project's .venv), stop now with a clear message
REM instead of silently falling through to a global Python that is
REM missing this project's packages.
if /i not "%VIRTUAL_ENV%"=="%~dp0.venv" (
    echo.
    echo ERROR: The virtual environment did not activate correctly.
    echo Expected VIRTUAL_ENV to be:
    echo   %~dp0.venv
    echo but it was:
    echo   %VIRTUAL_ENV%
    echo.
    echo The .venv may be broken or built from a Python install that no
    echo longer exists on this machine. Try deleting the .venv folder and
    echo running:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: The virtual environment is missing required packages
    echo ^(uvicorn not found^). Run this once to install them:
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Checking for an existing server on port 8001...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    echo Stopping existing process %%p on port 8001...
    taskkill /F /T /PID %%p >nul 2>&1
)

echo Starting server, please wait...
REM Opens the browser a few seconds after this window starts, giving the
REM server time to finish booting -- opening it instantly would just show
REM a "can't connect" error since uvicorn hasn't bound the port yet.
start "" /min cmd /c "timeout /t 5 /nobreak >nul & start "" http://localhost:8001"

uvicorn app.main:app --host 0.0.0.0 --port 8001

echo.
echo Server stopped.
pause
