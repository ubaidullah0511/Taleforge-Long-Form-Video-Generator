@echo off
REM One-time environment setup: creates .venv and installs dependencies.
REM Detects whether an NVIDIA GPU is present and installs the matching
REM torch build. requirements.txt already pins the GPU (CUDA) build as the
REM project's default, so on GPU-less machines we skip that pinned line and
REM let pip resolve a CPU-only torch wheel instead, which is what the
REM README's "No NVIDIA GPU/driver?" section documents doing by hand.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo  Clip Generation - Environment Setup
echo ============================================
echo.

echo Checking for an NVIDIA GPU...
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    nvidia-smi >nul 2>&1
)
if errorlevel 1 (
    set HAS_GPU=0
    echo No NVIDIA GPU detected ^(nvidia-smi not found or failed^).
    echo Installing CPU-only dependencies.
) else (
    set HAS_GPU=1
    echo NVIDIA GPU detected.
    echo Installing GPU-accelerated ^(CUDA^) dependencies.
)
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create the virtual environment. Is Python installed
        echo and available on PATH? Get it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists, reusing .venv
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate the virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip

if "%HAS_GPU%"=="1" (
    REM requirements.txt's first two lines pin the CUDA torch build
    REM ^(--extra-index-url + torch==...+cuXXX^); on a GPU machine we install
    REM the file as-is so that pin is honored.
    echo Installing requirements ^(GPU / CUDA build, as pinned in requirements.txt^)...
    pip install -r requirements.txt
) else (
    REM On a machine with no NVIDIA GPU, the pinned +cuXXX wheel simply
    REM will not install ^(no matching CPU build under that version+local
    REM tag^), so we strip the CUDA-specific lines and let pip resolve a
    REM plain CPU torch wheel for everything else in the file, matching the
    REM README's documented CPU-only fallback.
    echo Installing requirements ^(CPU-only torch build^)...
    findstr /v /i "extra-index-url torch==" requirements.txt > requirements.cpu.tmp.txt
    pip install torch
    pip install -r requirements.cpu.tmp.txt
    del requirements.cpu.tmp.txt
)

if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed. Scroll up for the pip error.
    pause
    exit /b 1
)

echo.
echo Checking for the bundled ffmpeg...
if not exist "bin\ffmpeg.exe" (
    echo.
    echo WARNING: bin\ffmpeg.exe is missing. Audio transcription and video
    echo rendering both require it. It ships bundled with this project ^(no
    echo separate install needed^) -- if it's missing, this copy of the project
    echo is incomplete or bin\ffmpeg.exe was removed ^(e.g. by antivirus^).
    echo Restore it from a clean copy of the project before running the server.
    echo.
) else (
    echo Found bin\ffmpeg.exe -- no separate ffmpeg install needed.
)

echo.
echo ============================================
echo  Setup complete.
echo ============================================
if "%HAS_GPU%"=="1" (
    echo Verifying GPU torch build...
    python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
)
pause
