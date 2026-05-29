@echo off
REM ###############################################################################
REM # Setup Script for RFH-FL Classification Project (Windows, using uv)
REM # Automates environment creation and dependency installation
REM ###############################################################################

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_NAME=rfh-fl-classification
set PYTHON_VERSION=3.12
set VENV_PATH=.venv
set INSTALL_TYPE=
set COMMAND_ARG=%1

REM Colors
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"
set "INFO=[INFO]"

cls
echo.
echo ========================================
echo RFH-FL Classification Project Setup (uv)
echo ========================================
echo.
echo Fast and reliable environment setup with uv
echo.

REM Step 1: Check Python
echo %INFO% Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python is not installed or not in PATH
    echo Please install Python 3.12 or higher from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_INSTALLED=%%i
echo %SUCCESS% Python !PYTHON_INSTALLED! found
echo.

REM Step 2: Check and install uv
echo %INFO% Checking uv installation...
where uv >nul 2>&1
if errorlevel 1 (
    echo %WARNING% uv not found. Installing uv...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
    if errorlevel 1 (
        echo %WARNING% Attempting alternative installation method...
        pip install uv >nul 2>&1
    )
)

where uv >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Failed to install uv. Please visit https://docs.astral.build/uv/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('uv --version 2^>nul') do set UV_VERSION=%%i
echo %SUCCESS% !UV_VERSION! found
echo.

REM Step 3: Parse command-line arguments
if /i "!COMMAND_ARG!"=="--cuda" (
    set INSTALL_TYPE=gpu
    echo %INFO% GPU (CUDA 12.8) mode selected
    echo.
) else if /i "!COMMAND_ARG!"=="--cpu" (
    set INSTALL_TYPE=cpu
    echo %INFO% CPU-only mode selected
    echo.
) else if /i "!COMMAND_ARG!"=="--help" (
    echo Usage: setup.bat [OPTIONS]
    echo.
    echo Options:
    echo   --cuda    Install with CUDA 12.8 support (GPU)
    echo   --cpu     Install CPU-only version
    echo   --help    Show this help message
    echo.
    pause
    exit /b 0
) else if not "!COMMAND_ARG!"=="" (
    echo %ERROR% Unknown argument: !COMMAND_ARG!
    pause
    exit /b 1
) else (
    REM Interactive mode
    echo ========================================
    echo Installation Type Selection
    echo ========================================
    where nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        echo GPU (CUDA 12.8) support available
        echo.
        echo 1 = GPU (CUDA 12.8)
        echo 2 = CPU only
        set /p INSTALL_TYPE="Select (1 or 2): "
        if not "!INSTALL_TYPE!"=="1" set INSTALL_TYPE=cpu
        if "!INSTALL_TYPE!"=="1" set INSTALL_TYPE=gpu
    ) else (
        set INSTALL_TYPE=cpu
        echo %WARNING% GPU not detected. Using CPU mode.
    )
    echo.
)

REM Step 4: Check if venv already exists
if exist "!VENV_PATH!" (
    echo %WARNING% Virtual environment already exists
    set /p RECREATE="Remove and recreate? (y/n): "
    if /i "!RECREATE!"=="y" (
        echo %INFO% Removing old environment...
        rmdir /s /q "!VENV_PATH!" >nul 2>&1
    ) else (
        echo %INFO% Using existing environment
        goto activate_venv
    )
)

REM Step 5: Create virtual environment
echo ========================================
echo Creating Virtual Environment
echo ========================================
echo %INFO% Creating with uv...
uv venv "!VENV_PATH!" --python !PYTHON_VERSION! >nul 2>&1
if !errorlevel! equ 0 (
    echo %SUCCESS% Virtual environment created
) else (
    echo %ERROR% Failed to create virtual environment
    pause
    exit /b 1
)
echo.

REM Step 6: Activate virtual environment
:activate_venv
echo %INFO% Activating virtual environment...
call "!VENV_PATH!\Scripts\activate.bat"
echo %SUCCESS% Virtual environment activated
echo.

REM Step 7: Install dependencies
if "!INSTALL_TYPE!"=="gpu" (
    echo ========================================
    echo Installing dependencies (GPU - CUDA 12.8)
    echo ========================================
    echo %INFO% Installing PyTorch GPU...
    uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 ^
        --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    if !errorlevel! equ 0 (
        echo %SUCCESS% PyTorch GPU installed
    ) else (
        echo %ERROR% Failed to install PyTorch GPU
    )
) else (
    echo ========================================
    echo Installing dependencies (CPU)
    echo ========================================
    echo %INFO% Installing PyTorch CPU...
    uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 ^
        --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
    if !errorlevel! equ 0 (
        echo %SUCCESS% PyTorch CPU installed
    ) else (
        echo %ERROR% Failed to install PyTorch CPU
    )
)
echo.

echo %INFO% Installing project dependencies...
uv pip install -e ".[dev]" --no-cache-dir
if !errorlevel! equ 0 (
    echo %SUCCESS% Project dependencies installed
) else (
    echo %WARNING% Some dependencies may not have installed
)
echo.

REM Step 8: Verify installation
echo ========================================
echo Verifying Installation
echo ========================================

python -c "import torch; print('PyTorch:', torch.__version__)" 2>nul
if !errorlevel! equ 0 (
    echo %SUCCESS% PyTorch verified
) else (
    echo %ERROR% PyTorch verification failed
)

python -c "import numpy; print('NumPy:', numpy.__version__)" 2>nul
if !errorlevel! equ 0 (
    echo %SUCCESS% NumPy verified
)

python -c "import pandas; print('Pandas:', pandas.__version__)" 2>nul
if !errorlevel! equ 0 (
    echo %SUCCESS% Pandas verified
)

echo.

REM Step 9: Print final instructions
echo ========================================
echo Setup Complete!
echo ========================================
echo %SUCCESS% Environment setup completed successfully!
echo.
echo Next Steps:
echo 1. Activate the environment:
echo    !VENV_PATH!\Scripts\activate.bat
echo.
echo 2. Verify the setup:
echo    python -c "import torch; print(torch.__version__)"
echo.
echo 3. Run inference:
echo    python INFERENCE.py --input_dir ./data/test/ --output_dir ./results/
echo.
if "!INSTALL_TYPE!"=="gpu" (
    echo 4. Check GPU:
    echo    python -c "import torch; print(torch.cuda.is_available())"
    echo.
)
echo Using uv for package management:
echo   - Add packages: uv pip install package_name
echo   - List packages: uv pip list
echo   - Update project: uv sync
echo.
echo Documentation: README.md, MODEL_CARD.txt, INSTALLATION.md
echo.

pause
