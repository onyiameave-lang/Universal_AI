@echo off
set PROJECT_DIR=MarketOracle-workspace

echo ===================================================
echo MarketOracle Ecosystem: Global Setup
echo ===================================================

if not exist "%PROJECT_DIR%" (
    echo [ERROR] Project directory "%PROJECT_DIR%" not found.
    echo Please run this script from the AI-ECOSYSTEM folder.
    pause
    exit /b 1
)

:: Move into the project directory to perform setup
pushd "%PROJECT_DIR%"

echo [1/3] Detecting Python installation...

:: Try the Python Launcher 'py' first
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto :process_setup
)

:: Try 'python' command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :process_setup
)

echo ===================================================
echo [ERROR] Python was not found.
echo 1. Download Python from: https://www.python.org/downloads/
echo 2. IMPORTANT: Check the box "Add Python to PATH".
echo ===================================================
popd
pause
exit /b 1

:process_setup
:: 1. Create Venv
if not exist venv (
    echo [1/3] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 ( echo [ERROR] Failed to create venv. && popd && pause && exit /b 1 )
) else (
    echo [1/3] Virtual environment already exists.
)

:: 2. Install dependencies
echo [2/3] Updating pip and installing dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip --default-timeout=1000 --retries 10

set REQS_FOUND=0
:: Search for requirements.txt files recursively within the project folder
for /r %%f in (requirements.txt) do (
    echo "%%f" | findstr /i /v "\\venv\\" >nul
    if %errorlevel% equ 0 (
        echo Installing: %%f
        venv\Scripts\python.exe -m pip install -r "%%f" --default-timeout=1000 --retries 10
        set /a REQS_FOUND+=1
    )
)

if %REQS_FOUND% equ 0 (
    echo [WARNING] No requirements.txt files found in %PROJECT_DIR%.
)

if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    popd
    pause
    exit /b 1
)

:: 3. Organize project folders
echo [3/3] Organizing project directories...
if exist experts\db_handler.py (
    venv\Scripts\python.exe -c "from experts.db_handler import ensure_dirs; ensure_dirs()"
)

echo ===================================================
echo SETUP COMPLETE!
echo To start, run: 
echo   cd %PROJECT_DIR%
echo   venv\Scripts\activate
echo ===================================================
popd
pause