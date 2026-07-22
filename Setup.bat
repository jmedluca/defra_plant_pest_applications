@echo off
setlocal

cd /d "%~dp0"

set "BOOTSTRAP_PY="
set "BOOTSTRAP_ARGS="

if exist ".venv\Scripts\python.exe" (
  set "BOOTSTRAP_PY=%~dp0.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "BOOTSTRAP_PY=py"
    set "BOOTSTRAP_ARGS=-3"
  ) else (
    where python >nul 2>nul
    if not errorlevel 1 (
      set "BOOTSTRAP_PY=python"
    ) else (
      if exist "C:\Anaconda\python.exe" (
        set "BOOTSTRAP_PY=C:\Anaconda\python.exe"
      ) else if exist "C:\Anaconda3\python.exe" (
        set "BOOTSTRAP_PY=C:\Anaconda3\python.exe"
      ) else if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "BOOTSTRAP_PY=%LocalAppData%\Programs\Python\Python312\python.exe"
      ) else if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
        set "BOOTSTRAP_PY=%LocalAppData%\Programs\Python\Python311\python.exe"
      ) else if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
        set "BOOTSTRAP_PY=%LocalAppData%\Programs\Python\Python310\python.exe"
      )
    )
  )
)

if not exist ".venv\Scripts\python.exe" (
  if "%BOOTSTRAP_PY%"=="" (
    echo Python could not be found on this machine.
    echo.
    echo Install Python 3.10 or later, then run Setup.bat again.
    echo Recommended:
    echo 1. Install Python from python.org
    echo 2. Tick "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
  )
  echo Creating local Python environment...
  call "%BOOTSTRAP_PY%" %BOOTSTRAP_ARGS% -m venv .venv
  if errorlevel 1 (
    echo Failed to create the local Python environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate the local Python environment.
  pause
  exit /b 1
)

echo Installing application requirements...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo Failed while preparing Python package tools.
  pause
  exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed while installing requirements.
  echo.
  echo Common causes:
  echo 1. No internet connection
  echo 2. Firewall or proxy blocking Python package downloads
  echo 3. Python installation problem
  echo.
  echo If DEFRA runs this on a restricted machine, they may need internet access or an internal package mirror.
  pause
  exit /b 1
)

echo.
echo Setup complete.
echo Use "Run Dashboard.bat" or "Run Simulator.bat".
pause
