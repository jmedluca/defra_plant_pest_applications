@echo off
setlocal

set PYTHON_EXE=%~dp0.venv\Scripts\python.exe
set APP_FILE=plant_pest_dashboard.py
set APP_URL=http://127.0.0.1:8000

cd /d "%~dp0"

if not exist "%PYTHON_EXE%" (
  echo The local environment has not been set up yet.
  echo Please run Setup.bat first.
  pause
  exit /b 1
)

echo Starting dashboard...
echo The dashboard should open in your browser automatically.
echo If it does not, open this address manually:
echo %APP_URL%
echo.
"%PYTHON_EXE%" -m shiny run --host 127.0.0.1 --port 8000 --launch-browser --no-dev-mode "%APP_FILE%"
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
  echo.
  echo Dashboard stopped with exit code %EXITCODE%.
  echo Review the error above.
  pause
)

exit /b %EXITCODE%
