@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"
set "FRONTEND_DIR=%ROOT_DIR%frontend"
set "STATIC_DIR=%BACKEND_DIR%\static"
set "LANDING_SRC=%FRONTEND_DIR%\index.html"
set "LANDING_DEST=%STATIC_DIR%\landing.html"
set "APP_SRC=%FRONTEND_DIR%\index.html"
set "APP_DEST=%STATIC_DIR%\index.html"
set "YOLO_CONFIG_DIR=%BACKEND_DIR%"
set "PORT="

set "PYTHON_EXE=%ROOT_DIR%..\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if not exist "%BACKEND_DIR%\main.py" (
  echo.
  echo Could not find backend\main.py
  pause
  exit /b 1
)

if not exist "%STATIC_DIR%" (
  mkdir "%STATIC_DIR%"
)

if exist "%LANDING_SRC%" (
  copy /Y "%LANDING_SRC%" "%LANDING_DEST%" >nul
  echo Synced landing page to backend\static\landing.html
)

if exist "%APP_SRC%" (
  copy /Y "%APP_SRC%" "%APP_DEST%" >nul
  echo Synced app page to backend\static\index.html
)

for %%F in (login-bg.avif room-classroom.png) do (
  if exist "%FRONTEND_DIR%\%%F" (
    copy /Y "%FRONTEND_DIR%\%%F" "%STATIC_DIR%\%%F" >nul
  )
)

cd /d "%BACKEND_DIR%"
set "DEBUG=false"

echo Checking Python environment...
"%PYTHON_EXE%" -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
  echo.
  echo Missing backend dependencies in this Python environment.
  echo Install them with:
  echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
  echo.
  echo After that, run start.bat again.
  pause
  exit /b 1
)

for %%P in (8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010) do (
  netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul 2>&1
  if errorlevel 1 if not defined PORT set "PORT=%%P"
)
if not defined PORT (
  echo.
  echo No free port found in the range 8000-8010.
  pause
  exit /b 1
)

if not "%PORT%"=="8000" echo Port 8000 is busy. Using port %PORT% instead.

echo Starting SentinelEye on http://127.0.0.1:%PORT%
echo App:     http://127.0.0.1:%PORT%/app
echo Landing: http://127.0.0.1:%PORT%/
"%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port %PORT%
