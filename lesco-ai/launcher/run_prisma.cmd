@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%~dp0launcher.py"
) else (
    py -3 "%~dp0launcher.py"
)

if errorlevel 1 (
    echo.
    echo Prisma no pudo iniciarse. Revisa launcher\logs\backend.log.
    pause
)
endlocal
