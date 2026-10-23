@echo off
rem Start the ScanBound app on Windows.  The first run creates a Python environment in .venv next to
rem this script and installs ScanBound into it (a few minutes); later runs start straight away.
rem
rem   run_scanbound.bat [data folder or project.json]
rem   run_scanbound.bat --update        reinstall the dependencies, then start
setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "UPDATE=0"
if /i "%~1"=="--update" (
    set "UPDATE=1"
    shift
)

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating the Python environment in %VENV% ...
    where py >nul 2>nul && (py -3 -m venv "%VENV%") || (python -m venv "%VENV%")
    if not exist "%VENV%\Scripts\python.exe" (
        echo ScanBound needs Python 3.10 or newer: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

rem reinstall when asked, on the first run, and after an install that did not finish
if not exist "%VENV%\.installed" set "UPDATE=1"
if "%UPDATE%"=="1" (
    echo Installing ScanBound and its dependencies ...
    if exist "%VENV%\.installed" del "%VENV%\.installed"
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip
    pushd "%HERE%"
    "%VENV%\Scripts\python.exe" -m pip install -e ".[gui,clusters,formats]"
    if errorlevel 1 (
        popd
        pause
        exit /b 1
    )
    popd
    type nul > "%VENV%\.installed"
)

"%VENV%\Scripts\python.exe" -m scanbound.app %1 %2 %3 %4 %5 %6 %7 %8 %9
