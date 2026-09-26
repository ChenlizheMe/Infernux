@echo off
setlocal EnableExtensions
cd /d "%~dp0\..\.."

rem Build local Windows artifacts. Official publication is performed only by
rem .github/workflows/release.yml after SignPath approval.
set "VERSION=%~1"
if not defined VERSION set /p "VERSION=Infernux version (for example 0.4.0): "
if not defined VERSION (
    echo [ERROR] A version number is required.
    exit /b 2
)

where conda >nul 2>nul
if not errorlevel 1 (
    call conda activate infernux
) else if exist "%ProgramData%\anaconda3\condabin\conda.bat" (
    call "%ProgramData%\anaconda3\condabin\conda.bat" activate infernux
) else if exist "%UserProfile%\anaconda3\condabin\conda.bat" (
    call "%UserProfile%\anaconda3\condabin\conda.bat" activate infernux
) else if exist "%UserProfile%\miniconda3\condabin\conda.bat" (
    call "%UserProfile%\miniconda3\condabin\conda.bat" activate infernux
) else (
    echo [ERROR] Conda is not available. Install or initialize Conda first.
    exit /b 3
)
if errorlevel 1 (
    echo [ERROR] Failed to activate the infernux Conda environment.
    exit /b 4
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0release_hub.ps1" -Version "%VERSION%"
exit /b %ERRORLEVEL%
