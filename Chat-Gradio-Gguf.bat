REM .\Chat-Gradio-Gguf.bat
@echo off
setlocal enabledelayedexpansion

REM ==== Static Configuration ====
set "TITLE=Chat-Gradio-Gguf"
title %TITLE%
mode con cols=82 lines=25
powershell -noprofile -command "& { $w = $Host.UI.RawUI; $b = $w.BufferSize; $b.Height = 6000; $w.BufferSize = $b; }"

:: DP0 TO SCRIPT BLOCK
set "ScriptDirectory=%~dp0"
set "ScriptDirectory=%ScriptDirectory:~0,-1%"
cd /d "%ScriptDirectory%"
echo Dp0'd to Script.

REM ==== Admin Check ====
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Error: Admin Required!
    timeout /t 2 >nul
    echo Right Click, Run As Administrator.
    timeout /t 2 >nul
    goto :end_of_script
)
echo Status: Administrator
timeout /t 1 >nul

REM ==== Function Definitions ====
goto :SkipFunctions

REM ==== Separator Functions ====
:DisplaySeparatorThick
echo ===============================================================================
goto :eof

:DisplaySeparatorThin
echo -------------------------------------------------------------------------------
goto :eof

REM ============================================================
REM  Detect best Python 3.11-3.13 from common locations
REM ============================================================
:DetectPython
set "PYTHON_EXE="
set "BEST_MAJOR=0"
set "BEST_MINOR=0"

REM Check each root directory separately (handles spaces/parentheses safely)
call :CheckRoot "%ProgramFiles%"
call :CheckRoot "%ProgramFiles(x86)%"
call :CheckRoot "%LOCALAPPDATA%\Programs\Python"
call :CheckRoot "%USERPROFILE%\AppData\Local\Programs\Python"
if defined ProgramW6432 call :CheckRoot "%ProgramW6432%"

REM Fallback to PATH if nothing found in standard locations
if not defined PYTHON_EXE (
    for %%P in (python.exe) do (
        set "PYTHON_PATH=%%~$PATH:P"
        if defined PYTHON_PATH call :CheckPythonVersion "!PYTHON_PATH!"
    )
)

if not defined PYTHON_EXE (
    echo Error: No compatible Python 3.11-3.13 found.
    echo Please install Python 3.11, 3.12, or 3.13 from python.org
    pause
    goto :end_of_script
)
echo Selected Python: %PYTHON_EXE%
timeout /t 1 >nul
goto :eof

:CheckRoot
set "root=%~1"
if not defined root goto :eof
if not exist "%root%\" goto :eof
for /d %%D in ("%root%\Python3*") do (
    if exist "%%D\python.exe" call :CheckPythonVersion "%%D\python.exe"
)
goto :eof

:CheckPythonVersion
set "pyexe=%~1"
for /f "tokens=2 delims= " %%V in ('"%pyexe%" --version 2^>^&1') do (
    for /f "tokens=1,2 delims=." %%A in ("%%V") do (
        set "MAJ=%%A"
        set "MIN=%%B"
        if !MAJ!==3 if !MIN! GEQ 11 if !MIN! LEQ 13 (
            if !MIN! GTR !BEST_MINOR! (
                set "BEST_MAJOR=!MAJ!"
                set "BEST_MINOR=!MIN!"
                set "PYTHON_EXE=!pyexe!"
            )
        )
    )
)
goto :eof

REM ==== Main Menu ====
:MainMenu
CLS
color 0F
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Batch Menu
call :DisplaySeparatorThick
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo     1. Run Main Program
echo.
echo     2. Run Installation
echo.
echo.
echo.
echo.
echo.
echo.
echo.
call :DisplaySeparatorThick
set /p "choice=Selection; Menu Options = 1-2, Exit Batch = X: "

if /i "%choice%"=="1" goto :RunMainProgram
if /i "%choice%"=="2" goto :RunInstallation
if /i "%choice%"=="X" goto :end_of_script

echo Invalid selection. Please try again.
timeout /t 2 >nul
goto :MainMenu

REM ==== Option 1: Run Main Program ====
:RunMainProgram
cls
color 06
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Launcher
call :DisplaySeparatorThick
echo.
echo Starting %TITLE%...
set PYTHONUNBUFFERED=1

REM Call the venv Python directly - no activate/deactivate needed.
REM Python resolves its own site-packages relative to its executable location,
REM so the venv works correctly even when the project folder has been moved.
.\.venv\Scripts\python.exe -u .\launcher.py windows
if errorlevel 1 (
    echo Error launching %TITLE%
    pause
)
set PYTHONUNBUFFERED=0
goto :MainMenu

REM ==== Option 2: Run Installation ====
:RunInstallation
cls
color 06
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Installer
call :DisplaySeparatorThick
echo.

REM Detect Python before installation
call :DetectPython

"%PYTHON_EXE%" .\installer.py windows
if errorlevel 1 (
    echo Error during installation
    pause
)
pause
goto :MainMenu

:SkipFunctions

REM ==== Start at Main Menu ====
goto :MainMenu

:end_of_script
echo Closing %TITLE%...
timeout /t 2 >nul
exit