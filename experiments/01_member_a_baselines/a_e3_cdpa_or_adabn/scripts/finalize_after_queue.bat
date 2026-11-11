@echo off
setlocal
set PYTHONUNBUFFERED=1
set PY=D:/KimiData/kimi/tasks/2026-10-09/20-25-28-8c596874/venv_exp/Scripts/python.exe
cd /d "%~dp0."
:wait
if exist "..\logs\QUEUE_DONE.txt" goto run
timeout /t 60 /nobreak >nul
goto wait
:run
echo [finalize] START %DATE% %TIME% >> ..\logs\finalize.log
"%PY%" evaluate_a_e3.py >> ..\logs\finalize.log 2>&1
"%PY%" aggregate_a_e3.py >> ..\logs\finalize.log 2>&1
"%PY%" analyze_confidence_a_e3.py >> ..\logs\finalize.log 2>&1
"%PY%" plot_a_e3.py >> ..\logs\finalize.log 2>&1
"%PY%" plot_cdpa_dynamics.py >> ..\logs\finalize.log 2>&1
echo [finalize] DONE %DATE% %TIME% > ..\logs\FINALIZE_DONE.txt
