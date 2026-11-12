@echo off
set PY=D:/KimiData/kimi/tasks/2026-10-09/20-25-28-8c596874/venv_exp/Scripts/python.exe
cd /d "%~dp0."
"%PY%" serve_monitor.py --port 8778 >> ..\logs\monitor.log 2>&1
