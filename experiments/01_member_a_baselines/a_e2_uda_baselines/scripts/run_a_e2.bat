@echo off
cd /d "D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\01_member_a_baselines\a_e2_uda_baselines"
"D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe" scripts\run_all.py --epochs 75 > logs\run_all_console.log 2>&1
exit /b %ERRORLEVEL%
