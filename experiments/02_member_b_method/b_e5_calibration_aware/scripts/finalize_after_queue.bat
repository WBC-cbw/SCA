@echo off
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware
set PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe
set MPLCONFIGDIR=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware\.mplconfig
if not exist .mplconfig mkdir .mplconfig
echo FINALIZE_START %DATE% %TIME% > logs\finalize.log
%PY% scripts\evaluate_b_e5.py >> logs\finalize.log 2>&1
%PY% scripts\aggregate_b_e5.py >> logs\finalize.log 2>&1
%PY% scripts\analyze_calibration_b_e5.py >> logs\finalize.log 2>&1
%PY% scripts\permutation_b_e5.py >> logs\finalize.log 2>&1
%PY% scripts\plot_b_e5.py >> logs\finalize.log 2>&1
%PY% scripts\make_manifest.py >> logs\finalize.log 2>&1
%PY% scripts\make_report.py >> logs\finalize.log 2>&1
echo FINALIZE_DONE %DATE% %TIME% >> logs\finalize.log
echo done > logs\FINALIZE_DONE.txt
