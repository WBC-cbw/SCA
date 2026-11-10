@echo off
setlocal
set "EXP=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e1_architecture_same_domain"
set "PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cd /d "%EXP%"
echo %DATE% %TIME% PILOT START resnet1d_s42_lr1e-3 >> "%EXP%\logs\run_all_status.log"
"%PY%" "%EXP%\scripts\train_b_e1.py" --arch resnet1d --seed 42 --lr 1e-3 --out-root "%EXP%\results_pilot" --tag pilot_lr1e-3 >> "%EXP%\logs\pilot_resnet1d_s42_lr1e-3.log" 2>&1
echo %DATE% %TIME% PILOT END rc=%errorlevel% >> "%EXP%\logs\run_all_status.log"
exit /b 0
