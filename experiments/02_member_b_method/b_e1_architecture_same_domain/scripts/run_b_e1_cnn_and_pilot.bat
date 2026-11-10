@echo off
setlocal enabledelayedexpansion
set "EXP=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e1_architecture_same_domain"
set "PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe"
set "STATUS=%EXP%\logs\run_all_status.log"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cd /d "%EXP%"
for %%S in (42 43 44) do call :cnn %%S
call :pilot
echo %DATE% %TIME% ALL DONE >> "%STATUS%"
exit /b 0

:cnn
if exist "%EXP%\results\cnn_s%~1\predictions_attack.npy" (
  echo %DATE% %TIME% SKIP cnn_s%~1 >> "%STATUS%"
  exit /b 0
)
echo %DATE% %TIME% START cnn_s%~1 >> "%STATUS%"
"%PY%" "%EXP%\scripts\train_b_e1.py" --arch cnn --seed %~1 --out-root "%EXP%\results" >> "%EXP%\logs\cnn_s%~1.log" 2>&1
echo %DATE% %TIME% END cnn_s%~1 rc=!errorlevel! >> "%STATUS%"
exit /b 0

:pilot
if exist "%EXP%\results_pilot\resnet1d_s42_pilot_lr1e-3\predictions_attack.npy" (
  echo %DATE% %TIME% SKIP resnet1d pilot >> "%STATUS%"
  exit /b 0
)
echo %DATE% %TIME% START resnet1d_s42_pilot_lr1e-3 >> "%STATUS%"
"%PY%" "%EXP%\scripts\train_b_e1.py" --arch resnet1d --seed 42 --lr 1e-3 --out-root "%EXP%\results_pilot" --tag pilot_lr1e-3 >> "%EXP%\logs\pilot_resnet1d_s42_lr1e-3.log" 2>&1
echo %DATE% %TIME% END resnet1d_s42_pilot_lr1e-3 rc=!errorlevel! >> "%STATUS%"
exit /b 0
