@echo off
setlocal enabledelayedexpansion
set "EXP=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e1_architecture_same_domain"
set "PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe"
set "STATUS=%EXP%\logs\run_all_status.log"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
cd /d "%EXP%"
for %%A in (resnet1d cnn) do (
  for %%S in (42 43 44) do call :one %%A %%S
)
echo %DATE% %TIME% ALL DONE >> "%STATUS%"
exit /b 0

:one
if exist "%EXP%\results\%~1_s%~2\predictions_attack.npy" (
  echo %DATE% %TIME% SKIP %~1_s%~2 >> "%STATUS%"
  exit /b 0
)
echo %DATE% %TIME% START %~1_s%~2 >> "%STATUS%"
"%PY%" "%EXP%\scripts\train_b_e1.py" --arch %~1 --seed %~2 --out-root "%EXP%\results" >> "%EXP%\logs\%~1_s%~2.log" 2>&1
echo %DATE% %TIME% END %~1_s%~2 rc=!errorlevel! >> "%STATUS%"
exit /b 0
