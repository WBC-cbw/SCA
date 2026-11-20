@echo off
setlocal
set PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\03_member_c_robustness\c_e4_target_sample_budget
if exist logs\QUEUE_D50_DONE.txt del logs\QUEUE_D50_DONE.txt
if exist logs\queue_d50_fail.txt del logs\queue_d50_fail.txt
echo started %DATE% %TIME% > logs\queue_d50_status.txt
for %%B in (50 500 5000) do (
  for %%S in (42 43 44) do (
    for %%M in (cdpa dann cdan_sca) do (
      echo [%DATE% %TIME%] start %%M b%%B s%%S >> logs\queue_d50_status.txt
      "%PY%" scripts\train_budget.py --method %%M --shift desync50 --budget %%B --seed %%S > logs\%%M_desync50_b%%B_s%%S.log 2>&1
      if errorlevel 1 (
        echo [%DATE% %TIME%] FAIL %%M b%%B s%%S >> logs\queue_d50_fail.txt
      ) else (
        echo [%DATE% %TIME%] ok %%M b%%B s%%S >> logs\queue_d50_status.txt
      )
    )
  )
)
echo done %DATE% %TIME% >> logs\queue_d50_status.txt
echo ok > logs\QUEUE_D50_DONE.txt
