@echo off
setlocal
cd /d D:/KimiData/kimi/tasks/2026-10-09/20-25-28-8c596874/sca_repo/experiments/02_member_b_method/b_e3_cdan_sca
set PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe
set MPLCONFIGDIR=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e3_cdan_sca\.mplconfig
if exist results\cdan_sca_adv2_desync50_s42\predictions_attack.npy (
  echo [%date% %time%] skip cdan_sca_adv2_desync50_s42 >> logs\adv2_status.txt
) else (
  echo [%date% %time%] start cdan_sca_adv2_desync50_s42 >> logs\adv2_status.txt
  "%PY%" scripts\train_cdan_sca_adv2.py --shift desync50 --seed 42 --out-root results --models-root D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\outputs\models\b_e3_adv2 >> logs\cdan_sca_adv2_desync50_s42.log 2>&1
  if errorlevel 1 (echo [%date% %time%] FAIL cdan_sca_adv2_desync50_s42 >> logs\adv2_status.txt) else (echo [%date% %time%] ok cdan_sca_adv2_desync50_s42 >> logs\adv2_status.txt)
)
if exist results\cdan_sca_adv2_desync50_s43\predictions_attack.npy (
  echo [%date% %time%] skip cdan_sca_adv2_desync50_s43 >> logs\adv2_status.txt
) else (
  echo [%date% %time%] start cdan_sca_adv2_desync50_s43 >> logs\adv2_status.txt
  "%PY%" scripts\train_cdan_sca_adv2.py --shift desync50 --seed 43 --out-root results --models-root D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\outputs\models\b_e3_adv2 >> logs\cdan_sca_adv2_desync50_s43.log 2>&1
  if errorlevel 1 (echo [%date% %time%] FAIL cdan_sca_adv2_desync50_s43 >> logs\adv2_status.txt) else (echo [%date% %time%] ok cdan_sca_adv2_desync50_s43 >> logs\adv2_status.txt)
)
if exist results\cdan_sca_adv2_desync50_s44\predictions_attack.npy (
  echo [%date% %time%] skip cdan_sca_adv2_desync50_s44 >> logs\adv2_status.txt
) else (
  echo [%date% %time%] start cdan_sca_adv2_desync50_s44 >> logs\adv2_status.txt
  "%PY%" scripts\train_cdan_sca_adv2.py --shift desync50 --seed 44 --out-root results --models-root D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\outputs\models\b_e3_adv2 >> logs\cdan_sca_adv2_desync50_s44.log 2>&1
  if errorlevel 1 (echo [%date% %time%] FAIL cdan_sca_adv2_desync50_s44 >> logs\adv2_status.txt) else (echo [%date% %time%] ok cdan_sca_adv2_desync50_s44 >> logs\adv2_status.txt)
)
if exist results\cdan_sca_adv2_desync100_s42\predictions_attack.npy (
  echo [%date% %time%] skip cdan_sca_adv2_desync100_s42 >> logs\adv2_status.txt
) else (
  echo [%date% %time%] start cdan_sca_adv2_desync100_s42 >> logs\adv2_status.txt
  "%PY%" scripts\train_cdan_sca_adv2.py --shift desync100 --seed 42 --out-root results --models-root D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\outputs\models\b_e3_adv2 >> logs\cdan_sca_adv2_desync100_s42.log 2>&1
  if errorlevel 1 (echo [%date% %time%] FAIL cdan_sca_adv2_desync100_s42 >> logs\adv2_status.txt) else (echo [%date% %time%] ok cdan_sca_adv2_desync100_s42 >> logs\adv2_status.txt)
)
if exist results\cdan_sca_adv2_desync100_s43\predictions_attack.npy (
  echo [%date% %time%] skip cdan_sca_adv2_desync100_s43 >> logs\adv2_status.txt
) else (
  echo [%date% %time%] start cdan_sca_adv2_desync100_s43 >> logs\adv2_status.txt
  "%PY%" scripts\train_cdan_sca_adv2.py --shift desync100 --seed 43 --out-root results --models-root D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\outputs\models\b_e3_adv2 >> logs\cdan_sca_adv2_desync100_s43.log 2>&1
  if errorlevel 1 (echo [%date% %time%] FAIL cdan_sca_adv2_desync100_s43 >> logs\adv2_status.txt) else (echo [%date% %time%] ok cdan_sca_adv2_desync100_s43 >> logs\adv2_status.txt)
)
if exist results\cdan_sca_adv2_desync100_s44\predictions_attack.npy (
  echo [%date% %time%] skip cdan_sca_adv2_desync100_s44 >> logs\adv2_status.txt
) else (
  echo [%date% %time%] start cdan_sca_adv2_desync100_s44 >> logs\adv2_status.txt
  "%PY%" scripts\train_cdan_sca_adv2.py --shift desync100 --seed 44 --out-root results --models-root D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\outputs\models\b_e3_adv2 >> logs\cdan_sca_adv2_desync100_s44.log 2>&1
  if errorlevel 1 (echo [%date% %time%] FAIL cdan_sca_adv2_desync100_s44 >> logs\adv2_status.txt) else (echo [%date% %time%] ok cdan_sca_adv2_desync100_s44 >> logs\adv2_status.txt)
)
echo done %date% %time% > logs\ADV2_DONE.txt
endlocal
