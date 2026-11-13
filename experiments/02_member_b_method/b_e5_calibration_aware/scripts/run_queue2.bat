@echo off
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware
set PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe
echo QUEUE2_START %DATE% %TIME% > logs\queue2.log
if exist logs\QUEUE2_DONE.txt del logs\QUEUE2_DONE.txt
echo [run] desync50 s44 %TIME% >> logs\queue2.log
%PY% scripts\train_cdan_sca_calib.py --shift desync50 --seed 44 > logs\calib_desync50_s44.log 2>&1
if errorlevel 1 (
  echo [retry] desync50 s44 %TIME% >> logs\queue2.log
  timeout /t 30 /nobreak > nul
  %PY% scripts\train_cdan_sca_calib.py --shift desync50 --seed 44 >> logs\calib_desync50_s44.log 2>&1
  if errorlevel 1 echo [FAIL] desync50 s44 %TIME% >> logs\queue2.log
)
echo [done] desync50 s44 %TIME% >> logs\queue2.log
echo [run] desync100 s42 %TIME% >> logs\queue2.log
%PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 42 > logs\calib_desync100_s42.log 2>&1
if errorlevel 1 (
  echo [retry] desync100 s42 %TIME% >> logs\queue2.log
  timeout /t 30 /nobreak > nul
  %PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 42 >> logs\calib_desync100_s42.log 2>&1
  if errorlevel 1 echo [FAIL] desync100 s42 %TIME% >> logs\queue2.log
)
echo [done] desync100 s42 %TIME% >> logs\queue2.log
echo [run] desync100 s43 %TIME% >> logs\queue2.log
%PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 43 > logs\calib_desync100_s43.log 2>&1
if errorlevel 1 (
  echo [retry] desync100 s43 %TIME% >> logs\queue2.log
  timeout /t 30 /nobreak > nul
  %PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 43 >> logs\calib_desync100_s43.log 2>&1
  if errorlevel 1 echo [FAIL] desync100 s43 %TIME% >> logs\queue2.log
)
echo [done] desync100 s43 %TIME% >> logs\queue2.log
echo [run] desync100 s44 %TIME% >> logs\queue2.log
%PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 44 > logs\calib_desync100_s44.log 2>&1
if errorlevel 1 (
  echo [retry] desync100 s44 %TIME% >> logs\queue2.log
  timeout /t 30 /nobreak > nul
  %PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 44 >> logs\calib_desync100_s44.log 2>&1
  if errorlevel 1 echo [FAIL] desync100 s44 %TIME% >> logs\queue2.log
)
echo [done] desync100 s44 %TIME% >> logs\queue2.log
echo QUEUE2_DONE %DATE% %TIME% >> logs\queue2.log
echo done > logs\QUEUE2_DONE.txt
