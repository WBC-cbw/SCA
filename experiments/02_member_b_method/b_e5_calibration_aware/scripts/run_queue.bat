@echo off
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware
set PY=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe
echo QUEUE_START %DATE% %TIME% > logs\queue.log
echo [run] desync50 s42 %TIME% >> logs\queue.log
%PY% scripts\train_cdan_sca_calib.py --shift desync50 --seed 42 > logs\calib_desync50_s42.log 2>&1
echo [done] desync50 s42 %TIME% >> logs\queue.log
echo [run] desync50 s43 %TIME% >> logs\queue.log
%PY% scripts\train_cdan_sca_calib.py --shift desync50 --seed 43 > logs\calib_desync50_s43.log 2>&1
echo [done] desync50 s43 %TIME% >> logs\queue.log
echo [run] desync50 s44 %TIME% >> logs\queue.log
%PY% scripts\train_cdan_sca_calib.py --shift desync50 --seed 44 > logs\calib_desync50_s44.log 2>&1
echo [done] desync50 s44 %TIME% >> logs\queue.log
echo [run] desync100 s42 %TIME% >> logs\queue.log
%PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 42 > logs\calib_desync100_s42.log 2>&1
echo [done] desync100 s42 %TIME% >> logs\queue.log
echo [run] desync100 s43 %TIME% >> logs\queue.log
%PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 43 > logs\calib_desync100_s43.log 2>&1
echo [done] desync100 s43 %TIME% >> logs\queue.log
echo [run] desync100 s44 %TIME% >> logs\queue.log
%PY% scripts\train_cdan_sca_calib.py --shift desync100 --seed 44 > logs\calib_desync100_s44.log 2>&1
echo [done] desync100 s44 %TIME% >> logs\queue.log
echo QUEUE_DONE %DATE% %TIME% >> logs\queue.log
