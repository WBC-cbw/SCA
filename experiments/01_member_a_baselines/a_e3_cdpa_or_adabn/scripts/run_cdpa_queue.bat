@echo off
setlocal
set PYTHONUNBUFFERED=1
set PY=D:/KimiData/kimi/tasks/2026-10-09/20-25-28-8c596874/venv_exp/Scripts/python.exe
cd /d "%~dp0."

echo [queue] START desync50 s42 %DATE% %TIME% >> ..\logs\queue.log
"%PY%" train_cdpa.py --shift desync50 --seed 42 --epochs 75 >> ..\logs\cdpa_desync50_s42.log 2>&1
echo [queue] END desync50 s42 exit=%ERRORLEVEL% %DATE% %TIME% >> ..\logs\queue.log
echo [queue] START desync50 s43 %DATE% %TIME% >> ..\logs\queue.log
"%PY%" train_cdpa.py --shift desync50 --seed 43 --epochs 75 >> ..\logs\cdpa_desync50_s43.log 2>&1
echo [queue] END desync50 s43 exit=%ERRORLEVEL% %DATE% %TIME% >> ..\logs\queue.log
echo [queue] START desync50 s44 %DATE% %TIME% >> ..\logs\queue.log
"%PY%" train_cdpa.py --shift desync50 --seed 44 --epochs 75 >> ..\logs\cdpa_desync50_s44.log 2>&1
echo [queue] END desync50 s44 exit=%ERRORLEVEL% %DATE% %TIME% >> ..\logs\queue.log
echo [queue] START desync100 s42 %DATE% %TIME% >> ..\logs\queue.log
"%PY%" train_cdpa.py --shift desync100 --seed 42 --epochs 75 >> ..\logs\cdpa_desync100_s42.log 2>&1
echo [queue] END desync100 s42 exit=%ERRORLEVEL% %DATE% %TIME% >> ..\logs\queue.log
echo [queue] START desync100 s43 %DATE% %TIME% >> ..\logs\queue.log
"%PY%" train_cdpa.py --shift desync100 --seed 43 --epochs 75 >> ..\logs\cdpa_desync100_s43.log 2>&1
echo [queue] END desync100 s43 exit=%ERRORLEVEL% %DATE% %TIME% >> ..\logs\queue.log
echo [queue] START desync100 s44 %DATE% %TIME% >> ..\logs\queue.log
"%PY%" train_cdpa.py --shift desync100 --seed 44 --epochs 75 >> ..\logs\cdpa_desync100_s44.log 2>&1
echo [queue] END desync100 s44 exit=%ERRORLEVEL% %DATE% %TIME% >> ..\logs\queue.log
echo [queue] ALL DONE %DATE% %TIME% > ..\logs\QUEUE_DONE.txt
