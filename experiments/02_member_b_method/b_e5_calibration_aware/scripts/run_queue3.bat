@echo off
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware
set PYW=D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\pythonw.exe
echo QUEUE3_START %DATE% %TIME% > logs\queue3.log
:loop1
findstr /c:"epochs_done" results\cdan_sca_calib_desync50_s43\run_meta.json >nul 2>&1
if not errorlevel 1 goto next1
set /a A1+=1
if %A1% GTR 10 goto fail1
echo [try %A1%] desync50 s43 %TIME% >> logs\queue3.log
%PYW% scripts\train_cdan_sca_calib.py --shift desync50 --seed 43 >> logs\calib_desync50_s43.log 2>&1
findstr /c:"epochs_done" results\cdan_sca_calib_desync50_s43\run_meta.json >nul 2>&1
if not errorlevel 1 goto next1
timeout /t 40 /nobreak > nul
goto loop1
:fail1
echo [FAIL] desync50 s43 %TIME% >> logs\queue3.log
:next1
:loop2
findstr /c:"epochs_done" results\cdan_sca_calib_desync50_s44\run_meta.json >nul 2>&1
if not errorlevel 1 goto next2
set /a A2+=1
if %A2% GTR 10 goto fail2
echo [try %A2%] desync50 s44 %TIME% >> logs\queue3.log
%PYW% scripts\train_cdan_sca_calib.py --shift desync50 --seed 44 >> logs\calib_desync50_s44.log 2>&1
findstr /c:"epochs_done" results\cdan_sca_calib_desync50_s44\run_meta.json >nul 2>&1
if not errorlevel 1 goto next2
timeout /t 40 /nobreak > nul
goto loop2
:fail2
echo [FAIL] desync50 s44 %TIME% >> logs\queue3.log
:next2
:loop3
findstr /c:"epochs_done" results\cdan_sca_calib_desync100_s42\run_meta.json >nul 2>&1
if not errorlevel 1 goto next3
set /a A3+=1
if %A3% GTR 10 goto fail3
echo [try %A3%] desync100 s42 %TIME% >> logs\queue3.log
%PYW% scripts\train_cdan_sca_calib.py --shift desync100 --seed 42 >> logs\calib_desync100_s42.log 2>&1
findstr /c:"epochs_done" results\cdan_sca_calib_desync100_s42\run_meta.json >nul 2>&1
if not errorlevel 1 goto next3
timeout /t 40 /nobreak > nul
goto loop3
:fail3
echo [FAIL] desync100 s42 %TIME% >> logs\queue3.log
:next3
:loop4
findstr /c:"epochs_done" results\cdan_sca_calib_desync100_s43\run_meta.json >nul 2>&1
if not errorlevel 1 goto next4
set /a A4+=1
if %A4% GTR 10 goto fail4
echo [try %A4%] desync100 s43 %TIME% >> logs\queue3.log
%PYW% scripts\train_cdan_sca_calib.py --shift desync100 --seed 43 >> logs\calib_desync100_s43.log 2>&1
findstr /c:"epochs_done" results\cdan_sca_calib_desync100_s43\run_meta.json >nul 2>&1
if not errorlevel 1 goto next4
timeout /t 40 /nobreak > nul
goto loop4
:fail4
echo [FAIL] desync100 s43 %TIME% >> logs\queue3.log
:next4
:loop5
findstr /c:"epochs_done" results\cdan_sca_calib_desync100_s44\run_meta.json >nul 2>&1
if not errorlevel 1 goto next5
set /a A5+=1
if %A5% GTR 10 goto fail5
echo [try %A5%] desync100 s44 %TIME% >> logs\queue3.log
%PYW% scripts\train_cdan_sca_calib.py --shift desync100 --seed 44 >> logs\calib_desync100_s44.log 2>&1
findstr /c:"epochs_done" results\cdan_sca_calib_desync100_s44\run_meta.json >nul 2>&1
if not errorlevel 1 goto next5
timeout /t 40 /nobreak > nul
goto loop5
:fail5
echo [FAIL] desync100 s44 %TIME% >> logs\queue3.log
:next5
echo QUEUE3_DONE %DATE% %TIME% >> logs\queue3.log
echo done > logs\QUEUE3_DONE.txt
