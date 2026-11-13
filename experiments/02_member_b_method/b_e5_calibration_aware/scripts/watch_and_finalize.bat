@echo off
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware
:wait
if exist logs\QUEUE_DONE.txt goto run
timeout /t 60 /nobreak > nul
goto wait
:run
timeout /t 20 /nobreak > nul
call scripts\finalize_after_queue.bat
