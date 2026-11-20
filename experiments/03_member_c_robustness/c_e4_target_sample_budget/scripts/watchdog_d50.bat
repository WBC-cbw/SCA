@echo off
setlocal enabledelayedexpansion
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\03_member_c_robustness\c_e4_target_sample_budget
if exist logs\QUEUE_D50_DONE.txt goto d100
if exist logs\QUEUE_D100_DONE.txt exit /b 0
set SHIFT=desync50
call :check
goto :eof

:d100
if exist logs\QUEUE_D100_DONE.txt exit /b 0
set SHIFT=desync100
call :check
goto :eof

:check
for /f %%i in ('powershell -NoProfile -Command "$f=Get-ChildItem logs\*_%SHIFT%_*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($f){[int]((Get-Date)-$f.LastWriteTime).TotalMinutes}else{999}"') do set AGE=%%i
if "%AGE%"=="" exit /b 0
if %AGE% GTR 900 exit /b 0
if %AGE% LSS 25 exit /b 0
echo [%date% %time%] STALL shift=%SHIFT% newest_log_age_min=%AGE% >> logs\watchdog.log
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kill_queue.ps1 >> logs\watchdog.log 2>&1
timeout /t 5 /nobreak >nul
if "%SHIFT%"=="desync50" (
  schtasks /end /tn CE4_D50 >> logs\watchdog.log 2>&1
  schtasks /run /tn CE4_D50 >> logs\watchdog.log 2>&1
) else (
  start "ce4_d100" /min cmd /c "scripts\run_queue_d100.bat >> logs\queue_d100_restart.log 2>&1"
)
schtasks /run /tn CE4_Mon >> logs\watchdog.log 2>&1
echo [%date% %time%] restarted %SHIFT% >> logs\watchdog.log
exit /b 0
