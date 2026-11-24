@echo off
setlocal
cd /d D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e3_cdan_sca
:wait
if exist logs\ADV2_DONE.txt goto go
timeout /t 120 /nobreak >nul
goto wait
:go
echo [%date% %time%] ADV2_DONE seen, finalize start >> logs\adv2_finalize.log
"D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\venv_exp\Scripts\python.exe" scripts\finalize_adv2.py >> logs\adv2_finalize.log 2>&1
echo [%date% %time%] finalize rc=%errorlevel% >> logs\adv2_finalize.log
endlocal
