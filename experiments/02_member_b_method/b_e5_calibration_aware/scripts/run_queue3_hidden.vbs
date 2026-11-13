Dim sh
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware"
sh.Run "cmd /c ""scripts\run_queue3.bat""", 0, True
