Dim sh, fso
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
sh.CurrentDirectory = "D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\02_member_b_method\b_e5_calibration_aware"
Do
  WScript.Sleep 60000
Loop Until fso.FileExists("logs\QUEUE3_DONE.txt")
sh.Run "cmd /c ""scripts\finalize_after_queue.bat""", 0, True
