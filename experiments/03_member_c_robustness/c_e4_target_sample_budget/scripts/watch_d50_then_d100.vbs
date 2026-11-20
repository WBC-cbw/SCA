Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
base = "D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\03_member_c_robustness\c_e4_target_sample_budget\"
Do
  If fso.FileExists(base & "logs\QUEUE_D50_DONE.txt") Then
    sh.Run """" & base & "scripts\run_queue_d100.bat""", 0, False
    Exit Do
  End If
  WScript.Sleep 60000
Loop
