' C-E4 收尾链：等 d50 完成 -> finalize -> 跑 d100 -> finalize（d100 数据也纳入）
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
base = "D:\KimiData\kimi\tasks\2026-10-09\20-25-28-8c596874\sca_repo\experiments\03_member_c_robustness\c_e4_target_sample_budget\"
logf = base & "logs\finalize_chain.log"
Sub Log(msg)
  Set f = fso.OpenTextFile(logf, 8, True)
  f.WriteLine Now & "  " & msg
  f.Close
End Sub

Log "chain started; waiting QUEUE_D50_DONE"
Do
  If fso.FileExists(base & "logs\QUEUE_D50_DONE.txt") Then Exit Do
  WScript.Sleep 60000
Loop
Log "d50 done -> finalize (d50 only)"
rc = sh.Run("cmd /c """ & base & "scripts\finalize_backup.bat""", 0, True)
Log "finalize rc=" & rc

Log "starting d100 queue"
rc2 = sh.Run("cmd /c """ & base & "scripts\run_queue_d100.bat""", 0, True)
Log "d100 queue rc=" & rc2

If fso.FileExists(base & "logs\QUEUE_D100_DONE.txt") Then
  Log "d100 done -> finalize (all shifts)"
  rc3 = sh.Run("cmd /c """ & base & "scripts\finalize_backup.bat""", 0, True)
  Log "finalize(all) rc=" & rc3
Else
  Log "d100 queue ended without DONE file; skip final finalize"
End If
Log "chain finished"
