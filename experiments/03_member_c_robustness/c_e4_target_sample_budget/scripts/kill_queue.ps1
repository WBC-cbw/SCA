# 结束 C-E4 训练/队列/收尾链进程（按命令行精确匹配），避免误杀监控或其他 python
$pats = @('*train_budget.py*', '*run_queue_d50.bat*', '*run_queue_d100.bat*', '*watch_finalize_chain.vbs*')
Get-CimInstance Win32_Process |
  Where-Object { $c = $_.CommandLine; $c -and ($pats | Where-Object { $c -like $_ }) } |
  ForEach-Object {
    Write-Output ("kill " + $_.ProcessId + " " + $_.Name + " " + $_.CommandLine)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
