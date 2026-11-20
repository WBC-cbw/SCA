# 只结束 train_budget.py 训练进程，避免误杀监控/收尾脚本
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*train_budget.py*' } |
  ForEach-Object {
    Write-Output ("kill " + $_.ProcessId + " " + $_.CommandLine)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
