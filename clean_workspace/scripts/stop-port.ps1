param(
  [int]$Port = 8000
)
$lines = netstat -ano | Select-String ":$Port\b"
if ($lines) {
  $pids = @()
  foreach ($l in $lines) {
    $parts = ($l.ToString() -split "\s+") | Where-Object { $_ -ne "" }
    if ($parts.Length -gt 4) { $pids += $parts[-1] }
  }
  $pids = $pids | Select-Object -Unique
  foreach ($procId in $pids) {
    try { Stop-Process -Id $procId -Force -ErrorAction Stop; Write-Host "Stopped PID $procId" } catch { }
  }
} else {
  Write-Host "No process found on port $Port"
}
