param(
  [string]$Url = "http://127.0.0.1:8000/health"
)
try {
  $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
  if ($resp.StatusCode) { Write-Host $resp.StatusCode; exit 0 }
  else { Write-Host 0; exit 1 }
} catch {
  $code = 0
  if ($_.Exception -and $_.Exception.Response -and $_.Exception.Response.StatusCode) {
    $code = [int]$_.Exception.Response.StatusCode
  }
  Write-Host $code
  exit 2
}
