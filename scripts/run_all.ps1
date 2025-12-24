param()

$logDir = "logs"
if (-not (Test-Path $logDir)) {
  New-Item -ItemType Directory -Path $logDir | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
.\scripts\run_generation.ps1 *> "logs\gen_$ts.log"
.\scripts\analyze.ps1 *> "logs\analyze_$ts.log"
