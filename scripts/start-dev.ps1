<#
Start-dev.ps1

Opens two new PowerShell windows:
- One runs the backend Flask app (keeps it running and shows logs)
- One serves the `health/template` folder over a simple static HTTP server (port 8000)

Usage: Right-click -> Run with PowerShell, or from an elevated PowerShell run:
  .\scripts\start-dev.ps1

#>

$repoRoot = Split-Path -Parent $PSScriptRoot

$backendDir = Join-Path $repoRoot 'backend'
$templateDir = Join-Path $repoRoot 'health\template'

Write-Host "Starting backend in new window: $backendDir"
Start-Process -FilePath powershell -ArgumentList "-NoExit","-Command","cd '$backendDir'; py -3 app.py" -WorkingDirectory $backendDir

Start-Sleep -Milliseconds 400

Write-Host "Starting static server in new window: $templateDir"
Start-Process -FilePath powershell -ArgumentList "-NoExit","-Command","cd '$templateDir'; py -3 -m http.server 8000" -WorkingDirectory $templateDir

Write-Host "Launched backend and static server. Open http://127.0.0.1:8000/health-risk-predictor.html in your browser."
