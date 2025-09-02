#!/usr/bin/env pwsh
# Test runner script
# This script activates the virtual environment and runs tests

Write-Host "🧪 Running Hip Torque API Tests..." -ForegroundColor Green

# Navigate to project root and activate virtual environment
Set-Location "C:\Users\danst\imu"
& .venv\Scripts\Activate.ps1

# Navigate to clean_workspace where tests are located
Set-Location clean_workspace

# Run tests
Write-Host "🔬 Running pytest..." -ForegroundColor Cyan
python -m pytest test_api.py -v

Write-Host "✅ Tests completed!" -ForegroundColor Green
