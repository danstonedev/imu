#!/usr/bin/env pwsh
# Development server startup script
# This script activates the virtual environment and starts the FastAPI server with auto-reload

Write-Host "🚀 Starting Hip Torque API Development Server..." -ForegroundColor Green

# Navigate to project root
Set-Location "C:\Users\danst\imu"

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Navigate to clean_workspace
Set-Location clean_workspace

# Check if required packages are installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor Yellow
$packages = @("fastapi", "uvicorn", "httpx")
foreach ($package in $packages) {
    $installed = pip list | Select-String $package
    if ($installed) {
        Write-Host "  ✅ $package installed" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $package not found" -ForegroundColor Red
    }
}

Write-Host "🌐 Starting server on http://localhost:8000" -ForegroundColor Cyan
Write-Host "📝 Auto-reload enabled - changes will restart the server" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the development server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
