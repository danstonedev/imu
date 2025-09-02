# Hip Torque API - Development Commands

## Quick Start Development Server
```powershell
# Navigate to project root and start dev server with auto-reload
cd "C:\Users\danst\imu"
.venv\Scripts\Activate.ps1
cd clean_workspace
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Run Tests
```powershell
# Navigate to project root and run tests
cd "C:\Users\danst\imu"
.venv\Scripts\Activate.ps1
cd clean_workspace
python -m pytest test_api.py -v
```

## Alternative: Use Convenience Scripts
```powershell
# Start development server
powershell -File "C:\Users\danst\imu\scripts\dev-server.ps1"

# Run tests
powershell -File "C:\Users\danst\imu\scripts\test.ps1"
```

## Test API Manually
```powershell
# Health check
curl http://localhost:8000/health

# Full analysis test
curl -X POST http://localhost:8000/api/analyze/ -F "height_m=1.7" -F "mass_kg=70"
```

## Debug Pipeline Directly
```powershell
cd "C:\Users\danst\imu"
.venv\Scripts\Activate.ps1
cd clean_workspace
python debug_pipeline_output.py
```

## Key URLs
- **Web Interface**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Recent Fixes Applied
✅ **Fixed missing CSV generation**: All 10 expected CSV files now generated
- `left_csv` / `right_csv` (torque time series)
- `left_cycle_csv` / `right_cycle_csv` (torque cycle analysis)  
- `left_angles_csv` / `right_angles_csv` (joint angles time series)
- `left_hip_cycle_csv` / `right_hip_cycle_csv` (hip angle cycles)
- `left_knee_cycle_csv` / `right_knee_cycle_csv` (knee angle cycles)

✅ **Fixed development environment**: Now properly uses virtual environment
- All commands use `.venv\Scripts\Activate.ps1`
- Auto-reload enabled for development
- All dependencies installed in virtual environment
