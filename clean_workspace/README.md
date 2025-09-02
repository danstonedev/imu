# Clean Hip Torque Core

Minimal, self-contained core for IMU-based hip torque and joint angles with robust gait cycle detection.

Folders

- `core/`: config constants, math, pipeline (FastAPI-ready backend helpers)
- `static/`: simple frontend to visualize results (served by `app.py`)
- `tests_clean/`: smoke test using sample data
- `docs/`: math notes

## Quick start

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

# Run tests
python -m pytest -q
# Run sample pipeline
python .\run_clean_pipeline.py --height 1.70 --mass 75.0 --data "..\sample data"

# Start API (http://127.0.0.1:8000)
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## Repo hygiene

- License: MIT (`LICENSE`)
- CI: GitHub Actions runs pytest on push/PR (`.github/workflows/ci.yml`)
- `.gitignore` excludes venvs, caches, build outputs

## Contributing

1. Branch off `main`
2. Add tests for behavior changes
3. Keep public APIs and file formats stable (CSV headers, JSON schema)
4. Ensure `python -m pytest -q` passes locally

## Publishing to GitHub

```powershell
# from repo root
git init
git add .
git commit -m "feat: initial clean core with CI and docs"
git branch -M main
git remote add origin https://github.com/<your-org>/<your-repo>.git
git push -u origin main

## Architecture

- Core pipeline path: `core/pipeline/pipeline.py` orchestrates calibration, per-side resampling, unified gait detection, stance, joint kinematics, and inverse dynamics.
- Active modules: `core/pipeline/unified_gait.py`, `core/pipeline/calibration.py`, `core/pipeline/io_utils.py`, `core/pipeline/stance_cycles.py`.
- Legacy modules: moved under `core/pipeline/legacy/` with shims for back-compat. Prefer unified_gait for event detection. Existing wrappers:
	- `core/pipeline/biomech_gait.py` → deprecated shim re-exporting `core/pipeline/legacy/biomech_gait.py`.
	- `core/pipeline/heel_strike_detection.py` → deprecated wrapper delegating to unified_gait/bilateral utilities.
