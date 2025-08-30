# Hip Torque Analysis API - Development Guide

## Server Setup

### ✅ Current Working Setup

**Single Method for Development:**
- Use VS Code Task: `Run API Server`
- This starts the FastAPI server with hot-reload on `http://127.0.0.1:8000`

### 🚀 Development Workflow

1. **Start Server:**
   - Press `Ctrl+Shift+P` → "Tasks: Run Task" → "Run API Server"
   - Or use Command Palette → "Run API Server"

2. **Stop Server:**
   - Use the stop script: `.\scripts\stop-port.ps1 -Port 8000`

3. **Health Check:**
   - Use the probe script: `.\scripts\probe.ps1 -Url http://127.0.0.1:8000/health`

4. **Run Tests:**
   - Use VS Code Task: `Run Tests`
   - Or manually: `python -m pytest test_api.py -v`

### 📁 Clean File Structure

**Server Files (Keep):**
- `app.py` - Main FastAPI application
- `.vscode/tasks.json` - VS Code tasks configuration
- `scripts/stop-port.ps1` - Utility to stop server
- `scripts/probe.ps1` - Utility to test endpoints

**Dependencies:**
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project configuration
- `.venv/` - Virtual environment

**Test Files:**
- `test_api.py` - API smoke tests

### ❌ Removed (Purged)

- `dev.py` - Redundant development script
- `scripts/start.ps1` - Redundant start script  
- `test_app_smoke.py` - Empty test file
- Direct `python app.py` execution support

### 🔧 Key Features

- **Hot Reload:** Server automatically reloads when code changes
- **Type Safety:** All type errors resolved
- **Stable Restart:** No more server crashes or port conflicts
- **Clean Architecture:** Single method for server management
- **Comprehensive Tests:** Working test suite with coverage

### 🚨 Important Notes

- Always use the VS Code task "Run API Server" for development
- The virtual environment is properly configured and required
- All dependencies are locked and synchronized
- Server runs on `127.0.0.1:8000` with CORS enabled
