#!/usr/bin/env python3
"""
Development server script with better error handling and environment validation.
"""
import sys
import os
import subprocess
from pathlib import Path
import socket

def check_environment():
    """Check if we're in the correct environment with required packages."""
    try:
        import fastapi
        import uvicorn
        import numpy
        import pandas
        print(f"✓ Environment OK: FastAPI {fastapi.__version__}, Uvicorn {uvicorn.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        # Try to hint correct working dir
        here = Path(__file__).resolve().parent
        req = here / "requirements.txt"
        if req.exists():
            print(f"Run: pip install -r {req}")
        else:
            print("Run: pip install -r requirements.txt")
        return False

def main():
    if not check_environment():
        sys.exit(1)
    
    # Default settings
    host = os.getenv("UVICORN_HOST", "127.0.0.1")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    reload = "--no-reload" not in sys.argv
    
    # Override from command line
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    
    # Ensure we run from the project folder containing app.py so imports work
    project_dir = Path(__file__).resolve().parent
    try:
        os.chdir(project_dir)
    except Exception:
        pass
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    # Quick port check (helpful error if already running)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            print(f"⚠ Port {port} is already in use on {host}. If an old server is stuck, run scripts/stop-port.ps1 -Port {port}.")
    except OSError:
        pass

    print(f"Starting development server on {host}:{port} (reload={'on' if reload else 'off'}) in {project_dir}")
    
    cmd = [
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
        # Explicitly watch the project dir and core module for changes
        cmd += ["--reload-dir", str(project_dir), "--reload-dir", str(project_dir / "core")]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"✗ Server failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
