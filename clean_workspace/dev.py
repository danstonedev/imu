#!/usr/bin/env python3
"""
Development server script with better error handling and environment validation.
"""
import sys
import os
import subprocess
from pathlib import Path

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
    
    print(f"Starting development server on {host}:{port} (reload={'on' if reload else 'off'})")
    
    cmd = [
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"✗ Server failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
