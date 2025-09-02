from __future__ import annotations
import os
import sys
from pathlib import Path
import uvicorn


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # If running in containers/production, prefer multiple workers; default 1 for Windows dev
    default_workers = 1
    try:
        workers = int(os.getenv("WORKERS", str(default_workers)))
    except Exception:
        workers = default_workers
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    # Ensure project root (clean_workspace) is on sys.path so we can import app
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Import the FastAPI app object
    from app import app as fastapi_app

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
