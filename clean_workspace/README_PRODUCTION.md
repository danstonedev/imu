# Production deployment guide

- App entry: FastAPI app in `app.py`.
- Prod runner: `python scripts/serve.py` (uses uvicorn).
- Config via environment variables (see below).

## Environment variables

- HOST (default 0.0.0.0)
- PORT (default 8000)
- WORKERS (default 1; set >1 on Linux/containers)
- LOG_LEVEL (info|warning|error)
- ALLOWED_ORIGINS (comma-separated; default `*`)
- ALLOW_CREDENTIALS (true/false; default true)
- ALLOWED_METHODS (comma-separated; default GET,POST,OPTIONS)
- ALLOWED_HEADERS (comma-separated; default `*`)
- GZIP_MIN_SIZE (bytes, default 1024)
- MAX_UPLOAD_MB (default 100)
- OPENAPI_ENABLED (false by default)
- DOCS_ENABLED (false by default)

## Run locally

```powershell
# Windows PowerShell
$env:PORT=8000; $env:DOCS_ENABLED=true; python scripts/serve.py
```

## Docker

```powershell
# Build
docker build -t imu-hip-torque:latest .
# Run
docker run -p 8000:8000 -e WORKERS=2 -e DOCS_ENABLED=false imu-hip-torque:latest
```

## Health and endpoints

- GET /health -> { status: ok }
- POST /api/analyze -> multipart form; supports zip, bulk files, or individual files. Falls back to sample data.

## Notes

- Large response bodies are gzip-compressed.
- Upload archive size limit enforced by MAX_UPLOAD_MB.
- CORS is configurable and defaults to permissive (`*`).
