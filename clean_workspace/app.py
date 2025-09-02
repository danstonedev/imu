from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import re
import io
import zipfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from core.pipeline.pipeline import run_pipeline_clean
from core.config.settings import settings

root_dir = Path(__file__).resolve().parent
sample_dir = root_dir / 'sample data'

app = FastAPI(
    title=settings.app_name,
    docs_url=("/docs" if settings.docs_enabled else None),
    redoc_url=("/redoc" if settings.docs_enabled else None),
    openapi_url=("/openapi.json" if settings.openapi_enabled else None),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=settings.allow_credentials,
    allow_methods=list(settings.allowed_methods),
    allow_headers=list(settings.allowed_headers),
)

# Compression for large JSON responses and CSV strings
app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_min_size)

# Restrict Host headers when ALLOWED_HOSTS is set to specific values
if settings.allowed_hosts and settings.allowed_hosts != ("*",):
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

def pick(pattern: str) -> str:
    matches = sorted(sample_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No sample file for pattern: {pattern}")
    return str(matches[0])

@app.post("/api/analyze/")
async def analyze_data(
    height_m: float = Form(...),
    mass_kg: float = Form(...),
    baseline_mode: Optional[str] = Form(None),
    cal_mode: Optional[str] = Form(None),
    # New: accept a single .zip containing multiple tests (nested folders ok)
    archive: Optional[UploadFile] = File(None),
    # New: support bulk upload of files (multiple or a selected folder)
    files: Optional[List[UploadFile]] = File(None),
    # Optional parallel list of client-side relative paths for the above files
    paths: Optional[List[str]] = Form(None),
    pelvis_file: Optional[UploadFile] = File(None),
    lfemur_file: Optional[UploadFile] = File(None),
    rfemur_file: Optional[UploadFile] = File(None),
    ltibia_file: Optional[UploadFile] = File(None),
    rtibia_file: Optional[UploadFile] = File(None),
):
    """Run analysis on uploaded files or fall back to sample data if none provided."""

    # JSON-safe converter (numpy arrays, scalars, nested)
    def to_json_safe(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_json_safe(x) for x in obj]
        return obj

    # Quick size guard for bulk payloads to avoid accidental huge uploads (best-effort)
    try:
        max_bytes = int(settings.max_upload_mb) * 1024 * 1024
        # Rough check on provided files/zip sizes if available via 'size' attribute or len(content)
        def _approx_size(u: Optional[UploadFile]) -> int:
            try:
                if u is None:
                    return 0
                # starlette UploadFile exposes spooled file; size isn't directly available
                # We avoid reading here to keep streaming behavior; rely on archive path below
                return 0
            except Exception:
                return 0
        total_nominal = 0
        for u in (files or []) + [archive, pelvis_file, lfemur_file, rfemur_file, ltibia_file, rtibia_file]:
            if isinstance(u, list):
                for x in u:
                    total_nominal += _approx_size(x)
            else:
                total_nominal += _approx_size(u)
        # Only enforce for archive after read where we know bytes
    except Exception:
        pass

    # Determine upload mode:
    # 1) Bulk files provided via `files`
    # 2) Individually provided 5 fields
    # 3) None provided -> use sample data
    # Treat empty UploadFile placeholders as None to avoid misrouting
    use_archive = bool(archive and getattr(archive, 'filename', '') )
    bulk_files = [
        f for f in (files or [])
        if f is not None and getattr(f, 'filename', '')
    ]
    indiv_list = [pelvis_file, lfemur_file, rfemur_file, ltibia_file, rtibia_file]
    use_bulk = (len(bulk_files) > 0) and not use_archive
    use_indiv_all = all(f is not None for f in indiv_list)
    use_none = (not use_archive) and (not use_bulk) and all(f is None for f in indiv_list)

    # Ensure local initialization to avoid any UnboundLocalError on rare paths
    batch_results: List[Dict[str, Any]] = []

    if not use_bulk and not use_indiv_all and not use_none:
        raise HTTPException(status_code=400, detail="Provide either: (a) a folder or multiple files in the 'files' field, (b) all 5 individual files, or (c) none to use sample data.")

    def idx_from_name(name: str) -> Optional[int]:
        if not isinstance(name, str):
            return None
        m = re.search(r"_(\d)(?:[_\.]|$)", name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def role_from_keywords(name: str) -> Optional[str]:
        if not isinstance(name, str):
            return None
        low = name.lower()
        # Heuristics if index pattern isn't present
        if 'pelvis' in low or 'trunk' in low or 'lumbar' in low:
            return 'pelvis'
        if 'lfemur' in low or ('femur' in low and ('left' in low or '_l' in low)):
            return 'lfemur'
        if 'rfemur' in low or ('femur' in low and ('right' in low or '_r' in low)):
            return 'rfemur'
        if 'ltibia' in low or ('tibia' in low and ('left' in low or '_l' in low)) or 'lshank' in low:
            return 'ltibia'
        if 'rtibia' in low or ('tibia' in low and ('right' in low or '_r' in low)) or 'rshank' in low:
            return 'rtibia'
        return None

    def group_key_from_path(name: str) -> str:
        """Derive a dataset key from a filename with possible subfolders.
        We use the immediate parent folder name if present; otherwise the stem without index.
        """
        if not isinstance(name, str) or not name:
            return "dataset"
        # Normalize separators
        p = name.replace('\\','/')
        parts = [s for s in p.split('/') if s]
        if len(parts) >= 2:
            # parent folder
            return parts[-2]
        # Fall back: strip trailing _digit markers in base name
        base = parts[-1]
        base = re.sub(r"\.[^.]+$", "", base)
        base = re.sub(r"_\d(?:[_\.]|$)", "", base)
        return base or "dataset"

    def build_datasets_from_uploads(upload_files: List[Any], rel_paths: Optional[List[str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
        role_by_idx = { 0: "pelvis", 1: "lfemur", 2: "rfemur", 3: "ltibia", 4: "rtibia" }
        # Group files by dataset key
        groups: Dict[str, List[Any]] = {}
        for i, f in enumerate(upload_files):
            # Prefer client-provided relative path if available (webkitdirectory)
            rp = rel_paths[i] if (rel_paths and i < len(rel_paths)) else getattr(f, 'filename', '')
            key = group_key_from_path(rp)
            groups.setdefault(key, []).append(f)
        datasets: List[Tuple[str, Dict[str, Any]]] = []
        for key, files_in_group in groups.items():
            detected: Dict[str, Any] = {}
            seen_idx: set[int] = set()
            # Pass 1: index-based mapping
            for f in files_in_group:
                idx = idx_from_name(getattr(f, 'filename', ''))
                if idx is None or idx not in role_by_idx:
                    continue
                if idx in seen_idx:
                    continue
                seen_idx.add(idx)
                detected[role_by_idx[idx]] = f
            # Pass 2: fill gaps via keywords
            if len(detected) < 5:
                missing = { r for r in role_by_idx.values() if r not in detected }
                if missing:
                    for f in sorted(files_in_group, key=lambda x: getattr(x, 'filename', '')):
                        role = role_from_keywords(getattr(f, 'filename', ''))
                        if role and role in missing and role not in detected:
                            detected[role] = f
                            missing.remove(role)
                            if not missing:
                                break
            if len(detected) == 5:
                datasets.append((key, detected))
        return datasets

    if use_archive:
        # Unpack zip and build datasets from contained CSVs
        arc = archive  # local non-optional alias for type checkers
        fn = (getattr(arc, 'filename', '') or '') if arc is not None else ''
        if arc is None or not fn:
            raise HTTPException(status_code=400, detail="Archive must be a .zip file")
        if not fn.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="Archive must be a .zip file")
        data_bytes = await arc.read()
        if len(data_bytes) > int(settings.max_upload_mb) * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Archive exceeds limit of {settings.max_upload_mb} MB")
        try:
            zf = zipfile.ZipFile(io.BytesIO(data_bytes))
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip archive")
        # Collect pseudo UploadFiles from zip entries
        file_objs: List[Any] = []
        csv_infos = [zi for zi in zf.infolist() if not zi.is_dir() and zi.filename.lower().endswith('.csv')]
        # Materialize as simple objects with .filename and .read()
        class ZipUP:
            def __init__(self, name: str, data: bytes):
                self.filename = name
                self._data = data
            async def read(self):
                return self._data
        for zi in csv_infos:
            try:
                with zf.open(zi, 'r') as fh:
                    file_objs.append(ZipUP(zi.filename, fh.read()))
            except Exception:
                continue
        # Build datasets
        ds = build_datasets_from_uploads(file_objs) if file_objs else []
        if not ds:
            raise HTTPException(status_code=400, detail="No valid datasets found in zip. Expect subfolders each containing 5 CSVs (pelvis/lfemur/rfemur/ltibia/rtibia). Filenames should include indices _0.._4 or role keywords.")
        batch_results = []
        options: dict = {'do_cal': True, 'yaw_align': True}
        if isinstance(baseline_mode, str) and baseline_mode in {"none","constant","linear"}:
            options['baseline_mode'] = baseline_mode
        if isinstance(cal_mode, str) and cal_mode in {"simple","advanced"}:
            options['cal_mode'] = cal_mode
        for key, mapping in ds:
            files_bytes = { role: (await mapping[role].read()) for role in ["pelvis","lfemur","rfemur","ltibia","rtibia"] }
            res = run_pipeline_clean(files_bytes, height_m, mass_kg, options)
            # Attach meta source name
            if isinstance(res, dict):
                res.setdefault('meta', {})
                res['meta']['dataset'] = key
                res['meta']['baseline_mode'] = (baseline_mode if baseline_mode in {"none","constant","linear"} else 'linear')
            batch_results.append({ 'name': key, 'results': res })
        # Convert to JSON-friendly (archive returns batch)
        return JSONResponse(content={'batch': to_json_safe(batch_results)})

    if use_bulk:
        # Try autodetect mapping from a mixed list of files
        datasets = build_datasets_from_uploads(bulk_files, paths or None)
        if not datasets:
            raise HTTPException(status_code=400, detail="Could not identify datasets from uploaded files. Ensure each test is in its own folder (or filenames include indices _0.._4) with all 5 CSVs.")
        # If multiple datasets, return batch; if single, continue with single result behavior
        if len(datasets) > 1:
            batch_results = []
            options: dict = {'do_cal': True, 'yaw_align': True}
            if isinstance(baseline_mode, str) and baseline_mode in {"none","constant","linear"}:
                options['baseline_mode'] = baseline_mode
            if isinstance(cal_mode, str) and cal_mode in {"simple","advanced"}:
                options['cal_mode'] = cal_mode
            for key, mapping in datasets:
                files_bytes = { role: (await mapping[role].read()) for role in ["pelvis","lfemur","rfemur","ltibia","rtibia"] }
                res = run_pipeline_clean(files_bytes, height_m, mass_kg, options)
                if isinstance(res, dict):
                    res.setdefault('meta', {})
                    res['meta']['dataset'] = key
                    res['meta']['baseline_mode'] = (baseline_mode if baseline_mode in {"none","constant","linear"} else 'linear')
                batch_results.append({ 'name': key, 'results': res })
            return JSONResponse(content={'batch': to_json_safe(batch_results)})
        # Single dataset
        key, mapping = datasets[0]
        files_bytes = { role: await mapping[role].read() for role in ["pelvis","lfemur","rfemur","ltibia","rtibia"] }
    elif use_indiv_all:
        # Verify all files are present and create non-nullable references
        if pelvis_file is None or lfemur_file is None or rfemur_file is None or ltibia_file is None or rtibia_file is None:
            raise HTTPException(status_code=400, detail="All 5 files must be provided when uploading.")
        # Autodetect roles from filename pattern across the five individual file fields
        uploads = [
            ("pelvis", pelvis_file),
            ("lfemur", lfemur_file),
            ("rfemur", rfemur_file),
            ("ltibia", ltibia_file),
            ("rtibia", rtibia_file),
        ]
        role_by_idx = { 0: "pelvis", 1: "lfemur", 2: "rfemur", 3: "ltibia", 4: "rtibia" }
        detected: dict[str, UploadFile] = {}
        all_idxs: set[int] = set()
        for _, f in uploads:
            idx = idx_from_name(getattr(f, 'filename', ''))
            if idx is None or idx not in role_by_idx:
                detected = {}
                break
            if idx in all_idxs:
                detected = {}
                break
            all_idxs.add(idx)
            detected[role_by_idx[idx]] = f

        if detected and len(detected) == 5:
            files_bytes = { role: await f.read() for role, f in detected.items() }
        else:
            # Fallback: trust form field mapping
            files_bytes = {
                "pelvis": await pelvis_file.read(),
                "lfemur": await lfemur_file.read(),
                "rfemur": await rfemur_file.read(),
                "ltibia": await ltibia_file.read(),
                "rtibia": await rtibia_file.read(),
            }
    else:
        # Fallback to sample data from disk (bytes)
        sample_paths = {
            'pelvis': pick('DEMO6_0_*.csv'),
            'lfemur': pick('DEMO6_1_*.csv'),
            'rfemur': pick('DEMO6_2_*.csv'),
            'ltibia': pick('DEMO6_3_*.csv'),
            'rtibia': pick('DEMO6_4_*.csv'),
        }
        files_bytes = {k: Path(v).read_bytes() for k, v in sample_paths.items()}

    options: dict = {'do_cal': True, 'yaw_align': True}
    if isinstance(baseline_mode, str) and baseline_mode in {"none","constant","linear"}:
        options['baseline_mode'] = baseline_mode
    if isinstance(cal_mode, str) and cal_mode in {"simple","advanced"}:
        options['cal_mode'] = cal_mode
    results = run_pipeline_clean(files_bytes, height_m, mass_kg, options)
    # Attach simple meta for UI
    try:
        meta = results.get('meta', {}) if isinstance(results, dict) else {}
        meta['baseline_mode'] = (baseline_mode if baseline_mode in {"none","constant","linear"} else 'linear')
        if isinstance(results, dict):
            results['meta'] = meta
    except Exception:
        pass

    return JSONResponse(content=to_json_safe(results))

@app.get("/")
async def read_index():
    index = root_dir / 'static' / 'index.html'
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "ok", "app": settings.app_name})

@app.get('/favicon.ico')
async def favicon():
    fav = root_dir / 'static' / 'favicon.ico'
    if fav.exists():
        return FileResponse(str(fav))
    raise HTTPException(status_code=404, detail="favicon not found")

app.mount("/static", StaticFiles(directory=str(root_dir / 'static')), name="static")

# Quiet Chrome/Edge DevTools probes (prevent 404 spam in logs)
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_probe():
    return Response(status_code=204)

# Simple health check endpoint for local probes
@app.get("/health")
async def health():
    return JSONResponse({"status":"ok"})
