"""
Diagnostic helpers shared by tests & CLI.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is on sys.path so that `core.pipeline.pipeline` can be imported
# when tests are run from the imu-gait-diagnostics folder.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # clean_workspace/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---- Import helpers -------------------------------------------------------
def try_import_pipeline():
    """
    Best-effort import of user's pipeline. Adjust these paths if needed.
    Returns a tuple (module, entry_callable_name) or raises ImportError with hints.
    """
    candidates = [
        ("core.pipeline.pipeline", "run_pipeline_clean"),
        ("core.pipeline.pipeline", "run_pipeline"),
        ("pipeline", "run_pipeline_clean"),
        ("pipeline", "run_pipeline"),
    ]
    last_err = None
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return mod, fn_name
        except Exception as e:  # noqa: BLE001 - diagnostic helper, broad by design
            last_err = e
            continue
    raise ImportError(
        "Could not locate your pipeline entry function. "
        "Tried: core.pipeline.pipeline:{run_pipeline_clean, run_pipeline} and pipeline:{...}.\n"
        f"Last error: {last_err}\n"
        "Fix: edit diagnostics/utils.py:try_import_pipeline() to point to your entry function."
    )


# ---- Execution helpers ----------------------------------------------------
def _call_with_possible_rom_flag(fn, kwargs: Dict[str, Any], rom_enabled: bool):
    """
    Call the pipeline function, trying to pass a ROM toggle if it exists.
    Fallback: uses environment variable IMU_ROM_ENABLED ('1'|'0').
    """
    sig = inspect.signature(fn)
    bound_kwargs = dict(kwargs)
    # Try common parameter names first
    for pname in ("rom_enabled", "compute_rom", "enable_rom"):
        if pname in sig.parameters:
            bound_kwargs[pname] = rom_enabled
            break
    # If no parameter available, use env var as a soft toggle
    os.environ["IMU_ROM_ENABLED"] = "1" if rom_enabled else "0"
    return fn(**bound_kwargs)


def run_twice_compare(pipeline_kwargs: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Run the pipeline twice on the same inputs, with ROM disabled and enabled.
    Returns (out_off, out_on) as dictionaries.

    The pipeline is expected to return a dict-like object including:
      - 'events' with keys HS_L, TO_L, HS_R, TO_R (arrays of timestamps or indices)
      - 'meta' with diagnostic metadata if available
    If your pipeline writes files instead, adjust here to read them.
    """
    mod, fn_name = try_import_pipeline()
    fn = getattr(mod, fn_name)
    out_off = _call_with_possible_rom_flag(fn, pipeline_kwargs, rom_enabled=False)
    out_on = _call_with_possible_rom_flag(fn, pipeline_kwargs, rom_enabled=True)
    # Normalize to dict
    out_off = _to_dict(out_off)
    out_on = _to_dict(out_on)
    return out_off, out_on


def _to_dict(obj: Any) -> Dict[str, Any]:
    # Accept dict-like, dataclass, or object with .to_dict()
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return obj.to_dict()
        except Exception:  # noqa: BLE001
            pass
    # Last resort: try json conversion via __dict__
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:  # noqa: BLE001
        return {"value": str(obj)}


# ---- Event comparison utilities ------------------------------------------
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)  # numpy arrays etc.


def compare_events(a: Dict, b: Dict, tol_ms: float = 10.0) -> Dict[str, Any]:
    """
    Compare HS/TO events between two pipeline outputs.
    Returns a summary dict with deltas and pass/fail flags.
    """
    eventsA = a.get("events", a)  # tolerate outputs where events is top-level
    eventsB = b.get("events", b)
    res = {"tolerance_ms": tol_ms, "sides": {}}
    for side in ("L", "R"):
        side_key_HS = f"HS_{side}"
        side_key_TO = f"TO_{side}"
        HS_A = _as_list(eventsA.get(side_key_HS))
        TO_A = _as_list(eventsA.get(side_key_TO))
        HS_B = _as_list(eventsB.get(side_key_HS))
        TO_B = _as_list(eventsB.get(side_key_TO))
        side_res: Dict[str, Any] = {}
        side_res["count_A"] = {"HS": len(HS_A), "TO": len(TO_A)}
        side_res["count_B"] = {"HS": len(HS_B), "TO": len(TO_B)}
        side_res["count_match"] = (len(HS_A) == len(HS_B)) and (len(TO_A) == len(TO_B))
        deltas_ms = {"HS": [], "TO": []}
        if side_res["count_match"]:
            for x, y in zip(HS_A, HS_B):
                deltas_ms["HS"].append(abs((float(y) - float(x)) * 1000.0))
            for x, y in zip(TO_A, TO_B):
                deltas_ms["TO"].append(abs((float(y) - float(x)) * 1000.0))
        side_res["max_delta_ms"] = {
            "HS": max(deltas_ms["HS"]) if deltas_ms["HS"] else None,
            "TO": max(deltas_ms["TO"]) if deltas_ms["TO"] else None,
        }

        def _pass(d):
            return (d is None) or (d <= tol_ms)

        side_res["pass"] = (
            side_res["count_match"]
            and _pass(side_res["max_delta_ms"]["HS"])
            and _pass(side_res["max_delta_ms"]["TO"])
        )
        res["sides"][side] = side_res
    return res


# ---- Additional diagnostics extraction -----------------------------------
def extract_detector_meta(out: Dict) -> Dict[str, Any]:
    """
    Pull detector-related metadata if present.
    Expected keys (optional): out['meta']['detector_meta'][side] or similar.
    Returns a normalized shallow dict for each side.
    """
    meta = out.get("meta", {})
    det = meta.get("detector_meta") or meta.get("events_meta") or {}
    norm = {}
    for side in ("L", "R"):
        m = det.get(side, {}) if isinstance(det, dict) else {}
        norm[side] = {
            "signal": m.get("signal"),
            "frame": m.get("frame"),
            "units": m.get("units"),
            "bandpass": m.get("bandpass"),
            "timebase": m.get("timebase"),
            "acc_is_free": meta.get("acceleration", {}).get("headers_freeacc"),
            "stance_thresholds": m.get("thresholds") or meta.get("stance_thresholds_used"),
        }
    return norm


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
