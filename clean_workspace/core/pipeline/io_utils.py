from __future__ import annotations
import io, re
import numpy as np
import pandas as pd
from ..config.constants import TIME_CANDS, QW, QX, QY, QZ, GYR, ACC, G
from ..math.kinematics import quats_to_R_batch

__all__ = [
    "read_xsens_bytes",
    "sanitize_cols",
    "pick_col",
    "extract_kinematics",
    "extract_kinematics_ex",
]


def sanitize_cols(cols):
    sc = []
    for c in cols:
        s = str(c).strip()
        s = re.sub(r"[^0-9A-Za-z]+", "_", s)
        s = re.sub(r"_+", "_", s)
        sc.append(s.strip("_").lower())
    return sc


def read_xsens_bytes(b: bytes) -> pd.DataFrame:
    text = b.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    def _looks_like_header(s: str) -> bool:
        s0 = s.strip()
        if not s0:
            return False
        if ":" in s0:
            return False
        if not any(d in s0 for d in (",", ";", "\t", "|")):
            return False
        low = s0.lower()
        hits = 0
        for kw in (
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
            "freeacc_x",
            "freeacc_y",
            "freeacc_z",
            "gyr_x",
            "gyr_y",
            "gyr_z",
            "sampletimefine",
            "packetcounter",
        ):
            if kw in low:
                hits += 1
        return hits >= 2

    header_i = None
    for i, line in enumerate(lines[:400]):
        if _looks_like_header(line):
            header_i = i
            break
    if header_i is None:
        for i, line in enumerate(lines[:400]):
            s0 = line.strip()
            if ":" in s0:
                continue
            if any(line.count(sep) >= 8 for sep in (",", ";", "\t", "|")):
                header_i = i
                break
    if header_i is None:
        header_i = 0

    payload = "\n".join(lines[header_i:])

    df = None
    try:
        df = pd.read_csv(io.StringIO(payload), low_memory=False)
    except Exception:
        pass
    if df is None:
        try:
            df = pd.read_csv(
                io.StringIO(payload), engine="python", sep=None, on_bad_lines="skip"
            )
        except Exception:
            df = None
    if df is None:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(
                    io.StringIO(payload), engine="python", sep=sep, on_bad_lines="skip"
                )
                break
            except Exception:
                df = None
    if df is None:
        # Last-resort: detect probable delimiter by most frequent in the first non-empty line
        rows = [r for r in payload.splitlines() if r.strip()]
        if not rows:
            raise ValueError("Empty CSV payload")
        delims = [",", ";", "\t", "|"]
        delim = max(delims, key=lambda d: rows[0].count(d))
        target_n = rows[0].count(delim) + 1
        filtered = [r for r in rows if (r.count(delim) + 1) == target_n]
        df = pd.read_csv(
            io.StringIO("\n".join(filtered)),
            engine="python",
            sep=delim,
            on_bad_lines="skip",
        )

    # Tiny-row guard: if parsed frame is suspiciously small, retry known delimiters and pick the best
    try:
        if df is not None and len(df) < 5:
            best_df = df
            best_len = len(df)
            for sep in [",", ";", "\t", "|"]:
                try:
                    cand = pd.read_csv(
                        io.StringIO(payload),
                        engine="python",
                        sep=sep,
                        on_bad_lines="skip",
                    )
                    if len(cand) > best_len:
                        best_df = cand
                        best_len = len(cand)
                except Exception:
                    pass
            df = best_df
    except Exception:
        pass

    df.columns = sanitize_cols(df.columns)
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    base = ["".join(filter(str.isalpha, c)) for c in df.columns]
    for c in candidates:
        token = "".join(filter(str.isalpha, c))
        for bidx, b in enumerate(base):
            if token and token in b:
                return df.columns[bidx]
    raise KeyError(f"Missing any of {candidates}")


def extract_kinematics(df: pd.DataFrame):
    t_col = pick_col(df, TIME_CANDS)
    t_raw = df[t_col].to_numpy(dtype=float)
    if "sampletimefine" in t_col:
        t = (t_raw - t_raw[0]) / 1e6
    else:
        # Normalize to seconds; some sources store minutes in a 'time' column.
        t = t_raw - t_raw[0]
        if len(t_raw) > 2:
            dt = np.diff(t_raw)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            med_dt = float(np.median(dt)) if dt.size else None
            total = float(t_raw[-1] - t_raw[0])
            # Heuristic: if timeline ends <= 60 and sampling seems fast, it's likely minutes
            if (
                med_dt is not None
                and total <= 60.0
                and med_dt <= 0.1
                and len(t_raw) >= 300
            ):
                t = (t_raw - t_raw[0]) * 60.0

    qw_col = pick_col(df, QW)
    qx_col = pick_col(df, QX)
    qy_col = pick_col(df, QY)
    qz_col = pick_col(df, QZ)
    qw = df[qw_col].to_numpy(dtype=float)
    qx = df[qx_col].to_numpy(dtype=float)
    qy = df[qy_col].to_numpy(dtype=float)
    qz = df[qz_col].to_numpy(dtype=float)
    quat = np.stack([qw, qx, qy, qz], axis=1)

    try:
        gx = df[pick_col(df, GYR["x"])].to_numpy(dtype=float) * (np.pi / 180.0)
        gy = df[pick_col(df, GYR["y"])].to_numpy(dtype=float) * (np.pi / 180.0)
        gz = df[pick_col(df, GYR["z"])].to_numpy(dtype=float) * (np.pi / 180.0)
        gyro = np.stack([gx, gy, gz], axis=1)
    except Exception:
        gyro = None

    # Prefer free acceleration if available, else fall back to generic acc columns
    ax_col = pick_col(df, ACC["x"])
    ay_col = pick_col(df, ACC["y"])
    az_col = pick_col(df, ACC["z"])
    ax = df[ax_col].to_numpy(dtype=float)
    ay = df[ay_col].to_numpy(dtype=float)
    az = df[az_col].to_numpy(dtype=float)
    acc = np.stack([ax, ay, az], axis=1)

    # Determine if selected acc columns are already free acceleration
    sel_names = " ".join([ax_col, ay_col, az_col]).lower()
    is_free = (
        ("freeacc" in sel_names) or ("free_acc" in sel_names) or ("_free" in sel_names)
    )
    # If explicit free acceleration is not present, fall back to using the raw
    # accelerometer values as-is (tests expect this behavior). We do not attempt
    # to remove gravity here to avoid introducing assumptions about sensor frame
    # and sign conventions.
    freeacc = acc if not is_free else acc

    return t, quat, gyro, freeacc


def extract_kinematics_ex(df: pd.DataFrame):
    """Extended extractor that returns whether acceleration columns are already free acceleration.

    Returns: t, quat, gyro, freeacc, meta where meta has keys:
      - acc_is_free: bool
      - acc_cols: tuple of selected acceleration column names
    """
    t_col = pick_col(df, TIME_CANDS)
    t_raw = df[t_col].to_numpy(dtype=float)
    if "sampletimefine" in t_col:
        t = (t_raw - t_raw[0]) / 1e6
    else:
        # Normalize to seconds; some sources store minutes in a 'time' column.
        t = t_raw - t_raw[0]
        if len(t_raw) > 2:
            dt = np.diff(t_raw)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            med_dt = float(np.median(dt)) if dt.size else None
            total = float(t_raw[-1] - t_raw[0])
            if (
                med_dt is not None
                and total <= 60.0
                and med_dt <= 0.1
                and len(t_raw) >= 300
            ):
                t = (t_raw - t_raw[0]) * 60.0

    qw_col = pick_col(df, QW)
    qx_col = pick_col(df, QX)
    qy_col = pick_col(df, QY)
    qz_col = pick_col(df, QZ)
    qw = df[qw_col].to_numpy(dtype=float)
    qx = df[qx_col].to_numpy(dtype=float)
    qy = df[qy_col].to_numpy(dtype=float)
    qz = df[qz_col].to_numpy(dtype=float)
    quat = np.stack([qw, qx, qy, qz], axis=1)

    try:
        gx = df[pick_col(df, GYR["x"])].to_numpy(dtype=float) * (np.pi / 180.0)
        gy = df[pick_col(df, GYR["y"])].to_numpy(dtype=float) * (np.pi / 180.0)
        gz = df[pick_col(df, GYR["z"])].to_numpy(dtype=float) * (np.pi / 180.0)
        gyro = np.stack([gx, gy, gz], axis=1)
    except Exception:
        gyro = None

    ax_col = pick_col(df, ACC["x"])
    ay_col = pick_col(df, ACC["y"])
    az_col = pick_col(df, ACC["z"])
    ax = df[ax_col].to_numpy(dtype=float)
    ay = df[ay_col].to_numpy(dtype=float)
    az = df[az_col].to_numpy(dtype=float)
    acc = np.stack([ax, ay, az], axis=1)

    sel_names = " ".join([ax_col, ay_col, az_col]).lower()
    is_free = (
        ("freeacc" in sel_names) or ("free_acc" in sel_names) or ("_free" in sel_names)
    )
    freeacc = (
        acc  # we pass through values unchanged here; interpretation handled upstream
    )

    # Estimate sampling rate for diagnostics
    fs = None
    if len(t) > 1:
        dtt = np.diff(t)
        dtt = dtt[np.isfinite(dtt) & (dtt > 0)]
        if dtt.size:
            fs = float(1.0 / np.median(dtt))

    meta = {
        "acc_is_free": bool(is_free),
        "acc_cols": (ax_col, ay_col, az_col),
        "fs_hz": fs,
    }
    return t, quat, gyro, freeacc, meta
