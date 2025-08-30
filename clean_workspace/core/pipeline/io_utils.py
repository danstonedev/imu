from __future__ import annotations
import io, re
import numpy as np
import pandas as pd
from ..config.constants import TIME_CANDS, QW, QX, QY, QZ, GYR, ACC

__all__ = ["read_xsens_bytes","sanitize_cols","pick_col","extract_kinematics"]

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
        if not any(d in s0 for d in (",",";","\t","|")):
            return False
        low = s0.lower()
        hits = 0
        for kw in ("quat_w","quat_x","quat_y","quat_z","freeacc_x","freeacc_y","freeacc_z","gyr_x","gyr_y","gyr_z","sampletimefine","packetcounter"):
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
            if any(line.count(sep) >= 8 for sep in (",",";","\t","|")):
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
            df = pd.read_csv(io.StringIO(payload), engine='python', sep=None, on_bad_lines='skip')
        except Exception:
            df = None
    if df is None:
        for sep in [',',';','\t','|']:
            try:
                df = pd.read_csv(io.StringIO(payload), engine='python', sep=sep, on_bad_lines='skip')
                break
            except Exception:
                df = None
    if df is None:
        rows = payload.splitlines()
        delims = [',',';','\t','|']
        delim = max(delims, key=lambda d: rows[0].count(d))
        target_n = rows[0].count(delim) + 1
        filtered = [r for r in rows if (r.count(delim) + 1) == target_n]
        df = pd.read_csv(io.StringIO("\n".join(filtered)), engine='python', sep=delim, on_bad_lines='skip')

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
    if 'sampletimefine' in t_col:
        t = (t_raw - t_raw[0]) / 1e6
    else:
        t = t_raw - t_raw[0]

    qw = df[pick_col(df, QW)].to_numpy(dtype=float)
    qx = df[pick_col(df, QX)].to_numpy(dtype=float)
    qy = df[pick_col(df, QY)].to_numpy(dtype=float)
    qz = df[pick_col(df, QZ)].to_numpy(dtype=float)
    quat = np.stack([qw,qx,qy,qz], axis=1)

    try:
        gx = df[pick_col(df, GYR['x'])].to_numpy(dtype=float) * (np.pi/180.0)
        gy = df[pick_col(df, GYR['y'])].to_numpy(dtype=float) * (np.pi/180.0)
        gz = df[pick_col(df, GYR['z'])].to_numpy(dtype=float) * (np.pi/180.0)
        gyro = np.stack([gx,gy,gz], axis=1)
    except Exception:
        gyro = None

    ax = df[pick_col(df, ACC['x'])].to_numpy(dtype=float)
    ay = df[pick_col(df, ACC['y'])].to_numpy(dtype=float)
    az = df[pick_col(df, ACC['z'])].to_numpy(dtype=float)
    freeacc = np.stack([ax,ay,az], axis=1)

    return t, quat, gyro, freeacc
