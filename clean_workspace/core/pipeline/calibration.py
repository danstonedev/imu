from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from ..config.constants import (
    STILL_THR_W, STILL_THR_A, STILL_SMOOTH_S,
    STILL_MIN_S_PRIMARY, STILL_MIN_S_SECONDARY, STILL_EDGE_FRAC,
    STILL_FALLBACK_WINDOW_S,
)
from ..math.kinematics import moving_avg

__all__ = [
    "detect_still","find_still_segments","pick_edge_segments",
    "calibration_windows_secs","calibrate_bias_trimmed","calibrate_all","CalibrationResult",
]

@dataclass
class CalibrationResult:
    """Container for calibration artifacts computed in a single pass.

    windows: List of dicts with start/end seconds for still windows (pelvis timeline).
    yaw_ref: Chosen yaw reference in radians (float).
    yaw_source: Source tag for yaw reference ('start'|'end'|'global').
    biases: Map from segment name to (gyro_bias[3], acc_bias[3]) arrays.
    """
    windows: List[dict]
    yaw_ref: float
    yaw_source: str
    biases: Dict[str, Tuple[np.ndarray, np.ndarray]]

def detect_still(t: np.ndarray, gyro: np.ndarray, freeacc: np.ndarray):
    t = np.asarray(t, float)
    gmag = np.linalg.norm(np.asarray(gyro, float), axis=1)
    amag = np.linalg.norm(np.asarray(freeacc, float), axis=1)
    # Estimate Fs
    if t.size < 2:
        Fs = 100.0
    else:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        Fs = float(1.0 / np.median(dt)) if dt.size else 100.0
    win = max(1, int(round(STILL_SMOOTH_S * Fs)))
    gmag_s = moving_avg(gmag, win=win)
    amag_s = moving_avg(amag, win=win)
    still = (gmag_s < STILL_THR_W) & (amag_s < STILL_THR_A)
    return still, {"Fs": Fs, "smooth_win_samples": int(win)}

def find_still_segments(still_mask: np.ndarray, min_len: int):
    n = int(len(still_mask))
    segs: list[tuple[int,int]] = []
    i = 0
    while i < n:
        if not still_mask[i]:
            i += 1; continue
        j = i
        while j < n and still_mask[j]:
            j += 1
        if (j - i) >= min_len:
            segs.append((i, j))
        i = j
    return segs

def pick_edge_segments(segs: list[tuple[int,int]], n: int, edge_frac: float = STILL_EDGE_FRAC):
    if not segs:
        return None, None
    centers = [0.5*(s+e) for (s,e) in segs]
    edge = int(round(edge_frac * n))
    start_candidates = [(e-s, i) for i,((s,e),c) in enumerate(zip(segs, centers)) if c <= edge]
    end_candidates   = [(e-s, i) for i,((s,e),c) in enumerate(zip(segs, centers)) if c >= (n-edge)]

    def pick(cands, prefer_first=True):
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1] if prefer_first else -x[1]))
            return segs[cands[0][1]]
        if prefer_first:
            idx = int(np.argmin([s for s,_ in segs]))
        else:
            idx = int(np.argmax([e for _,e in segs]))
        return segs[idx]

    sA = pick(start_candidates, prefer_first=True) if segs else None
    sB = pick(end_candidates, prefer_first=False) if segs else None
    if sA and sB and (sA[0] >= sB[1]):
        sA = min([sA, sB], key=lambda x: x[0])
        sB = max([sA, sB], key=lambda x: x[1])
    return sA, sB

def calibration_windows_secs(t: np.ndarray, still_mask: np.ndarray, Fs: float):
    # try primary and secondary min lengths
    def segs_for(min_s: float):
        k = int(max(1, round(min_s * Fs)))
        return find_still_segments(still_mask, min_len=k)
    segs = segs_for(STILL_MIN_S_PRIMARY)
    if not segs:
        segs = segs_for(STILL_MIN_S_SECONDARY)
    sA, sB = pick_edge_segments(segs, n=len(still_mask)) if segs else (None, None)

    if (sA is None) and (sB is None):
        N = len(still_mask)
        k = int(max(10, round(STILL_FALLBACK_WINDOW_S * max(Fs, 1.0))))
        k = min(k, max(1, N // 4))
        s1, e1 = 0, min(N, k)
        e2, s2 = N, max(0, N - k)
        if e1 >= s2:
            mid = N // 2
            e1 = min(e1, mid)
            s2 = max(s2, mid)
        sA, sB = (s1, e1), (s2, e2)

    wins = []
    def add_win(win, label):
        if not win or win[0] is None:
            return
        s, e = int(win[0]), int(win[1])
        s = max(0, s); e = max(s+1, min(e, len(t)))
        t0, t1 = float(t[s]), float(t[e-1])
        wins.append({
            'label': label,
            'start_s': t0,
            'end_s': t1,
            'duration_s': max(0.0, t1 - t0),
            'samples': int(e - s),
        })
    add_win(sA, 'start')
    add_win(sB, 'end')
    return wins

def calibrate_bias_trimmed(gyro_S: np.ndarray, freeacc_S: np.ndarray, still_ranges: list[tuple[int,int]]):
    def tmean(X, trim=0.1):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        lo = np.quantile(X, trim, axis=0)
        hi = np.quantile(X, 1.0-trim, axis=0)
        mask = (X >= lo) & (X <= hi)
        out = []
        for j in range(X.shape[1]):
            col = X[:, j]; m = mask[:, j]
            out.append(float(np.mean(col[m])) if np.any(m) else float(np.mean(col)))
        return np.array(out, float)

    if not still_ranges:
        return np.zeros(3), np.zeros(3)
    wsum = 0.0
    g_acc = np.zeros(3); a_acc = np.zeros(3)
    for (s,e) in still_ranges:
        w = float(max(1, e - s))
        g_acc += w * tmean(gyro_S[s:e], trim=0.1)
        a_acc += w * tmean(freeacc_S[s:e], trim=0.1)
        wsum += w
    if wsum <= 0:
        return np.zeros(3), np.zeros(3)
    return g_acc/wsum, a_acc/wsum


def calibrate_all(
    tP: np.ndarray,
    RP_raw: np.ndarray,
    stillP: np.ndarray | None,
    cal_windows: List[dict] | None,
    segments: Dict[str, Tuple[np.ndarray | None, np.ndarray | None]],
) -> CalibrationResult:
    """One-shot calibration: choose yaw reference and compute per-segment biases.

    - tP: pelvis time (s) for indexing windows
    - RP_raw: pelvis world rotations (T,3,3) BEFORE yaw alignment
    - stillP: pelvis still mask (T,)
    - cal_windows: windows as produced by calibration_windows_secs (seconds)
    - segments: name -> (gyro_S, freeacc_S) in sensor frame
    """
    # Import inside function to avoid circular import at module import time
    from .pipeline import compute_yaw_reference  # type: ignore

    # 1) windows (already in seconds); allow None -> empty list
    wins: List[dict] = list(cal_windows or [])

    # 2) yaw reference using existing, proven logic
    yaw_ref, yaw_source = compute_yaw_reference(tP, RP_raw, stillP, wins)

    # 3) convert second-based windows to index ranges on pelvis timeline
    ranges: List[tuple[int, int]] = []
    for w in wins:
        try:
            s = int(np.searchsorted(tP, float(w['start_s']), side='left'))
            e = int(np.searchsorted(tP, float(w['end_s']),   side='right'))
        except Exception:
            continue
        s = max(0, min(s, len(tP)))
        e = max(s + 1, min(e, len(tP)))
        ranges.append((s, e))

    def _trimmed_mean(X: np.ndarray | None, trim: float = 0.1) -> np.ndarray:
        if X is None:
            return np.zeros(3, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        out = []
        for j in range(min(3, X.shape[1])):
            if ranges:
                x = np.concatenate([X[s:e, j] for (s, e) in ranges])
            else:
                x = X[:, j]
            x = np.sort(np.asarray(x, float))
            n = len(x)
            k = int(round(trim * n))
            if n <= 2 * k:
                out.append(0.0)
            else:
                out.append(float(np.mean(x[k:n - k])))
        while len(out) < 3:
            out.append(0.0)
        return np.array(out, dtype=float)

    # 4) per-segment robust trimmed means for gyro and acceleration
    biases: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (gyro_S, freeacc_S) in (segments or {}).items():
        gb = _trimmed_mean(gyro_S, trim=0.1)
        ab = _trimmed_mean(freeacc_S, trim=0.1)
        biases[name] = (gb, ab)

    return CalibrationResult(windows=wins, yaw_ref=float(yaw_ref), yaw_source=str(yaw_source), biases=biases)
