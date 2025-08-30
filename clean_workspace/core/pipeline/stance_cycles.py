from __future__ import annotations
import numpy as np
from ..config.constants import (
    STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES,
    CYCLE_N, CYCLE_DROP_FIRST, CYCLE_DROP_LAST, CYCLE_MIN_DUR_S, CYCLE_MAX_DUR_S,
)

__all__ = ["composite_stance","contacts_from_stance","cycle_mean_sd"]

def composite_stance(omega_W: np.ndarray, a_free_W: np.ndarray,
                     th_w: float = STANCE_THR_W, th_a: float = STANCE_THR_A,
                     hyst: int = STANCE_HYST_SAMPLES) -> np.ndarray:
    wmag = np.linalg.norm(omega_W, axis=1)
    amag = np.linalg.norm(a_free_W, axis=1)
    raw = (wmag < th_w) & (amag < th_a)
    out = raw.copy()
    off = 0
    for i in range(1, len(raw)):
        if not out[i-1] and raw[i]:
            off = i
        if (raw[i] == False) and (out[i-1] == True):
            if (i - off) <= hyst:
                out[off:i] = True
    return out

def contacts_from_stance(stance: np.ndarray) -> np.ndarray:
    s = np.asarray(stance, dtype=bool)
    return np.flatnonzero((~s[:-1]) & s[1:]) + 1

def cycle_mean_sd(t: np.ndarray, sig: np.ndarray, contacts: np.ndarray, n: int = CYCLE_N,
                  drop_first: int = CYCLE_DROP_FIRST, drop_last: int = CYCLE_DROP_LAST,
                  min_dur_s: float = CYCLE_MIN_DUR_S, max_dur_s: float = CYCLE_MAX_DUR_S):
    if len(contacts) < 2:
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 0, 0
    used = []
    total = 0
    for i in range(len(contacts) - 1):
        total += 1
        if i < drop_first or i >= (len(contacts) - 1 - drop_last):
            continue
        s = int(contacts[i]); e = int(contacts[i+1])
        if e - s < 3:
            continue
        dur = float(t[min(e, len(t)-1)] - t[min(s, len(t)-1)])
        if not (min_dur_s <= dur <= max_dur_s):
            continue
        x = np.linspace(0.0, 1.0, e - s, dtype=float)
        xi = np.linspace(0.0, 1.0, n, dtype=float)
        seg = np.asarray(sig[s:e], dtype=float)
        if seg.ndim == 1:
            yi = np.interp(xi, x, seg)
        else:
            yi = np.vstack([np.interp(xi, x, seg[:, j]) for j in range(seg.shape[1])]).T
        used.append(yi)
    if not used:
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 0, total
    arr = np.stack(used, axis=0)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    return mean.astype(np.float32), sd.astype(np.float32), int(len(used)), int(total)
