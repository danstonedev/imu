from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from core.pipeline.pipeline import run_pipeline_clean

root = Path(__file__).resolve().parents[1]


def pick(pattern: str) -> str:
    m = sorted(root.joinpath('sample data').glob(pattern))
    if not m:
        raise FileNotFoundError(pattern)
    return str(m[0])


def summarize_angles(A: np.ndarray, t: np.ndarray, stance: np.ndarray, start_margin=2.0, end_margin=2.0):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        return None
    # define steady windows: first 10s after gait starts and last 10s before end
    t0, t1 = float(t[0]), float(t[-1])
    # detect gait start/end from stance mask edges
    def first_active(m):
        idx = np.where(np.asarray(m, dtype=bool))[0]
        return int(idx[0]) if idx.size else 0
    def last_active(m):
        idx = np.where(np.asarray(m, dtype=bool))[0]
        return int(idx[-1]) if idx.size else len(m)-1
    i_start = first_active(stance)
    i_end = last_active(stance)
    t_start = t[i_start] + start_margin
    t_end = t[i_end] - end_margin
    wA0 = (t >= t_start) & (t < t_start + 10.0)
    wA1 = (t <= t_end) & (t > max(t_end - 10.0, t[0]))
    def stats(win):
        if not np.any(win):
            return (np.full(3, np.nan), float('nan'))
        m = np.nanmean(A[win], axis=0)
        mx = float(np.nanmax(np.abs(A[win])))
        return (m, mx)
    m0, mx0 = stats(wA0)
    m1, mx1 = stats(wA1)
    # drift slope over mid 80%
    n = len(t)
    a = int(0.1*n); b = max(a+2, int(0.9*n))
    if b <= a:
        slope = np.full(3, np.nan)
    else:
        T = (t[b-1]-t[a]) / 60.0
        slope = (np.nanmean(A[b-50:b], axis=0) - np.nanmean(A[a:a+50], axis=0)) / max(T,1e-6)
    # stance means first/last 2s
    def stance_mean_at(ts, w):
        if not np.any(w):
            return np.full(3, np.nan)
        return np.nanmean(A[w & (stance>0)], axis=0)
    w_first2 = (t >= t_start) & (t < t_start + 2.0)
    w_last2 = (t <= t_end) & (t > max(t_end - 2.0, t[0]))
    stance_first = stance_mean_at(t, w_first2)
    stance_last = stance_mean_at(t, w_last2)
    return {
        'mean_first10': m0.tolist() if isinstance(m0, np.ndarray) else [np.nan]*3,
        'mean_last10': m1.tolist() if isinstance(m1, np.ndarray) else [np.nan]*3,
        'abs_mean_diff': (np.abs(m1-m0)).tolist() if isinstance(m0, np.ndarray) else [np.nan]*3,
        'drift_slope_deg_per_min': slope.tolist() if isinstance(slope, np.ndarray) else [np.nan]*3,
        'max_abs_first10': mx0,
        'max_abs_last10': mx1,
        'stance_mean_first2s': stance_first.tolist(),
        'stance_mean_last2s': stance_last.tolist(),
    }


def run_matrix():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['A','B','C','ALL'], default='ALL')
    args = ap.parse_args()
    paths = {
        'pelvis': pick('DEMO6_0_*.csv'),
        'lfemur': pick('DEMO6_1_*.csv'),
        'rfemur': pick('DEMO6_2_*.csv'),
        'ltibia': pick('DEMO6_3_*.csv'),
        'rtibia': pick('DEMO6_4_*.csv'),
    }
    # Common options
    base_opts = {'do_cal': True, 'yaw_align': True, 'debug_assert': True}
    # A: pre-Euler yaw-share only (simulated via baseline mode yaw_share_only)
    optsA = dict(base_opts, angles_baseline_mode='yaw_share_only', angles_highpass_fc=None)
    # B: stride-debias only
    optsB = dict(base_opts, angles_baseline_mode='stride_debias_only', angles_highpass_fc=None)
    # C: baseline + HP=0.03 Hz
    optsC = dict(base_opts, angles_baseline_mode=None, angles_highpass_fc=0.03)

    out = {}
    def collect(tag: str, opts: dict):
        res = run_pipeline_clean(paths, height_m=1.75, mass_kg=75.0, options=opts)
        tL = np.asarray(res['time_L'] if 'time_L' in res else res['tL'])
        tR = np.asarray(res['time_R'] if 'time_R' in res else res['tR'])
        stanceL = np.asarray(res['stance_L'], dtype=bool)
        stanceR = np.asarray(res['stance_R'], dtype=bool)
        for key in ['L_hip_angles_deg','R_hip_angles_deg','L_knee_angles_deg','R_knee_angles_deg']:
            A = np.asarray(res[key], dtype=float)
            if key.startswith('L_'):
                sm = summarize_angles(A, tL, stanceL)
            else:
                sm = summarize_angles(A, tR, stanceR)
            out.setdefault(tag, {})[key] = sm
    if args.mode in ('A','ALL'):
        collect('A_yaw_share_only', optsA)
    if args.mode in ('B','ALL'):
        collect('B_stride_debias_only', optsB)
    if args.mode in ('C','ALL'):
        collect('C_baseline_plus_HP', optsC)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    run_matrix()
