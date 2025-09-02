from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics
from core.math.kinematics import quats_to_R_batch, estimate_fs, resample_quat, resample_vec, gyro_from_quat
from core.math.baseline import BaselineConfig, apply_baseline_correction
from core.pipeline.unified_gait import detect_gait_cycles
from core.math.kinematics import hip_angles_xyz, knee_angles_xyz

# Helper: load a single trial with pelvis, femur, tibia files from a folder or explicit files
# For simplicity here we assume demo style 5 files are present in the folder.

def load_demo_trial(folder: Path):
    folder = Path(folder)
    files = sorted(folder.glob('DEMO6_*_*.csv'))
    pick = lambda idx: files[idx].read_bytes()
    def parse(b):
        df = read_xsens_bytes(b)
        t, q, g, a = extract_kinematics(df)
        return t, q, g, a
    tP, qP, gP, aP = parse(pick(0))
    tLf, qLf, gLf, aLf = parse(pick(1))
    tRf, qRf, gRf, aRf = parse(pick(2))
    tLt, qLt, gLt, aLt = parse(pick(3))
    tRt, qRt, gRt, aRt = parse(pick(4))
    return {
        'tP': tP, 'qP': qP,
        'tLf': tLf, 'qLf': qLf, 'gLf': gLf, 'aLf': aLf,
        'tRf': tRf, 'qRf': qRf, 'gRf': gRf, 'aRf': aRf,
        'tLt': tLt, 'qLt': qLt, 'gLt': gLt, 'aLt': aLt,
        'tRt': tRt, 'qRt': qRt, 'gRt': gRt, 'aRt': aRt,
    }


def per_stride_windows(hs_idx: np.ndarray, n: int) -> list[tuple[int,int]]:
    if hs_idx is None or len(hs_idx) < 2:
        return []
    hs = np.asarray(hs_idx, dtype=int)
    hs = hs[(hs >= 0) & (hs < n)]
    return [(int(a), int(b)) for a,b in zip(hs[:-1], hs[1:]) if int(b) > int(a)+1]


def metrics(t: np.ndarray, raw_deg: np.ndarray, corr_deg: np.ndarray, strides: list[tuple[int,int]]):
    # focus Y (1) and Z (2)
    YZr = raw_deg[:,1:3]
    YZc = corr_deg[:,1:3]
    dc_shift = np.mean(YZc, axis=0) - np.mean(YZr, axis=0)
    # drift slope over mid 80%
    n = len(t)
    i0 = int(0.1*n); i1 = int(0.9*n)
    def slope(y):
        x = t[i0:i1] - t[i0]
        yy = y[i0:i1]
        if x.size < 2:
            return np.nan
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, yy, rcond=None)[0]
        return float(m*60.0)  # per minute
    slope_y = slope(YZc[:,0])
    slope_z = slope(YZc[:,1])
    # per-stride mean RMS (on corrected)
    means = []
    for s,e in strides:
        if e-s < 2:
            continue
        m = np.mean(YZc[s:e], axis=0)
        means.append(m)
    means = np.array(means) if means else np.zeros((0,2))
    rms = np.sqrt(np.mean(means**2, axis=0)) if means.size else np.array([np.nan, np.nan])
    return dc_shift, (slope_y, slope_z), rms


def main():
    ap = argparse.ArgumentParser(description='Diagnose baseline corrections A/B')
    ap.add_argument('--trial', type=str, required=False, default=str(Path(__file__).resolve().parents[1]/'sample data'), help='Path to folder with demo 5 CSVs')
    ap.add_argument('--joint', type=str, choices=['hip','knee'], default='hip')
    ap.add_argument('--side', type=str, choices=['L','R'], default='L')
    args = ap.parse_args()

    data = load_demo_trial(Path(args.trial))
    # Resample tibia/pelvis to femur time per side
    if args.side == 'L':
        tF = data['tLf']; qF = data['qLf']
        tT = data['tLt']; qT = data['qLt']; gT = data['gLt']; aT = data['aLt']
    else:
        tF = data['tRf']; qF = data['qRf']
        tT = data['tRt']; qT = data['qRt']; gT = data['gRt']; aT = data['aRt']

    # Femur time baseline
    t = tF - tF[0]
    # Resample pelvis and tibia to femur time
    tT0 = tT - tT[0]
    tP = data['tP']; tP0 = tP - tP[0]
    qT_res = resample_quat(tT0, qT, t)
    qP_res = resample_quat(tP0, data['qP'], t)
    Rf = quats_to_R_batch(qF)
    Rt = quats_to_R_batch(qT_res)
    Rp = quats_to_R_batch(qP_res)
    # Signals: if gyro missing, derive from quats; resample vectors to femur time
    if gT is None:
        gT_orig = gyro_from_quat(tT, qT)  # body-frame rad/s
    else:
        gT_orig = gT
    aT_orig = aT
    gT_res = resample_vec(tT0, gT_orig, t)
    aT_res = resample_vec(tT0, aT_orig, t)
    # World-frame
    accW = (Rt @ aT_res[...,None]).squeeze(-1)
    gyroW = (Rt @ gT_res[...,None]).squeeze(-1)

    fs = estimate_fs(t)
    gait = detect_gait_cycles(t_left=t, accel_left=accW, gyro_left=gyroW,
                              t_right=t, accel_right=accW, gyro_right=gyroW, fs=fs)
    hs = np.asarray(gait['heel_strikes_left'], dtype=int)
    strides = per_stride_windows(hs, len(t))

    # raw angles
    if args.joint=='hip':
        A = hip_angles_xyz(Rp, Rf, side=args.side)
    else:
        A = knee_angles_xyz(Rf, Rt, side=args.side)
    A_deg = np.rad2deg(A).astype(np.float32)

    cfg = BaselineConfig(fs_hz=fs, use_yaw_share=True, yaw_share_fc_hz=0.05,
                         stride_debias_axes=("Y","Z"), highpass_fc_hz=None,
                         min_stride_samples=30, allow_stack=False)

    modes = ["none","yaw_share_only","stride_debias_only","highpass_only"]
    results = {}

    # Provide yaw traces for yaw_share_only
    from core.math.kinematics import yaw_from_R
    yaw_p = yaw_from_R(Rp)
    yaw_f = yaw_from_R(Rf)

    for m in modes:
        Ac = apply_baseline_correction(t, A, strides, cfg, yaw_pelvis=yaw_p, yaw_femur=yaw_f, mode=m)
        Ac_deg = np.rad2deg(Ac)
        dc, slope, rms = metrics(t, A_deg, Ac_deg, strides)
        results[m] = { 'angles_deg': Ac_deg, 'dc_shift_deg': dc, 'drift_slope_deg_per_min': slope, 'per_stride_mean_rms_deg': rms }

    # Print config and metrics
    print("fs_hz=", fs, "min_stride_samples=", cfg.min_stride_samples, "n_strides=", len(strides))
    if strides:
        lens = [e-s for s,e in strides]
        print("stride_len(samples): min/med/max=", int(np.min(lens)), int(np.median(lens)), int(np.max(lens)))
    print()
    rows = []
    for m in modes:
        dc = results[m]['dc_shift_deg']
        slope = results[m]['drift_slope_deg_per_min']
        rms = results[m]['per_stride_mean_rms_deg']
        rows.append([m, f"dc(Y,Z)={dc[0]:.2f},{dc[1]:.2f}", f"slope(Y,Z)={slope[0]:.2f},{slope[1]:.2f}", f"strideRMS(Y,Z)={rms[0]:.2f},{rms[1]:.2f}"])
    print(pd.DataFrame(rows, columns=['mode','dc_shift','drift_slope_per_min','per_stride_mean_rms']).to_string(index=False))

    # Plot overlay
    plt.figure(figsize=(12,6))
    titles = ['flex (X)','add (Y)','rot (Z)']
    for j in range(3):
        ax = plt.subplot(3,1,j+1)
        ax.plot(t, A_deg[:,j], 'k-', alpha=0.8, label='RAW')
        for m, color in zip(modes[1:], ['tab:orange','tab:green','tab:blue']):
            ax.plot(t, results[m]['angles_deg'][:,j], color=color, alpha=0.8, label=m)
        ax.set_ylabel(titles[j]+" (deg)")
        if j==0:
            ax.set_title(f"{args.side} {args.joint} baseline modes vs RAW")
        if j==2:
            ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    outdir = Path('analysis_out')
    outdir.mkdir(exist_ok=True)
    out_png = outdir / f"diagnose_baseline_{args.joint}_{args.side}.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print("saved:", out_png)

if __name__ == '__main__':
    main()
