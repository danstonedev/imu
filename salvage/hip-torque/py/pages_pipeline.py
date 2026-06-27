# pages_pipeline.py
# Client-side pipeline for Pyodide (browser). Uses only numpy/pandas + hip_inverse_dynamics.
# Steps: load CSVs -> optional overlap cleaning -> optional standing calibration -> hip ID -> return JSON-friendly dict.
import numpy as np, pandas as pd, io, json
import os
import re
from itertools import islice
from hip_inverse_dynamics import (
    SegmentKinematics, make_inertial_props, inverse_dynamics_lowerlimb_3D,
    com_from_joints_linear, GRAVITY, deleva_lower_limb_fractions
)

def _q_to_R(qw, qx, qy, qz):
    # Robust quaternion to rotation (returns 3x3; identity on invalid input)
    if not (np.isfinite(qw) and np.isfinite(qx) and np.isfinite(qy) and np.isfinite(qz)):
        return np.eye(3, dtype=float)
    nrm = float((qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5)
    if not np.isfinite(nrm) or nrm <= 0:
        return np.eye(3, dtype=float)
    w, x, y, z = qw/nrm, qx/nrm, qy/nrm, qz/nrm
    # Standard right-handed rotation matrix from unit quaternion (w,x,y,z)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def _read_xsens(path: str):
    # --- find actual header row (skip DeviceTag/Firmware... preamble) ---
    header_idx = None
    header_line = None
    preamble = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < 300:
                preamble.append(line)
            if ("PacketCounter" in line or "PacketCount" in line) and ("Quat" in line or "FreeAcc" in line):
                header_idx = i
                header_line = line
                break
        if header_idx is None:
            # If not found yet, read up to 300 lines and pick likely header by density + keywords
            for j, line in enumerate(islice(f, 0, 300)):
                if (line.count(",") >= 4 or line.count("\t") >= 4) and ("Quat" in line or "FreeAcc" in line):
                    header_idx = (i + 1) + j  # continue count
                    header_line = line
                    break
    if header_idx is None:
        sample = ''.join(preamble[:10])
        raise ValueError(f"Could not locate XSENS header row in {path}.\nFirst lines:\n{sample}")

    # Infer separator deterministically (fast parser)
    sep = ","
    # explicit sep= hint
    for l in preamble:
        if l.lower().startswith("sep=") and len(l.strip()) >= 5:
            sep = l.strip()[4]
            break
    else:
        if header_line is not None:
            sep = "\t" if header_line.count("\t") > header_line.count(",") else ","

    # First pass: header only to get raw columns (fast)
    df0 = pd.read_csv(path, sep=sep, skiprows=header_idx, nrows=0, engine="c")
    raw_cols = list(df0.columns)

    # --- sanitize & map columns ---

    def san(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"[^0-9A-Za-z]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_").lower()

    san_cols = [san(c) for c in raw_cols]
    san_to_orig = dict(zip(san_cols, raw_cols))

    def find_col(candidates, required=True):
        def norm(x): return x.replace("_", "").lower()
        for cand in candidates:
            c_norm = norm(cand)
            for s in san_cols:
                if s == cand.lower() or s.startswith(cand.lower()) or norm(s) == c_norm or norm(s).startswith(c_norm):
                    return san_to_orig[s]
        if required:
            raise ValueError(
                f"Missing required column. Tried any of: {candidates} in {path}\n"
                f"Available (sanitized): {san_cols}"
            )
        return None

    # quaternion + free-acc (accept Quat_W and q0..q3)
    qw = find_col(['quat_w','quaternion_w','orientation_w','ori_w','qw','quatw','quaternionw','orientationw','oriw','q0'])
    qx = find_col(['quat_x','quaternion_x','orientation_x','ori_x','qx','quatx','quaternionx','orientationx','orix','q1'])
    qy = find_col(['quat_y','quaternion_y','orientation_y','ori_y','qy','quaty','quaterniony','orientationy','oriy','q2'])
    qz = find_col(['quat_z','quaternion_z','orientation_z','ori_z','qz','quatz','quaternionz','orientationz','oriz','q3'])

    fax = find_col(['freeacc_x','free_acc_x','freeaccx'])
    fay = find_col(['freeacc_y','free_acc_y','freeaccy'])
    faz = find_col(['freeacc_z','free_acc_z','freeaccz'])

    # optional high-res timestamp
    stf = find_col(['sampletimefine','sample_time_fine','sample_timefine','sample_time','sampletime','timestamp','time'], required=False)

    # Build usecols with original names for fast parsing
    want_cols = [qw, qx, qy, qz, fax, fay, faz]
    if stf: want_cols.append(stf)

    # Second pass: load only needed columns with fast C engine
    df = pd.read_csv(path, sep=sep, skiprows=header_idx, usecols=want_cols, engine="c")
    # numeric conversion (ensure float32 dtype for compute-heavy columns)
    for c in [qw, qx, qy, qz, fax, fay, faz]:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32', copy=False)
    if stf:
        df[stf] = pd.to_numeric(df[stf], errors='coerce')

    # build time vector in seconds
    T = len(df)
    if stf is not None and df[stf].notna().any():
        v = df[stf].to_numpy()
        v = v - (v[0] if len(v) else 0)
        if len(v) > 1:
            dv = np.diff(v)
            med = float(np.nanmedian(np.abs(dv))) if np.isfinite(dv).any() else 0.0
        else:
            med = 0.0
        # choose scale that best matches ~60 Hz (0.0167 s)
        target_dt = 1.0/60.0
        scales = [1e-6, 1e-4, 1e-3, 1.0]
        if med > 0:
            errs = [abs((med*s) - target_dt) for s in scales]
            scale = scales[int(np.argmin(errs))]
        else:
            scale = 1.0/60.0  # fallback to indices
        t = v * scale if med > 0 else np.arange(T, dtype=float) * (1.0/60.0)
    else:
        t = np.arange(T, dtype=float) * (1.0/60.0)

    # filename-based offset: last 3 digits before .csv are the start milliseconds
    try:
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"(\d{3})$", base)
        if m:
            start_ms = int(m.group(1))
            t = t + (start_ms / 1000.0)
    except Exception:
        pass

    # quaternion to rotation matrices (vectorized)
    qwv = df[qw].to_numpy(dtype=np.float32, copy=False)
    qxv = df[qx].to_numpy(dtype=np.float32, copy=False)
    qyv = df[qy].to_numpy(dtype=np.float32, copy=False)
    qzv = df[qz].to_numpy(dtype=np.float32, copy=False)
    nrm = np.sqrt(qwv*qwv + qxv*qxv + qyv*qyv + qzv*qzv).astype(np.float32, copy=False)
    # avoid div by zero
    nrm[nrm == 0] = 1.0
    w = qwv / nrm; x = qxv / nrm; y = qyv / nrm; z = qzv / nrm
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.empty((T,3,3), dtype=np.float32)
    R[:,0,0] = 1 - 2*(yy + zz)
    R[:,0,1] = 2*(xy - wz)
    R[:,0,2] = 2*(xz + wy)
    R[:,1,0] = 2*(xy + wz)
    R[:,1,1] = 1 - 2*(xx + zz)
    R[:,1,2] = 2*(yz - wx)
    R[:,2,0] = 2*(xz - wy)
    R[:,2,1] = 2*(yz + wx)
    R[:,2,2] = 1 - 2*(xx + yy)

    # FreeAcc (sensor/body) -> world
    Ab = np.column_stack([
        df[fax].to_numpy(dtype=np.float32, copy=False),
        df[fay].to_numpy(dtype=np.float32, copy=False),
        df[faz].to_numpy(dtype=np.float32, copy=False)
    ]).astype(np.float32, copy=False)
    Aw = np.einsum('tij,tj->ti', R, Ab).astype(np.float32, copy=False)

    # approximate angular velocity from orientation derivative (vectorized)
    if T > 1:
        dt_arr = np.gradient(t)
        Rdot = np.gradient(R, axis=0) / dt_arr[:, None, None]
        Wm = np.einsum('tij,tjk->tik', Rdot, np.transpose(R, (0,2,1)))
        omega = np.empty((T,3), dtype=np.float32)
        omega[:,0] = 0.5*(Wm[:,2,1] - Wm[:,1,2])
        omega[:,1] = 0.5*(Wm[:,0,2] - Wm[:,2,0])
        omega[:,2] = 0.5*(Wm[:,1,0] - Wm[:,0,1])
    else:
        omega = np.zeros((T,3), dtype=np.float32)

    return {"t": t, "R": R, "omega": omega, "acc": Aw}
def _overlap_window(ds):
    starts = [d["t"][0] for d in ds]
    ends   = [d["t"][-1] for d in ds]
    return max(starts), min(ends)

def _trim(d, t0, t1):
    m = (d["t"]>=t0) & (d["t"]<=t1)
    out = {k:(v[m].copy() if hasattr(v,'__len__') else v) for k,v in d.items()}
    return out

def _avg_rot(Rs):
    M = Rs.mean(axis=0)
    U,S,Vt = np.linalg.svd(M)
    Ravg = U @ Vt
    if np.linalg.det(Ravg) < 0:
        U[:, -1] *= -1
        Ravg = U @ Vt
    return Ravg

def _static_mask(omega, dt, window_s=3.0, thresh=0.2):
    n = len(omega)
    w = max(1,int(window_s/dt))
    beg = np.arange(0,min(n,w)); end=np.arange(max(0,n-w),n)
    wn = np.linalg.norm(omega,axis=1)
    m = np.zeros(n,dtype=bool); m[beg]=wn[beg]<thresh; m[end]=wn[end]<thresh
    return m

def _calibrate(d):
    dt = np.median(np.diff(d["t"])) if len(d["t"])>1 else 1/60.0
    m = _static_mask(d["omega"], dt)
    Rs = d["R"][m] if m.any() else d["R"][:max(20,int(0.5/dt))]
    Rstat = _avg_rot(Rs)
    C = Rstat.T
    Rcal = np.einsum('tij,jk->tik', d["R"], C)
    # recompute omega (vectorized)
    dt_arr = np.gradient(d["t"]) if len(d["t"])>1 else np.array([1/60.0])
    Rdot = np.gradient(Rcal, axis=0) / dt_arr[:,None,None]
    Wm = np.einsum('tij,tjk->tik', Rdot, np.transpose(Rcal, (0,2,1)))
    omega = np.empty_like(d["omega"])  # (T,3)
    omega[:,0] = 0.5*(Wm[:,2,1] - Wm[:,1,2])
    omega[:,1] = 0.5*(Wm[:,0,2] - Wm[:,2,0])
    omega[:,2] = 0.5*(Wm[:,1,0] - Wm[:,0,1])
    # rotate FreeAcc using calibrated R (first infer body approx then reapply)
    Ab = np.einsum('tij,tj->ti', np.transpose(d["R"], (0,2,1)), d["acc"])
    Aw = np.einsum('tij,tj->ti', Rcal, Ab)
    return {"t": d["t"], "R": Rcal, "omega": omega, "acc": Aw}

def _build_chain(R_thigh, R_tibia, pelvis_h, L_thigh, L_shank, L_foot):
    T = R_thigh.shape[0]
    hip = np.tile(np.array([0.0, 0.0, pelvis_h], dtype=float), (T,1))
    thigh_dir0 = np.array([0.0, 0.0, -1.0], dtype=float)
    shank_dir0 = np.array([0.0, 0.0, -1.0], dtype=float)
    foot_dir0  = np.array([1.0, 0.0,  0.0], dtype=float)
    knee  = hip + np.einsum('tij,j->ti', R_thigh, thigh_dir0 * L_thigh)
    ankle = knee + np.einsum('tij,j->ti', R_tibia, shank_dir0 * L_shank)
    toe   = ankle + np.einsum('tij,j->ti', R_tibia, foot_dir0  * L_foot)
    return hip, knee, ankle, toe, R_tibia.copy()

def _detect_stance(accW, omegaW, acc_thresh=2.0, omega_thresh=1.5):
    a = np.linalg.norm(accW,axis=1); w=np.linalg.norm(omegaW,axis=1)
    s=((a<acc_thresh)&(w<omega_thresh)).astype(float)
    ker=np.ones(5)/5.0
    return np.convolve(s,ker,mode='same')

def _rocker(s, N):
    # Build monotonic 0->1 over each stance region; if none, simple linspace to avoid crashes
    u = np.zeros(N); on=False; t0=0
    for i,val in enumerate(s>0.5):
        if val and not on: on=True; t0=i
        if (not val) and on:
            dur = i-t0
            if dur>1:
                u[t0:i] = np.linspace(0,1,dur)
            on=False
    if on:
        dur = N - t0
        if dur>1:
            u[t0:] = np.linspace(0,1,dur)
    if not (u>0).any():
        u = np.linspace(0,1,N)  # fallback
    return u

def _cycle_norm(t, y, stance):
    s = (stance>0.5).astype(int)
    ds = np.diff(s, prepend=s[0])
    hs = np.where(ds==1)[0]
    if len(hs)<2: return None,None,None
    curves=[]
    for i in range(len(hs)-1):
        a,b = hs[i], hs[i+1]
        if b<=a+5: continue
        tt = t[a:b]-t[a]
        yy = y[a:b]
        pct = np.linspace(0,1,101)
        yn = np.interp(pct, tt/tt[-1], yy)
        curves.append(yn)
    if len(curves)==0: return None,None,None
    M = np.vstack(curves)
    return (np.linspace(0,100,101), M.mean(axis=0), M.std(axis=0))

def process_files(pelvis, L_thigh, R_thigh, L_tibia, R_tibia, height, mass, do_cal=True, do_overlap=True, fast_mode=False):
    # 1) Load
    P = _read_xsens(pelvis)
    LTh = _read_xsens(L_thigh)
    RTh = _read_xsens(R_thigh)
    LTi = _read_xsens(L_tibia)
    RTi = _read_xsens(R_tibia)

    # 2) Overlap cleaning
    if do_overlap:
        t0, t1 = _overlap_window([P, LTh, RTh, LTi, RTi])
        P = _trim(P, t0, t1)
        LTh = _trim(LTh, t0, t1)
        RTh = _trim(RTh, t0, t1)
        LTi = _trim(LTi, t0, t1)
        RTi = _trim(RTi, t0, t1)
        # equalize counts
        N = min(len(P["t"]), len(LTh["t"]), len(RTh["t"]), len(LTi["t"]), len(RTi["t"]))
        for d in (P, LTh, RTh, LTi, RTi):
            for k in ("t", "R", "omega", "acc"):
                d[k] = d[k][:N]

    # 3) Standing calibration
    if do_cal:
        P = _calibrate(P)
        LTh = _calibrate(LTh)
        RTh = _calibrate(RTh)
        LTi = _calibrate(LTi)
        RTi = _calibrate(RTi)

    # 3.5) Fast mode: decimate to ~60 Hz if higher
    if fast_mode:
        def decimate(d, target_hz=60.0):
            t = d["t"]
            if len(t) < 2:
                return d
            dt = float(np.median(np.diff(t)))
            if dt <= 0:
                return d
            hz = 1.0/dt
            if hz <= target_hz + 1e-3:
                return d
            step = max(1, int(round(hz/target_hz)))
            return {k: (v[::step].copy() if hasattr(v, '__len__') else v) for k, v in d.items()}
        P = decimate(P)
        LTh = decimate(LTh)
        RTh = decimate(RTh)
        LTi = decimate(LTi)
        RTi = decimate(RTi)

    # 4) Kinematics model
    scale = height / 1.75
    L_th = 0.45 * scale
    L_sh = 0.43 * scale
    L_fo = 0.25 * scale
    pelvis_h = 0.95 * scale
    heel_to_ankle = 0.07 * scale
    fr = deleva_lower_limb_fractions()

    def run_side(Th, Ti):
        Tn = len(Th["t"])
        dt = np.median(np.diff(Th["t"])) if Tn > 1 else 1/60.0
        hip, knee, ankle, toe, Rfoot = _build_chain(Th["R"], Ti["R"], pelvis_h, L_th, L_sh, L_fo)
        rcom_th = com_from_joints_linear(hip, knee, fr["thigh"]["com"])
        rcom_sh = com_from_joints_linear(knee, ankle, fr["shank"]["com"])
        rcom_fo = com_from_joints_linear(ankle, toe, fr["foot"]["com"])
        a_th, a_sh, a_fo = Th["acc"], Ti["acc"], Ti["acc"]
        stance = _detect_stance(Ti["acc"], Ti["omega"])
        u = _rocker(stance, Tn)
        # GRF
        Ftot = np.c_[mass * P["acc"][:, 0], mass * P["acc"][:, 1], mass * (P["acc"][:, 2] + GRAVITY[2])]
        Fw = stance[:, None] * Ftot
        # CoP (vectorized in world)
        x_heel = -heel_to_ankle
        x_toe = L_fo - heel_to_ankle
        copFx = (1.0 - u) * x_heel + u * x_toe
        copF = np.stack([copFx, np.zeros_like(copFx), np.zeros_like(copFx)], axis=1)
        rcop = np.einsum('tij,tj->ti', Rfoot, copF) + ankle
        # inertial properties
        prop_fo = make_inertial_props(height, mass, L_fo, "foot")
        prop_sh = make_inertial_props(height, mass, L_sh, "shank")
        prop_th = make_inertial_props(height, mass, L_th, "thigh")
        # dynamics (batched)
        alpha_th = np.gradient(Th["omega"], axis=0) / (np.gradient(Th["t"])[:, None] if Tn > 1 else 1/60.0)
        alpha_ti = np.gradient(Ti["omega"], axis=0) / (np.gradient(Ti["t"])[:, None] if Tn > 1 else 1/60.0)
        from hip_inverse_dynamics import inverse_dynamics_lowerlimb_3D_batch
        out = inverse_dynamics_lowerlimb_3D_batch(
            Rfoot, Ti["omega"], alpha_ti, rcom_fo, a_fo, ankle, toe,
            Ti["R"], Ti["omega"], alpha_ti, rcom_sh, a_sh, knee,
            Th["R"], Th["omega"], alpha_th, rcom_th, a_th, hip,
            prop_fo, prop_sh, prop_th,
            Fw, rcop, None
        )
        hipM_world = out["hip_M_W"]
        M_thigh = np.einsum('tji,ti->tj', Th["R"], hipM_world)
        return Th["t"], M_thigh[:, 1], stance

    tL, Mleft, stanceL = run_side(LTh, LTi)
    tR, Mright, stanceR = run_side(RTh, RTi)

    # Cycle stats (lighter in fast mode: keep mean only)
    pctL, meanL, sdL = _cycle_norm(tL, Mleft, stanceL)
    pctR, meanR, sdR = _cycle_norm(tR, Mright, stanceR)
    if fast_mode:
        if meanL is not None: sdL = np.zeros_like(meanL)
        if meanR is not None: sdR = np.zeros_like(meanR)
    if pctL is None or pctR is None:
        pct = list(np.linspace(0, 100, 101))
        meanL_list = sdL_list = meanR_list = sdR_list = [float('nan')] * 101
    else:
        pct = list(pctL)  # assume both are 0..100 same
        meanL_list = list(meanL if meanL is not None else np.full(101, np.nan))
        sdL_list = list(sdL if sdL is not None else np.full(101, np.nan))
        meanR_list = list(meanR if meanR is not None else np.full(101, np.nan))
        sdR_list = list(sdR if sdR is not None else np.full(101, np.nan))

    # CSVs
    left_df = pd.DataFrame({'time_s': tL, 'hip_My_Nm': Mleft})
    right_df = pd.DataFrame({'time_s': tR, 'hip_My_Nm': Mright})
    left_csv = left_df.to_csv(index=False)
    right_csv = right_df.to_csv(index=False)

    out = dict(
        time_s=list(tL),
        left_ts=list(Mleft),
        right_ts=list(Mright),
        cycle_pct=pct,
        left_mean=meanL_list,
        left_sd=sdL_list,
        right_mean=meanR_list,
        right_sd=sdR_list,
        left_csv=left_csv,
        right_csv=right_csv,
    )
    # Optionally expose flat arrays for GPU experiments (kept backward-compatible)
    try:
        # Provide any one segment’s quaternion/accel as a demo hook
        out["q_flat"] = list(np.column_stack([
            LTh["R"][:,0,0]*0+1.0,  # placeholder w=1 (identity) since we currently rotate via R
            LTh["R"][:,0,0]*0, LTh["R"][:,0,0]*0, LTh["R"][:,0,0]*0
        ]).astype(np.float32).ravel())
        out["acc_flat"] = list(LTh["acc"].astype(np.float32).ravel())
    except Exception:
        pass
    return out
