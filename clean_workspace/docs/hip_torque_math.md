# Hip Torque Math Spec

This document extracts the math that powers the current prototype so we can rebuild a clean implementation. It captures coordinate conventions, formulas, algorithms, shapes, and thresholds without any UI/server code.

Assumptions and scope:

- Teaching-grade inverse dynamics for hip moments without direct GRF, using IMU kinematics only (pelvis, femurs, tibias).
- Robust, simple methods favored over full biomechanical fidelity.
- Frames: S = sensor/body frame; W = world frame; J = joint coordinate system.
- Quaternion convention: [w, x, y, z]. Rotation matrices map S→W.

---

## Notation and conventions

- Time array t: seconds, 1-D shape (T,).
- Vectors: row-wise time series with shape (T, 3).
- Rotations: R_WS of shape (T, 3, 3) mapping S→W. Columns are basis vectors of S expressed in W.
- Gravity: g_W = [0, 0, -9.80665] m/s^2 (z-up world, gravity down).
- Femur anatomical axis e3 is taken as the 3rd column of R_femur (simplification).

---

## Quaternion to rotation

Input: quats q = (T,4) in [w,x,y,z]. Normalize first: q ← q / ||q||.

Rotation matrix R per sample (S→W):

- Precompute ww, xx, yy, zz, wx, wy, wz, xy, xz, yz.
- R =
	[ 1-2(yy+zz)   2(xy-wz)   2(xz+wy) ]
	[ 2(xy+wz)   1-2(xx+zz)   2(yz-wx) ]
	[ 2(xz-wy)     2(yz+wx)   1-2(xx+yy) ]

Function contract:

- quats_to_R_batch(quats: (T,4)) -> R_WS: (T,3,3)

Map sensor vectors to world: v_W = R_WS @ v_S.

---

## Angular velocity from quaternions (fallback)

When gyro isn’t available, approximate ω in body frame from consecutive orientations:

1. Compute R_i and R_{i+1} from quats.
2. Relative rotation dR = R_i^T R_{i+1}.
3. Angle θ = arccos((trace(dR) − 1)/2). If θ≈0, axis = 0.
4. Axis:
	ax = (dR[2,1] − dR[1,2]) / (2 sin θ)
	ay = (dR[0,2] − dR[2,0]) / (2 sin θ)
	az = (dR[1,0] − dR[0,1]) / (2 sin θ)
5. ω_i ≈ axis · (θ / Δt_i)

Shape: ω (T,3) in rad/s.

---

## Yaw extraction and alignment

Yaw from R_WS: ex = [1,0,0]; e = R_WS · ex; yaw = atan2(e_y, e_x).

To align a segment’s yaw to pelvis by a scalar Δψ (constant across time):

- Build Rz(Δψ) for each sample and left-multiply: R_aligned = Rz(Δψ) · R_WS.
- Δψ is chosen as the median yaw difference during “still” calibration windows (fallbacks provided below).

---

## Sampling rate estimation

Fs ≈ 1 / median(diff(t[diff(t)>0])). Default 100 Hz if unknown.

---

## Moving average smoother

For x with shape (T,) or (T,3): boxcar of length win samples (default win≈0.7 s·Fs):

- y = x * (1/win) convolved “same”. Applied component-wise for vectors.

---

## Still detection (for calibration windows)

Inputs: gyro (T,3) [rad/s], freeacc (T,3) [m/s²]; thresholds thr_w=0.08 rad/s, thr_a=0.30 m/s².

Algorithm:

1. Fs from t; win = round(0.7 s · Fs).
2. gmag = ||gyro||, amag = ||freeacc||.
3. Smooth to gmag_s, amag_s with moving average.
4. still_mask = (gmag_s < thr_w) AND (amag_s < thr_a).

Segment selection: find contiguous True runs with length ≥ min_len samples.

- Primary: min_len ≈ round(1.0 s · Fs) (clamped 30..300).
- Secondary fallback: min_len ≈ round(0.5 s · Fs).
- Choose start and end segments near first/last 25% of the recording, preferring longer segments.
- Final fallback: fixed edge windows of ≈0.75 s at start and end (non-overlapping).

Calibration biases (applied to femur/tibia only):

- For each chosen window, compute trimmed mean (10%) of signals.
- Average window means weighted by duration to get gyro bias gpb and accel bias apb.
- Apply: gyro_corr = gyro − gpb; acc_corr = freeacc − apb.

Outputs captured for debugging: Fs, win, thresholds, chosen windows in seconds (start_s, end_s, duration_s, samples).

---

## Stance detection (per shank)

Compute in world frame using tibia IMU:

- ω_W = R_tibia · ω_S
- a_free_W = R_tibia · a_free_S

Thresholding with short-gap hysteresis (hyst=8 samples):

- raw = (||ω_W|| < 2.0 rad/s) AND (||a_free_W|| < 0.8 m/s²)
- Fill brief False gaps inside True runs up to hyst samples.

Foot-strike events are rising edges of stance: contact indices i where stance[i-1]=False and stance[i]=True.

---

## Resampling utilities

- Quaternion slerp for (t_src, Q_src) onto t_dst (per-sample on unit quats; handle antipodal).
- Vector linear interpolation component-wise (np.interp) from t_src to t_dst.

---

## Hip joint coordinate system (JCS)

Teaching-grade simplification: use femur frame as joint frame so hip moments are resolved in femur anatomical axes.

- R_WJ = R_femur (columns are femur axes expressed in world).
- Resolve a world vector into JCS by M_J = R_WJ^T · M_W.

---

## Inverse dynamics (teaching-grade, no GRF)

Goal: estimate hip moment M_hip in world coordinates using femur kinematics and simple anthropometrics.

Inputs per sample:

- t (T,), R_femur (T,3,3), ω_femur_S (T,3), a_femur_S (T,3)
- R_tibia, ω_tibia_S, a_tibia_S are present but not used directly in the current hip moment estimate.
- Anthropometrics: height_m (H), mass_kg (M).

Parameters and approximations:.

- Femur length l_th ≈ clamp(0.245·H, 0.2..0.5) m.
- Femur mass m_th ≈ 0.10·M kg.
- Femur COM at r_com = 0.5·l_th along femur e3 axis.
- Rod inertia about proximal end: I_rod = m_th·l_th² / 3.
- Angular acceleration α computed from numeric derivative of ω in world frame (per component).
- COM linear acceleration a_com_W ≈ a_f_W + g_W, where a_f_W = R_femur · a_femur_S.
- Damping term −c·ω_f_W with c=0.02 for display stability.

Computation:

1. ω_f_W = R_femur · ω_femur_S; a_f_W = R_femur · a_femur_S.
2. α_f_W = d(ω_f_W)/dt (per-axis gradient over t).
3. Inertial term: M_inertial = I_rod · α_f_W (principal scalar approximation).
4. Linear COM term: r_W = e3_W · r_com (e3_W is 3rd column of R_femur),
	M_lin = r_W × (m_th · a_com_W), where a_com_W = a_f_W + g_W.
5. Damping: M_damp = −0.02 · ω_f_W.
6. Hip moment (world): M_W = M_inertial + M_lin + M_damp.

Output: M_W (T,3).

Notes:

- This ignores pelvis motion and GRFs; it’s for visualization/teaching.
- Sign and axis conventions depend on the chosen JCS.

---

## Baseline subtraction using calibration windows

To zero “static” standing loads in JCS, subtract median moment over selected calibration windows:

1. Build a boolean mask over t covering calibration windows in seconds.
2. base = median(M[mask], axis=0) per component.
3. M ← M − base.
Record base per-side for debugging.

---

## Gait cycle averaging

Contacts: rising edges of stance mask. Exclude the first and last 3 cycles.

Cycle selection: for each successive pair (s,e) of contacts,

- Require e−s ≥ 3 samples.
- Duration dur = t[e] − t[s] in [0.4, 2.0] s.
- Resample signal segment to n=101 phase points with linear interpolation.

Aggregate mean and standard deviation across selected cycles per side and per component (Mx, My, Mz, |M|).

---

## Summary of public math “contracts”

- quats_to_R_batch(quats: (T,4 [w,x,y,z])) -> (T,3,3) R_WS.
- world_vec(R_WS: (T,3,3), v_S: (T,3)) -> (T,3) v_W = R_WS @ v_S.
- yaw_from_R(R_WS: (T,3,3)) -> yaw: (T,) radians.
- apply_yaw_align(R_WS: (T,3,3), yaw_delta: scalar or (T,)) -> (T,3,3) aligned.
- detect_still(t, gyro, freeacc) -> still_mask (T,), Fs, config.
- pick_calibration_windows(still_mask, t, Fs) -> list of windows in seconds.
- calibrate_bias(gyro, freeacc, windows) -> gyro_bias, accel_bias (3,).
- stance_detection(R_tibia, gyro_S, acc_free_S) -> stance mask (T,).
- inverse_dynamics(t, R_femur, ω_S, a_S, height, mass) -> hip moment M_W (T,3).
- hip_jcs_from_R(R_pelvis, R_femur) -> R_WJ (T,3,3) [currently R_femur].
- resolve_in_jcs(M_W, R_WJ) -> M_J = R_WJ^T · M_W.
- contacts_from_stance(stance) -> indices of rising edges.
- cycle_mean_sd(t, sig, contacts, n=101, drop_first=3, drop_last=3, min_dur_s=0.4, max_dur_s=2.0) -> mean (n,), sd (n,), count_used, count_total.

---

## Default thresholds and constants

- Gravity g = [0,0,−9.80665] m/s²
- Still detection: thr_w = 0.08 rad/s, thr_a = 0.30 m/s², smooth_s = 0.7 s
- Calibration windows: prefer ≥1.0 s; fallback ≥0.5 s; fixed ≈0.75 s at edges
- Stance detection: thr_w = 2.0 rad/s, thr_a = 0.8 m/s², hyst = 8 samples
- Anthropometrics: l_th = clamp(0.245·H, 0.2..0.5) m; m_th = 0.10·M kg; I_rod = m_th·l_th²/3; r_com = 0.5·l_th
- Damping coefficient c = 0.02 (dimensionless multiplier on ω)
- Cycle averaging: n = 101 samples, drop first/last 3 cycles, min/max duration 0.4/2.0 s

---

## Implementation notes

- The world frame is implicitly z-up in use (gravity negative z). If you change world conventions, update gravity and yaw extraction accordingly.
- XSENS “freeacc” is assumed to be gravity-removed in sensor frame; adding g back reconstructs COM acceleration approximately.
- Pelvis yaw is used only to harmonize femur/tibia yaw (no roll/pitch alignment).
- Numeric differentiation uses np.gradient(t) per-axis. Smooth or regularize if needed for noisier data.

---

## Minimal rebuild order

1) IO and kinematic extraction (outside this spec)
2) Calibration windows + bias correction
3) Quaternion → rotation; yaw and alignment
4) Resample to shared timebase (SLERP and linear)
5) Stance detection (per shank)
6) Inverse dynamics hip moment in world
7) JCS resolution and baseline subtraction
8) Contacts, cycle averaging, and summary stats

This spec is sufficient to reproduce the math in a clean module without carrying over server/UI code.
