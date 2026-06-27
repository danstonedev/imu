# Salvaged code from `danstonedev/hip-torque`

**Provenance:** These files were copied **verbatim** from the
[`danstonedev/hip-torque`](https://github.com/danstonedev/hip-torque) repository
(`py/` and `js/gpu/` directories, default branch `main`, commit
`bd107c0b5e099fc4cea1976018744fe7fc421670`) on **2026-06-27**, ahead of archiving
that repository. The original repository will be archived (read-only); its full
git history is preserved there.

**Why:** A verified org-wide audit found that `hip-torque` uniquely contains a
3-segment (foot -> shank -> thigh) Newton-Euler inverse-dynamics chain with De
Leva anthropometrics, force-plate / insole center-of-pressure (CoP) handling, and
an IMU-only ground-reaction-force (GRF) / CoP estimator. This richer model is
found **nowhere else** in the `danstonedev` org. The `imu` repo (this repo,
hip-torque's partial successor) currently has only a simpler **femur-only** model
in `clean_workspace/core/math/inverse_dynamics.py`. These files are preserved here
so the unique biomechanics code survives in an active repository.

## Status: REFERENCE CODE ONLY

This is **salvaged reference code**. It is **NOT integrated** into `imu`'s
pipeline and is **not imported by any existing `imu` code**. Nothing under
`clean_workspace/` was modified by this salvage. In particular, `imu`'s existing
femur-only `clean_workspace/core/math/inverse_dynamics.py` is **unchanged**.

The Python modules below import each other by bare module name (e.g.
`from hip_inverse_dynamics import ...`), matching how they ran in hip-torque's
flat `py/` layout under Pyodide. They are kept together here for that reason and
are not wired into `imu`'s package structure.

## Contents

### `py/hip_inverse_dynamics.py`  (the crown jewel)
The unique 3-segment Newton-Euler inverse-dynamics model. Provides:
- De Leva (1996) lower-limb anthropometrics (mass fractions, COM locations,
  radii of gyration) and inertia-tensor construction.
- World-frame inertia transforms.
- GRF / CoP helpers for force plate and insole pressure maps.
- An IMU-only GRF/CoP estimator (teaching/demo, rocker foot model).
- Bottom-up Newton-Euler inverse dynamics for foot -> shank -> thigh yielding
  ankle, knee, and hip joint reaction forces and net moments, in both
  single-frame and vectorized (batched-over-time) forms.

### `py/pages_pipeline.py`
The browser (Pyodide) client-side pipeline built on top of the model above.
Handles XSENS CSV parsing (header auto-detection, quaternion -> rotation matrix),
overlap trimming across sensors, standing calibration, stance detection, a rocker
CoP model, hip-moment inverse dynamics per side, and gait-cycle normalization.
Uses only numpy / pandas plus `hip_inverse_dynamics`.

### `py/js_bridge.py`
Small Pyodide helper to convert NumPy values into JS-friendly typed arrays /
scalars when returning results to the browser. Included as supporting context for
`pages_pipeline.py`.

### `js/gpu/quats.webgpu.js`
WebGPU (WGSL) compute kernels for batched quaternion math: rotating vectors by
quaternions and computing relative (parent^-1 * child) quaternions on the GPU,
with the JS host code to dispatch them and read results back. Reference for
GPU-accelerated IMU orientation processing.

## Notes
- No binary files were present in hip-torque's `js/gpu/` (only the single text
  source `quats.webgpu.js`), so nothing was skipped.
- License: hip-torque is MIT-licensed; `hip_inverse_dynamics.py` carries an
  in-file MIT notice. The original repository's `LICENSE` governs reuse.
