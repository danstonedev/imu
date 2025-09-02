
# IMU Gait Diagnostics (Tests + CLI)

This package gives you a **turn‑key A/B test** to catch regressions in gait‑cycle detection
after adding ROM processing. It runs your pipeline twice on the *same* input (ROM OFF vs ON),
then checks heel‑strike (HS) and toe‑off (TO) invariance within ±10 ms. It also prints
detector metadata (frame, units, filters, timebase) if your pipeline exposes it.

## Files
- `tests/test_gait_events_invariance.py` – pytest that fails on event drift or count mismatch.
- `diagnostics/utils.py` – helpers to import your pipeline, run twice, compare outputs.
- `tools/diagnose_gait_detector.py` – CLI to print a report and exit non‑zero on failure.

## Quick start
1) Copy `imu-gait-diagnostics/` into your repo root (so you have `tests/` and `tools/`).
2) Tell the tests how to call your pipeline by setting an env var with kwargs, e.g.:
   ```bash
   export IMU_PIPELINE_KWARGS='{"path_pelvis":"data/pelvis.csv","path_lf":"data/lf.csv","path_rf":"data/rf.csv","path_lt":"data/lt.csv","path_rt":"data/rt.csv"}'
   ```
   Replace keys/paths with whatever your entry function expects.
3) (If needed) Edit `diagnostics/utils.py:try_import_pipeline()` to point to your entry function.
4) Run the tests:
   ```bash
   pytest -q
   ```

## ROM toggle
- If your entry function accepts a boolean argument like `rom_enabled` / `compute_rom` / `enable_rom`,
  the test will pass it directly.
- Otherwise, it sets the env var `IMU_ROM_ENABLED` to `'0'` then `'1'` for the two runs. Wire this in
  your pipeline to gate ROM processing.

## Expected outputs
The tests expect your pipeline return value to include:
- `events`: dict with `HS_L`, `TO_L`, `HS_R`, `TO_R` (timestamps in seconds preferred).
- `meta.detector_meta[L|R]` (optional): `signal`, `frame`, `units`, `bandpass`, `timebase`.
- `meta.acceleration.headers_freeacc` (optional): whether headers indicate FreeAcc.

## CLI usage
```bash
python tools/diagnose_gait_detector.py --kwargs "$IMU_PIPELINE_KWARGS" --tol_ms 10
echo $?  # 0 = pass, 1 = fail
```

## When it fails
- The report will show count mismatches or max per‑event deltas > tolerance.
- Compare meta OFF vs ON: if `frame`, `units`, `timebase`, or `bandpass` changed, fix that first.
- Freeze detector inputs (`.copy()`), detect **before** LR mirroring/ROM/world rotations,
  ensure body‑frame gyro in **rad/s**, femur timebase, and single gravity subtraction (or FreeAcc pass‑through).
