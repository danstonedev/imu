
#!/usr/bin/env python3
# CLI tool: run pipeline twice (ROM off/on) and print a comparison report.
# Usage:
#   python tools/diagnose_gait_detector.py --kwargs '{"path_pelvis":"...","path_lf":"..."}' --tol_ms 10

from __future__ import annotations
import os, json, argparse, sys
from diagnostics.utils import run_twice_compare, compare_events, extract_detector_meta, pretty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kwargs", required=False, default=os.environ.get("IMU_PIPELINE_KWARGS","{}"),
                    help="JSON kwargs to pass into your pipeline entry function")
    ap.add_argument("--tol_ms", type=float, default=10.0, help="Tolerance in milliseconds for event invariance")
    args = ap.parse_args()
    try:
        kwargs = json.loads(args.kwargs)
    except Exception as e:
        print("Failed to parse --kwargs JSON:", e, file=sys.stderr)
        sys.exit(2)
    out_off, out_on = run_twice_compare(kwargs)
    cmp = compare_events(out_off, out_on, tol_ms=args.tol_ms)
    det_off = extract_detector_meta(out_off)
    det_on  = extract_detector_meta(out_on)
    print("\n=== Gait Detector Invariance Report ===\n")
    print("Tolerance (ms):", args.tol_ms)
    print("\n-- Comparison (counts & max deltas) --")
    print(pretty(cmp))
    print("\n-- Detector meta (ROM OFF) --")
    print(pretty(det_off))
    print("\n-- Detector meta (ROM ON) --")
    print(pretty(det_on))
    okL = cmp["sides"]["L"]["pass"]
    okR = cmp["sides"]["R"]["pass"]
    sys.exit(0 if (okL and okR) else 1)

if __name__ == "__main__":
    main()
