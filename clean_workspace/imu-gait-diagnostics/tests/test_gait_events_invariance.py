
# PyTest: Gait event invariance before/after ROM processing.
# Fails if heel-strike/toe-off shift by more than ±10 ms or counts differ.
# Usage:
#   - Place this under tests/ and run `pytest -q`.
#   - Provide pipeline kwargs via env var IMU_PIPELINE_KWARGS as JSON (paths, config).
#   - Or edit the DEFAULT_KWARGS below.

from __future__ import annotations
import os, json, pytest
from diagnostics.utils import run_twice_compare, compare_events, extract_detector_meta, pretty

DEFAULT_KWARGS = json.loads(os.environ.get("IMU_PIPELINE_KWARGS", "{}"))
HAVE_KWARGS = bool(DEFAULT_KWARGS)
SKIP_REASON = (
    "inputs available"
    if HAVE_KWARGS
    else "Set IMU_PIPELINE_KWARGS or edit tools/kwargs.json to provide pipeline inputs (JSON)."
)

@pytest.mark.skipif(not HAVE_KWARGS, reason=SKIP_REASON)
def test_gait_events_invariance_10ms():
    out_off, out_on = run_twice_compare(DEFAULT_KWARGS)
    cmp = compare_events(out_off, out_on, tol_ms=10.0)
    det_off = extract_detector_meta(out_off)
    det_on  = extract_detector_meta(out_on)
    if not (cmp["sides"]["L"]["pass"] and cmp["sides"]["R"]["pass"]):
        print("\n--- Detector meta (ROM OFF) ---\n", pretty(det_off))
        print("\n--- Detector meta (ROM ON)  ---\n", pretty(det_on))
        print("\n--- Comparison ---\n", pretty(cmp))
    assert cmp["sides"]["L"]["pass"], f"L side events changed beyond tolerance: {pretty(cmp['sides']['L'])}"
    assert cmp["sides"]["R"]["pass"], f"R side events changed beyond tolerance: {pretty(cmp['sides']['R'])}"

@pytest.mark.skipif(not HAVE_KWARGS, reason=SKIP_REASON)
def test_detector_frames_units_consistent():
    out_off, out_on = run_twice_compare(DEFAULT_KWARGS)
    meta_off = extract_detector_meta(out_off)
    meta_on  = extract_detector_meta(out_on)
    for side in ("L","R"):
        off = meta_off.get(side,{}); on = meta_on.get(side,{})
        expected_frame = "body"
        expected_units = "rad/s"
        expected_timebase = "femur"
        if off.get("frame") is not None:
            assert off["frame"] == expected_frame, f"Detector frame incorrect (OFF, {side}): {off['frame']}"
        if on.get("frame") is not None:
            assert on["frame"] == expected_frame, f"Detector frame incorrect (ON, {side}): {on['frame']}"
        if off.get("units") is not None:
            assert off["units"] == expected_units, f"Units incorrect (OFF, {side}): {off['units']}"
        if on.get("units") is not None:
            assert on["units"] == expected_units, f"Units incorrect (ON, {side}): {on['units']}"
        if off.get("timebase") is not None:
            assert off["timebase"] == expected_timebase, f"Timebase incorrect (OFF, {side}): {off['timebase']}"
        if on.get("timebase") is not None:
            assert on["timebase"] == expected_timebase, f"Timebase incorrect (ON, {side}): {on['timebase']}"

@pytest.mark.skipif(not HAVE_KWARGS, reason=SKIP_REASON)
def test_events_counts_reasonable():
    # Sanity: step counts must be reasonable for both runs.
    out_off, out_on = run_twice_compare(DEFAULT_KWARGS)
    for label, out in (("ROM OFF", out_off), ("ROM ON", out_on)):
        events = out.get("events", out)
        HS_L = events.get("HS_L", []); HS_R = events.get("HS_R", [])
        assert len(HS_L) >= 4, f"{label}: too few HS_L ({len(HS_L)})"
        assert len(HS_R) >= 4, f"{label}: too few HS_R ({len(HS_R)})"
