from __future__ import annotations
from pathlib import Path
import numpy as np
from core.pipeline.pipeline import run_pipeline_clean

root = Path(__file__).resolve().parents[1]

def pick(pattern: str) -> str:
    matches = sorted(root.joinpath('sample data').glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return str(matches[0])

def test_smoke_run_pipeline_clean():
    paths = {
        'pelvis': pick('DEMO6_0_*.csv'),
        'lfemur': pick('DEMO6_1_*.csv'),
        'rfemur': pick('DEMO6_2_*.csv'),
        'ltibia': pick('DEMO6_3_*.csv'),
        'rtibia': pick('DEMO6_4_*.csv'),
    }
    out = run_pipeline_clean(paths, height_m=1.70, mass_kg=75.0, options={'do_cal': True, 'yaw_align': True})
    for k in ['L_mx','R_mx','stance_L','stance_R','left_csv','right_csv','cycles']:
        assert k in out
    assert out['L_mx'].ndim == 1 and out['R_mx'].ndim == 1
    assert out['stance_L'].dtype == np.uint8
    cycles = out['cycles']
    for side in ['L','R']:
        assert side in cycles
        assert 'Mx' in cycles[side] and 'mean' in cycles[side]['Mx'] and 'sd' in cycles[side]['Mx']
        assert len(cycles[side]['Mx']['mean']) == 101
