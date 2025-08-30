import numpy as np
from core.pipeline.unified_gait import detect_gait_cycles

def test_low_fs_filter_adjustments_and_params():
    fs = 20.0  # below 25 Hz to trigger low-fs safeguards
    N = 60
    t = np.arange(N, dtype=float) / fs
    # Minimal signals, no actual events required
    accel = np.zeros((N, 3), dtype=float)
    gyro = np.zeros((N, 3), dtype=float)

    res = detect_gait_cycles(
        t_left=t,
        accel_left=accel,
        gyro_left=gyro,
        t_right=t,
        accel_right=accel,
        gyro_right=gyro,
        fs=fs,
    )

    # Should always include sampling_frequency and detector params
    assert isinstance(res, dict)
    assert res.get('sampling_frequency') == fs
    dp = res.get('detector_params', {})
    fa = res.get('filter_adjustments', {})

    # Low-fs adjustments should be present and reason flagged
    assert isinstance(fa, dict) and fa.get('reason') == 'low_fs'
    assert 'vib_band' in fa and isinstance(fa['vib_band'], tuple)
    assert 'gyro_lpf' in fa

    # Params used should reflect the narrowed band and reduced LPF
    vib_band_used = dp.get('vib_band_used')
    gyro_lpf_used = dp.get('gyro_lpf_used')
    assert isinstance(vib_band_used, tuple) and len(vib_band_used) == 2
    assert vib_band_used[0] >= 4.0 and vib_band_used[1] <= 10.0
    assert gyro_lpf_used <= 6.0 and gyro_lpf_used <= fs/4.0 + 1e-6
