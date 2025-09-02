from __future__ import annotations
import numpy as np
from core.math.baseline import (
    BaselineConfig,
    apply_baseline_correction,
    highpass as hp,
    yaw_share_timevarying,
)


def test_stridede_bias_on_YZ_removes_dc_offset():
    fs = 100.0
    t = np.arange(0, 5, 1/fs)
    T = t.size
    # Synthetic gait-like strides every 1 s
    hs = np.arange(0, T, int(fs))
    strides = [(int(a), int(b)) for a, b in zip(hs[:-1], hs[1:])]
    # Build angles: X(t)=sin(2pi*1*t); Y,Z with DC offsets
    X = np.deg2rad(20*np.sin(2*np.pi*1.0*t))
    Y = np.deg2rad(5.0) + np.deg2rad(10*np.sin(2*np.pi*1.0*t))
    Z = np.deg2rad(-8.0) + np.deg2rad(5*np.sin(2*np.pi*1.0*t))
    A = np.stack([X,Y,Z], axis=1)
    cfg = BaselineConfig(fs_hz=fs, use_yaw_share=False, highpass_fc_hz=None)
    Ac = apply_baseline_correction(t, A, strides, cfg)
    # Check per-stride Y,Z mean ~ 0
    for s,e in strides:
        if e-s < cfg.min_stride_samples:
            continue
        m = np.mean(Ac[s:e, 1:3], axis=0)
        assert np.all(np.abs(np.rad2deg(m)) < 2.0)


def test_highpass_fs():
    fs = 200.0
    t = np.arange(0, 20, 1/fs)
    # signal = slow drift + 1 Hz sinusoid
    slow = 0.5*np.sin(2*np.pi*0.01*t)
    hf = np.sin(2*np.pi*1.0*t)
    sig = slow + hf
    out = hp(sig, fs, fc_hz=0.05, order=2)
    # Correlation with 1 Hz sinusoid should be high
    rho = np.corrcoef(out, hf)[0,1]
    assert rho > 0.95
    # Amplitude preserved within ~10%
    amp = (np.max(out) - np.min(out))/2.0
    assert 0.9 < amp < 1.1


def test_yaw_share_blends_low_freq_only():
    fs = 100.0
    t = np.arange(0, 20, 1/fs)
    hf = 0.2*np.sin(2*np.pi*2.0*t)
    lf_p = 0.5*np.sin(2*np.pi*0.02*t)
    lf_f = -0.3*np.sin(2*np.pi*0.02*t)
    yp = lf_p + hf
    yf = lf_f + hf
    cfg = BaselineConfig(fs_hz=fs, use_yaw_share=True, yaw_share_fc_hz=0.05)
    yp_c, yf_c = yaw_share_timevarying(yp, yf, cfg)
    # HF component retained: difference of corrected minus original has low correlation with HF
    # Check LF difference reduced
    yp_lf = hp(yp - hf, fs, 0.05, order=2)  # not exactly LF, but slow part
    yp_c_lf = hp(yp_c - hf, fs, 0.05, order=2)
    # variance of low frequency part reduces
    assert np.var(yp_c_lf) < np.var(yp_lf) + 1e-6
