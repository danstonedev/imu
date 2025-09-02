from __future__ import annotations
import numpy as np
from numpy.testing import assert_allclose
from core.math.baseline import BaselineConfig, apply_baseline_correction


def test_no_hp_by_default_preserves_dc():
    fs = 100.0
    t = np.arange(0, 10, 1/fs)
    # DC + 1 Hz sine on Y, Z; X zero
    x = np.zeros_like(t)
    y = np.deg2rad(10) + np.deg2rad(5)*np.sin(2*np.pi*1*t)
    z = np.deg2rad(-5) + np.deg2rad(2)*np.sin(2*np.pi*1.2*t)
    A = np.stack([x, y, z], axis=1)
    cfg = BaselineConfig(fs_hz=fs, use_yaw_share=False, highpass_fc_hz=None, allow_stack=False)
    out = apply_baseline_correction(t, A, stride_indices=None, cfg=cfg)
    # DC preserved within ~1 deg after wrap
    y_mean = np.arctan2(np.sin(out[:,1]).mean(), np.cos(out[:,1]).mean())
    z_mean = np.arctan2(np.sin(out[:,2]).mean(), np.cos(out[:,2]).mean())
    assert abs(np.rad2deg(y_mean) - 10) < 1.0
    assert abs(np.rad2deg(z_mean) - (-5)) < 1.0


def test_skip_short_strides():
    fs = 100.0
    t = np.arange(0, 2, 1/fs)
    A = np.zeros((t.size,3))
    strides = [(10, 20), (30, 35)]  # second is too short
    cfg = BaselineConfig(fs_hz=fs, use_yaw_share=False, highpass_fc_hz=None, min_stride_samples=20)
    out = apply_baseline_correction(t, A, stride_indices=strides, cfg=cfg)
    # shape preserved and no errors
    assert out.shape == A.shape
