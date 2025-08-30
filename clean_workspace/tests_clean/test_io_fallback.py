import numpy as np
from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics

def test_extract_kinematics_with_acc_fallback():
    # Minimal CSV with acc_* columns instead of freeacc_*
    csv = (
        "time_s,quat_w,quat_x,quat_y,quat_z,gyr_x,gyr_y,gyr_z,acc_x,acc_y,acc_z\n"
        "0.00,1,0,0,0,0.0,0.0,0.0,0.1,0.2,9.7\n"
        "0.01,1,0,0,0,0.0,0.0,0.0,0.1,0.2,9.7\n"
        "0.02,1,0,0,0,0.0,0.0,0.0,0.1,0.2,9.7\n"
    )
    df = read_xsens_bytes(csv.encode("utf-8"))
    t, quat, gyro, freeacc = extract_kinematics(df)
    assert t.shape[0] == 3
    assert quat.shape == (3, 4)
    # gyro may come through; allow None fallback
    if gyro is not None:
        assert gyro.shape == (3, 3)
    assert freeacc.shape == (3, 3)
    # Values should match the provided acc columns
    assert np.allclose(freeacc, np.array([[0.1, 0.2, 9.7]] * 3))
