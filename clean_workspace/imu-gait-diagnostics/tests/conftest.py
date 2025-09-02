# Ensure the diagnostics package is importable when running tests from repo root.
import os
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent  # imu-gait-diagnostics/
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

# Auto-populate IMU_PIPELINE_KWARGS from tools/kwargs.json if not provided
if not os.environ.get("IMU_PIPELINE_KWARGS"):
    kwargs_path = _pkg_dir / "tools" / "kwargs.json"
    if kwargs_path.exists():
        os.environ["IMU_PIPELINE_KWARGS"] = kwargs_path.read_text(encoding="utf-8")
