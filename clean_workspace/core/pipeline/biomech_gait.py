"""Deprecated shim that re-exports legacy biomech gait functions.

This module is kept to avoid breaking imports. It forwards calls to
core.pipeline.legacy.biomech_gait and warns once on import.
"""
from __future__ import annotations
import warnings

warnings.warn(
    "core.pipeline.biomech_gait is deprecated; use core.pipeline.unified_gait or core.pipeline.legacy.biomech_gait",
    DeprecationWarning,
    stacklevel=2,
)

from .legacy.biomech_gait import (  # type: ignore F401
    detect_heel_strikes_biomech,
    detect_hs_to_biomech,
    bilateral_gait_cycles,
    bilateral_gait_analysis,
    cycle_mean_sd_biomech,
)

__all__ = [
    "detect_heel_strikes_biomech",
    "detect_hs_to_biomech",
    "bilateral_gait_cycles",
    "bilateral_gait_analysis",
    "cycle_mean_sd_biomech",
]
