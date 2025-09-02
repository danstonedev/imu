from __future__ import annotations
import numpy as np

__all__ = ["enforce_lr_conventions"]

def enforce_lr_conventions(M_L: np.ndarray, M_R: np.ndarray,
                           anglesL: dict, anglesR: dict):
    """
    Enforce a single source of truth for L/R sign conventions.

        Rules:
        - Moments: mirror Right side in sagittal/transverse → flip X and Z components of moments.
            Also make flexion-positive for BOTH sides → flip Mx for Left and Right.
        - Angles (hip, knee): enforce a comparable L/R convention for plotting/averaging:
            • Flexion: positive for both sides → flip Right flex component only.
            • Adduction: already defined positive toward midline via JCS side handling → no flip.
            • Internal rotation: positive for both sides → flip Right rotation component only.

    Inputs:
      M_L, M_R: (T,>=3) hip moments in JCS for Left and Right.
      anglesL, anglesR: dicts with keys 'hip' and 'knee' mapping to (T,3) deg arrays.

    Returns: (M_L_adj, M_R_adj, anglesL_adj, anglesR_adj)
    """
    ML = np.asarray(M_L).copy()
    MR = np.asarray(M_R).copy()

    # Mirror Right X and Z to compare to Left
    if MR.ndim == 2 and MR.shape[1] >= 3:
        MR[:, 0] *= -1.0
        MR[:, 2] *= -1.0

    # Flexion-positive for both sides (Mx)
    if ML.ndim == 2 and ML.shape[1] >= 1:
        ML[:, 0] *= -1.0
    if MR.ndim == 2 and MR.shape[1] >= 1:
        MR[:, 0] *= -1.0

    # Angles: flip Right flex (make flexion positive across sides) and Right rotation to align L/R
    def _as_arr(A):
        try:
            if A is None:
                return None
            return np.asarray(A).copy()
        except Exception:
            return None

    outL = dict(anglesL or {})
    outR = dict(anglesR or {})
    for k in ("hip", "knee"):
        if k in outL:
            AL = _as_arr(outL.get(k))
            if isinstance(AL, np.ndarray) and AL.ndim == 2 and AL.shape[1] >= 3:
                outL[k] = AL  # Left unchanged
        if k in outR:
            AR = _as_arr(outR.get(k))
            if isinstance(AR, np.ndarray) and AR.ndim == 2 and AR.shape[1] >= 3:
                AR[:, 0] *= -1.0  # Right flex
                AR[:, 2] *= -1.0  # Right rotation
                outR[k] = AR

    return ML, MR, outL, outR
