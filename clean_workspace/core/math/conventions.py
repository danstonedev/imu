from __future__ import annotations
import numpy as np

__all__ = ["enforce_lr_conventions"]

def enforce_lr_conventions(M_L: np.ndarray, M_R: np.ndarray,
                           anglesL: dict, anglesR: dict):
    """
    Enforce a single source of truth for L/R sign conventions.

    Rules:
    - Mirror Right side in sagittal/transverse: flip X and Z components of moments.
    - Flexion-positive for BOTH sides: flip Mx for L and R.
    - For angles (hip, knee), flip the flexion component to be positive.

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

    def _flip_flex(A):
        if A is None:
            return None
        B = np.asarray(A).copy()
        if B.ndim == 2 and B.shape[1] >= 1:
            B[:, 0] *= -1.0
        return B

    outL = dict(anglesL or {})
    outR = dict(anglesR or {})
    for k in ("hip", "knee"):
        if k in outL:
            outL[k] = _flip_flex(outL[k])
        if k in outR:
            outR[k] = _flip_flex(outR[k])

    return ML, MR, outL, outR
