from pyodide.ffi import to_js
import numpy as np

def to_js_payload(**kwargs):
    """
    Convert NumPy values to JS-friendly objects for Pyodide (no JSON).
      - np.ndarray -> TypedArray (Float32Array / Int32Array, etc.). Flatten if ndim > 1.
      - np.floating / np.integer -> native Python float/int.
      - Other JSON-safe Python types are returned as-is.
    Usage (in your pyodide code cell):
        from py.js_bridge import to_js_payload
        out = to_js_payload(q=q, a=a, t=t, hipL=hip_L, hipR=hip_R)
        out  # Return the dict directly (do NOT json.dumps)
    """
    out = {}
    for k, v in kwargs.items():
        if v is None:
            continue

        # NumPy arrays -> TypedArray
        if isinstance(v, np.ndarray):
            # ensure float arrays are float32 (smaller/faster in JS & GPU)
            if v.dtype.kind == "f" and v.dtype != np.float32:
                v = v.astype(np.float32, copy=False)
            # flatten >1D for transfer (you can reshape in JS if you like)
            arr = v.ravel() if v.ndim > 1 else v
            out[k] = to_js(arr)
            continue

        # NumPy scalars -> Python native
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
            continue

        # Passthrough for Python scalars/strings/lists/dicts/bools
        out[k] = v
    return out
