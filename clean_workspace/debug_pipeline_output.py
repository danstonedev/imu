#!/usr/bin/env python3
"""
Debug script to examine pipeline output and identify what data is missing
"""

from pathlib import Path
from core.pipeline.pipeline import run_pipeline_clean
import json

def debug_pipeline():
    """Run the pipeline and examine the output structure."""
    
    # Use sample data paths like the app does
    root_dir = Path(__file__).resolve().parent
    sample_dir = root_dir / "sample data"
    
    def pick(pattern: str) -> str:
        matches = sorted(sample_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No sample file for pattern: {pattern}")
        return str(matches[0])
    
    # Build data dict with sample file paths
    data = {
        "pelvis": pick("DEMO6_0_*.csv"),
        "lfemur": pick("DEMO6_1_*.csv"),
        "rfemur": pick("DEMO6_2_*.csv"),
        "ltibia": pick("DEMO6_3_*.csv"),
        "rtibia": pick("DEMO6_4_*.csv"),
    }
    
    # Standard options
    options = {
        "do_cal": True,
        "yaw_align": True,
        "yaw_share": {"enabled": False},
        "angles_standing_neutral": True,
    }
    
    print("Running pipeline...")
    result = run_pipeline_clean(data, height_m=1.70, mass_kg=70.0, options=options)
    
    print(f"\nPipeline result type: {type(result)}")
    print(f"Pipeline result keys: {list(result.keys())}")
    
    # Check what's in the result
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"\n{key}: (dict with keys: {list(value.keys())})")
        elif hasattr(value, 'shape'):
            print(f"\n{key}: array shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"\n{key}: {type(value).__name__} with {len(value)} items")
        elif isinstance(value, str):
            print(f"\n{key}: string with {len(value)} characters")
        else:
            print(f"\n{key}: {type(value).__name__} = {value}")
    
    # Specifically check for the CSV data and cycles
    if 'left_csv' in result:
        lines = result['left_csv'].split('\n')
        print(f"\nleft_csv: {len(lines)} lines")
        print(f"First few lines:\n{chr(10).join(lines[:5])}")
    
    if 'right_csv' in result:
        lines = result['right_csv'].split('\n')
        print(f"\nright_csv: {len(lines)} lines")
        print(f"First few lines:\n{chr(10).join(lines[:5])}")
    
    if 'cycles' in result:
        cycles = result['cycles']
        print(f"\ncycles structure:")
        for side in ['L', 'R']:
            if side in cycles:
                side_cycles = cycles[side]
                print(f"  {side}: {list(side_cycles.keys())}")
                for component in ['Mx', 'My', 'Mz', 'Mmag']:
                    if component in side_cycles:
                        comp_data = side_cycles[component]
                        if isinstance(comp_data, dict) and 'mean' in comp_data:
                            mean_shape = comp_data['mean'].shape if hasattr(comp_data['mean'], 'shape') else 'no shape'
                            print(f"    {component}: mean shape {mean_shape}")
    
    if 'cycles_compare' in result:
        print(f"\ncycles_compare structure:")
        compare = result['cycles_compare']
        for anchor in ['anchor_L', 'anchor_R']:
            if anchor in compare:
                anchor_data = compare[anchor]
                print(f"  {anchor}: {list(anchor_data.keys())}")
    
    return result

if __name__ == "__main__":
    result = debug_pipeline()
