from __future__ import annotations
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
ws = repo_root / 'clean_workspace'
sys.path.insert(0, str(repo_root))

from clean_workspace.core.pipeline.pipeline import run_pipeline_clean  # type: ignore

def pick(pattern: str) -> str:
    matches = sorted((ws / 'sample data').glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No sample file for pattern: {pattern}")
    return str(matches[0])

def main():
    sample_paths = {
        'pelvis': pick('DEMO6_0_*.csv'),
        'lfemur': pick('DEMO6_1_*.csv'),
        'rfemur': pick('DEMO6_2_*.csv'),
        'ltibia': pick('DEMO6_3_*.csv'),
        'rtibia': pick('DEMO6_4_*.csv'),
    }
    files_bytes = {k: Path(v).read_bytes() for k, v in sample_paths.items()}
    options = {'do_cal': True, 'yaw_align': True}
    res = run_pipeline_clean(files_bytes, height_m=1.75, mass_kg=75.0, options=options)

    # Parse angle cycle CSVs and compute ROMs/baselines
    import csv, io
    def parse_cycle(csv_text: str):
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        def col(name: str):
            return [float(r.get(name, 'nan')) for r in rows]
        return {
            'hip_flex': col('hip_flex_mean(deg)'),
            'hip_rot':  col('hip_rot_mean(deg)'),
            'knee_flex': col('knee_flex_mean(deg)'),
            'knee_rot':  col('knee_rot_mean(deg)'),
        }

    Lhip = parse_cycle(res['left_hip_cycle_csv'])
    Rhip = parse_cycle(res['right_hip_cycle_csv'])
    Lknee = parse_cycle(res['left_knee_cycle_csv'])
    Rknee = parse_cycle(res['right_knee_cycle_csv'])

    def rom(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return float('nan')
        return round(max(vals) - min(vals), 3)

    summary = {
        'twist_deg': res.get('meta', {}).get('angles_calibration', {}).get('twist_deg', {}),
        'hip_rot_ROM_deg': {
            'L': rom(Lhip['hip_rot']),
            'R': rom(Rhip['hip_rot']),
        },
        'knee_rot_baseline_deg': {
            'L': round(Lknee['knee_rot'][0], 3) if Lknee['knee_rot'] else None,
            'R': round(Rknee['knee_rot'][0], 3) if Rknee['knee_rot'] else None,
        },
        'hip_flex_ROM_deg': {
            'L': rom(Lhip['hip_flex']),
            'R': rom(Rhip['hip_flex']),
        },
        'knee_flex_ROM_deg': {
            'L': rom(Lknee['knee_flex']),
            'R': rom(Rknee['knee_flex']),
        },
    }
    print(json.dumps(summary, separators=(',',':')))

if __name__ == '__main__':
    main()
