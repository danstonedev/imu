from __future__ import annotations
from pathlib import Path
import argparse
from core.pipeline.pipeline import run_pipeline_clean

def pick(root: Path, pattern: str) -> str:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return str(matches[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=float, required=True)
    ap.add_argument('--mass', type=float, required=True)
    ap.add_argument('--data', type=str, default='../sample data')
    args = ap.parse_args()
    root = Path(args.data)
    paths = {
        'pelvis': pick(root, 'DEMO6_0_*.csv'),
        'lfemur': pick(root, 'DEMO6_1_*.csv'),
        'rfemur': pick(root, 'DEMO6_2_*.csv'),
        'ltibia': pick(root, 'DEMO6_3_*.csv'),
        'rtibia': pick(root, 'DEMO6_4_*.csv'),
    }
    out = run_pipeline_clean(paths, height_m=args.height, mass_kg=args.mass, options={'do_cal': True, 'yaw_align': True})
    print('Left CSV rows:', len(out['left_csv'].splitlines()))
    print('Right CSV rows:', len(out['right_csv'].splitlines()))
    print('Cycles L used/total:', out['cycles']['L']['count_used'], '/', out['cycles']['L']['count_total'])
    print('Cycles R used/total:', out['cycles']['R']['count_used'], '/', out['cycles']['R']['count_total'])
    
    # Debug stance detection and toe-off percentages
    print('\n=== STANCE DETECTION DEBUG ===')
    if 'cycles_compare' in out:
        for key, data in out['cycles_compare'].items():
            if 'meta' in data and 'to_percent' in data['meta']:
                to_pct = data['meta']['to_percent']
                if to_pct is not None:
                    print(f'{key}: toe-off at {to_pct:.1f}%')
                else:
                    print(f'{key}: toe-off could not be determined')
    
    # Check if we have unusually low toe-off percentages
    low_to_found = False
    if 'cycles_compare' in out:
        for key, data in out['cycles_compare'].items():
            if 'meta' in data and 'to_percent' in data['meta']:
                to_pct = data['meta']['to_percent']
                if to_pct is not None and to_pct < 30:
                    low_to_found = True
                    print(f'WARNING: {key} has unusually low toe-off at {to_pct:.1f}%')
    
    if low_to_found:
        print('\nPossible issues:')
        print('1. Stance detection thresholds may be too strict')
        print('2. IMU placement or calibration issues')
        print('3. Data quality problems')
        print('4. Non-standard gait pattern')
        
        from core.config.constants import STANCE_THR_W, STANCE_THR_A
        print(f'\nCurrent stance thresholds:')
        print(f'  Gyro: {STANCE_THR_W} rad/s')
        print(f'  Accel: {STANCE_THR_A} m/s²')
        print('Consider adjusting these values if data appears normal')

if __name__ == '__main__':
    main()
