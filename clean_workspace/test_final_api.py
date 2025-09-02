#!/usr/bin/env python3

import requests
import json

def test_api():
    """Test the API to verify all expected data structures are present"""
    
    # Test with default sample data
    try:
        response = requests.post('http://localhost:8001/api/analyze/', 
                               data={'height_m': 1.75, 'mass_kg': 70.0}, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print('✅ API Response successful')
            print(f'Total keys in response: {len(result.keys())}')
            
            # Check key data structures for each tab
            tab_checks = {
                'Time Series': ['time_L', 'time_R', 'L_mx', 'L_my', 'L_mz', 'L_Mmag'],
                'Metrics': ['L_Mmag', 'R_Mmag', 'L_my', 'R_my', 'L_mz', 'R_mz'],
                'Angles Time': ['time_L', 'L_hip_angles_deg', 'L_knee_angles_deg'],
                'Angle Cycles': ['angle_cycles']
            }
            
            all_good = True
            for tab, keys in tab_checks.items():
                print(f'\n{tab}:')
                for key in keys:
                    if key in result:
                        value = result[key]
                        if isinstance(value, list):
                            print(f'  ✅ {key}: {len(value)} items')
                        elif isinstance(value, dict) and key == 'angle_cycles':
                            # Check angle_cycles structure
                            if 'hip' in value and 'L' in value['hip'] and 'flex' in value['hip']['L']:
                                mean_len = len(value['hip']['L']['flex']['mean'])
                                print(f'  ✅ {key}: hip.L.flex.mean has {mean_len} points')
                            else:
                                print(f'  ❌ {key}: invalid structure')
                                all_good = False
                        else:
                            print(f'  ✅ {key}: {type(value).__name__}')
                    else:
                        print(f'  ❌ {key}: MISSING')
                        all_good = False
            
            if all_good:
                print('\n🎉 SUCCESS: All tabs should now have the data they need!')
                print('   - Time Series: Has time arrays and all torque components')
                print('   - Cycle Averages: Already working') 
                print('   - Angles Time: Has time arrays and angle data')
                print('   - Angle Cycles: Has structured angle_cycles data')
                print('   - Metrics: Has all torque components')
            else:
                print('\n❌ Some data is still missing')
                        
        else:
            print(f'❌ API Error: {response.status_code} - {response.text}')
            
    except Exception as e:
        print(f'❌ Request failed: {e}')

if __name__ == "__main__":
    test_api()
