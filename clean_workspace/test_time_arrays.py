import requests
import json

# Test the API to check if time arrays are included
try:
    response = requests.post(
        "http://localhost:8001/api/analyze/",
        json={
            "use_sample_data": True,
            "height_m": 1.75,
            "mass_kg": 70
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("Response keys:", list(data.keys()))
        print("Time L length:", len(data.get("time_L", [])))
        print("Time R length:", len(data.get("time_R", [])))
        print("L_mx length:", len(data.get("L_mx", [])))
        print("R_mx length:", len(data.get("R_mx", [])))
        
        # Check if time arrays exist
        if "time_L" in data and "time_R" in data:
            print("✓ Time arrays successfully added to API response!")
        else:
            print("✗ Time arrays not found in API response")
    else:
        print(f"API Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")
