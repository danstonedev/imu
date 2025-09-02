#!/usr/bin/env python3
"""
Debug script to test the actual API endpoint and see what's being returned
"""

import requests
import json
from pathlib import Path

def test_api_endpoint():
    """Test the actual API endpoint to see what data is returned."""
    
    # Start the server first if not running
    url = "http://localhost:8000/api/analyze/"
    
    # Test with sample data (no files uploaded)
    data = {
        "height_m": 1.70,
        "mass_kg": 70.0
    }
    
    try:
        print("Testing API endpoint...")
        response = requests.post(url, data=data)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response type: {type(result)}")
            print(f"Response keys: {list(result.keys())}")
            
            # Check what's in the response
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"{key}: dict with keys {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"{key}: list with {len(value)} items")
                    if len(value) > 0:
                        print(f"  First item type: {type(value[0])}")
                elif isinstance(value, str):
                    print(f"{key}: string with {len(value)} characters")
                else:
                    print(f"{key}: {type(value)} = {value}")
            
            # Save the full response for inspection
            with open("api_response_debug.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
            print("\\nFull response saved to api_response_debug.json")
            
        else:
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure it's running on localhost:8000")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_api_endpoint()
