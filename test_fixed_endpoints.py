import requests
import json
import sys

# Base URL of the deployed server - change to localhost for local testing
# BASE_URL = "http://localhost:8000"  # For local testing
BASE_URL = "https://f1-mcp-server.onrender.com"  # For deployed server

def test_endpoint(base_url, endpoint, params=None, method="GET", json_data=None):
    """Test an endpoint and print the result"""
    url = f"{base_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params)
            print(f"\nTesting {endpoint}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    response_text = json.dumps(data, indent=2)
                    print("Response:", response_text[:500] + "..." if len(response_text) > 500 else response_text)
                    return True, None
                except Exception as e:
                    text = response.text
                    print(f"Error parsing response: {str(e)}")
                    print(f"Raw response: {text[:500]}...")
                    return False, f"Parsing error: {str(e)}"
            else:
                error_text = response.text
                print("Error:", error_text[:500] + "..." if len(error_text) > 500 else error_text)
                return False, f"Status {response.status_code}"
        elif method == "POST":
            response = requests.post(url, json=json_data)
            print(f"\nTesting {endpoint}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    response_text = json.dumps(data, indent=2)
                    print("Response:", response_text[:500] + "..." if len(response_text) > 500 else response_text)
                    return True, None
                except Exception as e:
                    text = response.text
                    print(f"Error parsing response: {str(e)}")
                    print(f"Raw response: {text[:500]}...")
                    return False, f"Parsing error: {str(e)}"
            else:
                error_text = response.text
                print("Error:", error_text[:500] + "..." if len(error_text) > 500 else error_text)
                return False, f"Status {response.status_code}"
    except Exception as e:
        print(f"Exception testing {endpoint}: {str(e)}")
        return False, f"Connection error: {str(e)}"

def run_tests():
    # Test the previously failing endpoints
    failing_tests = [
        # Test with proper circuit mapping now
        ("/mcp/function/get_circuit_info", {"circuit_id": "bahrain"}, "GET", None),
        
        # Test with fixed driver standings
        ("/mcp/function/get_driver_standings", {"year": 2023}, "GET", None),
        
        # Test with streaming response for live timing
        ("/mcp/function/get_live_timing", {"session_id": "latest"}, "GET", None),
    ]
    
    results = []
    for endpoint, params, method, json_data in failing_tests:
        try:
            success, error_msg = test_endpoint(BASE_URL, endpoint, params, method, json_data)
            results.append((endpoint, success, error_msg))
        except Exception as e:
            print(f"Failed to test {endpoint}: {str(e)}")
            results.append((endpoint, False, str(e)))
    
    # Print summary
    print("\n=== Test Summary ===")
    success_count = 0
    for endpoint, success, error_msg in results:
        status = "✅ PASS" if success else f"❌ FAIL: {error_msg}"
        if success:
            success_count += 1
        print(f"{status}: {endpoint}")
    
    print(f"\n{success_count}/{len(results)} endpoints working successfully ({success_count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest execution failed: {str(e)}")
        sys.exit(1) 