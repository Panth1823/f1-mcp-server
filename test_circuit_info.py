import requests
import json
import sys

# Base URL of the server - change to localhost for local testing
# BASE_URL = "http://localhost:8000"  # For local testing
BASE_URL = "https://f1-mcp-server.onrender.com"  # For deployed server

def test_endpoint(base_url, endpoint, params=None):
    """Test an endpoint and print the result"""
    url = f"{base_url}{endpoint}"
    
    try:
        response = requests.get(url, params=params)
        print(f"\nTesting {endpoint} with params {params}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                response_text = json.dumps(data, indent=2)
                print("Response:", response_text)
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
    # Test circuit info with various circuit identifiers
    circuit_tests = [
        {"circuit_id": "bahrain"},
        {"circuit_id": "china"},
        {"circuit_id": "japan"},
        {"circuit_id": "spa"},
        {"circuit_id": "monza"},
        {"circuit_id": "silverstone"},
        {"circuit_id": "monaco"},
        {"circuit_id": "cota"}  # Circuit of the Americas
    ]
    
    results = []
    for params in circuit_tests:
        try:
            success, error_msg = test_endpoint(BASE_URL, "/mcp/function/get_circuit_info", params)
            results.append((params["circuit_id"], success, error_msg))
        except Exception as e:
            print(f"Failed to test circuit {params['circuit_id']}: {str(e)}")
            results.append((params["circuit_id"], False, str(e)))
    
    # Print summary
    print("\n=== Test Summary ===")
    success_count = 0
    for circuit_id, success, error_msg in results:
        status = "✅ PASS" if success else f"❌ FAIL: {error_msg}"
        if success:
            success_count += 1
        print(f"{status}: {circuit_id}")
    
    print(f"\n{success_count}/{len(results)} circuits successfully retrieved ({success_count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest execution failed: {str(e)}")
        sys.exit(1) 