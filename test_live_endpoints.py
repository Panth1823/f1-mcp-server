import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Base URL of the deployed server
BASE_URL = "https://f1-mcp-server.onrender.com"

async def test_endpoint(session, base_url, endpoint, params=None, method="GET", json_data=None):
    """Test an endpoint and print the result"""
    url = f"{base_url}{endpoint}"
    if params and method == "GET":
        query_params = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query_params}"
    
    try:
        if method == "GET":
            async with session.get(url) as response:
                print(f"\nTesting {endpoint}")
                print(f"Status: {response.status}")
                if response.status == 200:
                    try:
                        data = await response.json()
                        print("Response:", json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data, indent=2)) > 500 else json.dumps(data, indent=2))
                        return True, None
                    except Exception as e:
                        text = await response.text()
                        print(f"Error parsing response: {str(e)}")
                        print(f"Raw response: {text[:500]}...")
                        return False, f"Parsing error: {str(e)}"
                else:
                    error_text = await response.text()
                    print("Error:", error_text[:500] + "..." if len(error_text) > 500 else error_text)
                    return False, f"Status {response.status}"
        elif method == "POST":
            async with session.post(url, json=json_data) as response:
                print(f"\nTesting {endpoint}")
                print(f"Status: {response.status}")
                if response.status == 200:
                    try:
                        data = await response.json()
                        print("Response:", json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data, indent=2)) > 500 else json.dumps(data, indent=2))
                        return True, None
                    except Exception as e:
                        text = await response.text()
                        print(f"Error parsing response: {str(e)}")
                        print(f"Raw response: {text[:500]}...")
                        return False, f"Parsing error: {str(e)}"
                else:
                    error_text = await response.text()
                    print("Error:", error_text[:500] + "..." if len(error_text) > 500 else error_text)
                    return False, f"Status {response.status}"
    except Exception as e:
        print(f"Exception testing {endpoint}: {str(e)}")
        return False, f"Connection error: {str(e)}"

async def run_tests():
    # Using 2024 with specific completed races and driver numbers
    test_data = {
        "year": 2024,
        "event": "bahrain",  # Bahrain GP is completed
        "driver": "1",       # Max Verstappen's number
        "lap": 1
    }
    
    async with aiohttp.ClientSession() as session:
        # Basic endpoints
        basic_tests = [
            ("/mcp/context", None, "GET", None),
            ("/mcp/functions", None, "GET", None),
        ]
        
        # Function endpoints
        function_tests = [
            ("/mcp/function/get_driver_standings", {"year": test_data["year"]}, "GET", None),
            ("/mcp/function/get_constructor_standings", {"year": test_data["year"]}, "GET", None),
            ("/mcp/function/get_race_calendar", {"year": test_data["year"]}, "GET", None),
            ("/mcp/function/get_event_details", {"year": test_data["year"], "event_identifier": test_data["event"]}, "GET", None),
            ("/mcp/function/get_session_results", {"year": test_data["year"], "event": test_data["event"], "session": "Race"}, "GET", None),
            ("/mcp/function/get_driver_performance", {"year": test_data["year"], "event": test_data["event"], "driver": test_data["driver"]}, "GET", None),
            ("/mcp/function/get_telemetry", {"year": test_data["year"], "event": test_data["event"], "driver": test_data["driver"], "lap": test_data["lap"]}, "GET", None),
            ("/mcp/function/compare_drivers", None, "POST", {"year": test_data["year"], "event": test_data["event"], "drivers": ["1", "11"]}),  # Verstappen and Perez
            ("/mcp/function/get_live_timing", {"session_id": "latest"}, "GET", None),
            ("/mcp/function/get_weather_data", {"year": test_data["year"], "event": test_data["event"]}, "GET", None),
            ("/mcp/function/get_circuit_info", {"circuit_id": test_data["event"]}, "GET", None),
            ("/mcp/function/get_testing_session", {"year": test_data["year"], "test_number": 1, "session_number": 1}, "GET", None),
            ("/mcp/function/get_testing_event", {"year": test_data["year"], "test_number": 1}, "GET", None),
            ("/mcp/function/get_events_remaining", {"include_testing": "true"}, "GET", None),
        ]
        
        results = []
        # Test basic endpoints
        for endpoint, params, method, json_data in basic_tests:
            try:
                success, error_msg = await test_endpoint(session, BASE_URL, endpoint, params, method, json_data)
                results.append((endpoint, success, error_msg))
            except Exception as e:
                print(f"Failed to test {endpoint}: {str(e)}")
                results.append((endpoint, False, str(e)))
        
        # Test function endpoints
        for endpoint, params, method, json_data in function_tests:
            try:
                success, error_msg = await test_endpoint(session, BASE_URL, endpoint, params, method, json_data)
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
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest execution failed: {str(e)}")
        sys.exit(1) 