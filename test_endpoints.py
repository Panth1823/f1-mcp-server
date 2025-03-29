import asyncio
import aiohttp
import json
from datetime import datetime
import sys

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
                    data = await response.json()
                    print("Response:", json.dumps(data, indent=2)[:500] + "...")  # Show first 500 chars
                    return True
                else:
                    error_text = await response.text()
                    print("Error:", error_text)
                    return False
        elif method == "POST":
            async with session.post(url, json=json_data) as response:
                print(f"\nTesting {endpoint}")
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print("Response:", json.dumps(data, indent=2)[:500] + "...")  # Show first 500 chars
                    return True
                else:
                    error_text = await response.text()
                    print("Error:", error_text)
                    return False
    except Exception as e:
        print(f"Exception testing {endpoint}: {str(e)}")
        return False

async def run_tests():
    base_url = "http://localhost:8000/mcp/function"
    
    # Using 2024 with specific completed races and driver numbers
    test_data = {
        "year": 2024,
        "event": "bahrain",  # Bahrain GP is completed
        "driver": "1",  # Max Verstappen's number
        "lap": 1
    }
    
    async with aiohttp.ClientSession() as session:
        # Test basic endpoints
        tests = [
            # Test race calendar
            ("get_race_calendar", {"year": test_data["year"]}, "GET", None),
            
            # Test event details
            ("get_event_details", {"year": test_data["year"], "event_identifier": test_data["event"]}, "GET", None),
            
            # Test session results
            ("get_session_results", {"year": test_data["year"], "event": test_data["event"], "session": "Race"}, "GET", None),
            
            # Test driver performance
            ("get_driver_performance", {"year": test_data["year"], "event": test_data["event"], "driver": test_data["driver"]}, "GET", None),
            
            # Test telemetry
            ("get_telemetry", {"year": test_data["year"], "event": test_data["event"], "driver": test_data["driver"], "lap": test_data["lap"]}, "GET", None),
            
            # Test driver comparison
            ("compare_drivers", None, "POST", {"year": test_data["year"], "event": test_data["event"], "drivers": ["1", "11"]}),  # Verstappen and Perez
            
            # Test weather data
            ("get_weather_data", {"year": test_data["year"], "event": test_data["event"]}, "GET", None),
            
            # Test circuit info
            ("get_circuit_info", {"circuit_id": test_data["event"]}, "GET", None),
            
            # Test standings
            ("get_driver_standings", {"year": test_data["year"]}, "GET", None),
            ("get_constructor_standings", {"year": test_data["year"]}, "GET", None),
        ]
        
        results = []
        for endpoint, params, method, json_data in tests:
            try:
                success = await test_endpoint(session, base_url, f"/{endpoint}", params, method, json_data)
                results.append((endpoint, success))
            except Exception as e:
                print(f"Failed to test {endpoint}: {str(e)}")
                results.append((endpoint, False))
        
        # Print summary
        print("\n=== Test Summary ===")
        for endpoint, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {endpoint}")

if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest execution failed: {str(e)}")
        sys.exit(1) 