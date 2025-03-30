import pytest
from jsonschema import validate
import json

pytestmark = pytest.mark.asyncio

async def test_telemetry_schema(session, base_url, test_data, load_schema):
    """Test that telemetry data conforms to the defined schema"""
    url = f"{base_url}/get_telemetry"
    params = {
        "year": test_data["year"],
        "event": test_data["event"],
        "driver": test_data["driver"],
        "lap": test_data["lap"]
    }
    
    async with session.get(url, params=params) as response:
        assert response.status == 200
        data = await response.json()
        
        # Validate against schema
        schema = load_schema("telemetry")
        validate(instance=data, schema=schema)

async def test_telemetry_data_sanity(session, base_url, test_data):
    """Test that telemetry data values are within reasonable ranges"""
    url = f"{base_url}/get_telemetry"
    params = {
        "year": test_data["year"],
        "event": test_data["event"],
        "driver": test_data["driver"],
        "lap": test_data["lap"]
    }
    
    async with session.get(url, params=params) as response:
        assert response.status == 200
        data = await response.json()
        
        for entry in data["telemetry"]:
            # Speed checks
            assert 0 <= entry["speed"] <= 400, f"Speed out of range: {entry['speed']}"
            
            # Throttle checks
            assert 0 <= entry["throttle"] <= 100, f"Throttle out of range: {entry['throttle']}"
            
            # Gear checks
            assert -1 <= entry["gear"] <= 8, f"Gear out of range: {entry['gear']}"
            
            # RPM checks
            assert 0 <= entry["rpm"] <= 15000, f"RPM out of range: {entry['rpm']}"
            
            # DRS checks
            assert entry["drs"] in [0, 1], f"Invalid DRS value: {entry['drs']}"

async def test_telemetry_sequence(session, base_url, test_data):
    """Test that telemetry data points form a valid sequence"""
    url = f"{base_url}/get_telemetry"
    params = {
        "year": test_data["year"],
        "event": test_data["event"],
        "driver": test_data["driver"],
        "lap": test_data["lap"]
    }
    
    async with session.get(url, params=params) as response:
        assert response.status == 200
        data = await response.json()
        
        # Check for minimum number of data points
        assert len(data["telemetry"]) > 50, "Too few telemetry data points"
        
        # Check for sequential speed changes (no impossible jumps)
        speeds = [entry["speed"] for entry in data["telemetry"]]
        for i in range(1, len(speeds)):
            speed_diff = abs(speeds[i] - speeds[i-1])
            assert speed_diff < 50, f"Impossible speed change detected: {speed_diff} km/h" 