import pytest
from jsonschema import validate
import json

pytestmark = pytest.mark.asyncio

async def test_session_results_schema(session, base_url, test_data, load_schema):
    """Test that session results conform to the defined schema"""
    url = f"{base_url}/get_session_results"
    params = {
        "year": test_data["year"],
        "event": test_data["event"],
        "session": "Race"
    }
    
    async with session.get(url, params=params) as response:
        assert response.status == 200
        data = await response.json()
        
        # Validate against schema
        schema = load_schema("session_results")
        validate(instance=data, schema=schema)
        
        # Additional assertions
        assert len(data["results"]) > 0, "No results returned"
        assert all(isinstance(r["position"], int) for r in data["results"]), "Invalid position format"
        assert all(r["driver_code"].isupper() and len(r["driver_code"]) == 3 for r in data["results"]), "Invalid driver code"

async def test_session_results_completeness(session, base_url, test_data):
    """Test that session results contain all expected data"""
    url = f"{base_url}/get_session_results"
    params = {
        "year": test_data["year"],
        "event": test_data["event"],
        "session": "Race"
    }
    
    async with session.get(url, params=params) as response:
        assert response.status == 200
        data = await response.json()
        
        # Check for complete driver field set
        first_result = data["results"][0]
        required_fields = {
            "position", "driver_number", "driver_code", "driver_name",
            "team", "grid_position", "status", "points", "time",
            "fastest_lap", "fastest_lap_time", "gap_to_leader"
        }
        
        assert all(field in first_result for field in required_fields), "Missing required fields"
        
        # Verify reasonable values
        assert 1 <= len(data["results"]) <= 20, "Invalid number of results"
        assert all(r["position"] > 0 for r in data["results"]), "Invalid position values"
        assert all(float(r["points"]) >= 0 if r["points"] is not None else True for r in data["results"]), "Invalid points"

async def test_session_results_order(session, base_url, test_data):
    """Test that results are properly ordered by position"""
    url = f"{base_url}/get_session_results"
    params = {
        "year": test_data["year"],
        "event": test_data["event"],
        "session": "Race"
    }
    
    async with session.get(url, params=params) as response:
        assert response.status == 200
        data = await response.json()
        
        positions = [r["position"] for r in data["results"]]
        assert positions == sorted(positions), "Results not ordered by position" 