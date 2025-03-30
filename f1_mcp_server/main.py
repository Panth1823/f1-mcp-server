import logging
import os
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
os.environ.setdefault('FASTF1_CACHE_DIR', 'fastf1_cache')

# Create FastAPI app
app = FastAPI()

# Define MCP function schemas
class F1Function:
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters

# Define available MCP functions
F1_MCP_FUNCTIONS = [
    F1Function(
        name="GET_CURRENT_SESSION",
        description="Get information about the current or most recent F1 session",
        parameters={}
    ),
    F1Function(
        name="GET_DRIVER_STANDINGS",
        description="Get current driver championship standings",
        parameters={
            "year": {"type": "integer", "description": "Championship year (defaults to current year)"}
        }
    ),
    F1Function(
        name="GET_CONSTRUCTOR_STANDINGS",
        description="Get current constructor championship standings",
        parameters={
            "year": {"type": "integer", "description": "Championship year (defaults to current year)"}
        }
    ),
    F1Function(
        name="GET_RACE_CALENDAR",
        description="Retrieve F1 season schedule",
        parameters={
            "year": {"type": "integer", "description": "Season year"}
        }
    ),
    F1Function(
        name="GET_EVENT_DETAILS",
        description="Fetch Grand Prix circuit/location data",
        parameters={
            "year": {"type": "integer", "description": "Event year"},
            "event_identifier": {"type": "string", "description": "Event identifier (e.g., 'bahrain', 'monza')"}
        }
    ),
    F1Function(
        name="GET_SESSION_RESULTS",
        description="Race/Qualifying/Practice results",
        parameters={
            "year": {"type": "integer", "description": "Event year"},
            "event": {"type": "string", "description": "Event identifier"},
            "session": {"type": "string", "description": "Session type (Race, Qualifying, Practice1, Practice2, Practice3, Sprint)"}
        }
    ),
    F1Function(
        name="GET_DRIVER_PERFORMANCE",
        description="Lap times/position changes",
        parameters={
            "year": {"type": "integer", "description": "Event year"},
            "event": {"type": "string", "description": "Event identifier"},
            "driver": {"type": "string", "description": "Driver identifier"}
        }
    ),
    F1Function(
        name="GET_TELEMETRY",
        description="Speed/throttle/gear data for specific lap",
        parameters={
            "year": {"type": "integer", "description": "Event year"},
            "event": {"type": "string", "description": "Event identifier"},
            "driver": {"type": "string", "description": "Driver identifier"},
            "lap": {"type": "integer", "description": "Lap number"}
        }
    ),
    F1Function(
        name="COMPARE_DRIVERS",
        description="Head-to-head performance analysis",
        parameters={
            "year": {"type": "integer", "description": "Event year"},
            "event": {"type": "string", "description": "Event identifier"},
            "drivers": {"type": "array", "items": {"type": "string"}, "description": "List of driver identifiers to compare"}
        }
    ),
    F1Function(
        name="GET_LIVE_TIMING",
        description="Real-time session updates (SSE)",
        parameters={
            "session_id": {"type": "string", "description": "Active session identifier"}
        }
    ),
    F1Function(
        name="GET_WEATHER_DATA",
        description="Historical/real-time race weather",
        parameters={
            "year": {"type": "integer", "description": "Event year"},
            "event": {"type": "string", "description": "Event identifier"}
        }
    ),
    F1Function(
        name="GET_CIRCUIT_INFO",
        description="Track layout/specifications",
        parameters={
            "circuit_id": {"type": "string", "description": "Circuit identifier"}
        }
    )
]

# Define MCP context
MCP_CONTEXT = {
    "name": "f1_racing",
    "version": "1.0.0",
    "description": "Formula 1 Racing Data MCP Server",
    "capabilities": {
        "real_time_data": True,
        "historical_data": True,
        "telemetry": True,
        "weather": True,
        "statistics": True
    },
    "data_sources": ["OpenF1", "FastF1"],
    "available_functions": [func.name for func in F1_MCP_FUNCTIONS],
    "supported_years": list(range(1950, datetime.now().year + 1)),
    "supported_data_types": [
        "session_info",
        "race_results",
        "qualifying_results",
        "practice_results",
        "driver_info",
        "team_info",
        "lap_times",
        "telemetry",
        "weather",
        "tire_data",
        "championship_standings"
    ]
}

# MCP Context endpoint
@app.get("/mcp/context")
async def get_mcp_context():
    """Get the MCP context information"""
    return MCP_CONTEXT

# MCP Function definitions endpoint
@app.get("/mcp/functions")
async def get_mcp_functions():
    """Get available MCP functions and their schemas"""
    return {
        "functions": [
            {
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters
            }
            for func in F1_MCP_FUNCTIONS
        ]
    }

# Import function implementations
from f1_mcp_server.core.functions import (
    get_current_session,
    get_driver_standings,
    get_constructor_standings,
    get_race_calendar,
    get_event_details,
    get_session_results,
    get_driver_performance,
    get_telemetry,
    compare_drivers,
    get_live_timing,
    get_weather_data,
    get_circuit_info
)

# Request body models
class CompareDriversRequest(BaseModel):
    year: int
    event: str
    drivers: List[str]

# Function endpoints
@app.get("/mcp/function/get_current_session")
async def handle_get_current_session():
    return await get_current_session()

@app.get("/mcp/function/get_driver_standings")
async def handle_get_driver_standings(year: Optional[int] = None):
    return await get_driver_standings(year)

@app.get("/mcp/function/get_constructor_standings")
async def handle_get_constructor_standings(year: Optional[int] = None):
    return await get_constructor_standings(year)

@app.get("/mcp/function/get_race_calendar")
async def handle_get_race_calendar(year: int):
    return await get_race_calendar(year)

@app.get("/mcp/function/get_event_details")
async def handle_get_event_details(year: int, event_identifier: str):
    return await get_event_details(year, event_identifier)

@app.get("/mcp/function/get_session_results")
async def handle_get_session_results(year: int, event: str, session: str):
    return await get_session_results(year, event, session)

@app.get("/mcp/function/get_driver_performance")
async def handle_get_driver_performance(year: int, event: str, driver: str):
    return await get_driver_performance(year, event, driver)

@app.get("/mcp/function/get_telemetry")
async def handle_get_telemetry(year: int, event: str, driver: str, lap: int):
    return await get_telemetry(year, event, driver, lap)

@app.post("/mcp/function/compare_drivers")
async def handle_compare_drivers(request: CompareDriversRequest):
    return await compare_drivers(request.year, request.event, request.drivers)

@app.get("/mcp/function/get_live_timing")
async def handle_get_live_timing(session_id: str):
    return await get_live_timing(session_id)

@app.get("/mcp/function/get_weather_data")
async def handle_get_weather_data(year: int, event: str):
    return await get_weather_data(year, event)

@app.get("/mcp/function/get_circuit_info")
async def handle_get_circuit_info(circuit_id: str):
    return await get_circuit_info(circuit_id) 