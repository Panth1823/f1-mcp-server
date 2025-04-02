import logging
import os
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
os.environ.setdefault('FASTF1_CACHE_DIR', os.getenv('FASTF1_CACHE_DIR', 'fastf1_cache'))

# Create cache directory if it doesn't exist
os.makedirs(os.environ['FASTF1_CACHE_DIR'], exist_ok=True)

# Import middleware
from f1_mcp_server.middleware.cache import F1CacheMiddleware
from f1_mcp_server.middleware.rate_limiter import RateLimitMiddleware
from f1_mcp_server.middleware.error_handler import ErrorHandlerMiddleware

# Create FastAPI app
app = FastAPI(
    title="Formula 1 MCP Server",
    description="A Formula 1 Machine Communication Protocol Server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(F1CacheMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    rate_limit=int(os.getenv("RATE_LIMIT", "5000")),  # Increased from 100 to 5000
    interval=int(os.getenv("RATE_LIMIT_INTERVAL", "60"))
)

# Define MCP function schemas
class F1Function:
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters

# Define available MCP functions
F1_MCP_FUNCTIONS = [
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
    ),
    F1Function(
        name="GET_TESTING_SESSION",
        description="Get information about F1 testing sessions",
        parameters={
            "year": {"type": "integer", "description": "Championship year"},
            "test_number": {"type": "integer", "description": "Number of the testing event (usually 1 or 2)"},
            "session_number": {"type": "integer", "description": "Number of the session within the testing event (usually 1-3)"}
        }
    ),
    F1Function(
        name="GET_TESTING_EVENT",
        description="Get information about an F1 testing event",
        parameters={
            "year": {"type": "integer", "description": "Championship year"},
            "test_number": {"type": "integer", "description": "Number of the testing event (usually 1 or 2)"}
        }
    ),
    F1Function(
        name="GET_EVENTS_REMAINING",
        description="Get information about remaining events in the season",
        parameters={
            "include_testing": {"type": "boolean", "description": "Whether to include testing sessions", "default": True}
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
    get_circuit_info,
    get_testing_session,
    get_testing_event,
    get_events_remaining
)

# Request body models
class CompareDriversRequest(BaseModel):
    year: int
    event: str
    drivers: List[str]

# Function endpoints
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

@app.get("/mcp/function/get_testing_session")
async def handle_get_testing_session(year: int, test_number: int, session_number: int):
    return await get_testing_session(year, test_number, session_number)

@app.get("/mcp/function/get_testing_event")
async def handle_get_testing_event(year: int, test_number: int):
    return await get_testing_event(year, test_number)

@app.get("/mcp/function/get_events_remaining")
async def handle_get_events_remaining(include_testing: bool = True):
    return await get_events_remaining(include_testing)