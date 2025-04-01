"""
F1 Data Models and Schemas
"""

from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field

class SessionData(BaseModel):
    """F1 session data model"""
    session_type: str
    session_name: str
    track_name: str
    session_date: Any
    session_status: str
    weather: Dict[str, Any]
    track_temp: Optional[float]
    air_temp: Optional[float]

class DriverData(BaseModel):
    """F1 driver data model"""
    driver_number: str
    driver_name: str
    team: str
    position: Optional[int]
    current_lap: int
    last_lap_time: Optional[float]
    best_lap_time: Optional[float]
    sector_times: Dict[str, Optional[float]]
    speed: Optional[float]
    tire_compound: Optional[str]
    pit_stops: int
    gap_to_leader: Optional[float]

class F1Data(BaseModel):
    """Main F1 data model"""
    session: SessionData
    drivers: Dict[str, DriverData]
    lap_count: int
    source_info: Dict[str, str]

class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number, starting from 1")
    page_size: int = Field(default=100, ge=1, le=1000, description="Number of items per page")

class PaginatedResponse(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: List[Any]

class TelemetryPoint(BaseModel):
    time: str
    speed: Optional[float]
    throttle: Optional[float]
    brake: Optional[bool]
    gear: Optional[int]
    rpm: Optional[float]
    drs: Optional[int]

class TelemetryResponse(PaginatedResponse):
    items: List[TelemetryPoint]
    driver: str
    lap_number: int