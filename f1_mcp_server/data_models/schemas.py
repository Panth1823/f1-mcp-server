"""
F1 Data Models and Schemas
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel

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