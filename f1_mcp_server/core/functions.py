"""
F1 MCP Server Core Functions
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from fastapi import HTTPException
import asyncio
import aiohttp
from aiocache import cached
import fastf1
from datetime import datetime
from .aggregator import DataAggregator
from pydantic import BaseModel, validator, Field
from enum import Enum
from f1_mcp_server.data_models.schemas import PaginationParams, TelemetryResponse, TelemetryPoint
import json
from json import JSONEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastF1 cache
fastf1.Cache.enable_cache('fastf1_cache')


class SessionType(str, Enum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    SPRINT = "Sprint"
    RACE = "Race"

class SessionRequest(BaseModel):
    year: int = Field(..., ge=1950, le=2100)
    event: str
    session: SessionType

    @validator('event')
    def validate_event(cls, v):
        if not v.strip():
            raise ValueError("Event identifier cannot be empty")
        return v.strip()

class TelemetryRequest(BaseModel):
    year: int = Field(..., ge=1950, le=2100)
    event: str
    driver: str
    lap: int = Field(..., ge=1)

    @validator('driver')
    def validate_driver(cls, v):
        if not v.isalnum():
            raise ValueError("Driver identifier must be alphanumeric")
        return v.upper()

class DriversComparisonRequest(BaseModel):
    year: int = Field(..., ge=1950, le=2100)
    event: str
    drivers: List[str] = Field(..., min_items=2, max_items=5)

    @validator('drivers')
    def validate_drivers(cls, drivers):
        if len(set(drivers)) != len(drivers):
            raise ValueError("Driver list contains duplicates")
        return [d.upper() for d in drivers if d.strip()]


class F1DataProvider:
    """Provider for F1 data"""

    def __init__(self):
        self.aggregator = DataAggregator()
        self.logger = logging.getLogger(__name__)

    @cached(ttl=3600)  # Cache for 1 hour
    async def get_race_calendar(self, year: int) -> Dict:
        """Get F1 season schedule"""
        try:
            # Get the schedule
            schedule = fastf1.get_event_schedule(year)
            if schedule is None or schedule.empty:
                raise HTTPException(
                    status_code=404, detail=f"No schedule found for year {year}")

            # Convert to a more readable format
            calendar = []
            for _, event in schedule.iterrows():
                calendar_entry = {
                    'round': int(event['RoundNumber']) if pd.notna(event['RoundNumber']) else None,
                    'event_name': event['EventName'],
                    'circuit': event['Location'],
                    'country': event['Country'],
                    'date': pd.to_datetime(event['EventDate']).isoformat() if pd.notna(event['EventDate']) else None,
                    'sessions': []
                }

                # Add all available sessions
                for i in range(1, 6):
                    session_date = event.get(f'Session{i}Date')
                    session_name = event.get(f'Session{i}')
                    if pd.notna(session_date) and session_name:
                        calendar_entry['sessions'].append({
                            'name': session_name,
                            'date': pd.to_datetime(session_date).isoformat()
                        })

                calendar.append(calendar_entry)

            return {
                'status': 'success',
                'year': year,
                'calendar': calendar
            }
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching race calendar: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error fetching race calendar: {str(e)}")

    @cached(ttl=3600)
    async def get_event_details(self, year: int, event_identifier: str) -> Dict:
        """Get detailed information about a specific Grand Prix"""
        try:
            # Get the schedule
            schedule = fastf1.get_event_schedule(year)
            if schedule is None or schedule.empty:
                raise HTTPException(
                    status_code=404, detail=f"No schedule found for year {year}")

            # Normalize the event identifier
            event_identifier = event_identifier.lower().strip()

            # Try to find the event by various fields
            event = None
            for _, row in schedule.iterrows():
                if (event_identifier in row['EventName'].lower() or
                    event_identifier in row['Location'].lower() or
                    event_identifier in row['Country'].lower()):
                    event = row
                    break

            if event is None:
                raise HTTPException(
                    status_code=404, detail=f"Event '{event_identifier}' not found in {year} schedule")

            # Format dates
            def format_date(date_str):
                if pd.isna(date_str):
                    return None
                return pd.to_datetime(date_str).isoformat()

            # Build the response
            response = {
                'status': 'success',
                'event_name': event['EventName'],
                'circuit_name': event['Location'],
                'country': event['Country'],
                'location': event['Location'],
                'round': int(event['RoundNumber']) if pd.notna(event['RoundNumber']) else None,
                'date': format_date(event['EventDate']),
                'sessions': []
            }

            # Add all available sessions
            for i in range(1, 6):
                session_date = event.get(f'Session{i}Date')
                session_name = event.get(f'Session{i}')
                if pd.notna(session_date) and session_name:
                    response['sessions'].append({
                        'name': session_name,
                        'date': format_date(session_date)
                    })

            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching event details: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error fetching event details: {str(e)}")

    @cached(ttl=1800)  # Cache for 30 minutes
    async def get_session_results(self, year: int, event: str, session: str) -> Dict:
        """Get results for a specific session"""
        try:
            # Validate inputs
            request = SessionRequest(year=year, event=event, session=session)
            
            # Load the session with all necessary data
            race_session = fastf1.get_session(request.year, request.event, request.session)
            race_session.load(laps=True, telemetry=True, weather=True, messages=True)

            # Get results and convert to a more readable format
            results = race_session.results
            if results is None or len(results) == 0:
                raise ValueError("No results available for this session")

            formatted_results = []
            for _, driver in results.iterrows():
                # Handle fastest lap data safely
                is_fastest = False
                fastest_lap_time = None
                if 'FastestLap' in driver and pd.notna(driver['FastestLap']):
                    is_fastest = True
                    fastest_lap_time = str(driver['FastestLap']) if pd.notna(
                        driver['FastestLap']) else None
                elif 'BestLapTime' in driver and pd.notna(driver['BestLapTime']):
                    fastest_lap_time = str(driver['BestLapTime'])

                # Get driver's best lap
                driver_laps = race_session.laps.pick_driver(driver['DriverNumber'])
                best_lap = None
                if not driver_laps.empty:
                    best_lap = driver_laps.pick_fastest()
                    if best_lap is not None:
                        best_lap = {
                            'lap_number': int(best_lap['LapNumber']),
                            'lap_time': str(best_lap['LapTime']),
                            'sector_1_time': str(best_lap['Sector1Time']) if pd.notna(best_lap['Sector1Time']) else None,
                            'sector_2_time': str(best_lap['Sector2Time']) if pd.notna(best_lap['Sector2Time']) else None,
                            'sector_3_time': str(best_lap['Sector3Time']) if pd.notna(best_lap['Sector3Time']) else None
                        }

                result = {
                    'position': int(driver['Position']) if pd.notna(driver['Position']) else None,
                    'driver_number': str(driver['DriverNumber']),
                    'driver_code': driver['Abbreviation'],
                    'driver_name': f"{driver['FirstName']} {driver['LastName']}",
                    'team': driver['TeamName'],
                    'grid_position': int(driver['GridPosition']) if pd.notna(driver['GridPosition']) else None,
                    'status': driver['Status'],
                    'points': float(driver['Points']) if pd.notna(driver['Points']) else 0.0,
                    'time': str(driver['Time']) if pd.notna(driver['Time']) else None,
                    'fastest_lap': is_fastest,
                    'fastest_lap_time': fastest_lap_time,
                    'gap_to_leader': str(driver['Time']) if pd.notna(driver['Time']) else None,
                    'best_lap': best_lap,
                    'total_laps': int(driver['TotalLaps']) if pd.notna(driver['TotalLaps']) else None
                }
                formatted_results.append(result)

            return {
                'status': 'success',
                'session': session,
                'event': event,
                'year': year,
                'results': formatted_results,
                'total_drivers': len(formatted_results)
            }
        except ValueError as ve:
            self.logger.error(f"Validation error in get_session_results: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            self.logger.error(f"Error fetching session results: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error fetching session results: {str(e)}")

    class DriverPerformanceRequest(BaseModel):
        year: int
        event: str
        driver: str

        @validator('year')
        def validate_year(cls, v):
            if v < 1950 or v > 2100:
                raise ValueError("Year must be between 1950 and 2100")
            return v

        @validator('driver')
        def validate_driver(cls, v):
            if not v.isalnum():
                raise ValueError("Driver identifier must be alphanumeric")
            return v

    @cached(ttl=1800)  # Cache for 30 minutes
    async def get_driver_performance(self, year: int, event: str, driver: str) -> Dict:
        """Get detailed performance data for a specific driver"""
        try:
            # Validate inputs
            request_data = self.DriverPerformanceRequest(year=year, event=event, driver=driver)

            # Load the race session
            session = fastf1.get_session(request_data.year, request_data.event, 'Race')
            session.load()

            # Get laps for the specific driver
            driver_laps = session.laps.pick_driver(request_data.driver)
            if driver_laps.empty:
                raise ValueError(f"No lap data found for driver {request_data.driver}")

            # Format lap data
            lap_data = []
            for idx, lap in driver_laps.iterrows():
                lap_info = {
                    'lap_number': int(lap['LapNumber']) if pd.notna(lap['LapNumber']) else None,
                    'lap_time': str(lap['LapTime']) if pd.notna(lap['LapTime']) else None,
                    'position': int(lap['Position']) if pd.notna(lap['Position']) else None,
                    'sector_1_time': str(lap['Sector1Time']) if pd.notna(lap['Sector1Time']) else None,
                    'sector_2_time': str(lap['Sector2Time']) if pd.notna(lap['Sector2Time']) else None,
                    'sector_3_time': str(lap['Sector3Time']) if pd.notna(lap['Sector3Time']) else None,
                    'speed_trap': float(lap['SpeedI2']) if pd.notna(lap['SpeedI2']) else None,
                    'is_personal_best': bool(lap['IsPersonalBest']) if pd.notna(lap['IsPersonalBest']) else False
                }
                lap_data.append(lap_info)

            return {
                'driver': request_data.driver,
                'total_laps': len(lap_data),
                'lap_data': lap_data
            }
        except ValueError as ve:
            self.logger.error(f"Validation error: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            self.logger.error(f"Unexpected error in get_driver_performance: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_telemetry(self, year: int, event: str, driver: str, lap: int, 
                           pagination: Optional[PaginationParams] = None) -> Dict:
        """Get detailed telemetry data for a specific lap with pagination support"""
        try:
            # Validate inputs
            request = TelemetryRequest(year=year, event=event, driver=driver, lap=lap)
            pagination = pagination or PaginationParams()
                
            # Load the race session
            session = fastf1.get_session(request.year, request.event, 'Race')
            session.load()

            # Get the specific lap's telemetry
            driver_laps = session.laps.pick_driver(request.driver)
            if driver_laps.empty:
                raise ValueError(f"No lap data found for driver {request.driver}")

            # Filter for the specific lap
            target_lap = driver_laps[driver_laps['LapNumber'] == request.lap]
            if target_lap.empty:
                raise ValueError(f"Lap {request.lap} not found for driver {request.driver}")

            # Get telemetry for the first matching lap
            target_lap = target_lap.iloc[0]
            telemetry = target_lap.get_telemetry()

            # Format telemetry data
            telemetry_points = []
            for idx, data in telemetry.iterrows():
                # Convert the time index to proper ISO format
                time_str = pd.Timestamp(data.name).isoformat() if isinstance(data.name, (pd.Timestamp, datetime)) else datetime.fromtimestamp(0).isoformat()
                
                point = {
                    "time": time_str,
                    "speed": float(data['Speed']) if pd.notna(data['Speed']) else 0.0,
                    "throttle": float(data['Throttle']) if pd.notna(data['Throttle']) else 0.0,
                    "brake": bool(data['Brake']) if pd.notna(data['Brake']) else False,
                    "gear": int(data['nGear']) if pd.notna(data['nGear']) else 0,
                    "rpm": float(data['RPM']) if pd.notna(data['RPM']) else 0.0,
                    "drs": int(data['DRS']) if pd.notna(data['DRS']) else 0
                }
                telemetry_points.append(point)

            # Apply pagination
            total_items = len(telemetry_points)
            total_pages = (total_items + pagination.page_size - 1) // pagination.page_size
            start_idx = (pagination.page - 1) * pagination.page_size
            end_idx = start_idx + pagination.page_size
            paginated_points = telemetry_points[start_idx:end_idx]

            return {
                "driver": request.driver,
                "lap_number": request.lap,
                "telemetry": paginated_points,
                "page": pagination.page,
                "page_size": pagination.page_size,
                "total_items": total_items,
                "total_pages": total_pages
            }
        except ValueError as ve:
            self.logger.error(f"Validation error in get_telemetry: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            self.logger.error(f"Error fetching telemetry: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching telemetry: {str(e)}")

    @cached(ttl=1800)
    async def compare_drivers(self, year: int, event: str, drivers: List[str]) -> Dict:
        """Compare performance between multiple drivers"""
        try:
            # Validate inputs
            request = DriversComparisonRequest(year=year, event=event, drivers=drivers)
            
            # Load the race session
            session = fastf1.get_session(request.year, request.event, 'Race')
            session.load()

            comparison_data = {}
            for driver in request.drivers:
                driver_laps = session.laps.pick_driver(driver)
                if len(driver_laps) == 0:
                    raise ValueError(f"No lap data found for driver {driver}")

                fastest_lap = driver_laps.pick_fastest()
                if fastest_lap is None:
                    raise ValueError(
                        f"No fastest lap found for driver {driver}")

                comparison_data[driver] = {
                    'fastest_lap_time': str(fastest_lap['LapTime']) if pd.notna(fastest_lap['LapTime']) else None,
                    'fastest_lap_number': int(fastest_lap['LapNumber']) if pd.notna(fastest_lap['LapNumber']) else None,
                    'sector_1_time': str(fastest_lap['Sector1Time']) if pd.notna(fastest_lap['Sector1Time']) else None,
                    'sector_2_time': str(fastest_lap['Sector2Time']) if pd.notna(fastest_lap['Sector2Time']) else None,
                    'sector_3_time': str(fastest_lap['Sector3Time']) if pd.notna(fastest_lap['Sector3Time']) else None,
                    'speed_trap': float(fastest_lap['SpeedI2']) if pd.notna(fastest_lap['SpeedI2']) else None
                }

            return comparison_data
        except Exception as e:
            self.logger.error(f"Error comparing drivers: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error comparing drivers: {str(e)}")

    async def get_live_timing(self, session_id: str):
        """Get real-time session updates using SSE"""
        try:
            while True:
                timing_data = await self.aggregator.get_live_timing(session_id)
                if timing_data:
                    yield {
                        "event": "timing",
                        "data": timing_data
                    }
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in live timing stream: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error in live timing stream: {str(e)}")

    @cached(ttl=1800)
    async def get_weather_data(self, year: int, event: str) -> Dict:
        """Get weather information for a specific event"""
        try:
            # Load the race session with weather data
            session = fastf1.get_session(year, event, 'Race')
            session.load()

            if session.weather_data is None or session.weather_data.empty:
                raise ValueError("No weather data available for this session")

            weather_data = []
            for idx, data in session.weather_data.iterrows():
                # Convert timestamp to string safely
                timestamp = idx
                if hasattr(timestamp, 'isoformat'):
                    time_str = timestamp.isoformat()
                elif isinstance(timestamp, (int, float)):
                    from datetime import datetime
                    time_str = datetime.fromtimestamp(timestamp).isoformat()
                else:
                    time_str = str(timestamp)

                point = {
                    'time': time_str,
                    'air_temp': float(data['AirTemp']) if pd.notna(data['AirTemp']) else None,
                    'track_temp': float(data['TrackTemp']) if pd.notna(data['TrackTemp']) else None,
                    'humidity': float(data['Humidity']) if pd.notna(data['Humidity']) else None,
                    'pressure': float(data['Pressure']) if pd.notna(data['Pressure']) else None,
                    'wind_speed': float(data['WindSpeed']) if pd.notna(data['WindSpeed']) else None,
                    'wind_direction': float(data['WindDirection']) if pd.notna(data['WindDirection']) else None
                }
                weather_data.append(point)

            return {
                'session': 'Race',
                'weather_data': weather_data
            }
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error fetching weather data: {str(e)}")

    @cached(ttl=86400)  # Cache for 24 hours
    async def get_circuit_info(self, circuit_id: str) -> Dict:
        """Get detailed information about a specific circuit"""
        try:
            # Get current year's schedule to find the circuit
            current_year = datetime.now().year
            schedule = fastf1.get_event_schedule(current_year)
            
            if schedule is None or schedule.empty:
                raise HTTPException(
                    status_code=404, detail="Could not fetch circuit information")

            # Find the circuit in the schedule
            circuit = None
            for _, event in schedule.iterrows():
                if circuit_id.lower() in event['Location'].lower():
                    circuit = event
                    break

            if circuit is None:
                raise HTTPException(
                    status_code=404, detail=f"Circuit '{circuit_id}' not found")

            # Get the most recent race session for this circuit
            session = fastf1.get_session(current_year, circuit['EventName'], 'Race')
            session.load()

            # Get circuit length and other details
            circuit_length = None
            if hasattr(session, 'track_length'):
                circuit_length = float(session.track_length)

            # Get DRS zones
            drs_zones = 0
            if hasattr(session, 'drs_zones'):
                drs_zones = len(session.drs_zones)

            # Get lap record if available
            lap_record = None
            if hasattr(session, 'laps') and not session.laps.empty:
                fastest_lap = session.laps.pick_fastest()
                if fastest_lap is not None:
                    lap_record = {
                        'time': str(fastest_lap['LapTime']),
                        'driver': f"{fastest_lap['Driver']}",
                        'year': current_year
                    }

            return {
                'status': 'success',
                'circuit_id': circuit_id,
                'name': circuit['Location'],
                'country': circuit['Country'],
                'length': circuit_length,  # km
                'turns': None,  # Not available in FastF1
                'drs_zones': drs_zones,
                'lap_record': lap_record,
                'first_grand_prix': None,  # Not available in FastF1
                'last_grand_prix': current_year
            }
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching circuit info: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error fetching circuit info: {str(e)}")

    @cached(ttl=3600)  # Cache for 1 hour
    async def get_testing_session(self, year: int, test_number: int, session_number: int) -> Dict[str, Any]:
        """Get information about F1 testing sessions"""
        try:
            # Get the testing event first
            schedule = fastf1.get_event_schedule(year, include_testing=True)
            testing_events = schedule[schedule['EventFormat'] == 'Testing']
            
            if testing_events.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No testing events found for year {year}"
                )
            
            # Get the specific testing event
            test_event = testing_events.iloc[test_number - 1]
            
            # Get the specific session date based on session number
            session_date = test_event.get(f'Session{session_number}Date')
            if pd.isna(session_date):
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_number} not found for testing event {test_number}"
                )
            
            return {
                'year': year,
                'test_number': test_number,
                'session_number': session_number,
                'event_name': test_event['EventName'],
                'circuit': test_event['Location'],
                'country': test_event['Country'],
                'date': pd.to_datetime(session_date).isoformat(),
                'session_name': f'Testing Day {session_number}',
                'weather_data': {
                    'air_temp': None,  # Weather data not available for testing sessions
                    'track_temp': None,
                    'humidity': None,
                    'rainfall': None
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching testing session data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching testing session data: {str(e)}"
            )

    @cached(ttl=3600)  # Cache for 1 hour
    async def get_testing_event(self, year: int, test_number: int) -> Dict[str, Any]:
        """Get information about an F1 testing event"""
        try:
            # Get the testing events from the schedule
            schedule = fastf1.get_event_schedule(year, include_testing=True)
            testing_events = schedule[schedule['EventFormat'] == 'Testing']
            
            if testing_events.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No testing events found for year {year}"
                )
            
            if test_number > len(testing_events):
                raise HTTPException(
                    status_code=404,
                    detail=f"Testing event {test_number} not found for year {year}"
                )
            
            # Get the specific testing event
            event = testing_events.iloc[test_number - 1]
            
            # Get all sessions for this testing event
            sessions = []
            for i in range(1, 6):  # Usually up to 5 possible sessions
                session_date = event.get(f'Session{i}Date')
                if pd.notna(session_date):
                    sessions.append({
                        'name': f'Testing Day {i}',
                        'date': pd.to_datetime(session_date).isoformat(),
                        'number': i
                    })
            
            return {
                'year': year,
                'test_number': test_number,
                'event_name': event['EventName'],
                'circuit': event['Location'],
                'country': event['Country'],
                'location': event['Location'],
                'date': pd.to_datetime(event['EventDate']).isoformat(),
                'sessions': sessions
            }
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching testing event data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching testing event data: {str(e)}"
            )

    @cached(ttl=1800)  # Cache for 30 minutes
    async def get_events_remaining(self, include_testing: bool = True) -> Dict[str, Any]:
        """Get information about remaining events in the season"""
        try:
            schedule = fastf1.get_event_schedule(datetime.now().year)
            current_date = datetime.now()
            
            # Filter for remaining events
            remaining_events = []
            for _, event in schedule.iterrows():
                event_date = pd.to_datetime(event['EventDate'])
                if event_date > current_date:
                    remaining_events.append({
                        'name': event['EventName'],
                        'circuit': event['Location'],
                        'country': event['Country'],
                        'location': event['Location'],
                        'date': event_date.isoformat(),
                        'round': event['RoundNumber'],
                        'format': event['EventFormat'] if 'EventFormat' in event else 'Traditional',
                        'sessions': [
                            {
                                'name': 'Practice 1',
                                'date': pd.to_datetime(event['Session1Date']).isoformat() if pd.notna(event['Session1Date']) else None
                            },
                            {
                                'name': 'Practice 2',
                                'date': pd.to_datetime(event['Session2Date']).isoformat() if pd.notna(event['Session2Date']) else None
                            },
                            {
                                'name': 'Practice 3',
                                'date': pd.to_datetime(event['Session3Date']).isoformat() if pd.notna(event['Session3Date']) else None
                            },
                            {
                                'name': 'Qualifying',
                                'date': pd.to_datetime(event['Session4Date']).isoformat() if pd.notna(event['Session4Date']) else None
                            },
                            {
                                'name': 'Sprint' if 'Sprint' in str(event.get('EventFormat', '')) else 'Race',
                                'date': pd.to_datetime(event['Session5Date']).isoformat() if pd.notna(event['Session5Date']) else None
                            }
                        ]
                    })
            
            return {
                'remaining_events': remaining_events
            }
        except Exception as e:
            self.logger.error(f"Error fetching remaining events: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching remaining events: {str(e)}"
            )


# Initialize the data provider
f1_provider = F1DataProvider()

# Export the functions
get_race_calendar = f1_provider.get_race_calendar
get_event_details = f1_provider.get_event_details
get_session_results = f1_provider.get_session_results
get_driver_performance = f1_provider.get_driver_performance
get_telemetry = f1_provider.get_telemetry
compare_drivers = f1_provider.compare_drivers
get_live_timing = f1_provider.get_live_timing
get_weather_data = f1_provider.get_weather_data
get_circuit_info = f1_provider.get_circuit_info
get_testing_session = f1_provider.get_testing_session
get_testing_event = f1_provider.get_testing_event
get_events_remaining = f1_provider.get_events_remaining


async def get_driver_standings(year: Optional[int] = None) -> Dict[str, Any]:
    """Get current driver championship standings"""
    try:
        if not year:
            year = datetime.now().year

        session = fastf1.get_session(year, "last", "Race")
        session.load(laps=True, telemetry=True, weather=True, messages=True)

        if session:
            standings = session.results
            if standings is not None:
                return {
                    "status": "success",
                    "data": standings.to_dict('records'),
                    "year": year
                }
        return {
            "status": "error",
            "message": f"No standings data found for year {year}"
        }
    except Exception as e:
        logger.error(f"Error in get_driver_standings: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles special types."""
    def default(self, obj):
        # Convert numpy types to Python standard types
        if hasattr(obj, "dtype") and hasattr(obj, "item"):
            return obj.item()
        # Convert sets to lists
        if isinstance(obj, set):
            return list(obj)
        # Convert other non-serializable types to strings
        try:
            return JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)

def safe_serialize(data):
    """Safely serialize data to JSON string, handling special types."""
    return json.dumps(data, cls=CustomJSONEncoder)

async def get_constructor_standings(year: Optional[int] = None) -> Dict[str, Any]:
    """Get constructor championship standings"""
    try:
        if not year:
            year = datetime.now().year

        # Get the most recent race session
        session = fastf1.get_session(year, "last", "Race")
        session.load()

        if session and session.results is not None:
            # Group results by constructor and calculate points
            constructor_points = {}
            for _, result in session.results.iterrows():
                constructor = result['TeamName']
                points = float(result['Points']) if pd.notna(result['Points']) else 0
                if constructor not in constructor_points:
                    constructor_points[constructor] = {
                        'points': 0,
                        'wins': 0,
                        'nationality': result.get('TeamNationality', 'Unknown'),
                        'constructor_id': constructor.lower().replace(' ', '_')
                    }
                constructor_points[constructor]['points'] += points
                if points > 0 and result['Position'] == 1:
                    constructor_points[constructor]['wins'] += 1

            # Convert to list and sort by points
            standings = []
            for constructor, data in constructor_points.items():
                standings.append({
                    'position': len([s for s in standings if s['points'] > data['points']]) + 1,
                    'team': constructor,
                    'points': data['points'],
                    'wins': data['wins'],
                    'nationality': data['nationality'],
                    'constructor_id': data['constructor_id']
                })

            standings.sort(key=lambda x: (-x['points'], -x['wins']))

            return {
                "status": "success",
                "data": standings,
                "year": year,
                "total_teams": len(standings),
                "round": str(session.event['RoundNumber']),
                "season": str(year)
            }

        return {
            "status": "error",
            "message": f"No standings data found for year {year}",
            "data": [],
            "year": year
        }
    except Exception as e:
        logger.error(f"Error in get_constructor_standings: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "data": [],
            "year": year
        }


async def get_race_results(year: int, grand_prix: str, session_type: str) -> Dict[str, Any]:
    """Get race results for a specific Grand Prix"""
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        if session:
            results = session.results
            return {
                "status": "success",
                "data": results.to_dict('records') if results is not None else [],
                "session_info": {
                    "year": year,
                    "grand_prix": grand_prix,
                    "session_type": session_type
                }
            }
        return {
            "status": "error",
            "message": f"No results found for {grand_prix} {session_type} {year}"
        }
    except Exception as e:
        logger.error(f"Error in get_race_results: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


async def get_driver_info(driver_number: str, session_key: Optional[int] = None) -> Dict[str, Any]:
    """Get detailed information about a specific driver"""
    try:
        # Try to get current session data first
        if session_key:
            driver_info = await openf1.get_driver_info(session_key)
            driver_data = next(
                (d for d in driver_info if str(
                    d.get('driver_number')) == driver_number),
                None
            )
            if driver_data:
                return {
                    "status": "success",
                    "data": driver_data,
                    "source": "OpenF1"
                }

        # Fallback to historical data
        current_year = datetime.now().year
        session = fastf1.get_session(current_year, "last", "Race")
        if session:
            driver_data = session.get_driver(driver_number)
            if driver_data:
                return {
                    "status": "success",
                    "data": driver_data,
                    "source": "FastF1"
                }

        return {
            "status": "error",
            "message": f"No driver information found for number {driver_number}"
        }
    except Exception as e:
        logger.error(f"Error in get_driver_info: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


async def get_lap_times(session_key: int, driver_number: Optional[str] = None) -> Dict[str, Any]:
    """Get lap times for a specific session"""
    try:
        lap_times = await openf1.get_lap_times(session_key)
        if driver_number:
            lap_times = [
                lap for lap in lap_times
                if str(lap.get('driver_number')) == driver_number
            ]

        return {
            "status": "success",
            "data": lap_times,
            "session_key": session_key,
            "driver_number": driver_number
        }
    except Exception as e:
        logger.error(f"Error in get_lap_times: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


async def search_historical_data(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """Search through historical F1 data"""
    try:
        if not year_from:
            year_from = 1950
        if not year_to:
            year_to = datetime.now().year

        results = []
        for year in range(year_from, year_to + 1):
            try:
                session = fastf1.get_session(year, "last", "Race")
                if session:
                    data = None
                    if category == "drivers":
                        data = session.results
                    elif category == "weather":
                        data = session.weather_data
                    elif category == "telemetry":
                        data = session.laps

                    if data is not None:
                        # Convert to records and filter based on query
                        records = data.to_dict('records')
                        filtered = [
                            r for r in records
                            if any(
                                str(query).lower() in str(v).lower()
                                for v in r.values()
                            )
                        ]
                        if filtered:
                            results.extend([
                                {**record, "year": year}
                                for record in filtered
                            ])
            except Exception as e:
                logger.warning(f"Error searching year {year}: {str(e)}")
                continue

        return {
            "status": "success",
            "data": results,
            "query": query,
            "year_range": {"from": year_from, "to": year_to},
            "category": category
        }
    except Exception as e:
        logger.error(f"Error in search_historical_data: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
