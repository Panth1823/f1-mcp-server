"""
Data Aggregator for F1 MCP Server
"""

import logging
from typing import Optional, Tuple, Dict
from ..adapters.openf1_adapter import OpenF1Adapter
from ..adapters.fastf1_adapter import FastF1Adapter
from ..data_models.schemas import F1Data, SessionData, DriverData
from datetime import datetime
import fastf1
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class DataAggregator:
    """Aggregates data from multiple F1 data sources"""
    
    def __init__(self):
        """Initialize the aggregator with data adapters"""
        self.openf1 = OpenF1Adapter()
        self.fastf1 = FastF1Adapter()
        self.last_session_key = None
        self.last_year = None
        self.last_gp = None
        self.last_session_type = None
        self.fastf1_session = None
        self._live_timing_cache = {}
        self._session = None
        
    async def initialize(self):
        """Initialize the data adapters"""
        await self.openf1.initialize()
        self._session = aiohttp.ClientSession()
        
    async def close(self):
        """Close the data adapters"""
        await self.openf1.close()
        if self._session:
            await self._session.close()
        
    async def get_current_session_info(self) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
        """
        Get information about the current session
        
        Returns:
            A tuple of (session_key, year, grand_prix, session_type)
        """
        session_info = await self.openf1.get_latest_session()
        if not session_info:
            return None, None, None, None
            
        session_key = session_info.get('session_key')
        year = session_info.get('year')
        grand_prix = session_info.get('meeting_name')
        session_type = session_info.get('session_name')
        
        return session_key, year, grand_prix, session_type
        
    async def get_current_data(self) -> Optional[F1Data]:
        """
        Get current F1 data from all sources
        
        Returns:
            An F1Data object containing combined data
        """
        try:
            # Get current session info
            session_key, year, grand_prix, session_type = await self.get_current_session_info()
            
            # If no live session is available, try to get the most recent historical session
            if not session_key:
                logger.info("No live session found, attempting to get most recent historical session")
                try:
                    # Get the most recent race weekend
                    current_year = datetime.now().year
                    schedule = fastf1.get_event_schedule(current_year)
                    if not schedule.empty:
                        latest_event = schedule.iloc[-1]
                        year = current_year
                        grand_prix = latest_event['EventName']
                        session_type = 'Race'  # Default to race session
                        logger.info(f"Using historical data from {grand_prix} {year}")
                    else:
                        logger.warning("No historical session data available")
                        return None
                except Exception as e:
                    logger.error(f"Error getting historical session: {str(e)}")
                    return None
            
            # Update FastF1 session if needed
            if (session_key != self.last_session_key or 
                year != self.last_year or 
                grand_prix != self.last_gp or 
                session_type != self.last_session_type):
                self.fastf1_session = self.fastf1.get_session(year, grand_prix, session_type)
                self.last_session_key = session_key
                self.last_year = year
                self.last_gp = grand_prix
                self.last_session_type = session_type
            
            # Get data from OpenF1 if available
            if session_key:
                session_status = await self.openf1.get_session_status(session_key)
                driver_info = await self.openf1.get_driver_info(session_key)
                telemetry = await self.openf1.get_driver_telemetry(session_key)
                lap_times = await self.openf1.get_lap_times(session_key)
            else:
                # Use historical data from FastF1
                session_status = {'status': 'Completed', 'date': datetime.now(), 'meeting_name': grand_prix}
                driver_info = self.fastf1.get_driver_info(self.fastf1_session) if self.fastf1_session else []
                lap_times = []
                telemetry = []
                if self.fastf1_session:
                    laps_df = self.fastf1.get_lap_times(self.fastf1_session)
                    if not laps_df.empty:
                        lap_times = laps_df.to_dict('records')
            
            # Create session data
            session_data = SessionData(
                session_type=session_type,
                session_name=f"{grand_prix} {session_type}",
                track_name=session_status.get('meeting_name', 'Unknown Track'),
                session_date=session_status.get('date', ''),
                session_status=session_status.get('status', 'Unknown'),
                weather={},  # Will be updated from FastF1
                track_temp=None,
                air_temp=None
            )
            
            # Add FastF1 weather data if available
            if self.fastf1_session:
                weather = self.fastf1.get_session_weather(self.fastf1_session)
                session_data.weather = weather
                session_data.track_temp = weather.get('track_temp')
                session_data.air_temp = weather.get('air_temp')
            
            # Create F1Data object
            f1_data = F1Data(
                session=session_data,
                drivers={},
                lap_count=max([lap.get('lap_number', 0) for lap in lap_times]) if lap_times else 0,
                source_info={"telemetry": "OpenF1" if session_key else "FastF1", "historical": "FastF1"}
            )
            
            # Process driver data
            for driver in driver_info:
                driver_number = str(driver.get('driver_number', '0'))
                driver_laps = [lap for lap in lap_times if lap.get('driver_number') == driver_number]
                driver_telemetry = [t for t in telemetry if t.get('driver_number') == driver_number]
                
                # Get the latest lap time
                last_lap = None
                best_lap = None
                if driver_laps:
                    valid_laps = [lap for lap in driver_laps if lap.get('lap_time')]
                    if valid_laps:
                        last_lap = valid_laps[-1].get('lap_time')
                        best_lap = min(lap.get('lap_time', float('inf')) for lap in valid_laps)
                
                # Get the latest telemetry
                speed = None
                if driver_telemetry:
                    speed = driver_telemetry[-1].get('speed')
                elif self.fastf1_session:
                    # Try to get speed from FastF1
                    car_data, _ = self.fastf1.get_driver_telemetry(self.fastf1_session, driver_number)
                    if car_data is not None and not car_data.empty:
                        speed = car_data['Speed'].iloc[-1]
                
                # Create driver data
                driver_data = DriverData(
                    driver_number=driver_number,
                    driver_name=driver.get('full_name', 'Unknown Driver'),
                    team=driver.get('team_name', 'Unknown Team'),
                    position=driver.get('position'),
                    current_lap=max([lap.get('lap_number', 0) for lap in driver_laps]) if driver_laps else 0,
                    last_lap_time=last_lap,
                    best_lap_time=best_lap if best_lap != float('inf') else None,
                    sector_times={},  # Will be populated from FastF1 if available
                    speed=speed,
                    tire_compound=None,  # Will be populated from FastF1 if available
                    pit_stops=len([lap for lap in driver_laps if lap.get('pit_out')]) if driver_laps else 0,
                    gap_to_leader=None  # Will be calculated if position data is available
                )
                
                # Add to F1Data
                f1_data.drivers[driver_number] = driver_data
            
            return f1_data
        except Exception as e:
            logger.error(f"Error in get_current_data: {str(e)}")
            return None
    
    async def get_live_timing(self, session_id: str) -> Optional[Dict]:
        """Get live timing data for a session"""
        try:
            # This should connect to F1's live timing service
            # For now, return cached data if available
            if session_id in self._live_timing_cache:
                return self._live_timing_cache[session_id]
            
            # TODO: Implement connection to real F1 Live Timing API
            # This requires credentials for F1's official data service
            # Example implementation:
            try:
                # Here you would call the actual F1 Live Timing API
                # This is a placeholder for the real implementation
                if not self._session:
                    self._session = aiohttp.ClientSession()
                
                # Example: async with self._session.get(f"{LIVE_TIMING_API_BASE_URL}/{session_id}") as response:
                #     if response.status == 200:
                #         live_data = await response.json()
                #         self._live_timing_cache[session_id] = live_data
                #         return live_data
                
                # For now, returning None until proper API implementation
                logger.warning("Live timing API connection not yet implemented")
                return None
                
            except Exception as e:
                logger.error(f"Error connecting to Live Timing API: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting live timing data: {str(e)}")
            return None