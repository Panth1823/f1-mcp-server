"""
FastF1 Adapter for F1 MCP Server
"""

import logging
import fastf1
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache

# Configure FastF1
fastf1.Cache.enable_cache('fastf1_cache')

logger = logging.getLogger(__name__)

class FastF1Adapter:
    """Adapter for FastF1 library"""
    
    def __init__(self):
        # Enable FastF1 cache
        fastf1.Cache.enable_cache('fastf1_cache')
    
    def get_session(self, year: int, event: str, session_type: str) -> Optional[fastf1.core.Session]:
        """Get a FastF1 session"""
        try:
            session = fastf1.get_session(year, event, session_type)
            session.load()
            return session
        except Exception as e:
            logger.error(f"Error getting FastF1 session: {str(e)}")
            return None
    
    def get_event_schedule(self, year: int) -> Optional[pd.DataFrame]:
        """Get F1 season schedule"""
        try:
            return fastf1.get_event_schedule(year)
        except Exception as e:
            logger.error(f"Error getting event schedule: {str(e)}")
            return None
    
    def get_session_weather(self, session: fastf1.core.Session) -> Optional[pd.DataFrame]:
        """Get weather data for a session"""
        try:
            return session.weather_data
        except Exception as e:
            logger.error(f"Error getting session weather: {str(e)}")
            return None
    
    def get_driver_info(self, session: fastf1.core.Session, driver_number: str) -> Optional[Dict[str, Any]]:
        """Get driver information"""
        try:
            return session.get_driver(driver_number)
        except Exception as e:
            logger.error(f"Error getting driver info: {str(e)}")
            return None
    
    def get_lap_times(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Get lap times for a session
        
        Args:
            session: A FastF1 session object
            
        Returns:
            A DataFrame containing lap time information
        """
        try:
            return session.laps
        except Exception as e:
            logger.error(f"Error getting lap times: {str(e)}")
            return pd.DataFrame()
    
    def get_driver_telemetry(self, session: fastf1.core.Session, driver: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get telemetry data for a specific driver
        
        Args:
            session: A FastF1 session object
            driver: The driver number
            
        Returns:
            A tuple of (car_data, position_data) DataFrames
        """
        try:
            laps = session.laps.pick_driver(driver)
            if laps.empty:
                return None, None
                
            fastest_lap = laps.pick_fastest()
            car_data = fastest_lap.get_car_data().add_distance()
            pos_data = fastest_lap.get_pos_data()
            
            return car_data, pos_data
        except Exception as e:
            logger.error(f"Error getting driver telemetry: {str(e)}")
            return None, None