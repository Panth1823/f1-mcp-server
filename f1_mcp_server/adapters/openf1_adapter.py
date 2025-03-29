"""
OpenF1 Adapter for F1 MCP Server
"""

import logging
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenF1Adapter:
    """Adapter for OpenF1 API"""
    
    def __init__(self):
        self.base_url = "https://api.openf1.org/v1"
        self._session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_latest_session(self) -> Optional[Dict[str, Any]]:
        """Get information about the latest F1 session"""
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/sessions?session_key=-1") as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        return data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting latest session: {str(e)}")
            return None
    
    async def get_driver_info(self, session_key: int) -> List[Dict[str, Any]]:
        """Get driver information for a session"""
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/drivers?session_key={session_key}") as response:
                if response.status == 200:
                    return await response.json()
            return []
        except Exception as e:
            logger.error(f"Error getting driver info: {str(e)}")
            return []
    
    async def get_lap_times(self, session_key: int) -> List[Dict[str, Any]]:
        """Get lap times for a session"""
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/laps?session_key={session_key}") as response:
                if response.status == 200:
                    return await response.json()
            return []
        except Exception as e:
            logger.error(f"Error getting lap times: {str(e)}")
            return []
    
    async def get_driver_telemetry(self, session_key: int) -> List[Dict[str, Any]]:
        """Get driver telemetry data"""
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/car_data?session_key={session_key}") as response:
                if response.status == 200:
                    return await response.json()
            return []
        except Exception as e:
            logger.error(f"Error getting telemetry: {str(e)}")
            return []
    
    async def get_session_status(self, session_key: int) -> Optional[Dict[str, Any]]:
        """Get session status information"""
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/sessions?session_key={session_key}") as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        return data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting session status: {str(e)}")
            return None