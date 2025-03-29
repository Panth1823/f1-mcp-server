import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from functools import lru_cache

class F1Analytics:
    """Analytics processor for F1 data"""
    
    def __init__(self):
        """Initialize the analytics processor"""
        self._drivers_df = pd.DataFrame()
        self._laps_df = pd.DataFrame()
        self._telemetry_df = pd.DataFrame()
    
    def update_data(self, drivers_data: List[Dict[str, Any]], laps_data: List[Dict[str, Any]], telemetry_data: List[Dict[str, Any]]):
        """
        Update the data used for analytics
        
        Args:
            drivers_data: List of driver data dictionaries
            laps_data: List of lap data dictionaries
            telemetry_data: List of telemetry data dictionaries
        """
        if drivers_data:
            self._drivers_df = pd.DataFrame(drivers_data)
        
        if laps_data:
            self._laps_df = pd.DataFrame(laps_data)
            
        if telemetry_data:
            self._telemetry_df = pd.DataFrame(telemetry_data)
    
    @lru_cache(maxsize=128)
    def get_driver_performance_stats(self, driver_number: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific driver
        
        Args:
            driver_number: Driver's number
            
        Returns:
            Dictionary containing performance statistics
        """
        if self._laps_df.empty or self._drivers_df.empty:
            return {}
            
        driver_laps = self._laps_df[self._laps_df['driver_number'] == driver_number]
        
        if driver_laps.empty:
            return {}
            
        valid_times = driver_laps['lap_time'].dropna()
        
        stats = {
            'total_laps': len(driver_laps),
            'best_lap': valid_times.min() if not valid_times.empty else None,
            'average_lap': valid_times.mean() if not valid_times.empty else None,
            'consistency': valid_times.std() if len(valid_times) > 1 else None
        }
        
        return stats
    
    def get_race_pace_comparison(self) -> Dict[str, List[float]]:
        """
        Compare race pace between drivers
        
        Returns:
            Dictionary mapping driver numbers to their lap time distributions
        """
        if self._laps_df.empty:
            return {}
            
        pace_comparison = {}
        for driver in self._laps_df['driver_number'].unique():
            driver_laps = self._laps_df[self._laps_df['driver_number'] == driver]['lap_time'].dropna()
            if not driver_laps.empty:
                pace_comparison[driver] = driver_laps.tolist()
                
        return pace_comparison
    
    def get_sector_analysis(self, driver_number: str) -> Dict[str, List[float]]:
        """
        Analyze sector times for a specific driver
        
        Args:
            driver_number: Driver's number
            
        Returns:
            Dictionary containing sector time distributions
        """
        if self._laps_df.empty:
            return {}
            
        driver_laps = self._laps_df[self._laps_df['driver_number'] == driver_number]
        
        if driver_laps.empty:
            return {}
            
        sectors = {}
        for sector in ['sector1_time', 'sector2_time', 'sector3_time']:
            sector_times = driver_laps[sector].dropna()
            if not sector_times.empty:
                sectors[sector] = sector_times.tolist()
                
        return sectors
    
    def get_telemetry_insights(self, driver_number: str) -> Dict[str, Any]:
        """
        Get insights from telemetry data for a specific driver
        
        Args:
            driver_number: Driver's number
            
        Returns:
            Dictionary containing telemetry insights
        """
        if self._telemetry_df.empty:
            return {}
            
        driver_telemetry = self._telemetry_df[self._telemetry_df['driver_number'] == driver_number]
        
        if driver_telemetry.empty:
            return {}
            
        insights = {
            'max_speed': driver_telemetry['speed'].max() if 'speed' in driver_telemetry.columns else None,
            'avg_speed': driver_telemetry['speed'].mean() if 'speed' in driver_telemetry.columns else None,
            'max_rpm': driver_telemetry['rpm'].max() if 'rpm' in driver_telemetry.columns else None,
            'avg_rpm': driver_telemetry['rpm'].mean() if 'rpm' in driver_telemetry.columns else None,
            'gear_distribution': driver_telemetry['gear'].value_counts().to_dict() if 'gear' in driver_telemetry.columns else None
        }
        
        return insights