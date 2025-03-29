"""
Mock Data Generator for F1 MCP Server
"""

import random
from datetime import datetime
from ..data_models.schemas import F1Data, SessionData, DriverData

class MockDataGenerator:
    """Generates mock F1 data for testing"""
    
    def __init__(self):
        self.drivers = [
            ("1", "Max Verstappen", "Red Bull Racing"),
            ("11", "Sergio Perez", "Red Bull Racing"),
            ("44", "Lewis Hamilton", "Mercedes"),
            ("63", "George Russell", "Mercedes"),
            ("16", "Charles Leclerc", "Ferrari"),
            ("55", "Carlos Sainz", "Ferrari"),
            ("4", "Lando Norris", "McLaren"),
            ("81", "Oscar Piastri", "McLaren"),
            ("14", "Fernando Alonso", "Aston Martin"),
            ("18", "Lance Stroll", "Aston Martin")
        ]
    
    def generate_data(self) -> F1Data:
        """Generate mock F1 data"""
        # Create session data
        session = SessionData(
            session_type="Race",
            session_name="Mock Grand Prix Race",
            track_name="Mock International Circuit",
            session_date=datetime.now(),
            session_status="Live",
            weather={
                "condition": "Clear",
                "humidity": 45,
                "wind_speed": 12,
                "wind_direction": "NE"
            },
            track_temp=35.5,
            air_temp=28.3
        )
        
        # Generate driver data
        drivers = {}
        for position, (number, name, team) in enumerate(self.drivers, 1):
            # Generate random lap times (1:30.000 to 1:32.000)
            last_lap = 90.0 + random.random() * 2
            best_lap = min(last_lap, 90.0 + random.random() * 1.5)
            
            # Generate random sector times
            s1 = 28.0 + random.random() * 0.5
            s2 = 31.0 + random.random() * 0.5
            s3 = 30.0 + random.random() * 0.5
            
            # Calculate gap to leader
            gap = (position - 1) * (0.5 + random.random() * 0.5) if position > 1 else 0.0
            
            drivers[number] = DriverData(
                driver_number=number,
                driver_name=name,
                team=team,
                position=position,
                current_lap=random.randint(30, 35),
                last_lap_time=last_lap,
                best_lap_time=best_lap,
                sector_times={
                    "S1": s1,
                    "S2": s2,
                    "S3": s3
                },
                speed=random.randint(280, 320),
                tire_compound=random.choice(["SOFT", "MEDIUM", "HARD"]),
                pit_stops=random.randint(0, 2),
                gap_to_leader=gap
            )
        
        # Create F1 data
        return F1Data(
            session=session,
            drivers=drivers,
            lap_count=35,
            source_info={"telemetry": "Mock", "historical": "Mock"}
        ) 