import asyncio
import logging
import json
from typing import Dict, Set, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from datetime import datetime

from f1_mcp_server.core.aggregator import F1DataAggregator

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class F1WebSocketServer:
    """WebSocket server for F1 data"""
    
    def __init__(self, aggregator: F1DataAggregator):
        """Initialize the server with a data aggregator"""
        self.aggregator = aggregator
        self.active_connections: Dict[str, WebSocket] = {}
        self.update_task = None
        self.is_running = False
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
        
        # Start update task if not running
        if not self.is_running:
            self.is_running = True
            self.update_task = asyncio.create_task(self._update_loop())
            
    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
            
        # Stop update task if no connections
        if not self.active_connections and self.update_task:
            self.is_running = False
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None
            
    async def toggle_mock_data(self, client_id: str):
        """Toggle mock data for testing"""
        if client_id in self.active_connections:
            self.aggregator.use_mock_data = not self.aggregator.use_mock_data
            status = "enabled" if self.aggregator.use_mock_data else "disabled"
            await self.active_connections[client_id].send_text(
                json.dumps({"type": "mock_status", "data": {"status": status}})
            )
            logger.info(f"Mock data {status} for client {client_id}")
            
    async def _update_loop(self):
        """Send F1 data updates to connected clients"""
        while self.is_running:
            try:
                # Get current data
                data = await self.aggregator.get_current_data()
                if data:
                    # Convert to dict for JSON serialization
                    data_dict = data.dict()
                    
                    # Send to all connected clients
                    dead_connections = set()
                    for client_id, websocket in self.active_connections.items():
                        try:
                            await websocket.send_text(json.dumps({"type": "update", "data": data_dict}, cls=DateTimeEncoder))
                        except Exception as e:
                            logger.error(f"Error sending to client {client_id}: {str(e)}")
                            dead_connections.add(client_id)
                            
                    # Clean up dead connections
                    for client_id in dead_connections:
                        await self.disconnect(client_id)
                        
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                
            # Wait before next update
            await asyncio.sleep(1)

def create_websocket_app():
    app = FastAPI()
    manager = F1WebSocketServer(F1DataAggregator())

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        await manager.start_updates()

    @app.on_event("shutdown")
    async def shutdown_event():
        await manager.stop_updates()

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(
        websocket: WebSocket,
        client_id: str = Path(..., description="Unique client identifier")
    ):
        try:
            await manager.connect(websocket, client_id)
            try:
                while True:
                    # Keep the connection alive and handle client messages if needed
                    data = await websocket.receive_text()
                    # You can add custom message handling here if needed
            except WebSocketDisconnect:
                manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"Error in websocket connection: {str(e)}")
                manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error establishing websocket connection: {str(e)}")
            try:
                await websocket.close()
            except:
                pass

    return app 