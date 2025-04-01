import asyncio
import logging
import json
from typing import Dict, Set, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from datetime import datetime
from contextlib import asynccontextmanager
from collections import defaultdict

from f1_mcp_server.core.aggregator import F1DataAggregator

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Pool of WebSocket connections with monitoring"""
    def __init__(self, pool_size: int = 1000):
        self.pool_size = pool_size
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        self.request_counts = defaultdict(int)
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Add connection to pool if capacity available"""
        if len(self.active_connections) >= self.pool_size:
            logger.warning("Connection pool full, rejecting connection")
            await websocket.close(code=1013)  # Try again later
            return False
            
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_stats[client_id] = {
            'connected_at': datetime.now(),
            'messages_sent': 0,
            'messages_received': 0,
            'last_activity': datetime.now()
        }
        logger.info(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
        return True
        
    async def disconnect(self, client_id: str):
        """Remove connection from pool"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            if client_id in self.connection_stats:
                del self.connection_stats[client_id]
            logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
            
    async def broadcast(self, message: Any):
        """Send message to all active connections"""
        dead_connections = set()
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
                self.connection_stats[client_id]['messages_sent'] += 1
                self.connection_stats[client_id]['last_activity'] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {str(e)}")
                dead_connections.add(client_id)
                
        # Clean up dead connections
        for client_id in dead_connections:
            await self.disconnect(client_id)
            
    def track_request(self, client_id: str):
        """Track request count for rate limiting"""
        self.request_counts[client_id] += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active_connections': len(self.active_connections),
            'pool_usage': len(self.active_connections) / self.pool_size * 100,
            'connection_stats': self.connection_stats,
            'request_counts': dict(self.request_counts)
        }

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class F1WebSocketServer:
    """WebSocket server for F1 data with connection pooling"""
    
    def __init__(self, aggregator: F1DataAggregator, pool_size: int = 1000):
        """Initialize the server with a data aggregator and connection pool"""
        self.aggregator = aggregator
        self.connection_pool = ConnectionPool(pool_size)
        self.update_task = None
        self.is_running = False
        self.heartbeat_interval = 30  # seconds
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection with pooling"""
        if await self.connection_pool.connect(websocket, client_id):
            # Start update task if not running
            if not self.is_running:
                self.is_running = True
                self.update_task = asyncio.create_task(self._update_loop())
            
    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection"""
        await self.connection_pool.disconnect(client_id)
            
        # Stop update task if no connections
        if not self.connection_pool.active_connections and self.update_task:
            self.is_running = False
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None
            
    async def toggle_mock_data(self, client_id: str):
        """Toggle mock data for testing"""
        if client_id in self.connection_pool.active_connections:
            self.aggregator.use_mock_data = not self.aggregator.use_mock_data
            status = "enabled" if self.aggregator.use_mock_data else "disabled"
            await self.connection_pool.active_connections[client_id].send_text(
                json.dumps({"type": "mock_status", "data": {"status": status}})
            )
            logger.info(f"Mock data {status} for client {client_id}")
            
    async def _update_loop(self):
        """Send F1 data updates to connected clients with heartbeat"""
        while self.is_running:
            try:
                # Get current data
                data = await self.aggregator.get_current_data()
                if data:
                    # Convert to dict for JSON serialization
                    data_dict = data.dict()
                    await self.connection_pool.broadcast({
                        "type": "update",
                        "data": data_dict,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                # Send heartbeat every interval
                if int(time.time()) % self.heartbeat_interval == 0:
                    await self.connection_pool.broadcast({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                        
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                
            # Wait before next update
            await asyncio.sleep(1)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'server_status': 'running' if self.is_running else 'stopped',
            'mock_data': self.aggregator.use_mock_data,
            'pool_stats': self.connection_pool.get_stats()
        }

def create_websocket_app(pool_size: int = 1000):
    app = FastAPI()
    manager = F1WebSocketServer(F1DataAggregator(), pool_size)

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
        # Initialize any startup tasks
        pass

    @app.on_event("shutdown")
    async def shutdown_event():
        # Clean up any resources
        pass

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(
        websocket: WebSocket,
        client_id: str = Path(..., description="Unique client identifier")
    ):
        try:
            await manager.connect(websocket, client_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    manager.connection_pool.track_request(client_id)
                    # Handle client messages if needed
            except WebSocketDisconnect:
                await manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"Error in websocket connection: {str(e)}")
                await manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error establishing websocket connection: {str(e)}")
            try:
                await websocket.close()
            except:
                pass

    @app.get("/ws/stats")
    async def get_websocket_stats():
        """Get WebSocket server statistics"""
        return manager.get_stats()

    return app