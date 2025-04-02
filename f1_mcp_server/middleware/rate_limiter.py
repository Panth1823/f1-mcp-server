"""
Rate limiter middleware for the F1 MCP Server
"""

import time
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import os

# Default rate limits
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5000"))  # Increased from 1000 to 5000 requests per minute
RATE_LIMIT_INTERVAL = int(os.getenv("RATE_LIMIT_INTERVAL", "60"))  # Increased from 20 to 60 seconds

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit=RATE_LIMIT, interval=RATE_LIMIT_INTERVAL):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.interval = interval
        self.request_counts = {}
        self.last_cleaned = time.time()
    
    async def dispatch(self, request: Request, call_next):
        # Clean old entries every minute
        current_time = time.time()
        if current_time - self.last_cleaned > 60:
            self._clean_old_entries(current_time)
            self.last_cleaned = current_time
        
        # Get client identifier (IP or authenticated user)
        client_id = self._get_client_id(request)
        
        # Check if client has exceeded rate limit
        if self._is_rate_limited(client_id, current_time):
            return Response(
                content="Rate limit exceeded. Please try again later.",
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(self.interval)}
            )
        
        # Process the request
        response = await call_next(request)
        return response
    
    def _get_client_id(self, request: Request):
        # Use the X-Forwarded-For header if available (common when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to client host
        return request.client.host
    
    def _is_rate_limited(self, client_id: str, current_time: float) -> bool:
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        # Add current request timestamp
        self.request_counts[client_id].append(current_time)
        
        # Filter requests within the rate limit interval
        recent_requests = [
            timestamp for timestamp in self.request_counts[client_id]
            if current_time - timestamp <= self.interval
        ]
        
        # Update the request list for this client
        self.request_counts[client_id] = recent_requests
        
        # Check if the number of recent requests exceeds the rate limit
        return len(recent_requests) > self.rate_limit
    
    def _clean_old_entries(self, current_time: float):
        """Remove entries older than the rate limit interval"""
        for client_id in list(self.request_counts.keys()):
            self.request_counts[client_id] = [
                timestamp for timestamp in self.request_counts[client_id]
                if current_time - timestamp <= self.interval
            ]
            
            # Remove client if no recent requests
            if not self.request_counts[client_id]:
                del self.request_counts[client_id]