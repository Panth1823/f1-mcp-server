"""Rate limiting middleware for F1 MCP Server"""

import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, Tuple[int, float]] = {}  # IP -> (count, start_time)
        
    async def __call__(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        current_time = time.time()
        
        # Check and update request count
        if client_ip in self.requests:
            count, start_time = self.requests[client_ip]
            # Reset if minute has passed
            if current_time - start_time >= 60:
                self.requests[client_ip] = (1, current_time)
            else:
                # Increment count
                count += 1
                if count > self.requests_per_minute:
                    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Too many requests",
                            "detail": f"Rate limit of {self.requests_per_minute} requests per minute exceeded"
                        }
                    )
                self.requests[client_ip] = (count, start_time)
        else:
            # First request from this IP
            self.requests[client_ip] = (1, current_time)
        
        # Process the request
        response = await call_next(request)
        return response