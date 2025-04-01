"""
Caching middleware for the F1 MCP Server
"""

import time
import json
import hashlib
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import os
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CacheEntry:
    """Represents a single cached response"""
    
    def __init__(self, response_body: bytes, headers: Dict[str, str], status_code: int):
        self.response_body = response_body
        self.headers = headers
        self.status_code = status_code
        self.timestamp = time.time()

class F1CacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware for caching API responses.
    
    This middleware caches GET requests to reduce load on external APIs like FastF1.
    It uses an in-memory cache with configurable TTL for different endpoints.
    """
    
    def __init__(
        self, 
        app,
        ttl_default: int = 300,  # 5 minutes default TTL
        cacheable_paths: List[str] = None,
        max_cache_size: int = 1000,
        ttl_overrides: Dict[str, int] = None
    ):
        super().__init__(app)
        self.cache: Dict[str, CacheEntry] = {}
        self.ttl_default = ttl_default
        self.max_cache_size = max_cache_size
        self.cacheable_paths = cacheable_paths or [
            "/mcp/function/get_race_calendar",
            "/mcp/function/get_event_details",
            "/mcp/function/get_session_results",
            "/mcp/function/get_driver_standings",
            "/mcp/function/get_constructor_standings",
            "/mcp/function/get_circuit_info"
        ]
        # Allow path-specific TTL configurations
        self.ttl_overrides = ttl_overrides or {
            # Historical data can be cached longer
            "/mcp/function/get_session_results": 3600,        # 1 hour
            "/mcp/function/get_race_calendar": 86400,         # 24 hours
            "/mcp/function/get_circuit_info": 604800,         # 1 week
            # Current data needs to be refreshed more often
            "/mcp/function/get_driver_standings": 300,        # 5 minutes
            "/mcp/function/get_constructor_standings": 300    # 5 minutes
        }
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # Cleanup cache every hour
        
    async def dispatch(self, request: Request, call_next):
        # Skip non-GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip if not a cacheable path
        if not any(request.url.path.startswith(path) for path in self.cacheable_paths):
            return await call_next(request)
        
        # Generate cache key from request
        cache_key = self._generate_cache_key(request)
        
        # Check if response is in cache and not expired
        cached_response = self._get_from_cache(cache_key, request.url.path)
        if cached_response:
            logger.debug(f"Cache hit for {request.url.path}")
            return Response(
                content=cached_response.response_body,
                status_code=cached_response.status_code,
                headers=cached_response.headers
            )
        
        # If not in cache, get the response
        logger.debug(f"Cache miss for {request.url.path}")
        response = await call_next(request)
        
        # Cache the response if status is 200 OK
        if response.status_code == 200:
            await self._cache_response(cache_key, response, request.url.path)
        
        # Periodic cache cleanup
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_cache()
            self.last_cleanup = current_time
        
        return response

    def _generate_cache_key(self, request: Request) -> str:
        """Generate a unique cache key based on the request path and query parameters"""
        # Combine path and query parameters for the key
        key_parts = [request.url.path]
        query_params = str(request.query_params)
        if query_params:
            key_parts.append(query_params)
            
        # Create a hash of the combined string
        key_str = "".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str, path: str) -> Optional[CacheEntry]:
        """Retrieve a response from cache if it exists and isn't expired"""
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        current_time = time.time()
        
        # Get TTL for this path (use default if not specified)
        ttl = self.ttl_overrides.get(path, self.ttl_default)
        
        # Check if entry is expired
        if current_time - entry.timestamp > ttl:
            # Remove expired entry
            del self.cache[cache_key]
            return None
        
        return entry
    
    async def _cache_response(self, cache_key: str, response: Response, path: str):
        """Cache a response"""
        # Read the response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
            
        # Create cache entry
        entry = CacheEntry(
            response_body=response_body,
            headers=dict(response.headers),
            status_code=response.status_code
        )
        
        # Store in cache
        self.cache[cache_key] = entry
        
        # If cache is too large, remove oldest entries
        if len(self.cache) > self.max_cache_size:
            self._trim_cache()
        
        # Recreate the response since we consumed the body iterator
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    
    def _trim_cache(self):
        """Remove oldest entries when cache exceeds max size"""
        # Sort entries by timestamp
        sorted_entries = sorted(
            [(k, v.timestamp) for k, v in self.cache.items()],
            key=lambda x: x[1]
        )
        
        # Remove oldest 10% of entries
        entries_to_remove = int(len(sorted_entries) * 0.1)
        for i in range(entries_to_remove):
            if i < len(sorted_entries):
                del self.cache[sorted_entries[i][0]]
    
    def _cleanup_cache(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            # Find the path for this cache entry
            path = None
            for p in self.cacheable_paths:
                if key.startswith(p):
                    path = p
                    break
            
            if not path:
                # If we can't determine the path, use default TTL
                ttl = self.ttl_default
            else:
                # Use path-specific TTL or default
                ttl = self.ttl_overrides.get(path, self.ttl_default)
            
            # Check if expired
            if current_time - entry.timestamp > ttl:
                keys_to_remove.append(key)
        
        # Remove expired entries
        for key in keys_to_remove:
            del self.cache[key]
            
        logger.info(f"Cache cleanup: removed {len(keys_to_remove)} expired entries.")