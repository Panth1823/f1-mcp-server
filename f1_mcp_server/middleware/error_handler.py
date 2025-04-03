"""
Error handling middleware for the F1 MCP Server
"""

import logging
import traceback
import time
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for consistent error handling across the application.
    
    This middleware catches exceptions and formats them into consistent JSON responses.
    It also logs errors appropriately based on their severity.
    """
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(int(time.time() * 1000))
        start_time = time.time()
        
        # Add request_id to request state for logging
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            # Known HTTP exceptions (e.g. raised by our code)
            logger.warning(
                f"Request {request_id} failed with HTTP error {exc.status_code}: {exc.detail}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content=self._format_error_response(
                    error_code=f"HTTP_{exc.status_code}",
                    message=str(exc.detail),
                    request_id=request_id
                )
            )
            
        except RequestValidationError as exc:
            # FastAPI validation errors
            logger.warning(
                f"Request {request_id} failed validation: {str(exc)}"
            )
            return JSONResponse(
                status_code=422,
                content=self._format_error_response(
                    error_code="VALIDATION_ERROR",
                    message="Request validation failed",
                    request_id=request_id,
                    details=exc.errors()
                )
            )
            
        except Exception as exc:
            # Unexpected errors
            request_duration = time.time() - start_time
            
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path} "
                f"[ID: {request_id}, Duration: {request_duration:.3f}s]: {str(exc)}",
                exc_info=True
            )
            
            # In development, include the traceback
            tb = traceback.format_exc() if logger.level <= logging.DEBUG else None
            
            return JSONResponse(
                status_code=500,
                content=self._format_error_response(
                    error_code="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred",
                    request_id=request_id,
                    traceback=tb
                )
            )
    
    def _format_error_response(
        self, 
        error_code: str, 
        message: str, 
        request_id: str, 
        details: Any = None,
        traceback: str = None
    ) -> Dict[str, Any]:
        """Format a consistent error response"""
        response = {
            "error": {
                "code": error_code,
                "message": message,
                "request_id": request_id
            },
            "success": False
        }
        
        # Add details if provided
        if details:
            response["error"]["details"] = details
            
        # Add traceback in debug mode only
        if traceback and logger.level <= logging.DEBUG:
            response["error"]["traceback"] = traceback
            
        return response
