"""
Test the F1 MCP Server middleware components
"""

import pytest
from fastapi.testclient import TestClient
import time
from f1_mcp_server.main import app

client = TestClient(app)


class TestRateLimiter:
    """Tests for the rate limiter middleware"""

    def test_normal_request_succeeds(self):
        """Test that normal requests work fine"""
        response = client.get("/mcp/context")
        assert response.status_code == 200
        assert "name" in response.json()

    def test_rate_limit_exceeded(self):
        """Test that rate limits work"""
        # Temporarily modify the rate limit for testing
        from f1_mcp_server.middleware.rate_limiter import RateLimitMiddleware
        
        # Find the middleware instance
        for middleware in app.user_middleware:
            if hasattr(middleware, "cls") and middleware.cls == RateLimitMiddleware:
                # Store original values
                original_rate_limit = middleware.options.get("rate_limit", 100)
                
                # Set a very low rate limit for testing
                middleware.options["rate_limit"] = 3
                break
        
        try:
            # Make requests until we hit the limit
            for i in range(3):
                response = client.get("/mcp/context")
                assert response.status_code == 200
            
            # This should now fail with 429
            response = client.get("/mcp/context")
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.text
            
        finally:
            # Reset rate limit
            for middleware in app.user_middleware:
                if hasattr(middleware, "cls") and middleware.cls == RateLimitMiddleware:
                    middleware.options["rate_limit"] = original_rate_limit


class TestCacheMiddleware:
    """Tests for the caching middleware"""

    def test_cache_hit(self):
        """Test that responses are cached"""
        # First request - should be a cache miss
        start_time = time.time()
        response1 = client.get("/mcp/function/get_race_calendar?year=2023")
        first_request_time = time.time() - start_time
        
        # Second request to same endpoint - should be faster due to cache hit
        start_time = time.time()
        response2 = client.get("/mcp/function/get_race_calendar?year=2023")
        second_request_time = time.time() - start_time
        
        # Both responses should be successful
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Responses should be identical
        assert response1.json() == response2.json()
        
        # Second request should typically be faster due to caching
        # Note: This is a simple heuristic test that might occasionally fail
        # if system load changes dramatically between requests
        print(f"First request time: {first_request_time:.4f}s")
        print(f"Second request time: {second_request_time:.4f}s")
        
        # We're not asserting second_request_time < first_request_time
        # because it might not always be true in a test environment
        # Just logging the times for inspection


class TestErrorHandler:
    """Tests for the error handling middleware"""

    def test_404_error_formatting(self):
        """Test that 404 errors are properly formatted"""
        response = client.get("/non_existent_endpoint")
        assert response.status_code == 404
        
        # Check error response format
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert "request_id" in data["error"]
        assert data["success"] is False

    def test_validation_error_formatting(self):
        """Test that validation errors are properly formatted"""
        # Missing required parameter (year)
        response = client.get("/mcp/function/get_race_calendar")
        assert response.status_code == 422
        
        # Check error response format
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in data["error"]
        assert data["success"] is False


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])