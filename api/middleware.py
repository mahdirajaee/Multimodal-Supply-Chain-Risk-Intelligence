#!/usr/bin/env python3
"""
Custom Middleware for Enhanced API Server
Includes rate limiting, logging, metrics, and security middleware
"""

import time
import uuid
import logging
from typing import Dict, Any, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
RATE_LIMIT_EXCEEDED = Counter('rate_limit_exceeded_total', 'Rate limit exceeded count')

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with sliding window"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(lambda: deque())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        now = time.time()
        
        # Clean old entries
        client_calls = self.clients[client_ip]
        while client_calls and client_calls[0] < now - self.period:
            client_calls.popleft()
        
        # Check rate limit
        if len(client_calls) >= self.calls:
            RATE_LIMIT_EXCEEDED.inc()
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Max {self.calls} requests per {self.period} seconds",
                    "retry_after": self.period
                }
            )
        
        # Add current request
        client_calls.append(now)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(self.calls - len(client_calls))
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware with correlation IDs"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Log request
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        logger.info(
            f"Request started [{correlation_id}] {request.method} {request.url.path} "
            f"from {client_ip}"
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(
                f"Request completed [{correlation_id}] {response.status_code} "
                f"in {duration:.3f}s"
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed [{correlation_id}] after {duration:.3f}s: {str(e)}"
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extract endpoint for metrics
        endpoint = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
            REQUEST_DURATION.observe(duration)
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
            REQUEST_DURATION.observe(duration)
            raise

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers and validation middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Validate request size
        if hasattr(request, 'headers'):
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "message": "Request payload too large. Maximum size is 10MB"
                    }
                )
        
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Health check bypass middleware for monitoring"""
    
    def __init__(self, app):
        super().__init__(app)
        self.health_endpoints = {"/health", "/metrics", "/ready", "/live"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Bypass rate limiting and auth for health checks
        if request.url.path in self.health_endpoints:
            request.state.bypass_rate_limit = True
            request.state.bypass_auth = True
        
        return await call_next(request)

class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add compression hint
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" in accept_encoding and response.headers.get("content-type", "").startswith("application/json"):
            response.headers["Content-Encoding"] = "gzip"
        
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation and sanitization middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.blocked_user_agents = ["bot", "crawler", "spider"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check user agent
        user_agent = request.headers.get("user-agent", "").lower()
        for blocked_agent in self.blocked_user_agents:
            if blocked_agent in user_agent:
                logger.warning(f"Blocked request from user agent: {user_agent}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "forbidden",
                        "message": "Access denied"
                    }
                )
        
        # Validate content type for POST requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "unsupported_media_type",
                        "message": "Content-Type must be application/json"
                    }
                )
        
        return await call_next(request)
