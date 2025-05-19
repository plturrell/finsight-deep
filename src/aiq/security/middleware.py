"""
Security middleware for AIQToolkit API.

This module provides:
- Security headers
- CORS configuration
- Request validation
- IP whitelisting/blacklisting
- Request sanitization
"""

import time
import json
import hashlib
from typing import List, Optional, Set
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from aiq.settings.security_config import get_security_config
from aiq.security.audit_logger import (
    log_security_event,
    SecurityEventType,
    SeverityLevel
)
from aiq.security.rate_limiter import RateLimitMiddleware, rate_limiter
from aiq.utils.exception_handlers import (
    ErrorContext,
    format_error_response
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    def __init__(self, app):
        super().__init__(app)
        self.config = get_security_config()
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        if self.config.enable_security_headers:
            # Basic security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # HSTS
            if self.config.enable_https:
                response.headers["Strict-Transport-Security"] = (
                    f"max-age={self.config.hsts_max_age}; includeSubDomains"
                )
            
            # CSP
            if self.config.csp_policy:
                response.headers["Content-Security-Policy"] = self.config.csp_policy
            else:
                response.headers["Content-Security-Policy"] = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: https:; "
                    "font-src 'self' data:; "
                    "connect-src 'self' https:;"
                )
            
            # Remove server header
            response.headers.pop("Server", None)
        
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize incoming requests"""
    
    def __init__(self, app):
        super().__init__(app)
        self.max_content_length = 10 * 1024 * 1024  # 10MB
        self.suspicious_patterns = [
            "script>", "<script",
            "javascript:", "vbscript:",
            "onload=", "onerror=",
            "../", "..\\",
            "union select", "drop table"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            log_security_event(
                event_type=SecurityEventType.SECURITY_ALERT,
                action="request_rejected",
                result="content_too_large",
                severity=SeverityLevel.WARNING,
                ip_address=request.client.host,
                details={"content_length": content_length}
            )
            return JSONResponse(
                status_code=413,
                content={"error": "Request entity too large"}
            )
        
        # Check for suspicious patterns in URL
        url_str = str(request.url)
        for pattern in self.suspicious_patterns:
            if pattern.lower() in url_str.lower():
                log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    action="request_rejected",
                    result="suspicious_pattern",
                    severity=SeverityLevel.WARNING,
                    ip_address=request.client.host,
                    details={"pattern": pattern, "url": url_str}
                )
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid request"}
                )
        
        # Check body for suspicious content (for POST/PUT)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8')
                    for pattern in self.suspicious_patterns:
                        if pattern.lower() in body_str.lower():
                            log_security_event(
                                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                                action="request_rejected",
                                result="suspicious_body_content",
                                severity=SeverityLevel.WARNING,
                                ip_address=request.client.host,
                                details={"pattern": pattern}
                            )
                            return JSONResponse(
                                status_code=400,
                                content={"error": "Invalid request content"}
                            )
                    
                    # Restore body for downstream processing
                    request._body = body
            except Exception:
                pass
        
        response = await call_next(request)
        return response


class IPFilterMiddleware(BaseHTTPMiddleware):
    """IP whitelist/blacklist middleware"""
    
    def __init__(
        self,
        app,
        whitelist: Optional[Set[str]] = None,
        blacklist: Optional[Set[str]] = None
    ):
        super().__init__(app)
        self.whitelist = whitelist or set()
        self.blacklist = blacklist or set()
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else None
        
        if client_ip:
            # Check blacklist first
            if client_ip in self.blacklist:
                log_security_event(
                    event_type=SecurityEventType.SECURITY_ALERT,
                    action="request_blocked",
                    result="blacklisted_ip",
                    severity=SeverityLevel.WARNING,
                    ip_address=client_ip
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied"}
                )
            
            # Check whitelist if configured
            if self.whitelist and client_ip not in self.whitelist:
                log_security_event(
                    event_type=SecurityEventType.SECURITY_ALERT,
                    action="request_blocked",
                    result="not_whitelisted_ip",
                    severity=SeverityLevel.WARNING,
                    ip_address=client_ip
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied"}
                )
        
        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests for audit purposes"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Generate request ID
        request_id = hashlib.md5(
            f"{time.time()}{request.client.host}".encode()
        ).hexdigest()[:16]
        
        # Add request ID to state
        request.state.request_id = request_id
        
        # Extract request details
        details = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get user info if available
        user_id = getattr(request.state, "user_id", None)
        
        # Log request
        log_security_event(
            event_type=SecurityEventType.API_CALL,
            action="request_received",
            result="processing",
            user_id=user_id,
            ip_address=request.client.host if request.client else None,
            resource=request.url.path,
            details=details
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            log_security_event(
                event_type=SecurityEventType.API_CALL,
                action="request_completed",
                result="success",
                user_id=user_id,
                ip_address=request.client.host if request.client else None,
                resource=request.url.path,
                details={
                    **details,
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            log_security_event(
                event_type=SecurityEventType.API_CALL,
                action="request_failed",
                result="error",
                severity=SeverityLevel.ERROR,
                user_id=user_id,
                ip_address=request.client.host if request.client else None,
                resource=request.url.path,
                details={
                    **details,
                    "error": str(e),
                    "duration": duration
                },
                error_message=str(e)
            )
            raise


def setup_security_middleware(app: FastAPI):
    """Configure all security middleware for the application"""
    config = get_security_config()
    
    # CORS
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Trusted hosts
    if config.enable_https:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure as needed
        )
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request validation
    app.add_middleware(RequestValidationMiddleware)
    
    # Rate limiting
    if config.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=rate_limiter
        )
    
    # Request logging
    if config.enable_audit_logging:
        app.add_middleware(RequestLoggingMiddleware)
    
    # IP filtering (if configured)
    # This would be configured based on environment
    # app.add_middleware(
    #     IPFilterMiddleware,
    #     whitelist=set(config.ip_whitelist) if config.ip_whitelist else None,
    #     blacklist=set(config.ip_blacklist) if config.ip_blacklist else None
    # )


# Export public interface
__all__ = [
    "SecurityHeadersMiddleware",
    "RequestValidationMiddleware",
    "IPFilterMiddleware",
    "RequestLoggingMiddleware",
    "setup_security_middleware"
]