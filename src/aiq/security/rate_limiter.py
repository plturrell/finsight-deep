"""
Rate limiting module for API security.

This module provides:
- Token bucket rate limiting
- Sliding window rate limiting
- Distributed rate limiting support
- Configurable limits per endpoint/user
- Automatic cleanup of expired data
"""

import time
import asyncio
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import json

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from aiq.utils.exception_handlers import (
    RateLimitError,
    handle_errors,
    async_handle_errors
)
from aiq.security.audit_logger import (
    log_security_event,
    SecurityEventType,
    SeverityLevel
)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds
    burst: Optional[int] = None  # Burst capacity for token bucket
    key_func: Optional[Callable[[Request], str]] = None  # Function to extract rate limit key


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float, burst: Optional[int] = None):
        self.capacity = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.burst = burst or capacity
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        with self.lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens are available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed = tokens - self.tokens
            return needed / self.refill_rate


class SlidingWindow:
    """Sliding window implementation for rate limiting"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # Window size in seconds
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def check_and_update(self) -> bool:
        """Check if request is allowed and update window"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.window_size:
                self.requests.popleft()
            
            # Check if we can accept new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def get_wait_time(self) -> float:
        """Get wait time until next request is allowed"""
        with self.lock:
            if not self.requests:
                return 0.0
            
            if len(self.requests) < self.max_requests:
                return 0.0
            
            # Wait until the oldest request expires
            oldest = self.requests[0]
            wait_time = (oldest + self.window_size) - time.time()
            return max(0.0, wait_time)


class RateLimiter:
    """Main rate limiter class"""
    
    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        storage_backend: Optional[Any] = None  # Redis, memcached, etc.
    ):
        self.default_config = default_config or RateLimitConfig(
            requests=100,
            window=60
        )
        self.storage_backend = storage_backend
        
        # In-memory storage
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindow] = {}
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}
        
        # Cleanup task
        self.cleanup_interval = 300  # 5 minutes
        self.cleanup_task = None
    
    async def start(self):
        """Start the rate limiter"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the rate limiter"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    def configure_endpoint(
        self,
        endpoint: str,
        config: RateLimitConfig
    ):
        """Configure rate limit for specific endpoint"""
        self.endpoint_configs[endpoint] = config
    
    @async_handle_errors(reraise=True)
    async def check_rate_limit(
        self,
        request: Request,
        endpoint: Optional[str] = None,
        key: Optional[str] = None
    ) -> bool:
        """Check if request is within rate limit"""
        # Get configuration
        config = self._get_config(endpoint)
        
        # Get rate limit key
        if not key:
            key = self._get_key(request, config)
        
        # Use distributed storage if available
        if self.storage_backend:
            return await self._check_distributed(key, config)
        
        # Use in-memory storage
        return self._check_local(key, config)
    
    def _get_config(self, endpoint: Optional[str]) -> RateLimitConfig:
        """Get rate limit configuration for endpoint"""
        if endpoint and endpoint in self.endpoint_configs:
            return self.endpoint_configs[endpoint]
        return self.default_config
    
    def _get_key(self, request: Request, config: RateLimitConfig) -> str:
        """Get rate limit key from request"""
        if config.key_func:
            return config.key_func(request)
        
        # Default: use IP address
        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        
        if user_id:
            return f"user:{user_id}"
        return f"ip:{client_ip}"
    
    def _check_local(self, key: str, config: RateLimitConfig) -> bool:
        """Check rate limit using local storage"""
        # Use token bucket if burst is configured
        if config.burst:
            bucket_key = f"bucket:{key}"
            
            if bucket_key not in self.token_buckets:
                refill_rate = config.requests / config.window
                self.token_buckets[bucket_key] = TokenBucket(
                    capacity=config.requests,
                    refill_rate=refill_rate,
                    burst=config.burst
                )
            
            bucket = self.token_buckets[bucket_key]
            if not bucket.consume():
                wait_time = bucket.get_wait_time()
                self._log_rate_limit_exceeded(key, config, wait_time)
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {wait_time:.1f} seconds",
                    details={"retry_after": wait_time}
                )
            
            return True
        
        # Use sliding window
        window_key = f"window:{key}"
        
        if window_key not in self.sliding_windows:
            self.sliding_windows[window_key] = SlidingWindow(
                window_size=config.window,
                max_requests=config.requests
            )
        
        window = self.sliding_windows[window_key]
        if not window.check_and_update():
            wait_time = window.get_wait_time()
            self._log_rate_limit_exceeded(key, config, wait_time)
            raise RateLimitError(
                f"Rate limit exceeded. Retry after {wait_time:.1f} seconds",
                details={"retry_after": wait_time}
            )
        
        return True
    
    async def _check_distributed(
        self,
        key: str,
        config: RateLimitConfig
    ) -> bool:
        """Check rate limit using distributed storage"""
        # This would be implemented based on the storage backend
        # For now, fall back to local check
        return self._check_local(key, config)
    
    def _log_rate_limit_exceeded(
        self,
        key: str,
        config: RateLimitConfig,
        wait_time: float
    ):
        """Log rate limit exceeded event"""
        log_security_event(
            event_type=SecurityEventType.SECURITY_ALERT,
            action="rate_limit_exceeded",
            result="blocked",
            severity=SeverityLevel.WARNING,
            details={
                "key": key,
                "limit": config.requests,
                "window": config.window,
                "wait_time": wait_time
            }
        )
    
    async def _cleanup_loop(self):
        """Periodically clean up old rate limit data"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Error in rate limiter cleanup: {e}")
    
    async def _cleanup(self):
        """Clean up expired rate limit data"""
        # Clean up token buckets that haven't been used
        now = time.time()
        buckets_to_remove = []
        
        for key, bucket in self.token_buckets.items():
            if now - bucket.last_refill > self.cleanup_interval:
                buckets_to_remove.append(key)
        
        for key in buckets_to_remove:
            del self.token_buckets[key]
        
        # Clean up sliding windows
        windows_to_remove = []
        
        for key, window in self.sliding_windows.items():
            with window.lock:
                if not window.requests:
                    windows_to_remove.append(key)
        
        for key in windows_to_remove:
            del self.sliding_windows[key]


# Global rate limiter instance
rate_limiter = RateLimiter()


# FastAPI middleware
class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        # Get endpoint path
        endpoint = request.url.path
        
        try:
            # Check rate limit
            await self.rate_limiter.check_rate_limit(request, endpoint)
        except RateLimitError as e:
            return HTTPException(
                status_code=429,
                detail=str(e),
                headers={"Retry-After": str(int(e.details.get("retry_after", 60)))}
            )
        
        # Continue with request
        response = await call_next(request)
        return response


# Decorator for rate limiting
def rate_limit(
    requests: int = 100,
    window: int = 60,
    burst: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for rate limiting endpoints"""
    config = RateLimitConfig(
        requests=requests,
        window=window,
        burst=burst,
        key_func=key_func
    )
    
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            await rate_limiter.check_rate_limit(request, key=None)
            return await func(request, *args, **kwargs)
        
        # Store config for the endpoint
        endpoint = f"{func.__module__}.{func.__name__}"
        rate_limiter.configure_endpoint(endpoint, config)
        
        return wrapper
    return decorator


# Export public interface
__all__ = [
    "RateLimiter",
    "RateLimitConfig",
    "TokenBucket",
    "SlidingWindow",
    "RateLimitMiddleware",
    "rate_limit",
    "rate_limiter"
]