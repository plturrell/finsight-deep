"""Error handling and resilience patterns for document management."""

import asyncio
import logging
from typing import TypeVar, Callable, Optional, Any, Dict
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
import traceback
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class DocumentError(Exception):
    """Base exception for document management errors."""
    pass


class UploadError(DocumentError):
    """Error during document upload."""
    pass


class ProcessingError(DocumentError):
    """Error during document processing."""
    pass


class StorageError(DocumentError):
    """Error with document storage."""
    pass


class EmbeddingError(DocumentError):
    """Error generating embeddings."""
    pass


class JenaError(DocumentError):
    """Error with Jena operations."""
    pass


class AuthenticationError(DocumentError):
    """Authentication related error."""
    pass


class ValidationError(DocumentError):
    """Validation error."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


# Global circuit breakers for external services
circuit_breakers = {
    "jena": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
    "nvidia_rag": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "embeddings": CircuitBreaker(failure_threshold=4, recovery_timeout=45),
}


def with_circuit_breaker(service_name: str):
    """Decorator to apply circuit breaker to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(service_name)
            if breaker:
                return await breaker.call(func, *args, **kwargs)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_retry(
    *,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    exceptions=(Exception,),
    before=None,
    after=None
):
    """Decorator for retry logic with configurable parameters."""
    def decorator(func):
        @wraps(func)
        @retry(
            stop=stop,
            wait=wait,
            retry=retry_if_exception_type(exceptions),
            before=before or before_log(logger, logging.DEBUG),
            after=after or after_log(logger, logging.DEBUG)
        )
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @wraps(func)
        @retry(
            stop=stop,
            wait=wait,
            retry=retry_if_exception_type(exceptions),
            before=before or before_log(logger, logging.DEBUG),
            after=after or after_log(logger, logging.DEBUG)
        )
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class ErrorHandler:
    """Centralized error handling with logging and metrics."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, Dict[str, Any]] = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str = "ERROR"
    ) -> Dict[str, Any]:
        """Handle and log error with context."""
        error_type = type(error).__name__
        error_key = f"{error_type}:{context.get('operation', 'unknown')}"
        
        # Increment error count
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create error details
        error_details = {
            "error_type": error_type,
            "message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "count": self.error_counts[error_key],
            "traceback": traceback.format_exc()
        }
        
        # Store last error
        self.last_errors[error_key] = error_details
        
        # Log error
        logger.error(
            f"{error_type} in {context.get('operation', 'unknown')}: {error}",
            extra=error_details
        )
        
        # Convert to user-friendly message
        user_message = self._get_user_message(error)
        
        return {
            "error": user_message,
            "error_code": error_key,
            "details": context,
            "timestamp": error_details["timestamp"]
        }
    
    def _get_user_message(self, error: Exception) -> str:
        """Convert exception to user-friendly message."""
        error_messages = {
            UploadError: "Failed to upload document. Please try again.",
            ProcessingError: "Error processing document. Please check the file format.",
            StorageError: "Storage error occurred. Please contact support.",
            EmbeddingError: "Failed to generate embeddings. Service temporarily unavailable.",
            JenaError: "Database error occurred. Please try again later.",
            AuthenticationError: "Authentication failed. Please check your credentials.",
            ValidationError: str(error),  # Return actual validation message
        }
        
        error_type = type(error)
        if error_type in error_messages:
            return error_messages[error_type]
        
        # Generic message for unknown errors
        return "An unexpected error occurred. Please try again later."
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts,
            "recent_errors": list(self.last_errors.values())[-10:]  # Last 10 errors
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_document_errors(operation: str):
    """Decorator to handle document-related errors."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except DocumentError as e:
                # Handle known document errors
                context = {
                    "operation": operation,
                    "args": str(args)[:100],  # Limit args length
                    "error_type": type(e).__name__
                }
                error_response = error_handler.handle_error(e, context)
                raise DocumentError(error_response["error"])
            except Exception as e:
                # Handle unexpected errors
                context = {
                    "operation": operation,
                    "args": str(args)[:100],
                    "error_type": "Unexpected"
                }
                error_response = error_handler.handle_error(e, context, severity="CRITICAL")
                raise DocumentError(error_response["error"])
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DocumentError as e:
                context = {
                    "operation": operation,
                    "args": str(args)[:100],
                    "error_type": type(e).__name__
                }
                error_response = error_handler.handle_error(e, context)
                raise DocumentError(error_response["error"])
            except Exception as e:
                context = {
                    "operation": operation,
                    "args": str(args)[:100],
                    "error_type": "Unexpected"
                }
                error_response = error_handler.handle_error(e, context, severity="CRITICAL")
                raise DocumentError(error_response["error"])
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, list] = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        if key not in self.calls:
            self.calls[key] = []
        
        # Remove old calls
        self.calls[key] = [
            call_time for call_time in self.calls[key]
            if now - call_time < timedelta(seconds=self.time_window)
        ]
        
        # Check if allowed
        if len(self.calls[key]) < self.max_calls:
            self.calls[key].append(now)
            return True
        
        return False


# Global rate limiters
rate_limiters = {
    "upload": RateLimiter(max_calls=100, time_window=60),  # 100 uploads per minute
    "search": RateLimiter(max_calls=300, time_window=60),  # 300 searches per minute
    "crawl": RateLimiter(max_calls=10, time_window=300),   # 10 crawls per 5 minutes
}


def with_rate_limit(limit_type: str):
    """Decorator to apply rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            # Get user ID or IP for rate limiting
            user_id = getattr(request.state, "user_id", None) or request.client.host
            
            limiter = rate_limiters.get(limit_type)
            if limiter and not limiter.is_allowed(user_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


# Timeout decorator
def with_timeout(seconds: int):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise DocumentError(f"Operation timed out after {seconds} seconds")
        return wrapper
    return decorator


# Health check for circuit breakers
def get_circuit_status() -> Dict[str, str]:
    """Get status of all circuit breakers."""
    return {
        name: breaker.state.value
        for name, breaker in circuit_breakers.items()
    }


# Graceful degradation helper
class GracefulDegradation:
    """Helper for graceful degradation of features."""
    
    @staticmethod
    async def with_fallback(
        primary_func: Callable,
        fallback_func: Callable,
        *args,
        **kwargs
    ):
        """Execute primary function with fallback on failure."""
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}. Using fallback.")
            return await fallback_func(*args, **kwargs)