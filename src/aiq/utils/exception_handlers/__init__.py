"""
Exception handlers and error management utilities for AIQToolkit.

This module provides centralized error handling capabilities including:
- Custom exception types for different error categories
- Error tracking and analytics
- Consistent error response formatting
- Automatic error recovery strategies
"""

from typing import Type, Optional, Dict, Any, List, Callable
import traceback
import logging
import json
from datetime import datetime
from functools import wraps
import asyncio

from aiq.data_models.common import BaseModel


logger = logging.getLogger(__name__)


# Base exception classes
class AIQToolkitError(Exception):
    """Base exception class for all AIQToolkit errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.error_code,
            "message": str(self),
            "details": self.details,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
        }


# Specific exception types
class ValidationError(AIQToolkitError):
    """Raised when input validation fails"""
    pass


class ConfigurationError(AIQToolkitError):
    """Raised when configuration is invalid"""
    pass


class AuthenticationError(AIQToolkitError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(AIQToolkitError):
    """Raised when authorization fails"""
    pass


class ResourceNotFoundError(AIQToolkitError):
    """Raised when a requested resource is not found"""
    pass


class RateLimitError(AIQToolkitError):
    """Raised when rate limit is exceeded"""
    pass


class ExternalServiceError(AIQToolkitError):
    """Raised when external service call fails"""
    pass


class TimeoutError(AIQToolkitError):
    """Raised when operation times out"""
    pass


# Error tracking
class ErrorTracker:
    """Tracks and analyzes errors for monitoring"""
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
    
    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Track an error occurrence"""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        self.errors.append(error_info)
        self.error_counts[error_info["type"]] = self.error_counts.get(error_info["type"], 0) + 1
        
        # Log the error
        logger.error(f"Error tracked: {error_info['type']} - {error_info['message']}", extra=error_info)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors"""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "recent_errors": self.errors[-10:]  # Last 10 errors
        }


# Global error tracker instance
error_tracker = ErrorTracker()


# Decorators for error handling
def handle_errors(
    exceptions_to_catch: tuple = (Exception,),
    default_return: Any = None,
    reraise: bool = False,
    log_level: int = logging.ERROR
):
    """
    Decorator to handle errors consistently across functions.
    
    Args:
        exceptions_to_catch: Tuple of exception types to catch
        default_return: Default value to return on error
        reraise: Whether to re-raise the exception after handling
        log_level: Logging level for errors
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions_to_catch as e:
                # Track error
                error_tracker.track_error(e, {
                    "function": func.__name__,
                    "args": str(args)[:100],  # Truncate for logging
                    "kwargs": str(kwargs)[:100]
                })
                
                # Log error
                logger.log(log_level, f"Error in {func.__name__}: {str(e)}")
                
                # Convert to AIQToolkit error if needed
                if not isinstance(e, AIQToolkitError):
                    e = AIQToolkitError(
                        message=str(e),
                        error_code=f"{type(e).__name__}_in_{func.__name__}",
                        details={"original_error": type(e).__name__}
                    )
                
                if reraise:
                    raise e
                
                return default_return
        
        return wrapper
    return decorator


def async_handle_errors(
    exceptions_to_catch: tuple = (Exception,),
    default_return: Any = None,
    reraise: bool = False,
    log_level: int = logging.ERROR
):
    """Async version of handle_errors decorator"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions_to_catch as e:
                # Track error
                error_tracker.track_error(e, {
                    "function": func.__name__,
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                })
                
                # Log error
                logger.log(log_level, f"Error in {func.__name__}: {str(e)}")
                
                # Convert to AIQToolkit error if needed
                if not isinstance(e, AIQToolkitError):
                    e = AIQToolkitError(
                        message=str(e),
                        error_code=f"{type(e).__name__}_in_{func.__name__}",
                        details={"original_error": type(e).__name__}
                    )
                
                if reraise:
                    raise e
                
                return default_return
        
        return wrapper
    return decorator


# Error recovery strategies
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def can_recover(self, error: Exception) -> bool:
        """Check if this strategy can recover from the error"""
        raise NotImplementedError
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry strategy for transient errors"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def can_recover(self, error: Exception) -> bool:
        """Check if error is retryable"""
        return isinstance(error, (TimeoutError, ExternalServiceError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry the operation"""
        if "retry_count" not in context:
            context["retry_count"] = 0
        
        if context["retry_count"] >= self.max_retries:
            raise error
        
        context["retry_count"] += 1
        return "retry"


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback to alternative implementation"""
    
    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func
    
    def can_recover(self, error: Exception) -> bool:
        """Check if fallback is available"""
        return True
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Execute fallback function"""
        return self.fallback_func(**context.get("kwargs", {}))


# HTTP error response formatter
def format_error_response(error: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Format error for HTTP API response"""
    if isinstance(error, AIQToolkitError):
        response = error.to_dict()
    else:
        response = {
            "error": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    if request_id:
        response["request_id"] = request_id
    
    return response


# Error context manager
class ErrorContext:
    """Context manager for consistent error handling"""
    
    def __init__(self, operation_name: str, raise_on_error: bool = True):
        self.operation_name = operation_name
        self.raise_on_error = raise_on_error
        self.start_time = None
        self.error = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_val:
            self.error = exc_val
            error_tracker.track_error(exc_val, {
                "operation": self.operation_name,
                "duration": duration
            })
            logger.error(f"Operation {self.operation_name} failed after {duration}s: {exc_val}")
            
            if not self.raise_on_error:
                return True  # Suppress exception
        else:
            logger.info(f"Operation {self.operation_name} completed successfully in {duration}s")
        
        return False


# Validation helpers
def validate_input(data: Dict[str, Any], schema: Type[BaseModel]) -> BaseModel:
    """Validate input data against a schema"""
    try:
        return schema(**data)
    except Exception as e:
        raise ValidationError(
            message="Input validation failed",
            details={"error": str(e), "data": data}
        )


# Export public interface
__all__ = [
    # Exception classes
    "AIQToolkitError",
    "ValidationError",
    "ConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "RateLimitError",
    "ExternalServiceError",
    "TimeoutError",
    
    # Error handling utilities
    "error_tracker",
    "handle_errors",
    "async_handle_errors",
    "format_error_response",
    "ErrorContext",
    "validate_input",
    
    # Recovery strategies
    "ErrorRecoveryStrategy",
    "RetryStrategy",
    "FallbackStrategy"
]