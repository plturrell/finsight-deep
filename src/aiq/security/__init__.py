"""
AIQToolkit Security Module

Provides comprehensive security features including:
- Authentication and authorization
- Rate limiting
- Audit logging
- Security middleware
- Encryption and data protection
"""

from aiq.security.auth import (
    AuthManager,
    User,
    UserRole,
    Permission,
    auth_manager,
    get_current_user,
    get_api_key_user,
    get_current_user_flexible,
    require_permission
)

from aiq.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    SlidingWindow,
    RateLimitMiddleware,
    rate_limit,
    rate_limiter
)

from aiq.security.audit_logger import (
    AuditLogger,
    ComplianceReporter,
    SecurityEventType,
    SeverityLevel,
    AuditEvent,
    audit_logger,
    log_security_event,
    async_log_security_event
)

from aiq.security.middleware import (
    SecurityHeadersMiddleware,
    RequestValidationMiddleware,
    IPFilterMiddleware,
    RequestLoggingMiddleware,
    setup_security_middleware
)


# Initialize security subsystems
async def initialize_security():
    """Initialize all security subsystems"""
    # Start audit logger
    await audit_logger.start()
    
    # Start rate limiter
    await rate_limiter.start()
    
    # Create default admin user if none exists
    if not auth_manager.users:
        try:
            auth_manager.create_user(
                username="admin",
                email="admin@aiqtoolkit.local",
                password="changeme123!",  # Should be changed on first login
                roles=[UserRole.ADMIN]
            )
            log_security_event(
                event_type=SecurityEventType.ACCESS_CONTROL,
                action="create_default_admin",
                result="success",
                severity=SeverityLevel.WARNING,
                details={"message": "Default admin user created. Please change password!"}
            )
        except Exception as e:
            log_security_event(
                event_type=SecurityEventType.ERROR,
                action="create_default_admin",
                result="failed",
                severity=SeverityLevel.ERROR,
                error_message=str(e)
            )


async def shutdown_security():
    """Shutdown security subsystems gracefully"""
    # Stop audit logger
    await audit_logger.stop()
    
    # Stop rate limiter
    await rate_limiter.stop()


# Export public interface
__all__ = [
    # Auth
    "AuthManager",
    "User",
    "UserRole",
    "Permission",
    "auth_manager",
    "get_current_user",
    "get_api_key_user",
    "get_current_user_flexible",
    "require_permission",
    
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "TokenBucket",
    "SlidingWindow",
    "RateLimitMiddleware",
    "rate_limit",
    "rate_limiter",
    
    # Audit logging
    "AuditLogger",
    "ComplianceReporter",
    "SecurityEventType",
    "SeverityLevel",
    "AuditEvent",
    "audit_logger",
    "log_security_event",
    "async_log_security_event",
    
    # Middleware
    "SecurityHeadersMiddleware",
    "RequestValidationMiddleware",
    "IPFilterMiddleware",
    "RequestLoggingMiddleware",
    "setup_security_middleware",
    
    # Initialization
    "initialize_security",
    "shutdown_security"
]