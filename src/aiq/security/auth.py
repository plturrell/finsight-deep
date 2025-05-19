"""
Authentication and authorization module for AIQToolkit.

This module provides:
- JWT token-based authentication
- RBAC (Role-Based Access Control)
- API key management
- Session management
- Multi-factor authentication support
"""

import secrets
import hashlib
import time
import json
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta, timezone
from enum import Enum
import re
import base64

import jwt
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator

from aiq.settings.security_config import SecurityConfig
from aiq.utils.exception_handlers import (
    AuthenticationError,
    AuthorizationError,
    handle_errors,
    async_handle_errors
)
from aiq.security.audit_logger import (
    log_security_event,
    SecurityEventType,
    SeverityLevel
)


# Security configuration
security_config = SecurityConfig()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
JWT_SECRET = security_config.jwt_secret or secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions"""
    # Workflow permissions
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_UPDATE = "workflow:update"
    WORKFLOW_DELETE = "workflow:delete"
    WORKFLOW_EXECUTE = "workflow:execute"
    
    # API permissions
    API_FULL_ACCESS = "api:full"
    API_READ_ONLY = "api:read"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_AUDIT = "admin:audit"


# Role-permission mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: set(Permission),  # All permissions
    UserRole.USER: {
        Permission.WORKFLOW_CREATE,
        Permission.WORKFLOW_READ,
        Permission.WORKFLOW_UPDATE,
        Permission.WORKFLOW_DELETE,
        Permission.WORKFLOW_EXECUTE,
        Permission.API_FULL_ACCESS,
    },
    UserRole.SERVICE: {
        Permission.WORKFLOW_READ,
        Permission.WORKFLOW_EXECUTE,
        Permission.API_FULL_ACCESS,
    },
    UserRole.READONLY: {
        Permission.WORKFLOW_READ,
        Permission.API_READ_ONLY,
    }
}


class User(BaseModel):
    """User model"""
    user_id: str
    username: str
    email: str
    roles: List[UserRole] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    api_keys: List[str] = Field(default_factory=list)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError('Invalid email format')
        return v
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = set()
        for role in self.roles:
            user_permissions.update(ROLE_PERMISSIONS.get(role, set()))
        return permission in user_permissions


class Session(BaseModel):
    """User session model"""
    session_id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class APIKey(BaseModel):
    """API key model"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    permissions: List[Permission] = Field(default_factory=list)
    is_active: bool = True


class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.lockout_duration = timedelta(minutes=30)
        self.max_failed_attempts = 5
    
    @handle_errors(reraise=True)
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[UserRole] = None
    ) -> User:
        """Create a new user"""
        # Check if user exists
        if self._find_user_by_username(username):
            raise ValueError("Username already exists")
        
        if self._find_user_by_email(email):
            raise ValueError("Email already exists")
        
        # Create user
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            roles=roles or [UserRole.USER]
        )
        
        # Store password hash separately (in production, use proper storage)
        password_hash = pwd_context.hash(password)
        # This is a placeholder - store in database
        user._password_hash = password_hash
        
        self.users[user.user_id] = user
        
        log_security_event(
            event_type=SecurityEventType.ACCESS_CONTROL,
            action="create_user",
            result="success",
            user_id=user.user_id,
            details={"username": username, "roles": [r.value for r in user.roles]}
        )
        
        return user
    
    @async_handle_errors(reraise=True)
    async def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Authenticate user and create session"""
        # Check lockout
        if self._is_locked_out(username):
            log_security_event(
                event_type=SecurityEventType.AUTHENTICATION,
                action="login_attempt",
                result="locked_out",
                details={"username": username, "ip_address": ip_address}
            )
            raise AuthenticationError("Account is locked due to multiple failed attempts")
        
        # Find user
        user = self._find_user_by_username(username)
        if not user:
            self._record_failed_attempt(username)
            raise AuthenticationError("Invalid credentials")
        
        # Verify password (in production, retrieve hash from database)
        password_hash = getattr(user, '_password_hash', None)
        if not password_hash or not pwd_context.verify(password, password_hash):
            self._record_failed_attempt(username)
            log_security_event(
                event_type=SecurityEventType.AUTHENTICATION,
                action="login_attempt",
                result="invalid_password",
                user_id=user.user_id,
                ip_address=ip_address
            )
            raise AuthenticationError("Invalid credentials")
        
        # Check if user is active
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        # Clear failed attempts
        self.failed_attempts.pop(username, None)
        
        # Create session
        session = await self._create_session(user, ip_address, user_agent)
        
        # Generate JWT token
        token = self._generate_jwt_token(user, session)
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        
        log_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            action="login",
            result="success",
            user_id=user.user_id,
            session_id=session.session_id,
            ip_address=ip_address
        )
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRATION_MINUTES * 60,
            "user": user.dict(exclude={'mfa_secret', '_password_hash'})
        }
    
    def verify_token(self, token: str) -> User:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
            session_id = payload.get("session_id")
            
            if not user_id or not session_id:
                raise AuthenticationError("Invalid token")
            
            # Check session
            session = self.sessions.get(session_id)
            if not session or not session.is_active:
                raise AuthenticationError("Invalid or expired session")
            
            if session.expires_at < datetime.now(timezone.utc):
                session.is_active = False
                raise AuthenticationError("Session expired")
            
            # Get user
            user = self.users.get(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("Invalid user")
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def generate_api_key(
        self,
        user: User,
        name: str,
        permissions: List[Permission] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Generate API key for user"""
        # Generate key
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Create API key record
        api_key_obj = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            user_id=user.user_id,
            name=name,
            created_at=datetime.now(timezone.utc),
            permissions=permissions or [],
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_in_days) if expires_in_days else None
        )
        
        self.api_keys[api_key_obj.key_id] = api_key_obj
        user.api_keys.append(api_key_obj.key_id)
        
        log_security_event(
            event_type=SecurityEventType.ACCESS_CONTROL,
            action="generate_api_key",
            result="success",
            user_id=user.user_id,
            details={"key_name": name, "permissions": [p.value for p in permissions or []]}
        )
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> User:
        """Verify API key and return user"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Find API key
        api_key_obj = None
        for key in self.api_keys.values():
            if key.key_hash == key_hash and key.is_active:
                api_key_obj = key
                break
        
        if not api_key_obj:
            raise AuthenticationError("Invalid API key")
        
        # Check expiration
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
            api_key_obj.is_active = False
            raise AuthenticationError("API key expired")
        
        # Get user
        user = self.users.get(api_key_obj.user_id)
        if not user or not user.is_active:
            raise AuthenticationError("Invalid user")
        
        # Update last used
        api_key_obj.last_used = datetime.now(timezone.utc)
        
        return user
    
    def authorize(
        self,
        user: User,
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """Authorize user for specific permission"""
        if not user.has_permission(permission):
            log_security_event(
                event_type=SecurityEventType.AUTHORIZATION,
                action="permission_check",
                result="denied",
                user_id=user.user_id,
                resource=resource,
                details={"permission": permission.value}
            )
            raise AuthorizationError(f"User lacks permission: {permission.value}")
        
        log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="permission_check",
            result="granted",
            user_id=user.user_id,
            resource=resource,
            details={"permission": permission.value}
        )
        
        return True
    
    async def _create_session(
        self,
        user: User,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> Session:
        """Create user session"""
        session = Session(
            session_id=secrets.token_urlsafe(16),
            user_id=user.user_id,
            token=secrets.token_urlsafe(32),
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session.session_id] = session
        return session
    
    def _generate_jwt_token(self, user: User, session: Session) -> str:
        """Generate JWT token"""
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": [r.value for r in user.roles],
            "session_id": session.session_id,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES)
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _find_user_by_email(self, email: str) -> Optional[User]:
        """Find user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def _is_locked_out(self, username: str) -> bool:
        """Check if user is locked out"""
        attempts = self.failed_attempts.get(username, [])
        if len(attempts) < self.max_failed_attempts:
            return False
        
        # Check if lockout period has passed
        last_attempt = max(attempts)
        if datetime.now(timezone.utc) - last_attempt > self.lockout_duration:
            self.failed_attempts[username] = []
            return False
        
        return True
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now(timezone.utc))
        
        # Keep only recent attempts
        cutoff = datetime.now(timezone.utc) - self.lockout_duration
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff
        ]


# Global auth manager instance
auth_manager = AuthManager()


# FastAPI dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> User:
    """Get current user from JWT token"""
    try:
        user = auth_manager.verify_token(credentials.credentials)
        return user
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def get_api_key_user(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """Get user from API key"""
    if not api_key:
        return None
    
    try:
        user = auth_manager.verify_api_key(api_key)
        return user
    except AuthenticationError:
        return None


async def get_current_user_flexible(
    token_user: Optional[User] = Depends(get_current_user),
    api_key_user: Optional[User] = Depends(get_api_key_user)
) -> User:
    """Get current user from either JWT or API key"""
    user = token_user or api_key_user
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_permission(permission: Permission):
    """Require specific permission decorator"""
    async def permission_checker(user: User = Depends(get_current_user_flexible)):
        auth_manager.authorize(user, permission)
        return user
    return permission_checker


class JWTManager:
    """Simple JWT manager for basic token operations"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", expiry_minutes: int = 60):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiry_minutes = expiry_minutes
    
    def create_token(self, payload: Dict[str, Any]) -> str:
        """Create a JWT token with expiration"""
        now = datetime.now(timezone.utc)
        jwt_payload = {
            **payload,
            "iat": now,
            "exp": now + timedelta(minutes=self.expiry_minutes),
            "jti": secrets.token_urlsafe(16)
        }
        return jwt.encode(jwt_payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")


# Export public interface
__all__ = [
    "AuthManager",
    "User",
    "UserRole",
    "Permission",
    "auth_manager",
    "get_current_user",
    "get_api_key_user",
    "get_current_user_flexible",
    "require_permission",
    "JWTManager"
]