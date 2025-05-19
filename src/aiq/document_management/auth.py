"""Authentication and authorization for document management system."""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import bcrypt
import redis
from functools import wraps

from ..settings.global_settings import GlobalSettings

logger = logging.getLogger(__name__)

# Security settings
security = HTTPBearer()
settings = GlobalSettings()

# JWT settings
JWT_SECRET = settings.get("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
REFRESH_TOKEN_EXPIRATION_DAYS = 30

# Redis for token blacklist
redis_client = redis.Redis(
    host=settings.get("REDIS_HOST", "localhost"),
    port=settings.get("REDIS_PORT", 6379),
    db=1,  # Use DB 1 for auth
    decode_responses=True
)


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    tenant_id: Optional[str] = None


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # Subject (user ID)
    exp: datetime
    iat: datetime
    roles: List[str]
    permissions: List[str]
    tenant_id: Optional[str] = None


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        self.secret_key = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        expire = now + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        payload = {
            "sub": user.username,
            "exp": expire,
            "iat": now,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions,
            "tenant_id": user.tenant_id
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRATION_DAYS)
        
        payload = {
            "sub": user.username,
            "exp": expire,
            "iat": now,
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token in Redis
        redis_client.setex(
            f"refresh_token:{user.username}",
            REFRESH_TOKEN_EXPIRATION_DAYS * 24 * 3600,
            token
        )
        
        return token
    
    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            if redis_client.get(f"blacklist:{token}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str):
        """Revoke a token by adding it to blacklist."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp")
            
            if exp:
                # Calculate remaining TTL
                ttl = exp - datetime.utcnow().timestamp()
                if ttl > 0:
                    redis_client.setex(f"blacklist:{token}", int(ttl), "1")
                    
        except jwt.InvalidTokenError:
            pass
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )


# Global auth manager instance
auth_manager = AuthManager()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """Get current authenticated user from token."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    # Here you would typically fetch user from database
    # For now, create from token payload
    return User(
        username=payload.sub,
        email=payload.email if hasattr(payload, 'email') else f"{payload.sub}@example.com",
        roles=payload.roles,
        permissions=payload.permissions,
        tenant_id=payload.tenant_id
    )


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
            if permission not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_role(role: str):
    """Decorator to require specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
            if role not in user.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


# Permission constants
class Permissions:
    """Document management permissions."""
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_ADMIN = "document:admin"
    
    RESEARCH_READ = "research:read"
    RESEARCH_WRITE = "research:write"
    
    CRAWLER_EXECUTE = "crawler:execute"
    
    METADATA_READ = "metadata:read"
    METADATA_WRITE = "metadata:write"
    
    SEARCH_EXECUTE = "search:execute"
    SEARCH_ADMIN = "search:admin"


# Role definitions
class Roles:
    """Predefined roles."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    CRAWLER = "crawler"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    Roles.ADMIN: [
        Permissions.DOCUMENT_READ,
        Permissions.DOCUMENT_WRITE,
        Permissions.DOCUMENT_DELETE,
        Permissions.DOCUMENT_ADMIN,
        Permissions.RESEARCH_READ,
        Permissions.RESEARCH_WRITE,
        Permissions.CRAWLER_EXECUTE,
        Permissions.METADATA_READ,
        Permissions.METADATA_WRITE,
        Permissions.SEARCH_EXECUTE,
        Permissions.SEARCH_ADMIN
    ],
    Roles.RESEARCHER: [
        Permissions.DOCUMENT_READ,
        Permissions.DOCUMENT_WRITE,
        Permissions.RESEARCH_READ,
        Permissions.RESEARCH_WRITE,
        Permissions.METADATA_READ,
        Permissions.METADATA_WRITE,
        Permissions.SEARCH_EXECUTE
    ],
    Roles.VIEWER: [
        Permissions.DOCUMENT_READ,
        Permissions.RESEARCH_READ,
        Permissions.METADATA_READ,
        Permissions.SEARCH_EXECUTE
    ],
    Roles.CRAWLER: [
        Permissions.CRAWLER_EXECUTE,
        Permissions.DOCUMENT_WRITE,
        Permissions.METADATA_WRITE
    ]
}


def get_permissions_for_roles(roles: List[str]) -> List[str]:
    """Get all permissions for given roles."""
    permissions = set()
    for role in roles:
        if role in ROLE_PERMISSIONS:
            permissions.update(ROLE_PERMISSIONS[role])
    return list(permissions)


# Auth endpoints for API
async def login(username: str, password: str) -> Dict[str, str]:
    """Login endpoint logic."""
    # Here you would verify against database
    # For demo, using simple check
    
    # In production, fetch from database
    user = User(
        username=username,
        email=f"{username}@example.com",
        roles=[Roles.RESEARCHER],
        permissions=get_permissions_for_roles([Roles.RESEARCHER])
    )
    
    access_token = auth_manager.create_access_token(user)
    refresh_token = auth_manager.create_refresh_token(user)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


async def refresh(refresh_token: str) -> Dict[str, str]:
    """Refresh access token."""
    try:
        payload = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token type"
            )
        
        # Verify refresh token exists in Redis
        stored_token = redis_client.get(f"refresh_token:{payload['sub']}")
        if stored_token != refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        # In production, fetch user from database
        user = User(
            username=payload["sub"],
            email=f"{payload['sub']}@example.com",
            roles=[Roles.RESEARCHER],
            permissions=get_permissions_for_roles([Roles.RESEARCHER])
        )
        
        access_token = auth_manager.create_access_token(user)
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


async def logout(token: str, user: User = Depends(get_current_user)):
    """Logout by revoking tokens."""
    # Revoke access token
    auth_manager.revoke_token(token)
    
    # Remove refresh token
    redis_client.delete(f"refresh_token:{user.username}")
    
    return {"message": "Successfully logged out"}