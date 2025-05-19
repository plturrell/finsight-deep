"""Security Manager for Digital Human system"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import redis
from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())

Base = declarative_base()


class UserRole(str, Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    ANALYST = "analyst"
    GUEST = "guest"


class PermissionScope(str, Enum):
    """Permission scopes for actions"""
    READ_PORTFOLIO = "read:portfolio"
    WRITE_PORTFOLIO = "write:portfolio"
    EXECUTE_TRADES = "execute:trades"
    VIEW_ANALYSIS = "view:analysis"
    MANAGE_SETTINGS = "manage:settings"
    ADMIN_USERS = "admin:users"
    ACCESS_AI = "access:ai"
    ACCESS_RESEARCH = "access:research"


@dataclass
class User:
    """User data model"""
    id: str
    email: str
    username: str
    role: UserRole
    permissions: Set[PermissionScope]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class AuthToken:
    """Authentication token model"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600
    scope: str = ""


class UserModel(Base):
    """SQLAlchemy User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default=UserRole.USER.value)
    permissions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, default=dict)
    mfa_secret = Column(String, nullable=True)
    api_keys = Column(JSON, default=list)


class APIKey(Base):
    """API Key model"""
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    key_hash = Column(String, nullable=False)
    name = Column(String, nullable=False)
    permissions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)


class SessionToken(Base):
    """Session token model"""
    __tablename__ = "session_tokens"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    token_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)


class SecurityManager:
    """Main security manager for the Digital Human system"""
    
    def __init__(self, database_url: str, redis_url: Optional[str] = None):
        # Database setup
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Encryption
        self.cipher_suite = Fernet(ENCRYPTION_KEY)
        
        # Redis for token blacklist and rate limiting
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: {
                PermissionScope.READ_PORTFOLIO,
                PermissionScope.WRITE_PORTFOLIO,
                PermissionScope.EXECUTE_TRADES,
                PermissionScope.VIEW_ANALYSIS,
                PermissionScope.MANAGE_SETTINGS,
                PermissionScope.ADMIN_USERS,
                PermissionScope.ACCESS_AI,
                PermissionScope.ACCESS_RESEARCH
            },
            UserRole.PREMIUM: {
                PermissionScope.READ_PORTFOLIO,
                PermissionScope.WRITE_PORTFOLIO,
                PermissionScope.EXECUTE_TRADES,
                PermissionScope.VIEW_ANALYSIS,
                PermissionScope.MANAGE_SETTINGS,
                PermissionScope.ACCESS_AI,
                PermissionScope.ACCESS_RESEARCH
            },
            UserRole.USER: {
                PermissionScope.READ_PORTFOLIO,
                PermissionScope.WRITE_PORTFOLIO,
                PermissionScope.VIEW_ANALYSIS,
                PermissionScope.MANAGE_SETTINGS,
                PermissionScope.ACCESS_AI
            },
            UserRole.ANALYST: {
                PermissionScope.READ_PORTFOLIO,
                PermissionScope.VIEW_ANALYSIS,
                PermissionScope.ACCESS_RESEARCH
            },
            UserRole.GUEST: {
                PermissionScope.VIEW_ANALYSIS
            }
        }
    
    async def create_tables(self):
        """Create database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    # User Management
    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.USER,
        metadata: Optional[Dict] = None
    ) -> User:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        password_hash = self.pwd_context.hash(password)
        
        async with self.async_session() as session:
            user = UserModel(
                id=user_id,
                email=email,
                username=username,
                password_hash=password_hash,
                role=role.value,
                permissions=list(self.role_permissions[role]),
                created_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            session.add(user)
            await session.commit()
            
            return self._model_to_user(user)
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        async with self.async_session() as session:
            result = await session.execute(
                select(UserModel).where(
                    (UserModel.username == username) | (UserModel.email == username)
                )
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model:
                return None
            
            if not self.pwd_context.verify(password, user_model.password_hash):
                return None
            
            # Update last login
            user_model.last_login = datetime.utcnow()
            await session.commit()
            
            return self._model_to_user(user_model)
    
    async def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate user with JWT token"""
        try:
            # Check if token is blacklisted
            if self.redis_client and await self._is_token_blacklisted(token):
                return None
            
            # Decode token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
            
            if not user_id:
                return None
            
            # Get user from database
            async with self.async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.id == user_id)
                )
                user_model = result.scalar_one_or_none()
                
                if not user_model or not user_model.is_active:
                    return None
                
                return self._model_to_user(user_model)
        
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        async with self.async_session() as session:
            result = await session.execute(
                select(APIKey).where(
                    APIKey.key_hash == key_hash,
                    APIKey.is_active == True
                )
            )
            api_key_model = result.scalar_one_or_none()
            
            if not api_key_model:
                return None
            
            # Check expiration
            if api_key_model.expires_at and api_key_model.expires_at < datetime.utcnow():
                return None
            
            # Update last used
            api_key_model.last_used = datetime.utcnow()
            
            # Get user
            user_result = await session.execute(
                select(UserModel).where(UserModel.id == api_key_model.user_id)
            )
            user_model = user_result.scalar_one_or_none()
            
            if not user_model or not user_model.is_active:
                return None
            
            await session.commit()
            
            return self._model_to_user(user_model)
    
    # Token Management
    async def generate_tokens(self, user: User) -> AuthToken:
        """Generate access and refresh tokens"""
        # Access token
        access_payload = {
            "sub": user.id,
            "role": user.role.value,
            "permissions": list(user.permissions),
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        }
        access_token = jwt.encode(access_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Refresh token
        refresh_payload = {
            "sub": user.id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=30)
        }
        refresh_token = jwt.encode(refresh_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Store session
        await self._store_session(user.id, access_token, refresh_token)
        
        return AuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=JWT_EXPIRATION_HOURS * 3600
        )
    
    async def refresh_tokens(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Get user
            async with self.async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.id == user_id)
                )
                user_model = result.scalar_one_or_none()
                
                if not user_model or not user_model.is_active:
                    return None
                
                user = self._model_to_user(user_model)
                
            # Generate new tokens
            return await self.generate_tokens(user)
        
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def revoke_token(self, token: str):
        """Revoke a token by adding it to blacklist"""
        if self.redis_client:
            # Add to blacklist with expiration
            try:
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                exp = payload.get("exp")
                
                if exp:
                    ttl = exp - time.time()
                    if ttl > 0:
                        await self.redis_client.setex(
                            f"blacklist:{token}",
                            int(ttl),
                            "1"
                        )
            except:
                pass
    
    # Authorization
    async def authorize_action(
        self,
        user: User,
        action: str,
        resource: Optional[str] = None
    ) -> bool:
        """Check if user is authorized for an action"""
        # Check if user has permission
        if action not in user.permissions:
            return False
        
        # Additional resource-based checks
        if resource:
            return await self._check_resource_permission(user, action, resource)
        
        return True
    
    async def check_rate_limit(
        self,
        user_id: str,
        action: str,
        limit: int = 100,
        window: int = 3600
    ) -> bool:
        """Check rate limiting for user actions"""
        if not self.redis_client:
            return True
        
        key = f"rate_limit:{user_id}:{action}"
        
        try:
            current = await self.redis_client.incr(key)
            
            if current == 1:
                await self.redis_client.expire(key, window)
            
            return current <= limit
        except:
            return True
    
    # Encryption
    async def encrypt_sensitive_data(self, data: Dict) -> str:
        """Encrypt sensitive data"""
        json_data = json.dumps(data)
        encrypted = self.cipher_suite.encrypt(json_data.encode())
        return base64.b64encode(encrypted).decode()
    
    async def decrypt_sensitive_data(self, encrypted_data: str) -> Dict:
        """Decrypt sensitive data"""
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(decoded)
            return json.loads(decrypted.decode())
        except:
            return {}
    
    # API Key Management
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[List[PermissionScope]] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create API key for user"""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        async with self.async_session() as session:
            api_key_model = APIKey(
                id=str(uuid.uuid4()),
                user_id=user_id,
                key_hash=key_hash,
                name=name,
                permissions=permissions or [],
                created_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            session.add(api_key_model)
            await session.commit()
        
        return api_key
    
    async def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        async with self.async_session() as session:
            result = await session.execute(
                select(APIKey).where(APIKey.key_hash == key_hash)
            )
            api_key_model = result.scalar_one_or_none()
            
            if api_key_model:
                api_key_model.is_active = False
                await session.commit()
    
    # Multi-Factor Authentication
    async def enable_mfa(self, user_id: str) -> str:
        """Enable MFA for user"""
        import pyotp
        
        secret = pyotp.random_base32()
        
        async with self.async_session() as session:
            result = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if user_model:
                user_model.mfa_secret = secret
                await session.commit()
        
        return secret
    
    async def verify_mfa(self, user_id: str, token: str) -> bool:
        """Verify MFA token"""
        import pyotp
        
        async with self.async_session() as session:
            result = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model or not user_model.mfa_secret:
                return False
            
            totp = pyotp.TOTP(user_model.mfa_secret)
            return totp.verify(token, valid_window=1)
    
    # Session Management
    async def get_active_sessions(self, user_id: str) -> List[Dict]:
        """Get active sessions for user"""
        async with self.async_session() as session:
            result = await session.execute(
                select(SessionToken).where(
                    SessionToken.user_id == user_id,
                    SessionToken.is_active == True,
                    SessionToken.expires_at > datetime.utcnow()
                )
            )
            sessions = result.scalars().all()
            
            return [
                {
                    "id": s.id,
                    "created_at": s.created_at.isoformat(),
                    "expires_at": s.expires_at.isoformat(),
                    "ip_address": s.ip_address,
                    "user_agent": s.user_agent
                }
                for s in sessions
            ]
    
    async def terminate_session(self, session_id: str):
        """Terminate a specific session"""
        async with self.async_session() as session:
            result = await session.execute(
                select(SessionToken).where(SessionToken.id == session_id)
            )
            session_token = result.scalar_one_or_none()
            
            if session_token:
                session_token.is_active = False
                await session.commit()
    
    # Helper Methods
    def _model_to_user(self, model: UserModel) -> User:
        """Convert database model to User object"""
        return User(
            id=model.id,
            email=model.email,
            username=model.username,
            role=UserRole(model.role),
            permissions=set(model.permissions),
            created_at=model.created_at,
            last_login=model.last_login,
            is_active=model.is_active,
            metadata=model.metadata
        )
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        if not self.redis_client:
            return False
        
        try:
            exists = await self.redis_client.exists(f"blacklist:{token}")
            return bool(exists)
        except:
            return False
    
    async def _store_session(self, user_id: str, access_token: str, refresh_token: str):
        """Store session information"""
        async with self.async_session() as session:
            session_token = SessionToken(
                id=str(uuid.uuid4()),
                user_id=user_id,
                token_hash=hashlib.sha256(access_token.encode()).hexdigest(),
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
            )
            
            session.add(session_token)
            await session.commit()
    
    async def _check_resource_permission(
        self,
        user: User,
        action: str,
        resource: str
    ) -> bool:
        """Check permission for specific resource"""
        # Implement resource-specific permission checks
        # For example, check if user owns the portfolio they're trying to modify
        
        if action == PermissionScope.WRITE_PORTFOLIO:
            # Check if user owns the portfolio
            return await self._user_owns_portfolio(user.id, resource)
        
        return True
    
    async def _user_owns_portfolio(self, user_id: str, portfolio_id: str) -> bool:
        """Check if user owns a portfolio"""
        # This would query the portfolio database
        # For now, return True for demonstration
        return True


# Middleware for FastAPI
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    security_manager: SecurityManager = Depends(get_security_manager)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    
    # Try JWT token first
    user = await security_manager.authenticate_token(token)
    
    # Try API key if JWT fails
    if not user:
        user = await security_manager.authenticate_api_key(token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    return user


async def require_permission(permission: PermissionScope):
    """Require specific permission"""
    async def permission_checker(
        user: User = Depends(get_current_user),
        security_manager: SecurityManager = Depends(get_security_manager)
    ):
        if not await security_manager.authorize_action(user, permission):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    
    return permission_checker


# Global instance getter
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get security manager instance"""
    global _security_manager
    
    if not _security_manager:
        database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./digital_human.db")
        redis_url = os.getenv("REDIS_URL", None)
        _security_manager = SecurityManager(database_url, redis_url)
    
    return _security_manager