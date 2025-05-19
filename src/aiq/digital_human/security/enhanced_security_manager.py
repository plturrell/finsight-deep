"""Production-grade Security Manager for Digital Human Financial Advisor"""

import asyncio
import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import json

import jwt
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aiq.data_models.common import BaseModel
from aiq.utils.debugging_utils import log_function_call


# Security configuration
ALLOWED_CHARACTERS = re.compile(r"^[a-zA-Z0-9\s\.,!?;:'\"-@$%&()\[\]{}]*$")
MAX_INPUT_LENGTH = 10000
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class EnhancedSecurityManager:
    """
    Manages security for the Digital Human system including:
    - Authentication and authorization
    - Input validation and sanitization
    - Encryption and decryption
    - Session management
    - Rate limiting
    - Audit logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = config.get("encryption_key", Fernet.generate_key())
        self.jwt_secret = config.get("jwt_secret", secrets.token_urlsafe(32))
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize Redis for session and rate limiting
        self.redis_client = redis.from_url(
            config.get("redis_url", "redis://localhost:6379/0"),
            decode_responses=True
        )
        
        # Initialize database for audit logs
        db_url = config.get("database_url", "postgresql://localhost/digital_human")
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Blocked patterns for security
        self.blocked_patterns = [
            r"<\s*script",  # Script tags
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",   # Event handlers
            r"eval\s*\(",   # Eval function
            r"exec\s*\(",   # Exec function
            r"__import__",  # Python import
            r"subprocess",  # System commands
            r"os\.",        # OS module
            r"sys\.",       # Sys module
            r"\.\.\/",      # Directory traversal
            r"union\s+select",  # SQL injection
            r"drop\s+table",    # SQL injection
            r"insert\s+into",   # SQL injection
        ]
        
        # Compile blocked patterns
        self.blocked_regex = re.compile(
            "|".join(self.blocked_patterns),
            re.IGNORECASE | re.MULTILINE
        )
        
        # Rate limiting configuration
        self.rate_limits = {
            "login": {"requests": 5, "window": 300},  # 5 attempts per 5 minutes
            "message": {"requests": 30, "window": 60},  # 30 messages per minute
            "api": {"requests": 100, "window": 60}  # 100 API calls per minute
        }
    
    def validate_input(self, user_input: str) -> str:
        """Validate and sanitize user input"""
        if not user_input:
            return ""
            
        # Check length
        if len(user_input) > MAX_INPUT_LENGTH:
            raise ValueError(f"Input exceeds maximum length of {MAX_INPUT_LENGTH}")
        
        # Check for blocked patterns
        if self.blocked_regex.search(user_input):
            raise ValueError("Input contains forbidden patterns")
        
        # Check allowed characters
        if not ALLOWED_CHARACTERS.match(user_input):
            # Strip potentially dangerous characters
            user_input = re.sub(r"[^\w\s\.,!?;:'\"-@$%&()\[\]{}]", "", user_input)
        
        # Additional sanitization
        user_input = user_input.strip()
        user_input = re.sub(r"\s+", " ", user_input)  # Normalize whitespace
        
        return user_input
    
    def generate_token(self, user_id: str, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for user"""
        payload = {
            "sub": user_id,
            "data": user_data,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)
        
        # Store token in Redis for revocation capability
        self.redis_client.setex(
            f"token:{payload['jti']}",
            int(timedelta(hours=JWT_EXPIRY_HOURS).total_seconds()),
            json.dumps({"user_id": user_id, "revoked": False})
        )
        
        return token