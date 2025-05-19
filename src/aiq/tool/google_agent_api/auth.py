# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import jwt
import asyncio
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

log = logging.getLogger(__name__)


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


@dataclass
class AuthToken:
    user_id: str
    permissions: Set[Permission]
    roles: Set[str]
    expires_at: float
    issued_at: float
    token_id: str
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions or Permission.ADMIN in self.permissions
    
    def has_role(self, role: str) -> bool:
        return role in self.roles or "admin" in self.roles


class AuthenticationManager:
    """Production authentication with JWT tokens and RSA signatures"""
    
    def __init__(self, 
                 private_key_path: Optional[str] = None,
                 public_key_path: Optional[str] = None,
                 token_ttl: int = 3600):
        self.token_ttl = token_ttl
        self._tokens = {}  # Token cache
        self._revoked_tokens = set()  # Revocation list
        
        # Load or generate RSA keys
        if private_key_path and public_key_path:
            with open(private_key_path, 'rb') as f:
                self.private_key = f.read()
            with open(public_key_path, 'rb') as f:
                self.public_key = f.read()
        else:
            # Generate keys for development
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            self.public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
    
    async def create_token(self, 
                          user_id: str,
                          permissions: Set[Permission],
                          roles: Set[str]) -> str:
        """Create a new JWT token"""
        
        token_id = str(uuid.uuid4())
        issued_at = time.time()
        expires_at = issued_at + self.token_ttl
        
        # Create token object
        auth_token = AuthToken(
            user_id=user_id,
            permissions=permissions,
            roles=roles,
            expires_at=expires_at,
            issued_at=issued_at,
            token_id=token_id
        )
        
        # Create JWT payload
        payload = {
            "user_id": user_id,
            "permissions": [p.value for p in permissions],
            "roles": list(roles),
            "exp": expires_at,
            "iat": issued_at,
            "jti": token_id
        }
        
        # Sign token
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        
        # Cache token
        self._tokens[token_id] = auth_token
        
        return token
    
    async def verify_token(self, token: str) -> Optional[AuthToken]:
        """Verify and decode JWT token"""
        
        try:
            # Decode token
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
            
            token_id = payload["jti"]
            
            # Check revocation
            if token_id in self._revoked_tokens:
                return None
            
            # Check cache
            if token_id in self._tokens:
                auth_token = self._tokens[token_id]
                if not auth_token.is_expired():
                    return auth_token
            
            # Reconstruct token
            auth_token = AuthToken(
                user_id=payload["user_id"],
                permissions={Permission(p) for p in payload["permissions"]},
                roles=set(payload["roles"]),
                expires_at=payload["exp"],
                issued_at=payload["iat"],
                token_id=token_id
            )
            
            # Cache token
            self._tokens[token_id] = auth_token
            
            return auth_token
            
        except jwt.ExpiredSignatureError:
            log.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            log.warning(f"Invalid token: {e}")
            return None
    
    async def revoke_token(self, token_id: str):
        """Revoke a token"""
        self._revoked_tokens.add(token_id)
        self._tokens.pop(token_id, None)
    
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens from cache"""
        current_time = time.time()
        expired_tokens = [
            token_id for token_id, token in self._tokens.items()
            if token.expires_at < current_time
        ]
        
        for token_id in expired_tokens:
            del self._tokens[token_id]


class AuthorizationManager:
    """Role-based access control (RBAC) and policy management"""
    
    def __init__(self):
        self.policies = {}
        self.role_permissions = {
            "admin": {Permission.ADMIN},
            "operator": {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            "analyst": {Permission.READ, Permission.EXECUTE},
            "viewer": {Permission.READ}
        }
        
        # Resource-based permissions
        self.resource_permissions = {}
    
    async def check_permission(self,
                              auth_token: AuthToken,
                              resource: str,
                              permission: Permission) -> bool:
        """Check if user has permission for resource"""
        
        # Admin can do anything
        if Permission.ADMIN in auth_token.permissions:
            return True
        
        # Check direct permissions
        if permission in auth_token.permissions:
            return True
        
        # Check role-based permissions
        for role in auth_token.roles:
            role_perms = self.role_permissions.get(role, set())
            if permission in role_perms:
                return True
        
        # Check resource-specific permissions
        resource_perms = self.resource_permissions.get(resource, {})
        user_resource_perms = resource_perms.get(auth_token.user_id, set())
        
        return permission in user_resource_perms
    
    async def grant_permission(self,
                              user_id: str,
                              resource: str,
                              permission: Permission):
        """Grant permission to user for specific resource"""
        
        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = {}
        
        if user_id not in self.resource_permissions[resource]:
            self.resource_permissions[resource][user_id] = set()
        
        self.resource_permissions[resource][user_id].add(permission)
    
    async def revoke_permission(self,
                               user_id: str,
                               resource: str,
                               permission: Permission):
        """Revoke permission from user for specific resource"""
        
        if resource in self.resource_permissions:
            if user_id in self.resource_permissions[resource]:
                self.resource_permissions[resource][user_id].discard(permission)


class AuthMiddleware:
    """Authentication middleware for API requests"""
    
    def __init__(self, 
                 auth_manager: AuthenticationManager,
                 authz_manager: AuthorizationManager):
        self.auth_manager = auth_manager
        self.authz_manager = authz_manager
    
    async def authenticate(self, request: Dict[str, Any]) -> Optional[AuthToken]:
        """Extract and verify token from request"""
        
        # Check Authorization header
        auth_header = request.get("headers", {}).get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        return await self.auth_manager.verify_token(token)
    
    async def authorize(self,
                       auth_token: AuthToken,
                       resource: str,
                       permission: Permission) -> bool:
        """Check authorization for resource access"""
        
        return await self.authz_manager.check_permission(
            auth_token, resource, permission
        )
    
    async def middleware(self, request: Dict[str, Any], next_handler):
        """Middleware function for request processing"""
        
        # Skip auth for health checks
        if request.get("path") == "/health":
            return await next_handler(request)
        
        # Authenticate
        auth_token = await self.authenticate(request)
        
        if not auth_token:
            return {
                "status": 401,
                "body": {"error": "Unauthorized"}
            }
        
        # Add auth token to request context
        request["auth_token"] = auth_token
        
        return await next_handler(request)