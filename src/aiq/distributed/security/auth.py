# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Authentication for distributed AIQToolkit
"""

import jwt
import grpc
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import hashlib
import secrets

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication configuration"""
    secret_key: str
    algorithm: str = "HS256"
    token_expiry_seconds: int = 3600
    enable_auth: bool = True


class AuthManager:
    """Manages authentication for distributed communication"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.enabled = config.enable_auth
        
    def generate_token(self, 
                      node_id: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate authentication token"""
        if not self.enabled:
            return ""
            
        payload = {
            "node_id": node_id,
            "issued_at": time.time(),
            "expires_at": time.time() + self.config.token_expiry_seconds,
            "metadata": metadata or {}
        }
        
        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        return token
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify authentication token"""
        if not self.enabled:
            return True, None
            
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            
            # Check expiration
            if payload.get("expires_at", 0) < time.time():
                logger.warning("Token expired")
                return False, None
                
            return True, payload
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return False, None
    
    def generate_node_key(self, node_id: str) -> str:
        """Generate a unique key for a node"""
        # Combine node ID with secret and hash
        combined = f"{node_id}:{self.config.secret_key}:{time.time()}"
        return hashlib.sha256(combined.encode()).hexdigest()


class AuthInterceptor(grpc.ServerInterceptor):
    """gRPC interceptor for authentication"""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept gRPC calls for authentication"""
        # Extract token from metadata
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get("authorization", "").replace("Bearer ", "")
        
        # Verify token
        valid, payload = self.auth_manager.verify_token(token)
        
        if not valid and self.auth_manager.enabled:
            # Return error for invalid token
            def abort_invalid_token(request, context):
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid or missing authentication token"
                )
            return grpc.unary_unary_rpc_method_handler(abort_invalid_token)
        
        # Add auth context
        if payload:
            handler_call_details.invocation_metadata.append(
                ("auth_context", str(payload))
            )
        
        # Continue with the original handler
        return continuation(handler_call_details)


class AuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    """gRPC client interceptor for adding authentication"""
    
    def __init__(self, auth_manager: AuthManager, node_id: str):
        self.auth_manager = auth_manager
        self.node_id = node_id
        self.token = None
        self.token_expires = 0
    
    def _refresh_token(self):
        """Refresh authentication token if needed"""
        if time.time() >= self.token_expires - 60:  # Refresh 1 minute before expiry
            self.token = self.auth_manager.generate_token(self.node_id)
            self.token_expires = time.time() + self.auth_manager.config.token_expiry_seconds
    
    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Add authentication to outgoing requests"""
        if not self.auth_manager.enabled:
            return continuation(client_call_details, request)
            
        # Refresh token if needed
        self._refresh_token()
        
        # Add token to metadata
        metadata = list(client_call_details.metadata or [])
        metadata.append(("authorization", f"Bearer {self.token}"))
        
        # Create new call details with auth metadata
        new_details = grpc.ClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=metadata,
            credentials=client_call_details.credentials
        )
        
        return continuation(new_details, request)


class APIKeyAuth:
    """Simple API key authentication"""
    
    def __init__(self):
        self.api_keys: Dict[str, str] = {}  # node_id -> api_key
        
    def generate_api_key(self, node_id: str) -> str:
        """Generate a new API key for a node"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[node_id] = api_key
        return api_key
    
    def verify_api_key(self, node_id: str, api_key: str) -> bool:
        """Verify an API key"""
        stored_key = self.api_keys.get(node_id)
        return stored_key == api_key
    
    def revoke_api_key(self, node_id: str):
        """Revoke an API key"""
        self.api_keys.pop(node_id, None)


class RBACManager:
    """Role-Based Access Control manager"""
    
    def __init__(self):
        self.roles: Dict[str, set] = {
            "admin": {"*"},  # All permissions
            "manager": {"node.register", "node.unregister", "task.assign", "cluster.status"},
            "worker": {"task.execute", "heartbeat.send", "status.report"},
            "viewer": {"cluster.status", "task.status"}
        }
        
        self.node_roles: Dict[str, str] = {}  # node_id -> role
    
    def assign_role(self, node_id: str, role: str):
        """Assign a role to a node"""
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        self.node_roles[node_id] = role
    
    def check_permission(self, node_id: str, action: str) -> bool:
        """Check if a node has permission for an action"""
        role = self.node_roles.get(node_id, "viewer")
        permissions = self.roles.get(role, set())
        
        # Check for wildcard or specific permission
        return "*" in permissions or action in permissions
    
    def get_node_role(self, node_id: str) -> str:
        """Get the role assigned to a node"""
        return self.node_roles.get(node_id, "viewer")


def create_secure_auth_config() -> AuthConfig:
    """Create a secure authentication configuration"""
    # Generate random secret key
    secret_key = secrets.token_urlsafe(32)
    
    return AuthConfig(
        secret_key=secret_key,
        algorithm="HS256",
        token_expiry_seconds=3600,
        enable_auth=True
    )


# Example usage
if __name__ == "__main__":
    # Create authentication manager
    config = create_secure_auth_config()
    auth_manager = AuthManager(config)
    
    # Generate token
    token = auth_manager.generate_token("worker-1", {"role": "worker"})
    print(f"Generated token: {token[:20]}...")
    
    # Verify token
    valid, payload = auth_manager.verify_token(token)
    print(f"Token valid: {valid}")
    print(f"Payload: {payload}")
    
    # Test RBAC
    rbac = RBACManager()
    rbac.assign_role("worker-1", "worker")
    
    print(f"Can execute task: {rbac.check_permission('worker-1', 'task.execute')}")
    print(f"Can register node: {rbac.check_permission('worker-1', 'node.register')}")