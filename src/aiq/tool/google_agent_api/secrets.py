# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Dict, Optional
import asyncio
import json
from abc import ABC, abstractmethod
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
import hvac  # HashiCorp Vault client
# Removed AWS Secrets Manager import

log = logging.getLogger(__name__)


class SecretProvider(ABC):
    """Abstract base for secret providers"""
    
    @abstractmethod
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value"""
        pass
    
    @abstractmethod
    async def set_secret(self, key: str, value: str):
        """Set a secret value"""
        pass
    
    @abstractmethod
    async def delete_secret(self, key: str):
        """Delete a secret"""
        pass


class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault secret provider"""
    
    def __init__(self, vault_url: str, vault_token: str, mount_point: str = "secret"):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.mount_point = mount_point
        
        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Vault"""
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                self.client.secrets.kv.v2.read_secret_version,
                path=key,
                mount_point=self.mount_point
            )
            
            data = response.get("data", {}).get("data", {})
            return data.get("value")
            
        except Exception as e:
            log.error(f"Failed to get secret {key}: {e}")
            return None
    
    async def set_secret(self, key: str, value: str):
        """Set secret in Vault"""
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            None,
            self.client.secrets.kv.v2.create_or_update_secret,
            path=key,
            secret={"value": value},
            mount_point=self.mount_point
        )
    
    async def delete_secret(self, key: str):
        """Delete secret from Vault"""
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            None,
            self.client.secrets.kv.v2.delete_metadata_and_all_versions,
            path=key,
            mount_point=self.mount_point
        )


# AWS Secrets Manager provider removed


class EncryptedFileProvider(SecretProvider):
    """Local encrypted file provider (for development)"""
    
    def __init__(self, file_path: str, master_password: str):
        self.file_path = file_path
        self.secrets = {}
        
        # Derive encryption key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self.cipher = Fernet(key)
        
        # Load existing secrets
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from encrypted file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'rb') as f:
                    encrypted_data = f.read()
                    
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self.secrets = json.loads(decrypted_data.decode())
                
            except Exception as e:
                log.error(f"Failed to load secrets: {e}")
                self.secrets = {}
    
    def _save_secrets(self):
        """Save secrets to encrypted file"""
        try:
            data = json.dumps(self.secrets).encode()
            encrypted_data = self.cipher.encrypt(data)
            
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            with open(self.file_path, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            log.error(f"Failed to save secrets: {e}")
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from encrypted file"""
        return self.secrets.get(key)
    
    async def set_secret(self, key: str, value: str):
        """Set secret in encrypted file"""
        self.secrets[key] = value
        self._save_secrets()
    
    async def delete_secret(self, key: str):
        """Delete secret from encrypted file"""
        self.secrets.pop(key, None)
        self._save_secrets()


class SecretManager:
    """Central secret management with caching and rotation"""
    
    def __init__(self, provider: SecretProvider, cache_ttl: int = 300):
        self.provider = provider
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._rotation_callbacks = {}
    
    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret with caching"""
        
        # Check cache
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
        
        # Get from provider
        value = await self.provider.get_secret(key)
        
        if value:
            # Cache the value
            self._cache[key] = (value, time.time())
        
        return value
    
    async def set_secret(self, key: str, value: str):
        """Set secret and clear cache"""
        await self.provider.set_secret(key, value)
        
        # Clear cache
        self._cache.pop(key, None)
        
        # Trigger rotation callbacks
        if key in self._rotation_callbacks:
            await self._rotation_callbacks[key](value)
    
    async def rotate_secret(self, key: str, new_value: str):
        """Rotate a secret with notification"""
        old_value = await self.get_secret(key)
        
        # Set new value
        await self.set_secret(key, new_value)
        
        # Log rotation
        log.info(f"Rotated secret: {key}")
        
        return old_value
    
    def on_rotation(self, key: str, callback):
        """Register callback for secret rotation"""
        self._rotation_callbacks[key] = callback
    
    async def bulk_get_secrets(self, keys: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple secrets efficiently"""
        tasks = [self.get_secret(key) for key in keys]
        values = await asyncio.gather(*tasks)
        
        return dict(zip(keys, values))


# Factory function to create appropriate provider
def create_secret_provider(provider_type: str, **kwargs) -> SecretProvider:
    """Create secret provider based on configuration"""
    
    if provider_type == "vault":
        return VaultSecretProvider(
            vault_url=kwargs.get("vault_url"),
            vault_token=kwargs.get("vault_token"),
            mount_point=kwargs.get("mount_point", "secret")
        )
    
    # AWS provider option removed
    
    elif provider_type == "encrypted_file":
        return EncryptedFileProvider(
            file_path=kwargs.get("file_path", "/tmp/secrets.enc"),
            master_password=kwargs.get("master_password")
        )
    
    else:
        raise ValueError(f"Unknown secret provider: {provider_type}")