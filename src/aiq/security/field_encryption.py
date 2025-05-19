"""
Field-level encryption for sensitive data in AIQToolkit.

This module provides:
- Transparent encryption/decryption of specific fields
- Key rotation support
- Multiple encryption algorithms
- Searchable encryption for certain fields
- Performance-optimized encryption
"""

import base64
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timezone
from functools import lru_cache

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, Field

from aiq.settings.security_config import get_security_config
from aiq.security.audit_logger import (
    log_security_event,
    SecurityEventType,
    SeverityLevel
)
from aiq.utils.exception_handlers import (
    handle_errors,
    ValidationError
)


class EncryptionKey(BaseModel):
    """Encryption key metadata"""
    key_id: str
    key_data: bytes
    algorithm: str = "AES-256-GCM"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rotated_at: Optional[datetime] = None
    is_active: bool = True
    purpose: str = "field_encryption"


class EncryptedField(BaseModel):
    """Encrypted field data structure"""
    ciphertext: str  # Base64 encoded
    key_id: str
    algorithm: str
    nonce: Optional[str] = None  # For GCM mode
    tag: Optional[str] = None  # Authentication tag for GCM
    searchable_hash: Optional[str] = None  # For searchable encryption


class FieldEncryptionConfig(BaseModel):
    """Configuration for field-level encryption"""
    enabled: bool = True
    default_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    searchable_fields: List[str] = Field(default_factory=list)
    sensitive_fields: List[str] = Field(default_factory=lambda: [
        "ssn",
        "credit_card",
        "bank_account",
        "password",
        "api_key",
        "private_key",
        "medical_record",
        "driver_license"
    ])


class FieldEncryptionManager:
    """Manage field-level encryption operations"""
    
    def __init__(self, config: Optional[FieldEncryptionConfig] = None):
        self.config = config or FieldEncryptionConfig()
        self.keys: Dict[str, EncryptionKey] = {}
        self.active_key_id: Optional[str] = None
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize encryption keys"""
        # Load or generate master key
        security_config = get_security_config()
        
        if security_config.encryption_key:
            # Use provided key
            key_data = base64.b64decode(security_config.encryption_key)
        else:
            # Generate new key
            key_data = os.urandom(32)  # 256 bits
            
            log_security_event(
                event_type=SecurityEventType.SECURITY_ALERT,
                action="generate_encryption_key",
                result="success",
                severity=SeverityLevel.WARNING,
                details={"message": "Generated new encryption key"}
            )
        
        # Create initial key
        key = EncryptionKey(
            key_id=self._generate_key_id(),
            key_data=key_data,
            algorithm=self.config.default_algorithm
        )
        
        self.keys[key.key_id] = key
        self.active_key_id = key.key_id
    
    @handle_errors(reraise=True)
    def encrypt_field(
        self,
        field_name: str,
        value: Any,
        searchable: bool = False
    ) -> Union[EncryptedField, Any]:
        """Encrypt a field value"""
        if not self.config.enabled:
            return value
        
        # Check if field should be encrypted
        if field_name not in self.config.sensitive_fields:
            return value
        
        # Convert value to bytes
        if isinstance(value, str):
            plaintext = value.encode('utf-8')
        elif isinstance(value, (int, float, bool)):
            plaintext = str(value).encode('utf-8')
        elif isinstance(value, (dict, list)):
            plaintext = json.dumps(value).encode('utf-8')
        else:
            plaintext = str(value).encode('utf-8')
        
        # Get active key
        key = self.keys[self.active_key_id]
        
        # Encrypt based on algorithm
        if key.algorithm == "AES-256-GCM":
            encrypted = self._encrypt_aes_gcm(plaintext, key)
        elif key.algorithm == "Fernet":
            encrypted = self._encrypt_fernet(plaintext, key)
        else:
            raise ValueError(f"Unsupported algorithm: {key.algorithm}")
        
        # Add searchable hash if needed
        if searchable or field_name in self.config.searchable_fields:
            encrypted.searchable_hash = self._create_searchable_hash(plaintext)
        
        log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            action="encrypt_field",
            result="success",
            resource=field_name,
            details={
                "algorithm": key.algorithm,
                "key_id": key.key_id,
                "searchable": bool(encrypted.searchable_hash)
            }
        )
        
        return encrypted
    
    @handle_errors(reraise=True)
    def decrypt_field(
        self,
        field_name: str,
        encrypted_data: Union[EncryptedField, Any]
    ) -> Any:
        """Decrypt a field value"""
        if not self.config.enabled:
            return encrypted_data
        
        # Check if data is encrypted
        if not isinstance(encrypted_data, EncryptedField):
            return encrypted_data
        
        # Get encryption key
        key = self.keys.get(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Unknown key ID: {encrypted_data.key_id}")
        
        # Decrypt based on algorithm
        if encrypted_data.algorithm == "AES-256-GCM":
            plaintext = self._decrypt_aes_gcm(encrypted_data, key)
        elif encrypted_data.algorithm == "Fernet":
            plaintext = self._decrypt_fernet(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported algorithm: {encrypted_data.algorithm}")
        
        # Convert back to original type
        try:
            # Try JSON first
            return json.loads(plaintext)
        except json.JSONDecodeError:
            # Return as string
            return plaintext.decode('utf-8')
    
    def _encrypt_aes_gcm(self, plaintext: bytes, key: EncryptionKey) -> EncryptedField:
        """Encrypt using AES-256-GCM"""
        # Generate nonce
        nonce = os.urandom(12)  # 96 bits for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptedField(
            ciphertext=base64.b64encode(ciphertext).decode('utf-8'),
            key_id=key.key_id,
            algorithm=key.algorithm,
            nonce=base64.b64encode(nonce).decode('utf-8'),
            tag=base64.b64encode(encryptor.tag).decode('utf-8')
        )
    
    def _decrypt_aes_gcm(self, encrypted: EncryptedField, key: EncryptionKey) -> bytes:
        """Decrypt using AES-256-GCM"""
        # Decode components
        ciphertext = base64.b64decode(encrypted.ciphertext)
        nonce = base64.b64decode(encrypted.nonce)
        tag = base64.b64decode(encrypted.tag)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def _encrypt_fernet(self, plaintext: bytes, key: EncryptionKey) -> EncryptedField:
        """Encrypt using Fernet (simplified)"""
        f = Fernet(base64.b64encode(key.key_data[:32]))  # Fernet needs base64-encoded 32 bytes
        ciphertext = f.encrypt(plaintext)
        
        return EncryptedField(
            ciphertext=base64.b64encode(ciphertext).decode('utf-8'),
            key_id=key.key_id,
            algorithm=key.algorithm
        )
    
    def _decrypt_fernet(self, encrypted: EncryptedField, key: EncryptionKey) -> bytes:
        """Decrypt using Fernet"""
        f = Fernet(base64.b64encode(key.key_data[:32]))
        ciphertext = base64.b64decode(encrypted.ciphertext)
        
        return f.decrypt(ciphertext)
    
    def _create_searchable_hash(self, plaintext: bytes) -> str:
        """Create searchable hash for encrypted data"""
        # Use HMAC for deterministic hashing
        h = hashlib.sha256()
        h.update(self.keys[self.active_key_id].key_data)
        h.update(plaintext)
        return h.hexdigest()
    
    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        return hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    
    @handle_errors(reraise=True)
    def rotate_keys(self) -> str:
        """Rotate encryption keys"""
        # Generate new key
        new_key_data = os.urandom(32)
        new_key = EncryptionKey(
            key_id=self._generate_key_id(),
            key_data=new_key_data,
            algorithm=self.config.default_algorithm
        )
        
        # Add new key
        self.keys[new_key.key_id] = new_key
        
        # Mark old key as rotated
        old_key = self.keys[self.active_key_id]
        old_key.rotated_at = datetime.now(timezone.utc)
        old_key.is_active = False
        
        # Set new active key
        self.active_key_id = new_key.key_id
        
        log_security_event(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            action="rotate_encryption_keys",
            result="success",
            details={
                "old_key_id": old_key.key_id,
                "new_key_id": new_key.key_id
            }
        )
        
        return new_key.key_id
    
    def search_encrypted_field(
        self,
        field_name: str,
        search_value: Any,
        encrypted_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search encrypted fields using searchable hash"""
        if not self.config.enabled:
            return [r for r in encrypted_records if r.get(field_name) == search_value]
        
        # Create search hash
        if isinstance(search_value, str):
            search_bytes = search_value.encode('utf-8')
        else:
            search_bytes = str(search_value).encode('utf-8')
        
        search_hash = self._create_searchable_hash(search_bytes)
        
        # Search records
        matches = []
        for record in encrypted_records:
            field_value = record.get(field_name)
            if isinstance(field_value, EncryptedField):
                if field_value.searchable_hash == search_hash:
                    matches.append(record)
        
        return matches


# Pydantic model integration
class EncryptedFieldMixin:
    """Mixin for Pydantic models with encrypted fields"""
    
    _encryption_manager: Optional[FieldEncryptionManager] = None
    _encrypted_fields: List[str] = []
    
    def __init__(self, **data):
        # Initialize encryption manager
        if not self._encryption_manager:
            self._encryption_manager = field_encryption_manager
        
        # Process fields before validation
        processed_data = {}
        for field_name, value in data.items():
            if field_name in self._encrypted_fields and not isinstance(value, EncryptedField):
                # Encrypt field
                processed_data[field_name] = self._encryption_manager.encrypt_field(
                    field_name,
                    value,
                    searchable=True
                )
            else:
                processed_data[field_name] = value
        
        super().__init__(**processed_data)
    
    def get_decrypted_field(self, field_name: str) -> Any:
        """Get decrypted value of a field"""
        if field_name not in self._encrypted_fields:
            return getattr(self, field_name)
        
        encrypted_value = getattr(self, field_name)
        return self._encryption_manager.decrypt_field(field_name, encrypted_value)
    
    def set_encrypted_field(self, field_name: str, value: Any):
        """Set and encrypt a field value"""
        if field_name not in self._encrypted_fields:
            setattr(self, field_name, value)
            return
        
        encrypted_value = self._encryption_manager.encrypt_field(
            field_name,
            value,
            searchable=True
        )
        setattr(self, field_name, encrypted_value)
    
    def dict_with_decrypted_fields(self, **kwargs) -> Dict[str, Any]:
        """Get dictionary with decrypted fields"""
        data = self.dict(**kwargs)
        
        for field_name in self._encrypted_fields:
            if field_name in data:
                data[field_name] = self.get_decrypted_field(field_name)
        
        return data


# Example encrypted model
class SecureUserProfile(BaseModel, EncryptedFieldMixin):
    """User profile with encrypted sensitive fields"""
    
    _encrypted_fields = ["ssn", "credit_card", "bank_account"]
    
    user_id: str
    username: str
    email: str
    ssn: Union[str, EncryptedField]
    credit_card: Union[str, EncryptedField]
    bank_account: Union[str, EncryptedField]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Performance optimization
@lru_cache(maxsize=1000)
def cached_decrypt(encrypted_data: str, key_id: str) -> str:
    """Cached decryption for frequently accessed data"""
    encrypted = EncryptedField(
        ciphertext=encrypted_data,
        key_id=key_id,
        algorithm="AES-256-GCM"
    )
    return field_encryption_manager.decrypt_field("cached", encrypted)


# Global encryption manager
field_encryption_manager = FieldEncryptionManager()


# Database integration helpers
def encrypt_db_row(row: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
    """Encrypt sensitive fields in a database row"""
    encrypted_row = row.copy()
    
    for field in sensitive_fields:
        if field in row and row[field] is not None:
            encrypted_row[field] = field_encryption_manager.encrypt_field(
                field,
                row[field],
                searchable=True
            ).dict()
    
    return encrypted_row


def decrypt_db_row(row: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
    """Decrypt sensitive fields in a database row"""
    decrypted_row = row.copy()
    
    for field in sensitive_fields:
        if field in row and isinstance(row[field], dict):
            encrypted_field = EncryptedField(**row[field])
            decrypted_row[field] = field_encryption_manager.decrypt_field(
                field,
                encrypted_field
            )
    
    return decrypted_row


# Export public interface
__all__ = [
    "FieldEncryptionManager",
    "FieldEncryptionConfig",
    "EncryptedField",
    "EncryptedFieldMixin",
    "SecureUserProfile",
    "field_encryption_manager",
    "encrypt_db_row",
    "decrypt_db_row",
    "cached_decrypt"
]