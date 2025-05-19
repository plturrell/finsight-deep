"""Tests for security configuration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import jwt
import hashlib
import base64
from cryptography.fernet import Fernet

from aiq.settings.security_config import (
    SecurityConfig,
    EncryptionManager,
    AuthenticationManager,
    AuditLogger,
    RBACManager,
    SecureConfigLoader
)


class TestSecurityConfig:
    """Test security configuration settings."""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration instance."""
        return SecurityConfig(
            encryption_enabled=True,
            auth_required=True,
            rbac_enabled=True,
            audit_logging=True
        )
    
    def test_initialization(self, security_config):
        """Test security config initialization."""
        assert security_config.encryption_enabled
        assert security_config.auth_required
        assert security_config.rbac_enabled
        assert security_config.audit_logging
    
    def test_default_configuration(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert config.encryption_enabled is True  # Secure by default
        assert config.auth_required is True
        assert config.rbac_enabled is False  # Optional by default
        assert config.audit_logging is True
    
    def test_validate_configuration(self, security_config):
        """Test configuration validation."""
        # Valid configuration
        assert security_config.validate()
        
        # Invalid configuration
        invalid_config = SecurityConfig(
            encryption_enabled=True,
            auth_required=False,  # Should require auth if encryption enabled
            rbac_enabled=True
        )
        with pytest.raises(ValueError, match="Authentication required when encryption enabled"):
            invalid_config.validate()
    
    def test_to_dict(self, security_config):
        """Test configuration export to dictionary."""
        config_dict = security_config.to_dict()
        assert config_dict['encryption_enabled'] is True
        assert config_dict['auth_required'] is True
        assert config_dict['rbac_enabled'] is True
        assert config_dict['audit_logging'] is True
    
    def test_from_dict(self):
        """Test configuration import from dictionary."""
        config_dict = {
            'encryption_enabled': False,
            'auth_required': True,
            'rbac_enabled': True,
            'audit_logging': False
        }
        config = SecurityConfig.from_dict(config_dict)
        assert config.encryption_enabled is False
        assert config.auth_required is True
        assert config.rbac_enabled is True
        assert config.audit_logging is False


class TestEncryptionManager:
    """Test encryption management."""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager instance."""
        return EncryptionManager()
    
    def test_key_generation(self, encryption_manager):
        """Test encryption key generation."""
        key = encryption_manager.generate_key()
        assert isinstance(key, bytes)
        assert len(key) == 44  # Fernet key length (base64 encoded)
    
    def test_encrypt_decrypt(self, encryption_manager):
        """Test encryption and decryption."""
        plaintext = "sensitive data"
        key = encryption_manager.generate_key()
        
        encrypted = encryption_manager.encrypt(plaintext, key)
        assert encrypted != plaintext
        assert isinstance(encrypted, bytes)
        
        decrypted = encryption_manager.decrypt(encrypted, key)
        assert decrypted == plaintext
    
    def test_encrypt_file(self, encryption_manager, tmp_path):
        """Test file encryption."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("sensitive file content")
        
        key = encryption_manager.generate_key()
        
        # Encrypt file
        encrypted_file = tmp_path / "test.txt.enc"
        encryption_manager.encrypt_file(test_file, encrypted_file, key)
        
        assert encrypted_file.exists()
        assert encrypted_file.read_bytes() != test_file.read_bytes()
        
        # Decrypt file
        decrypted_file = tmp_path / "test_decrypted.txt"
        encryption_manager.decrypt_file(encrypted_file, decrypted_file, key)
        
        assert decrypted_file.read_text() == "sensitive file content"
    
    def test_key_rotation(self, encryption_manager):
        """Test key rotation."""
        old_key = encryption_manager.generate_key()
        new_key = encryption_manager.generate_key()
        
        plaintext = "data to rotate"
        encrypted = encryption_manager.encrypt(plaintext, old_key)
        
        # Rotate key
        rotated = encryption_manager.rotate_key(encrypted, old_key, new_key)
        
        # Should decrypt with new key
        decrypted = encryption_manager.decrypt(rotated, new_key)
        assert decrypted == plaintext
        
        # Should not decrypt with old key
        with pytest.raises(Exception):
            encryption_manager.decrypt(rotated, old_key)
    
    def test_secure_deletion(self, encryption_manager, tmp_path):
        """Test secure file deletion."""
        test_file = tmp_path / "secure_delete.txt"
        test_file.write_text("delete me securely")
        
        with patch('os.urandom') as mock_urandom:
            mock_urandom.return_value = b'\x00' * 1024
            encryption_manager.secure_delete(test_file)
        
        assert not test_file.exists()
        mock_urandom.assert_called()


class TestAuthenticationManager:
    """Test authentication management."""
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager instance."""
        return AuthenticationManager(secret_key="test_secret_key")
    
    def test_token_generation(self, auth_manager):
        """Test JWT token generation."""
        user_id = "user123"
        role = "admin"
        
        token = auth_manager.generate_token(user_id, role)
        assert isinstance(token, str)
        
        # Decode token to verify claims
        payload = jwt.decode(token, "test_secret_key", algorithms=["HS256"])
        assert payload["user_id"] == user_id
        assert payload["role"] == role
        assert "exp" in payload
    
    def test_token_validation(self, auth_manager):
        """Test token validation."""
        token = auth_manager.generate_token("user123", "admin")
        
        # Valid token
        is_valid, payload = auth_manager.validate_token(token)
        assert is_valid
        assert payload["user_id"] == "user123"
        
        # Invalid token
        is_valid, error = auth_manager.validate_token("invalid_token")
        assert not is_valid
        assert error is not None
    
    def test_token_expiration(self, auth_manager):
        """Test token expiration."""
        # Create token with short expiration
        token = auth_manager.generate_token("user123", "admin", expires_in=1)
        
        # Should be valid immediately
        is_valid, _ = auth_manager.validate_token(token)
        assert is_valid
        
        # Mock time to simulate expiration
        with patch('time.time', return_value=auth_manager._get_timestamp() + 3600):
            is_valid, error = auth_manager.validate_token(token)
            assert not is_valid
            assert "expired" in str(error).lower()
    
    def test_password_hashing(self, auth_manager):
        """Test password hashing and verification."""
        password = "secure_password123"
        
        # Hash password
        hashed = auth_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) == 64  # SHA256 hex length
        
        # Verify password
        assert auth_manager.verify_password(password, hashed)
        assert not auth_manager.verify_password("wrong_password", hashed)
    
    def test_api_key_management(self, auth_manager):
        """Test API key generation and validation."""
        api_key = auth_manager.generate_api_key()
        assert isinstance(api_key, str)
        assert len(api_key) >= 32
        
        # Store and validate API key
        auth_manager.store_api_key(api_key, "service123")
        assert auth_manager.validate_api_key(api_key)
        assert not auth_manager.validate_api_key("invalid_key")


class TestAuditLogger:
    """Test audit logging functionality."""
    
    @pytest.fixture
    def audit_logger(self, tmp_path):
        """Create audit logger instance."""
        log_file = tmp_path / "audit.log"
        return AuditLogger(log_file=str(log_file))
    
    def test_log_event(self, audit_logger):
        """Test event logging."""
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"
            
            audit_logger.log_event(
                user_id="user123",
                action="login",
                resource="system",
                result="success"
            )
        
        # Verify log entry
        log_content = open(audit_logger.log_file).read()
        assert "user123" in log_content
        assert "login" in log_content
        assert "success" in log_content
        assert "2024-01-01T00:00:00" in log_content
    
    def test_log_security_event(self, audit_logger):
        """Test security event logging."""
        audit_logger.log_security_event(
            event_type="unauthorized_access",
            user_id="user456",
            details={"ip": "192.168.1.1", "endpoint": "/api/admin"}
        )
        
        log_content = open(audit_logger.log_file).read()
        assert "SECURITY" in log_content
        assert "unauthorized_access" in log_content
        assert "192.168.1.1" in log_content
    
    def test_log_rotation(self, audit_logger):
        """Test log rotation."""
        # Write many log entries
        for i in range(100):
            audit_logger.log_event(
                user_id=f"user{i}",
                action="action",
                resource="resource",
                result="success"
            )
        
        # Trigger rotation
        with patch('os.path.getsize', return_value=1024*1024*11):  # 11MB
            audit_logger._check_rotation()
        
        # Should have rotated log
        import glob
        log_files = glob.glob(f"{audit_logger.log_file}.*")
        assert len(log_files) > 0
    
    def test_query_logs(self, audit_logger):
        """Test log querying."""
        # Add various log entries
        audit_logger.log_event("user1", "login", "system", "success")
        audit_logger.log_event("user2", "logout", "system", "success")
        audit_logger.log_event("user1", "update", "profile", "failure")
        
        # Query by user
        user1_logs = audit_logger.query_logs(user_id="user1")
        assert len(user1_logs) == 2
        
        # Query by action
        login_logs = audit_logger.query_logs(action="login")
        assert len(login_logs) == 1
        
        # Query by result
        failure_logs = audit_logger.query_logs(result="failure")
        assert len(failure_logs) == 1


class TestRBACManager:
    """Test Role-Based Access Control."""
    
    @pytest.fixture
    def rbac_manager(self):
        """Create RBAC manager instance."""
        return RBACManager()
    
    def test_role_creation(self, rbac_manager):
        """Test role creation and management."""
        rbac_manager.create_role("admin", ["read", "write", "delete"])
        rbac_manager.create_role("user", ["read"])
        
        assert rbac_manager.get_role_permissions("admin") == {"read", "write", "delete"}
        assert rbac_manager.get_role_permissions("user") == {"read"}
    
    def test_user_role_assignment(self, rbac_manager):
        """Test user role assignment."""
        rbac_manager.create_role("admin", ["read", "write", "delete"])
        rbac_manager.assign_role("user123", "admin")
        
        assert rbac_manager.get_user_role("user123") == "admin"
        assert rbac_manager.has_permission("user123", "write")
        assert not rbac_manager.has_permission("user123", "execute")
    
    def test_permission_check(self, rbac_manager):
        """Test permission checking."""
        rbac_manager.create_role("reader", ["read"])
        rbac_manager.create_role("writer", ["read", "write"])
        
        rbac_manager.assign_role("user1", "reader")
        rbac_manager.assign_role("user2", "writer")
        
        assert rbac_manager.has_permission("user1", "read")
        assert not rbac_manager.has_permission("user1", "write")
        assert rbac_manager.has_permission("user2", "write")
    
    def test_role_hierarchy(self, rbac_manager):
        """Test role hierarchy and inheritance."""
        rbac_manager.create_role("super_admin", ["*"])  # All permissions
        rbac_manager.create_role("admin", ["read", "write", "delete"])
        rbac_manager.create_role("moderator", ["read", "write"], parent="admin")
        
        # Moderator should inherit from admin
        assert rbac_manager.has_permission("moderator", "delete")
    
    def test_permission_decorator(self, rbac_manager):
        """Test permission decorator."""
        rbac_manager.create_role("admin", ["admin_access"])
        rbac_manager.assign_role("admin_user", "admin")
        
        @rbac_manager.require_permission("admin_access")
        def admin_function(user_id):
            return f"Admin function executed by {user_id}"
        
        # Should work for admin
        result = admin_function("admin_user")
        assert result == "Admin function executed by admin_user"
        
        # Should fail for non-admin
        with pytest.raises(PermissionError):
            admin_function("regular_user")


class TestSecureConfigLoader:
    """Test secure configuration loading."""
    
    @pytest.fixture
    def config_loader(self):
        """Create secure config loader."""
        return SecureConfigLoader()
    
    def test_load_encrypted_config(self, config_loader, tmp_path):
        """Test loading encrypted configuration."""
        # Create encrypted config
        config_data = {
            "database": {"host": "localhost", "password": "secret123"},
            "api_keys": {"openai": "sk-1234567890"}
        }
        
        config_file = tmp_path / "config.enc"
        key = config_loader.create_encrypted_config(config_data, config_file)
        
        # Load encrypted config
        loaded_config = config_loader.load_config(config_file, key)
        assert loaded_config == config_data
    
    def test_environment_variable_substitution(self, config_loader):
        """Test environment variable substitution."""
        config_data = {
            "database": {
                "host": "${DB_HOST:localhost}",
                "password": "${DB_PASSWORD}"
            }
        }
        
        with patch.dict(os.environ, {"DB_HOST": "prod-db", "DB_PASSWORD": "prod-pass"}):
            processed_config = config_loader.process_config(config_data)
            assert processed_config["database"]["host"] == "prod-db"
            assert processed_config["database"]["password"] == "prod-pass"
    
    def test_secure_defaults(self, config_loader):
        """Test secure default values."""
        config_data = {
            "security": {
                "encryption": "${ENCRYPTION_ENABLED:true}",
                "auth_required": "${AUTH_REQUIRED:true}"
            }
        }
        
        processed_config = config_loader.process_config(config_data)
        assert processed_config["security"]["encryption"] == "true"
        assert processed_config["security"]["auth_required"] == "true"
    
    def test_config_validation(self, config_loader):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "security": {"encryption_enabled": True},
            "database": {"host": "localhost", "port": 5432}
        }
        assert config_loader.validate_config(valid_config)
        
        # Invalid config (missing required fields)
        invalid_config = {
            "security": {}  # Missing encryption_enabled
        }
        with pytest.raises(ValueError, match="Missing required configuration"):
            config_loader.validate_config(invalid_config)
    
    def test_config_schema_validation(self, config_loader):
        """Test configuration schema validation."""
        schema = {
            "type": "object",
            "required": ["security", "database"],
            "properties": {
                "security": {
                    "type": "object",
                    "required": ["encryption_enabled"],
                    "properties": {
                        "encryption_enabled": {"type": "boolean"}
                    }
                },
                "database": {
                    "type": "object",
                    "required": ["host", "port"],
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"}
                    }
                }
            }
        }
        
        valid_config = {
            "security": {"encryption_enabled": True},
            "database": {"host": "localhost", "port": 5432}
        }
        
        assert config_loader.validate_schema(valid_config, schema)
        
        invalid_config = {
            "security": {"encryption_enabled": "yes"},  # Should be boolean
            "database": {"host": "localhost", "port": "5432"}  # Should be integer
        }
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            config_loader.validate_schema(invalid_config, schema)