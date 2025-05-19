"""
Security Configuration for AIQToolkit
Manages sensitive data across all components
"""

import os
from typing import Any, Optional
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Centralized security configuration"""
    
    # API Security
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    
    # OpenAI/LLM Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    nim_api_key: Optional[str] = None
    
    # Database Credentials
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    milvus_host: Optional[str] = None
    milvus_port: int = 19530
    
    # AWS Credentials
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # Google Cloud
    google_application_credentials: Optional[str] = None
    
    # Azure
    azure_subscription_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    
    # Blockchain/Consensus
    contract_address: Optional[str] = None
    private_key: Optional[str] = None
    infura_project_id: Optional[str] = None
    alchemy_api_key: Optional[str] = None
    
    # Third-party Services
    github_token: Optional[str] = None
    jira_api_token: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    # Security Settings
    enable_https: bool = True
    cors_origins: list[str] = None
    rate_limit_enabled: bool = True
    
    # Authentication Settings
    enable_auth: bool = True
    session_timeout: int = 1800  # 30 minutes
    max_login_attempts: int = 5
    lockout_duration: int = 1800  # 30 minutes
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    rate_limit_burst: int = 150
    
    # Encryption
    enable_data_encryption: bool = True
    encryption_key: Optional[str] = None
    
    # Audit Settings
    enable_audit_logging: bool = True
    audit_log_file: str = "audit.log"
    audit_retention_days: int = 90
    
    # Security Headers
    enable_security_headers: bool = True
    csp_policy: Optional[str] = None
    hsts_max_age: int = 31536000
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8080"]
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # API Security
        config.api_key = os.getenv("AIQ_API_KEY")
        config.jwt_secret = os.getenv("AIQ_JWT_SECRET", os.urandom(32).hex())
        
        # LLM Keys
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        config.nim_api_key = os.getenv("NIM_API_KEY")
        
        # Database
        config.database_url = os.getenv("DATABASE_URL")
        config.redis_url = os.getenv("REDIS_URL")
        config.milvus_host = os.getenv("MILVUS_HOST")
        config.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        
        # AWS
        config.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        config.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        config.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        # GCP
        config.google_application_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Azure
        config.azure_subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        config.azure_tenant_id = os.getenv("AZURE_TENANT_ID")
        config.azure_client_id = os.getenv("AZURE_CLIENT_ID")
        config.azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        # Blockchain
        config.contract_address = os.getenv("CONSENSUS_CONTRACT_ADDRESS")
        config.private_key = os.getenv("CONSENSUS_PRIVATE_KEY")
        config.infura_project_id = os.getenv("INFURA_PROJECT_ID")
        config.alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
        
        # Third-party
        config.github_token = os.getenv("GITHUB_TOKEN")
        config.jira_api_token = os.getenv("JIRA_API_TOKEN")
        config.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        # Security Settings
        config.enable_https = os.getenv("ENABLE_HTTPS", "true").lower() == "true"
        config.cors_origins = [
            origin.strip() 
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
        ]
        config.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        
        # Authentication
        config.enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        config.session_timeout = int(os.getenv("SESSION_TIMEOUT", "1800"))
        config.max_login_attempts = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
        config.lockout_duration = int(os.getenv("LOCKOUT_DURATION", "1800"))
        
        # Rate Limiting
        config.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        config.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        config.rate_limit_burst = int(os.getenv("RATE_LIMIT_BURST", "150"))
        
        # Encryption
        config.enable_data_encryption = os.getenv("ENABLE_DATA_ENCRYPTION", "true").lower() == "true"
        config.encryption_key = os.getenv("DATA_ENCRYPTION_KEY")
        
        # Audit
        config.enable_audit_logging = os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true"
        config.audit_log_file = os.getenv("AUDIT_LOG_FILE", "audit.log")
        config.audit_retention_days = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))
        
        # Security Headers
        config.enable_security_headers = os.getenv("ENABLE_SECURITY_HEADERS", "true").lower() == "true"
        config.csp_policy = os.getenv("CSP_POLICY")
        config.hsts_max_age = int(os.getenv("HSTS_MAX_AGE", "31536000"))
        
        return config
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        # Check API security
        if not self.api_key:
            warnings.append("AIQ_API_KEY not set - API endpoints unprotected")
        
        if not self.jwt_secret:
            warnings.append("AIQ_JWT_SECRET not set - using random secret")
        
        # Check LLM keys
        if not any([self.openai_api_key, self.anthropic_api_key, self.nim_api_key]):
            warnings.append("No LLM API keys set - AI features may not work")
        
        # Check database
        if not any([self.database_url, self.redis_url, self.milvus_host]):
            warnings.append("No database connections configured")
        
        # Check blockchain (if consensus features used)
        if not all([self.contract_address, self.private_key]) and os.getenv("ENABLE_CONSENSUS"):
            warnings.append("Consensus configuration incomplete")
        
        return warnings
    
    def mask_sensitive(self, value: Optional[str]) -> str:
        """Mask sensitive values for logging"""
        if not value:
            return "Not set"
        if len(value) < 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"
    
    def to_dict(self, mask_sensitive: bool = True) -> dict[str, Any]:
        """Convert to dictionary with optional masking"""
        result = {}
        
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            
            if mask_sensitive and field in [
                "api_key", "jwt_secret", "openai_api_key", "anthropic_api_key",
                "nim_api_key", "aws_secret_access_key", "azure_client_secret",
                "private_key", "github_token", "jira_api_token"
            ]:
                value = self.mask_sensitive(value)
            
            result[field] = value
        
        return result
    
    def save_to_file(self, filepath: str, mask_sensitive: bool = True):
        """Save configuration to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(mask_sensitive), f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {filepath}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error loading config from {filepath}: {e}")


# Global configuration instance
_config = None


def get_security_config() -> SecurityConfig:
    """Get global security configuration"""
    global _config
    if _config is None:
        _config = SecurityConfig.from_env()
        
        # Try to load from default locations
        config_locations = [
            os.getenv("AIQ_CONFIG_FILE"),
            "config/security.json",
            "/Users/apple/projects/AIQToolkit/config/security.json"
        ]
        
        for config_file in config_locations:
            if config_file and os.path.exists(config_file):
                _config.load_from_file(config_file)
                break
        
        # Validate and log warnings
        warnings = _config.validate()
        for warning in warnings:
            logger.warning(f"Security config: {warning}")
        
        # Log configuration (masked)
        logger.info("Security configuration loaded")
        logger.debug(f"Config: {_config.to_dict(mask_sensitive=True)}")
    
    return _config


def require_api_key(func):
    """Decorator to require API key for endpoints"""
    from functools import wraps
    from fastapi import HTTPException, Header
    
    @wraps(func)
    async def wrapper(*args, api_key: str = Header(None), **kwargs):
        config = get_security_config()
        
        if config.api_key and api_key != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return await func(*args, **kwargs)
    
    return wrapper