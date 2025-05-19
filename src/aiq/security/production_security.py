"""
Production Security Implementation
Complete security stack for neural supercomputer deployment
"""

import os
import secrets
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta

# Cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, utils
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate
import jwt

# Blockchain
from web3 import Web3
from eth_account import Account
from eth_typing import Address
import binascii

# Security frameworks
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pyotp
import bcrypt
from argon2 import PasswordHasher
import pykeepass

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import logging
from typing import Callable

# Metrics
security_events_counter = Counter('security_events_total', 'Total security events', ['event_type'])
auth_attempts_counter = Counter('auth_attempts_total', 'Authentication attempts', ['status'])
encryption_operations_histogram = Histogram('encryption_duration_seconds', 'Encryption operation duration')
active_sessions_gauge = Gauge('active_sessions', 'Number of active sessions')

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for production deployment"""
    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_function: str = "PBKDF2"
    pbkdf2_iterations: int = 100000
    
    # Authentication
    jwt_algorithm: str = "RS256"
    jwt_expiration_hours: int = 24
    mfa_required: bool = True
    password_min_length: int = 16
    password_require_special: bool = True
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    ddos_protection: bool = True
    
    # Blockchain
    ethereum_network: str = "mainnet"
    contract_address: Optional[str] = None
    gas_price_gwei: float = 50.0
    confirmation_blocks: int = 12
    
    # Monitoring
    audit_log_enabled: bool = True
    security_monitoring_interval: int = 60
    anomaly_detection: bool = True


class ProductionSecurityManager:
    """
    Comprehensive security manager for production deployment
    Handles encryption, authentication, blockchain, and monitoring
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_hasher = PasswordHasher()
        
        # Initialize components
        self._init_encryption()
        self._init_authentication()
        self._init_blockchain()
        self._init_monitoring()
        self._init_rate_limiting()
        
        # Session management
        self.active_sessions: Dict[str, Dict] = {}
        self.blacklisted_tokens: set = set()
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._security_monitoring_loop())
    
    def _init_encryption(self):
        """Initialize encryption systems"""
        # Generate master key from environment
        master_key_seed = os.environ.get("MASTER_KEY_SEED")
        if not master_key_seed:
            raise ValueError("MASTER_KEY_SEED environment variable not set")
        
        # Derive master key using KDF
        salt = b"supercomputer-neural-salt"  # Should be stored securely
        kdf = hashes.PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.pbkdf2_iterations
        )
        self.master_key = kdf.derive(master_key_seed.encode())
        
        # Initialize hardware security module if available
        self.hsm = self._init_hsm()
    
    def _init_hsm(self):
        """Initialize Hardware Security Module if available"""
        try:
            import pkcs11
            # Initialize HSM connection
            lib = pkcs11.lib(os.environ.get("HSM_MODULE_PATH"))
            token = lib.get_token(token_label=os.environ.get("HSM_TOKEN_LABEL"))
            session = token.open(user_pin=os.environ.get("HSM_PIN"))
            return session
        except:
            logger.warning("HSM not available, using software encryption")
            return None
    
    def _init_authentication(self):
        """Initialize authentication systems"""
        # Load RSA keys for JWT
        private_key_path = os.environ.get("JWT_PRIVATE_KEY_PATH")
        public_key_path = os.environ.get("JWT_PUBLIC_KEY_PATH")
        
        with open(private_key_path, 'rb') as f:
            self.jwt_private_key = serialization.load_pem_private_key(
                f.read(),
                password=os.environ.get("JWT_KEY_PASSWORD").encode(),
                backend=default_backend()
            )
        
        with open(public_key_path, 'rb') as f:
            self.jwt_public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        
        # Initialize MFA
        self.totp_secrets: Dict[str, str] = {}
        
        # Load user database
        self.user_db = self._load_user_database()
    
    def _init_blockchain(self):
        """Initialize blockchain connection"""
        # Connect to Ethereum node
        self.w3 = Web3(Web3.HTTPProvider(os.environ.get("ETHEREUM_RPC_URL")))
        
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        # Load account
        self.account = Account.from_key(os.environ.get("ETHEREUM_PRIVATE_KEY"))
        
        # Load smart contract
        contract_address = self.config.contract_address or os.environ.get("CONTRACT_ADDRESS")
        with open('contracts/SecurityAudit.json', 'r') as f:
            contract_abi = json.load(f)['abi']
        
        self.contract = self.w3.eth.contract(
            address=Web3.toChecksumAddress(contract_address),
            abi=contract_abi
        )
    
    def _init_monitoring(self):
        """Initialize security monitoring"""
        # Setup logging
        self.audit_logger = logging.getLogger("security.audit")
        handler = logging.FileHandler("/var/log/security/audit.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)
        
        # Initialize anomaly detection
        if self.config.anomaly_detection:
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(contamination=0.1)
            self.security_features: List[List[float]] = []
    
    def _init_rate_limiting(self):
        """Initialize rate limiting"""
        from redis import Redis
        
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_client = Redis.from_url(redis_url)
        
        self.limiter = Limiter(
            app=None,  # Will be set when Flask app is created
            key_func=get_remote_address,
            storage_uri=redis_url,
            default_limits=[f"{self.config.rate_limit_per_minute}/minute"]
        )
    
    # Encryption Operations
    
    def encrypt_data(self, data: bytes, associated_data: Optional[bytes] = None) -> Dict[str, bytes]:
        """Encrypt data using AES-256-GCM"""
        with encryption_operations_histogram.time():
            # Generate random nonce
            nonce = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Add associated data if provided
            if associated_data:
                encryptor.authenticate_additional_data(associated_data)
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            return {
                'nonce': nonce,
                'ciphertext': ciphertext,
                'tag': encryptor.tag
            }
    
    def decrypt_data(self, encrypted_data: Dict[str, bytes], associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-256-GCM"""
        with encryption_operations_histogram.time():
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(encrypted_data['nonce'], encrypted_data['tag']),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            if associated_data:
                decryptor.authenticate_additional_data(associated_data)
            
            return decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()
    
    # Authentication
    
    def create_user(self, username: str, password: str, email: str, role: str = "user") -> Dict[str, Any]:
        """Create a new user with secure password hashing"""
        # Validate password
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
        
        # Hash password
        password_hash = self.password_hasher.hash(password)
        
        # Generate MFA secret
        totp_secret = pyotp.random_base32()
        self.totp_secrets[username] = totp_secret
        
        # Create user record
        user = {
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "mfa_enabled": self.config.mfa_required,
            "totp_secret": totp_secret,
            "status": "active"
        }
        
        # Store in database
        self.user_db[username] = user
        self._save_user_database()
        
        # Log creation
        self.audit_logger.info(f"User created: {username}")
        security_events_counter.labels(event_type="user_created").inc()
        
        return {"username": username, "totp_qr": self._generate_totp_qr(username, totp_secret)}
    
    def authenticate_user(self, username: str, password: str, totp_code: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return JWT token"""
        user = self.user_db.get(username)
        
        if not user:
            auth_attempts_counter.labels(status="failed").inc()
            return None
        
        # Verify password
        try:
            self.password_hasher.verify(user["password_hash"], password)
        except:
            auth_attempts_counter.labels(status="failed").inc()
            return None
        
        # Check if password needs rehashing
        if self.password_hasher.check_needs_rehash(user["password_hash"]):
            user["password_hash"] = self.password_hasher.hash(password)
            self._save_user_database()
        
        # Verify MFA if required
        if self.config.mfa_required and user.get("mfa_enabled", False):
            if not totp_code:
                return None
            
            totp = pyotp.TOTP(self.totp_secrets.get(username))
            if not totp.verify(totp_code, valid_window=1):
                auth_attempts_counter.labels(status="mfa_failed").inc()
                return None
        
        # Generate JWT token
        payload = {
            "username": username,
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)
        }
        
        token = jwt.encode(payload, self.jwt_private_key, algorithm=self.config.jwt_algorithm)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            "username": username,
            "token": token,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        active_sessions_gauge.inc()
        auth_attempts_counter.labels(status="success").inc()
        self.audit_logger.info(f"User authenticated: {username}")
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        if token in self.blacklisted_tokens:
            return None
        
        try:
            payload = jwt.decode(token, self.jwt_public_key, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    # Blockchain Operations
    
    async def log_security_event_blockchain(self, event_type: str, event_data: Dict[str, Any]):
        """Log security event to blockchain"""
        # Prepare event data
        event = {
            "event_type": event_type,
            "timestamp": int(datetime.utcnow().timestamp()),
            "data_hash": hashlib.sha256(json.dumps(event_data).encode()).hexdigest()
        }
        
        # Build transaction
        transaction = self.contract.functions.logSecurityEvent(
            event["event_type"],
            event["timestamp"],
            event["data_hash"]
        ).build_transaction({
            'from': self.account.address,
            'gas': 100000,
            'gasPrice': Web3.toWei(self.config.gas_price_gwei, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(self.account.address)
        })
        
        # Sign and send transaction
        signed = self.w3.eth.account.sign_transaction(transaction, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        
        # Verify confirmations
        for _ in range(self.config.confirmation_blocks):
            await asyncio.sleep(15)
            block_number = self.w3.eth.block_number
            if block_number - receipt.blockNumber < self.config.confirmation_blocks:
                continue
        
        self.audit_logger.info(f"Security event logged to blockchain: {tx_hash.hex()}")
        return tx_hash.hex()
    
    async def verify_blockchain_audit(self, tx_hash: str) -> Dict[str, Any]:
        """Verify security audit from blockchain"""
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            # Decode logs
            logs = self.contract.events.SecurityEventLogged().process_receipt(receipt)
            
            if logs:
                event = logs[0]
                return {
                    "event_type": event["args"]["eventType"],
                    "timestamp": event["args"]["timestamp"],
                    "data_hash": event["args"]["dataHash"],
                    "block_number": event["blockNumber"],
                    "tx_hash": tx_hash
                }
        except:
            return None
    
    # Security Monitoring
    
    async def _security_monitoring_loop(self):
        """Background security monitoring"""
        while True:
            try:
                # Check for anomalies
                if self.config.anomaly_detection:
                    await self._detect_anomalies()
                
                # Check session timeouts
                await self._check_session_timeouts()
                
                # Monitor failed authentication attempts
                await self._monitor_auth_attempts()
                
                # Check blockchain synchronization
                await self._check_blockchain_sync()
                
                await asyncio.sleep(self.config.security_monitoring_interval)
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
    
    async def _detect_anomalies(self):
        """Detect security anomalies using ML"""
        # Collect current features
        features = [
            len(self.active_sessions),
            auth_attempts_counter._value.get(("failed",), 0),
            auth_attempts_counter._value.get(("success",), 0),
            security_events_counter._value.get(("intrusion_attempt",), 0)
        ]
        
        self.security_features.append(features)
        
        # Train/update model periodically
        if len(self.security_features) > 1000:
            self.anomaly_detector.fit(self.security_features[-10000:])
            
            # Check current state
            anomaly_score = self.anomaly_detector.decision_function([features])[0]
            
            if anomaly_score < -0.5:  # Anomaly threshold
                await self.log_security_event_blockchain(
                    "anomaly_detected",
                    {"score": anomaly_score, "features": features}
                )
                security_events_counter.labels(event_type="anomaly_detected").inc()
    
    def _validate_password(self, password: str) -> bool:
        """Validate password meets security requirements"""
        if len(password) < self.config.password_min_length:
            return False
        
        if self.config.password_require_special:
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                return False
        
        # Check for common passwords
        if password.lower() in ["password", "admin", "12345678", "qwerty"]:
            return False
        
        return True
    
    def _generate_totp_qr(self, username: str, secret: str) -> str:
        """Generate TOTP QR code"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username,
            issuer_name="Neural Supercomputer"
        )
        return totp_uri
    
    # Additional security features
    
    def enable_2fa_for_user(self, username: str) -> str:
        """Enable 2FA for a user"""
        user = self.user_db.get(username)
        if not user:
            raise ValueError("User not found")
        
        # Generate new TOTP secret if not exists
        if username not in self.totp_secrets:
            totp_secret = pyotp.random_base32()
            self.totp_secrets[username] = totp_secret
        else:
            totp_secret = self.totp_secrets[username]
        
        user["mfa_enabled"] = True
        self._save_user_database()
        
        return self._generate_totp_qr(username, totp_secret)
    
    def revoke_token(self, token: str):
        """Revoke a JWT token"""
        self.blacklisted_tokens.add(token)
        
        # Remove associated sessions
        for session_id, session in list(self.active_sessions.items()):
            if session["token"] == token:
                del self.active_sessions[session_id]
                active_sessions_gauge.dec()
    
    async def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit"""
        audit_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "active_sessions": len(self.active_sessions),
            "failed_auth_attempts": auth_attempts_counter._value.get(("failed",), 0),
            "security_events": dict(security_events_counter._value),
            "blockchain_sync": self.w3.is_connected(),
            "encryption_operations": dict(encryption_operations_histogram._count)
        }
        
        # Log to blockchain
        tx_hash = await self.log_security_event_blockchain("security_audit", audit_results)
        audit_results["blockchain_tx"] = tx_hash
        
        return audit_results