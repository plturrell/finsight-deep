"""
Configuration for Nash-Ethereum Consensus System
Loads sensitive data from environment variables
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsensusConfig:
    """Configuration for consensus system"""
    
    # Ethereum Configuration
    contract_address: Optional[str] = None
    private_key: Optional[str] = None
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 1337
    gas_limit: int = 1000000
    gas_price_gwei: float = 20.0
    
    # Nash Configuration
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    learning_rate: float = 0.1
    
    # Security Configuration
    min_stake: float = 0.1
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 10
    signature_required: bool = True
    
    # Layer 2 Configuration
    use_layer2: bool = False
    layer2_network: str = "optimism"
    layer2_rpc_url: Optional[str] = None
    
    # Monitoring Configuration
    enable_monitoring: bool = True
    prometheus_port: int = 9090
    
    @classmethod
    def from_env(cls) -> "ConsensusConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Load Ethereum settings
        config.contract_address = os.getenv("CONSENSUS_CONTRACT_ADDRESS")
        config.private_key = os.getenv("CONSENSUS_PRIVATE_KEY")
        config.rpc_url = os.getenv("CONSENSUS_RPC_URL", config.rpc_url)
        config.chain_id = int(os.getenv("CONSENSUS_CHAIN_ID", str(config.chain_id)))
        config.gas_limit = int(os.getenv("CONSENSUS_GAS_LIMIT", str(config.gas_limit)))
        config.gas_price_gwei = float(os.getenv("CONSENSUS_GAS_PRICE", str(config.gas_price_gwei)))
        
        # Load Nash settings
        config.max_iterations = int(os.getenv("NASH_MAX_ITERATIONS", str(config.max_iterations)))
        config.convergence_threshold = float(os.getenv("NASH_CONVERGENCE_THRESHOLD", str(config.convergence_threshold)))
        config.learning_rate = float(os.getenv("NASH_LEARNING_RATE", str(config.learning_rate)))
        
        # Load security settings
        config.min_stake = float(os.getenv("CONSENSUS_MIN_STAKE", str(config.min_stake)))
        config.rate_limit_window = int(os.getenv("CONSENSUS_RATE_LIMIT_WINDOW", str(config.rate_limit_window)))
        config.rate_limit_max_requests = int(os.getenv("CONSENSUS_RATE_LIMIT_MAX", str(config.rate_limit_max_requests)))
        config.signature_required = os.getenv("CONSENSUS_REQUIRE_SIGNATURE", "true").lower() == "true"
        
        # Load Layer 2 settings
        config.use_layer2 = os.getenv("CONSENSUS_USE_LAYER2", "false").lower() == "true"
        config.layer2_network = os.getenv("CONSENSUS_LAYER2_NETWORK", config.layer2_network)
        config.layer2_rpc_url = os.getenv("CONSENSUS_LAYER2_RPC_URL")
        
        # Load monitoring settings
        config.enable_monitoring = os.getenv("CONSENSUS_ENABLE_MONITORING", "true").lower() == "true"
        config.prometheus_port = int(os.getenv("CONSENSUS_PROMETHEUS_PORT", str(config.prometheus_port)))
        
        # Validate critical settings
        if not config.validate():
            logger.warning("Configuration validation failed - using defaults for missing values")
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not self.private_key:
            errors.append("CONSENSUS_PRIVATE_KEY not set")
        
        if not self.contract_address:
            errors.append("CONSENSUS_CONTRACT_ADDRESS not set")
        
        if self.use_layer2 and not self.layer2_rpc_url:
            errors.append("Layer 2 enabled but CONSENSUS_LAYER2_RPC_URL not set")
        
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)"""
        return {
            "contract_address": self.contract_address[:10] + "..." if self.contract_address else None,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "gas_limit": self.gas_limit,
            "gas_price_gwei": self.gas_price_gwei,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "learning_rate": self.learning_rate,
            "min_stake": self.min_stake,
            "rate_limit_window": self.rate_limit_window,
            "rate_limit_max_requests": self.rate_limit_max_requests,
            "signature_required": self.signature_required,
            "use_layer2": self.use_layer2,
            "layer2_network": self.layer2_network,
            "enable_monitoring": self.enable_monitoring,
            "prometheus_port": self.prometheus_port
        }


# Default configuration for development
DEFAULT_CONFIG = ConsensusConfig()

# Production configuration loaded from environment
PRODUCTION_CONFIG = ConsensusConfig.from_env()


def get_config() -> ConsensusConfig:
    """Get active configuration based on environment"""
    env = os.getenv("AIQ_ENV", "development")
    
    if env == "production":
        return PRODUCTION_CONFIG
    else:
        return DEFAULT_CONFIG