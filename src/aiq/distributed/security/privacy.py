"""Privacy-preserving mechanisms for distributed AIQToolkit"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
import hmac

logger = logging.getLogger(__name__)


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy"""
    epsilon: float = 1.0       # Privacy budget
    delta: float = 1e-5        # Privacy parameter
    clip_norm: float = 1.0     # Gradient clipping norm
    noise_multiplier: float = 1.0
    mechanism: str = "gaussian"  # "gaussian" or "laplace"


@dataclass
class SecureAggregationConfig:
    """Configuration for secure aggregation"""
    threshold: int = 3         # Minimum clients for reconstruction
    key_size: int = 2048       # RSA key size
    session_timeout: int = 300  # Seconds


class PrivacyManager:
    """Manages privacy-preserving mechanisms"""
    
    def __init__(self, dp_config: DifferentialPrivacyConfig):
        self.dp_config = dp_config
        self.privacy_accountant = PrivacyAccountant(dp_config)
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to tensor for differential privacy"""
        if self.dp_config.mechanism == "gaussian":
            return self._add_gaussian_noise(tensor)
        elif self.dp_config.mechanism == "laplace":
            return self._add_laplace_noise(tensor)
        else:
            raise ValueError(f"Unknown mechanism: {self.dp_config.mechanism}")
            
    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for differential privacy"""
        sensitivity = self._compute_sensitivity(tensor)
        
        # Calculate noise scale
        noise_scale = (
            sensitivity * self.dp_config.noise_multiplier / 
            self.dp_config.epsilon
        )
        
        # Generate and add noise
        noise = torch.randn_like(tensor) * noise_scale
        return tensor + noise
        
    def _add_laplace_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Laplace noise for differential privacy"""
        sensitivity = self._compute_sensitivity(tensor)
        
        # Calculate noise scale
        noise_scale = sensitivity / self.dp_config.epsilon
        
        # Generate Laplace noise
        noise = torch.distributions.Laplace(
            torch.zeros_like(tensor),
            torch.ones_like(tensor) * noise_scale
        ).sample()
        
        return tensor + noise
        
    def _compute_sensitivity(self, tensor: torch.Tensor) -> float:
        """Compute sensitivity of tensor"""
        # L2 sensitivity for bounded functions
        return self.dp_config.clip_norm
        
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity"""
        clipped = {}
        
        # Compute total norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        if total_norm > self.dp_config.clip_norm:
            clip_factor = self.dp_config.clip_norm / total_norm
            for name, grad in gradients.items():
                clipped[name] = grad * clip_factor
        else:
            clipped = gradients
            
        return clipped
        
    def track_privacy_spent(self, num_iterations: int):
        """Track privacy budget spent"""
        self.privacy_accountant.track_spending(num_iterations)
        
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent"""
        return self.privacy_accountant.get_spent()


class PrivacyAccountant:
    """Tracks privacy budget spending"""
    
    def __init__(self, config: DifferentialPrivacyConfig):
        self.config = config
        self.steps = 0
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        
    def track_spending(self, num_steps: int):
        """Track privacy spending for given steps"""
        self.steps += num_steps
        
        # Simple composition for Gaussian mechanism
        if self.config.mechanism == "gaussian":
            self.epsilon_spent = self.config.epsilon * np.sqrt(self.steps)
            self.delta_spent = self.config.delta * self.steps
        else:
            # Basic composition for other mechanisms
            self.epsilon_spent = self.config.epsilon * self.steps
            self.delta_spent = self.config.delta * self.steps
            
    def get_spent(self) -> Tuple[float, float]:
        """Get spent privacy budget"""
        return self.epsilon_spent, self.delta_spent
        
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        total_epsilon = self.config.epsilon * 100  # Assume max 100 iterations
        total_delta = self.config.delta * 100
        
        return (
            max(0, total_epsilon - self.epsilon_spent),
            max(0, total_delta - self.delta_spent)
        )


class SecureAggregator:
    """Implements secure aggregation for federated learning"""
    
    def __init__(self, config: SecureAggregationConfig):
        self.config = config
        self.private_key = None
        self.public_key = None
        self.client_keys: Dict[str, Any] = {}
        self.session_keys: Dict[str, bytes] = {}
        self._generate_keys()
        
    def _generate_keys(self):
        """Generate RSA key pair"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
    def register_client(self, client_id: str, public_key_pem: bytes):
        """Register client's public key"""
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        self.client_keys[client_id] = public_key
        logger.info(f"Registered public key for client {client_id}")
        
    def generate_session_key(self, client_id: str) -> bytes:
        """Generate session key for client"""
        # Generate random session key
        session_key = os.urandom(32)  # 256-bit key
        self.session_keys[client_id] = session_key
        
        # Encrypt with client's public key
        client_key = self.client_keys[client_id]
        encrypted_key = client_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_key
        
    def aggregate_encrypted(self, encrypted_updates: Dict[str, bytes]) -> bytes:
        """Aggregate encrypted updates"""
        if len(encrypted_updates) < self.config.threshold:
            raise ValueError(f"Need at least {self.config.threshold} clients")
            
        # Decrypt updates
        decrypted = {}
        for client_id, encrypted in encrypted_updates.items():
            session_key = self.session_keys.get(client_id)
            if not session_key:
                logger.warning(f"No session key for client {client_id}")
                continue
                
            decrypted[client_id] = self._decrypt_update(encrypted, session_key)
            
        # Aggregate decrypted updates
        aggregated = self._aggregate_updates(decrypted)
        
        # Encrypt result
        return self._encrypt_result(aggregated)
        
    def _decrypt_update(self, encrypted: bytes, session_key: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt client update"""
        # Setup AES decryption
        iv = encrypted[:16]
        ciphertext = encrypted[16:]
        
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Deserialize tensors
        return self._deserialize_tensors(plaintext)
        
    def _aggregate_updates(self, updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate decrypted updates"""
        aggregated = {}
        num_clients = len(updates)
        
        # Get parameter names from first client
        first_client = next(iter(updates.values()))
        
        for param_name in first_client.keys():
            # Average parameters
            param_sum = torch.zeros_like(first_client[param_name])
            
            for client_updates in updates.values():
                param_sum += client_updates[param_name]
                
            aggregated[param_name] = param_sum / num_clients
            
        return aggregated
        
    def _encrypt_result(self, result: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt aggregated result"""
        # Serialize tensors
        serialized = self._serialize_tensors(result)
        
        # Generate new session key for result
        session_key = os.urandom(32)
        iv = os.urandom(16)
        
        # Encrypt with AES
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad to block size
        padded = self._pad_data(serialized)
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        
        return iv + ciphertext
        
    def _serialize_tensors(self, tensors: Dict[str, torch.Tensor]) -> bytes:
        """Serialize tensors to bytes"""
        # Simple serialization - in production use protobuf
        data = {}
        for name, tensor in tensors.items():
            data[name] = {
                "shape": list(tensor.shape),
                "data": tensor.cpu().numpy().tolist()
            }
        return json.dumps(data).encode()
        
    def _deserialize_tensors(self, data: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize tensors from bytes"""
        parsed = json.loads(data.decode())
        tensors = {}
        
        for name, tensor_data in parsed.items():
            arr = np.array(tensor_data["data"])
            arr = arr.reshape(tensor_data["shape"])
            tensors[name] = torch.from_numpy(arr)
            
        return tensors
        
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data to AES block size"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length]) * padding_length
        return data + padding


class HomomorphicEncryption:
    """Homomorphic encryption for privacy-preserving computation"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        # In practice, use a proper HE library like SEAL or HElib
        self._setup_keys()
        
    def _setup_keys(self):
        """Setup homomorphic encryption keys"""
        # Placeholder - real implementation would use HE library
        self.public_key = None
        self.private_key = None
        self.evaluation_key = None
        
    def encrypt(self, value: float) -> Any:
        """Encrypt a value"""
        # Placeholder for HE encryption
        return value
        
    def decrypt(self, ciphertext: Any) -> float:
        """Decrypt a ciphertext"""
        # Placeholder for HE decryption
        return ciphertext
        
    def add(self, ct1: Any, ct2: Any) -> Any:
        """Add two ciphertexts"""
        # Homomorphic addition
        return ct1 + ct2
        
    def multiply(self, ct: Any, scalar: float) -> Any:
        """Multiply ciphertext by scalar"""
        # Homomorphic scalar multiplication
        return ct * scalar
        
    def aggregate_encrypted_gradients(self, 
                                    encrypted_grads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate encrypted gradients"""
        if not encrypted_grads:
            return {}
            
        # Initialize with first gradient
        aggregated = {}
        for param_name in encrypted_grads[0].keys():
            aggregated[param_name] = encrypted_grads[0][param_name]
            
        # Add remaining gradients
        for grad_dict in encrypted_grads[1:]:
            for param_name, encrypted_grad in grad_dict.items():
                aggregated[param_name] = self.add(
                    aggregated[param_name],
                    encrypted_grad
                )
                
        # Scale by number of clients
        num_clients = len(encrypted_grads)
        for param_name in aggregated:
            aggregated[param_name] = self.multiply(
                aggregated[param_name],
                1.0 / num_clients
            )
            
        return aggregated


class MultiPartyComputation:
    """Secure multi-party computation protocols"""
    
    def __init__(self, num_parties: int, threshold: int):
        self.num_parties = num_parties
        self.threshold = threshold
        self.shares: Dict[int, Dict[str, Any]] = {}
        
    def create_shares(self, secret: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """Create Shamir secret shares"""
        shares = []
        
        # Simple additive secret sharing for illustration
        # Real implementation would use polynomial secret sharing
        random_shares = []
        for i in range(self.num_parties - 1):
            share = torch.randn_like(secret)
            random_shares.append(share)
            shares.append((i, share))
            
        # Last share ensures sum equals secret
        last_share = secret - sum(random_shares)
        shares.append((self.num_parties - 1, last_share))
        
        return shares
        
    def reconstruct_secret(self, shares: List[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Reconstruct secret from shares"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
            
        # Simple reconstruction for additive shares
        secret = torch.zeros_like(shares[0][1])
        for _, share in shares:
            secret += share
            
        return secret
        
    def secure_comparison(self, x_shares: List[Tuple[int, torch.Tensor]], 
                         y_shares: List[Tuple[int, torch.Tensor]]) -> List[Tuple[int, bool]]:
        """Secure comparison protocol"""
        # Simplified - real implementation would use garbled circuits
        # or GMW protocol
        result_shares = []
        
        for i in range(self.num_parties):
            # Each party compares their shares locally
            comparison = x_shares[i][1] > y_shares[i][1]
            result_shares.append((i, comparison))
            
        return result_shares
        
    def secure_aggregation(self, 
                          client_shares: Dict[str, List[Tuple[int, torch.Tensor]]]) -> torch.Tensor:
        """Secure aggregation using MPC"""
        # Aggregate shares from each party
        party_sums = {}
        
        for party_id in range(self.num_parties):
            party_sum = None
            
            for client_id, shares in client_shares.items():
                share = next(s for pid, s in shares if pid == party_id)
                
                if party_sum is None:
                    party_sum = share
                else:
                    party_sum += share
                    
            party_sums[party_id] = party_sum
            
        # Reconstruct aggregated result
        shares_list = [(pid, sum_val) for pid, sum_val in party_sums.items()]
        return self.reconstruct_secret(shares_list)