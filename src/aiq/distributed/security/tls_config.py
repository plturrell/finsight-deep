# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TLS/SSL configuration for secure distributed communication
"""

import os
import ssl
from dataclasses import dataclass
from typing import Optional, Tuple
import grpc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TLSConfig:
    """TLS configuration for secure communication"""
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None
    server_cert_path: Optional[str] = None
    server_key_path: Optional[str] = None
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True
    
    def validate(self) -> bool:
        """Validate TLS configuration files exist"""
        paths = [
            self.ca_cert_path,
            self.client_cert_path,
            self.client_key_path,
            self.server_cert_path,
            self.server_key_path
        ]
        
        for path in paths:
            if path and not Path(path).exists():
                logger.error(f"TLS file not found: {path}")
                return False
        return True
    
    def load_certificates(self) -> Tuple[bytes, bytes, bytes]:
        """Load certificate files"""
        ca_cert = None
        client_cert = None
        client_key = None
        
        if self.ca_cert_path:
            with open(self.ca_cert_path, 'rb') as f:
                ca_cert = f.read()
                
        if self.client_cert_path:
            with open(self.client_cert_path, 'rb') as f:
                client_cert = f.read()
                
        if self.client_key_path:
            with open(self.client_key_path, 'rb') as f:
                client_key = f.read()
                
        return ca_cert, client_cert, client_key


class TLSManager:
    """Manages TLS/SSL for secure distributed communication"""
    
    def __init__(self, config: Optional[TLSConfig] = None):
        self.config = config or TLSConfig()
        self.enabled = bool(config)
        
    def create_server_credentials(self) -> grpc.ServerCredentials:
        """Create secure server credentials"""
        if not self.enabled:
            logger.warning("TLS not enabled, using insecure connection")
            return None
            
        if not self.config.validate():
            raise ValueError("Invalid TLS configuration")
            
        # Load certificates
        with open(self.config.ca_cert_path, 'rb') as f:
            ca_cert = f.read()
        with open(self.config.server_cert_path, 'rb') as f:
            server_cert = f.read()
        with open(self.config.server_key_path, 'rb') as f:
            server_key = f.read()
        
        # Create credentials
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True
        )
        
        logger.info("Created secure server credentials")
        return credentials
    
    def create_client_credentials(self) -> grpc.ChannelCredentials:
        """Create secure client credentials"""
        if not self.enabled:
            logger.warning("TLS not enabled, using insecure connection")
            return None
            
        if not self.config.validate():
            raise ValueError("Invalid TLS configuration")
            
        # Load certificates
        ca_cert, client_cert, client_key = self.config.load_certificates()
        
        # Create credentials
        credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert
        )
        
        logger.info("Created secure client credentials")
        return credentials
    
    def create_secure_channel(self, target: str) -> grpc.Channel:
        """Create a secure gRPC channel"""
        if not self.enabled:
            return grpc.insecure_channel(target)
            
        credentials = self.create_client_credentials()
        channel = grpc.secure_channel(target, credentials)
        
        logger.info(f"Created secure channel to {target}")
        return channel
    
    def create_secure_server(self, port: int) -> grpc.Server:
        """Create a secure gRPC server"""
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        
        if self.enabled:
            credentials = self.create_server_credentials()
            server.add_secure_port(f'[::]:{port}', credentials)
            logger.info(f"Created secure server on port {port}")
        else:
            server.add_insecure_port(f'[::]:{port}')
            logger.warning(f"Created insecure server on port {port}")
            
        return server


def load_tls_config_from_env() -> Optional[TLSConfig]:
    """Load TLS configuration from environment variables"""
    ca_cert = os.getenv("AIQ_TLS_CA_CERT")
    client_cert = os.getenv("AIQ_TLS_CLIENT_CERT")
    client_key = os.getenv("AIQ_TLS_CLIENT_KEY")
    server_cert = os.getenv("AIQ_TLS_SERVER_CERT")
    server_key = os.getenv("AIQ_TLS_SERVER_KEY")
    
    if not all([ca_cert, client_cert, client_key]):
        return None
        
    return TLSConfig(
        ca_cert_path=ca_cert,
        client_cert_path=client_cert,
        client_key_path=client_key,
        server_cert_path=server_cert,
        server_key_path=server_key
    )


def generate_self_signed_certificates(output_dir: str = "./certs"):
    """Generate self-signed certificates for testing"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate CA key and certificate
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_cert = x509.CertificateBuilder().subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AIQToolkit CA")])
    ).issuer_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AIQToolkit CA")])
    ).public_key(
        ca_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=0),
        critical=True
    ).sign(ca_key, hashes.SHA256())
    
    # Save CA certificate
    with open(output_path / "ca.crt", "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
    
    # Generate server key and certificate
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_cert = x509.CertificateBuilder().subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    ).issuer_name(
        ca_cert.subject
    ).public_key(
        server_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("*.aiqtoolkit.local"),
        ]),
        critical=False
    ).sign(ca_key, hashes.SHA256())
    
    # Save server files
    with open(output_path / "server.key", "wb") as f:
        f.write(server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    with open(output_path / "server.crt", "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))
    
    # Generate client key and certificate
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client_cert = x509.CertificateBuilder().subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "client")])
    ).issuer_name(
        ca_cert.subject
    ).public_key(
        client_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).sign(ca_key, hashes.SHA256())
    
    # Save client files
    with open(output_path / "client.key", "wb") as f:
        f.write(client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    with open(output_path / "client.crt", "wb") as f:
        f.write(client_cert.public_bytes(serialization.Encoding.PEM))
    
    logger.info(f"Generated self-signed certificates in {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate test certificates
    cert_dir = generate_self_signed_certificates()
    print(f"Certificates generated in: {cert_dir}")