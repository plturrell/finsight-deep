#!/usr/bin/env python3
"""
Secure Secrets Manager for AWS Deployment
Handles all API keys and credentials securely
"""

import os
import json
import boto3
from cryptography.fernet import Fernet
from typing import Dict, Any
import base64
import logging

logger = logging.getLogger(__name__)


class SecureSecretsManager:
    """Manages secrets securely for Digital Human deployment"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.secrets_client = boto3.client('secretsmanager', region_name=region)
        self.ssm_client = boto3.client('ssm', region_name=region)
        
        # Initialize encryption
        self.cipher_suite = self._init_encryption()
    
    def _init_encryption(self) -> Fernet:
        """Initialize encryption for local secrets"""
        key = os.environ.get('ENCRYPTION_KEY')
        if not key:
            # Generate new key if not exists
            key = Fernet.generate_key()
            logger.warning("Generated new encryption key - save this: %s", key.decode())
        
        return Fernet(key)
    
    def store_nvidia_credentials(self, api_key: str):
        """Store NVIDIA credentials securely"""
        secret_name = "digital-human/nvidia-credentials"
        
        try:
            # Create the secret
            response = self.secrets_client.create_secret(
                Name=secret_name,
                Description="NVIDIA API credentials for Digital Human",
                SecretString=json.dumps({
                    "api_key": api_key,
                    "audio2face_key": api_key,  # Same key for Audio2Face-3D
                    "service": "nvidia-ace"
                }),
                Tags=[
                    {'Key': 'Project', 'Value': 'digital-human'},
                    {'Key': 'Service', 'Value': 'nvidia'}
                ]
            )
            logger.info(f"Created secret: {response['ARN']}")
        except self.secrets_client.exceptions.ResourceExistsException:
            # Update existing secret
            response = self.secrets_client.update_secret(
                SecretId=secret_name,
                SecretString=json.dumps({
                    "api_key": api_key,
                    "audio2face_key": api_key,
                    "service": "nvidia-ace"
                })
            )
            logger.info(f"Updated secret: {response['ARN']}")
        
        return response['ARN']
    
    def store_aws_credentials(self, access_key: str, secret_key: str):
        """Store AWS credentials securely (for cross-account access)"""
        secret_name = "digital-human/aws-cross-account"
        
        try:
            response = self.secrets_client.create_secret(
                Name=secret_name,
                Description="AWS cross-account access for Digital Human",
                SecretString=json.dumps({
                    "access_key_id": access_key,
                    "secret_access_key": secret_key,
                    "region": self.region
                }),
                Tags=[
                    {'Key': 'Project', 'Value': 'digital-human'},
                    {'Key': 'Service', 'Value': 'aws'}
                ]
            )
            logger.info(f"Created AWS credentials secret: {response['ARN']}")
        except self.secrets_client.exceptions.ResourceExistsException:
            response = self.secrets_client.put_secret_value(
                SecretId=secret_name,
                SecretString=json.dumps({
                    "access_key_id": access_key,
                    "secret_access_key": secret_key,
                    "region": self.region
                })
            )
            logger.info("Updated AWS credentials")
        
        return response['ARN']
    
    def store_all_api_keys(self, api_keys: Dict[str, str]):
        """Store all API keys in one secret"""
        secret_name = "digital-human/api-keys"
        
        # Encrypt sensitive keys
        encrypted_keys = {}
        for key, value in api_keys.items():
            encrypted_keys[key] = self.cipher_suite.encrypt(value.encode()).decode()
        
        try:
            response = self.secrets_client.create_secret(
                Name=secret_name,
                Description="All API keys for Digital Human system",
                SecretString=json.dumps(encrypted_keys),
                Tags=[
                    {'Key': 'Project', 'Value': 'digital-human'},
                    {'Key': 'Type', 'Value': 'api-keys'}
                ]
            )
            logger.info(f"Created API keys secret: {response['ARN']}")
        except self.secrets_client.exceptions.ResourceExistsException:
            response = self.secrets_client.put_secret_value(
                SecretId=secret_name,
                SecretString=json.dumps(encrypted_keys)
            )
            logger.info("Updated API keys")
        
        return response['ARN']
    
    def create_secure_parameter(self, name: str, value: str, secure: bool = True):
        """Create SSM parameter for configuration"""
        parameter_name = f"/digital-human/{name}"
        
        try:
            response = self.ssm_client.put_parameter(
                Name=parameter_name,
                Description=f"Parameter for Digital Human: {name}",
                Value=value,
                Type='SecureString' if secure else 'String',
                Overwrite=True,
                Tags=[
                    {'Key': 'Project', 'Value': 'digital-human'}
                ]
            )
            logger.info(f"Created parameter: {parameter_name}")
        except Exception as e:
            logger.error(f"Error creating parameter {parameter_name}: {e}")
            raise
        
        return parameter_name
    
    def setup_kubernetes_secrets(self, namespace: str = "digital-human"):
        """Create Kubernetes secrets from AWS Secrets Manager"""
        # Get all secrets
        secrets = self.list_secrets()
        
        k8s_manifests = []
        
        for secret in secrets:
            secret_value = self.get_secret(secret['Name'])
            
            # Create Kubernetes secret manifest
            k8s_secret = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": secret['Name'].replace('/', '-'),
                    "namespace": namespace
                },
                "type": "Opaque",
                "data": {}
            }
            
            # Base64 encode values
            for key, value in secret_value.items():
                k8s_secret["data"][key] = base64.b64encode(value.encode()).decode()
            
            k8s_manifests.append(k8s_secret)
        
        return k8s_manifests
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve secret from AWS Secrets Manager"""
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise
    
    def list_secrets(self) -> list:
        """List all Digital Human secrets"""
        try:
            response = self.secrets_client.list_secrets(
                Filters=[
                    {
                        'Key': 'tag-key',
                        'Values': ['Project']
                    },
                    {
                        'Key': 'tag-value',
                        'Values': ['digital-human']
                    }
                ]
            )
            return response['SecretList']
        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            raise
    
    def rotate_secret(self, secret_name: str):
        """Rotate a secret"""
        try:
            response = self.secrets_client.rotate_secret(
                SecretId=secret_name,
                RotationRules={
                    'AutomaticallyAfterDays': 90
                }
            )
            logger.info(f"Rotation enabled for {secret_name}")
            return response
        except Exception as e:
            logger.error(f"Error rotating secret {secret_name}: {e}")
            raise


def main():
    """Setup secrets for Digital Human deployment"""
    import sys
    
    # Initialize manager
    manager = SecureSecretsManager()
    
    # Check if we're setting up NVIDIA credentials
    if len(sys.argv) > 1 and sys.argv[1] == "nvidia":
        # Get NVIDIA API key from environment or prompt
        nvidia_key = os.environ.get('NVIDIA_API_KEY') or input("Enter NVIDIA API key: ")
        manager.store_nvidia_credentials(nvidia_key)
        print("✓ NVIDIA credentials stored")
    
    # Check if we're setting up AWS credentials
    elif len(sys.argv) > 1 and sys.argv[1] == "aws":
        # Get AWS credentials
        access_key = os.environ.get('AWS_ACCESS_KEY_ID') or input("Enter AWS Access Key: ")
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or input("Enter AWS Secret Key: ")
        manager.store_aws_credentials(access_key, secret_key)
        print("✓ AWS credentials stored")
    
    # Setup all API keys
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        api_keys = {
            "neural_api_key": os.environ.get('NEURAL_API_KEY', ''),
            "google_api_key": os.environ.get('GOOGLE_API_KEY', ''),
            "yahoo_api_key": os.environ.get('YAHOO_API_KEY', ''),
            "alpha_vantage_key": os.environ.get('ALPHA_VANTAGE_API_KEY', ''),
            "polygon_key": os.environ.get('POLYGON_API_KEY', ''),
            "quandl_key": os.environ.get('QUANDL_API_KEY', '')
        }
        
        # Remove empty keys
        api_keys = {k: v for k, v in api_keys.items() if v}
        
        if api_keys:
            manager.store_all_api_keys(api_keys)
            print(f"✓ Stored {len(api_keys)} API keys")
        else:
            print("No API keys found in environment")
    
    # List secrets
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        secrets = manager.list_secrets()
        print("\nStored secrets:")
        for secret in secrets:
            print(f"  - {secret['Name']}")
            print(f"    Created: {secret['CreatedDate']}")
            print(f"    Last updated: {secret.get('LastChangedDate', 'Never')}")
    
    else:
        print("Usage:")
        print("  python secure_secrets_manager.py nvidia    # Store NVIDIA credentials")
        print("  python secure_secrets_manager.py aws      # Store AWS credentials")
        print("  python secure_secrets_manager.py all      # Store all API keys")
        print("  python secure_secrets_manager.py list     # List all secrets")


if __name__ == "__main__":
    main()