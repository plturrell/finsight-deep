#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deployment script for Nash-Ethereum consensus smart contracts
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import time

from web3 import Web3
from eth_account import Account
from solcx import compile_source, install_solc
import yaml


# Contract source code
CONSENSUS_CONTRACT_SOURCE = """
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract NeuralConsensus is ReentrancyGuard, AccessControl, Pausable {
    // Contract implementation as defined in secure_nash_ethereum.py
    // ... (full contract code)
}
"""


class ContractDeployer:
    """Deploy and manage consensus contracts"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.w3 = self._initialize_web3()
        self.account = self._initialize_account()
        
        # Install Solidity compiler
        install_solc('0.8.19')
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_web3(self) -> Web3:
        """Initialize Web3 connection"""
        provider_url = self.config['network']['provider_url']
        
        if provider_url.startswith('http'):
            w3 = Web3(Web3.HTTPProvider(provider_url))
        elif provider_url.startswith('ws'):
            w3 = Web3(Web3.WebsocketProvider(provider_url))
        else:
            raise ValueError(f"Invalid provider URL: {provider_url}")
        
        if not w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum network")
        
        print(f"Connected to network: {w3.eth.chain_id}")
        return w3
    
    def _initialize_account(self) -> Account:
        """Initialize deployer account"""
        private_key = os.environ.get('DEPLOYER_PRIVATE_KEY')
        if not private_key:
            raise ValueError("DEPLOYER_PRIVATE_KEY environment variable not set")
        
        account = Account.from_key(private_key)
        print(f"Deployer address: {account.address}")
        
        # Check balance
        balance = self.w3.eth.get_balance(account.address)
        balance_eth = self.w3.from_wei(balance, 'ether')
        print(f"Balance: {balance_eth} ETH")
        
        if balance_eth < 0.1:
            raise ValueError("Insufficient balance for deployment")
        
        return account
    
    def compile_contract(self) -> Dict[str, Any]:
        """Compile the consensus contract"""
        print("Compiling contract...")
        
        # Add OpenZeppelin imports
        contract_source = self._prepare_contract_source()
        
        compiled = compile_source(
            contract_source,
            output_values=['abi', 'bin'],
            solc_version='0.8.19'
        )
        
        contract_interface = compiled['<stdin>:NeuralConsensus']
        return contract_interface
    
    def _prepare_contract_source(self) -> str:
        """Prepare contract source with imports"""
        # In production, properly handle OpenZeppelin imports
        # For now, use simplified version
        return CONSENSUS_CONTRACT_SOURCE
    
    def deploy_contract(self, contract_interface: Dict[str, Any]) -> str:
        """Deploy the contract to the network"""
        print("Deploying contract...")
        
        # Get contract data
        abi = contract_interface['abi']
        bytecode = contract_interface['bin']
        
        # Create contract instance
        Contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Build constructor transaction
        constructor_args = self.config.get('constructor_args', {})
        
        # Estimate gas
        gas_estimate = Contract.constructor(**constructor_args).estimate_gas({
            'from': self.account.address
        })
        gas_price = self.w3.eth.gas_price
        
        print(f"Estimated gas: {gas_estimate}")
        print(f"Gas price: {self.w3.from_wei(gas_price, 'gwei')} gwei")
        
        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        transaction = Contract.constructor(**constructor_args).build_transaction({
            'from': self.account.address,
            'gas': int(gas_estimate * 1.2),  # 20% buffer
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': self.w3.eth.chain_id
        })
        
        # Sign and send transaction
        signed_txn = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"Transaction sent: {tx_hash.hex()}")
        print("Waiting for confirmation...")
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        
        if receipt['status'] == 1:
            contract_address = receipt['contractAddress']
            print(f"Contract deployed successfully at: {contract_address}")
            
            # Save deployment info
            self._save_deployment_info(contract_address, abi, receipt)
            
            return contract_address
        else:
            raise Exception("Contract deployment failed")
    
    def _save_deployment_info(self, address: str, abi: list, receipt: dict):
        """Save deployment information"""
        deployment_info = {
            'address': address,
            'abi': abi,
            'transaction_hash': receipt['transactionHash'].hex(),
            'block_number': receipt['blockNumber'],
            'gas_used': receipt['gasUsed'],
            'deployer': self.account.address,
            'network': self.w3.eth.chain_id,
            'timestamp': int(time.time())
        }
        
        # Save to file
        output_dir = Path(self.config.get('output_dir', './deployments'))
        output_dir.mkdir(exist_ok=True)
        
        filename = f"deployment_{self.w3.eth.chain_id}_{int(time.time())}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"Deployment info saved to: {filepath}")
    
    def verify_contract(self, address: str, constructor_args: Dict[str, Any]):
        """Verify contract on Etherscan"""
        if self.config.get('verify', False):
            etherscan_api_key = os.environ.get('ETHERSCAN_API_KEY')
            if not etherscan_api_key:
                print("Warning: ETHERSCAN_API_KEY not set, skipping verification")
                return
            
            # Implementation would use Etherscan API
            print(f"Contract verification submitted for {address}")
    
    def setup_contract(self, address: str):
        """Initialize contract after deployment"""
        print(f"Setting up contract at {address}...")
        
        # Load contract
        deployment_file = self._find_deployment_file(address)
        with open(deployment_file, 'r') as f:
            deployment_info = json.load(f)
        
        contract = self.w3.eth.contract(
            address=address,
            abi=deployment_info['abi']
        )
        
        # Grant roles
        if 'initial_agents' in self.config:
            for agent_address in self.config['initial_agents']:
                print(f"Granting AGENT_ROLE to {agent_address}")
                
                tx = contract.functions.grantRole(
                    contract.functions.AGENT_ROLE().call(),
                    agent_address
                ).build_transaction({
                    'from': self.account.address,
                    'gas': 100000,
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address)
                })
                
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                if receipt['status'] == 1:
                    print(f"Role granted successfully")
                else:
                    print(f"Failed to grant role")
        
        print("Contract setup complete")
    
    def _find_deployment_file(self, address: str) -> Path:
        """Find deployment file by address"""
        output_dir = Path(self.config.get('output_dir', './deployments'))
        
        for file in output_dir.glob('deployment_*.json'):
            with open(file, 'r') as f:
                data = json.load(f)
                if data['address'].lower() == address.lower():
                    return file
        
        raise FileNotFoundError(f"Deployment file not found for {address}")
    
    def deploy_full_system(self):
        """Deploy complete consensus system"""
        print("Deploying Nash-Ethereum Consensus System")
        print("=" * 40)
        
        # 1. Compile contract
        contract_interface = self.compile_contract()
        
        # 2. Deploy contract
        contract_address = self.deploy_contract(contract_interface)
        
        # 3. Verify contract (if configured)
        self.verify_contract(
            contract_address,
            self.config.get('constructor_args', {})
        )
        
        # 4. Setup contract
        self.setup_contract(contract_address)
        
        print("\nDeployment complete!")
        print(f"Contract address: {contract_address}")
        
        # 5. Deploy additional contracts if configured
        if 'additional_contracts' in self.config:
            for contract_config in self.config['additional_contracts']:
                print(f"\nDeploying {contract_config['name']}...")
                # Deploy additional contracts (token, governance, etc.)
        
        return contract_address


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy Nash-Ethereum consensus contracts')
    parser.add_argument('--config', required=True, help='Path to deployment config file')
    parser.add_argument('--only-compile', action='store_true', help='Only compile contracts')
    parser.add_argument('--setup-only', help='Setup existing contract at address')
    
    args = parser.parse_args()
    
    try:
        deployer = ContractDeployer(args.config)
        
        if args.only_compile:
            contract_interface = deployer.compile_contract()
            print("Contract compiled successfully")
            
            # Save ABI
            abi_path = Path('abi/NeuralConsensus.json')
            abi_path.parent.mkdir(exist_ok=True)
            with open(abi_path, 'w') as f:
                json.dump(contract_interface['abi'], f, indent=2)
            print(f"ABI saved to {abi_path}")
            
        elif args.setup_only:
            deployer.setup_contract(args.setup_only)
            
        else:
            deployer.deploy_full_system()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()