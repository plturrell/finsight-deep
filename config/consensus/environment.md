# Nash-Ethereum Consensus Environment Setup

## Overview

This guide provides step-by-step instructions for setting up the Nash-Ethereum consensus system in your development environment.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Docker (for local Ethereum node)
- Node.js 16+ (for smart contract development)

## 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv consensus-env
source consensus-env/bin/activate  # On Windows: consensus-env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install web3 eth-account prometheus-client aiohttp pyyaml
pip install pytest pytest-asyncio pytest-cov

# Install AIQToolkit with consensus features
pip install -e .[consensus]
```

## 2. Ethereum Development Environment

### Option A: Local Development (Recommended for testing)

```bash
# Install Ganache CLI
npm install -g ganache

# Start local Ethereum node
ganache --accounts 10 --deterministic --gasLimit 10000000
```

### Option B: Testnet (Polygon Mumbai)

1. Get testnet ETH from faucet: https://faucet.polygon.technology/
2. Configure RPC endpoint in `config/consensus/deployment_config.yaml`

### Option C: Docker Setup

```bash
# Pull and run Ethereum node
docker run -d -p 8545:8545 --name eth-node \
  -e NETWORK=mainnet \
  -e INFURA_PROJECT_ID=your_project_id \
  ethereum/client-go

# For local development
docker run -d -p 8545:8545 --name ganache \
  trufflesuite/ganache:latest \
  --accounts 10 --deterministic
```

## 3. Smart Contract Setup

```bash
# Install Solidity compiler
npm install -g solc

# Install OpenZeppelin contracts
npm install @openzeppelin/contracts

# Compile contracts
cd scripts
python deploy_consensus_contracts.py --config ../config/consensus/deployment_config.yaml --only-compile
```

## 4. Configuration

### Environment Variables

Create `.env` file:

```bash
# Ethereum Configuration
DEPLOYER_PRIVATE_KEY=0x... # Your private key for deployment
ETHEREUM_RPC_URL=http://localhost:8545
CHAIN_ID=1337  # Local: 1337, Mumbai: 80001

# API Keys
INFURA_API_KEY=your_infura_key
ALCHEMY_API_KEY=your_alchemy_key
ETHERSCAN_API_KEY=your_etherscan_key

# Monitoring
PROMETHEUS_PORT=8000
ALERT_WEBHOOK_URL=https://your-alerts.com/webhook

# Layer 2 (Optional)
POLYGON_RPC_URL=https://polygon-rpc.com
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc
```

### Configuration Files

1. **Deployment Config**: `config/consensus/deployment_config.yaml`
   - Network settings
   - Contract parameters
   - Gas limits
   - Initial agents

2. **Agent Config**: `config/consensus/agents.yaml`
   ```yaml
   agents:
     - id: "agent_0"
       type: "content_moderator"
       model: "models/moderator_v1.pt"
       reputation: 100
     
     - id: "agent_1"
       type: "recommender"
       model: "models/recommender_v1.pt"
       reputation: 100
   ```

3. **Monitoring Config**: `config/consensus/monitoring.yaml`
   ```yaml
   monitoring:
     prometheus_port: 8000
     retention_days: 7
     alert_rules:
       - name: "high_gas_usage"
         threshold: 1000000
         severity: "warning"
   ```

## 5. GPU Setup (Optional)

### NVIDIA CUDA

```bash
# Check CUDA installation
nvidia-smi

# Install CUDA toolkit (if needed)
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Test PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### TensorCore Optimization

```python
# Enable TensorCore optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## 6. Running the System

### Deploy Contracts

```bash
# Deploy to local network
python scripts/deploy_consensus_contracts.py \
  --config config/consensus/deployment_config.yaml

# Deploy to testnet
python scripts/deploy_consensus_contracts.py \
  --config config/consensus/deployment_config_testnet.yaml
```

### Start Monitoring

```bash
# Start Prometheus monitoring
python -m aiq.neural.consensus_monitoring

# Access metrics at http://localhost:8000
```

### Run Examples

```bash
# Basic example
python examples/consensus/practical_examples.py

# Financial consensus (higher stakes)
python examples/consensus/financial_consensus.py

# Content moderation
python examples/consensus/content_moderation.py
```

## 7. Testing

```bash
# Run all tests
pytest tests/aiq/neural/

# Run specific test suite
pytest tests/aiq/neural/test_nash_ethereum_consensus.py

# Run with coverage
pytest --cov=aiq.neural --cov-report=html tests/
```

## 8. Production Deployment

### Security Checklist

- [ ] Audit smart contracts
- [ ] Use hardware wallet for deployment
- [ ] Enable multi-signature for admin functions
- [ ] Set up monitoring and alerts
- [ ] Configure rate limiting
- [ ] Enable HTTPS for RPC endpoints
- [ ] Use secrets management (e.g., AWS Secrets Manager)

### Scaling Considerations

1. **Layer 2 Networks**
   - Deploy on Polygon/Arbitrum for lower gas costs
   - Use state channels for frequent operations

2. **Batch Processing**
   - Group multiple operations
   - Use multicall contracts
   - Implement queue management

3. **Caching**
   - Cache gas prices
   - Store frequently accessed data
   - Use Redis for distributed caching

## 9. Troubleshooting

### Common Issues

1. **Gas Price Too High**
   ```python
   # Adjust gas optimization
   consensus = GasOptimizedConsensus(
       batch_size=20,  # Increase batch size
       layer2_config=layer2_config  # Use L2
   )
   ```

2. **Consensus Not Converging**
   ```python
   # Adjust Nash parameters
   consensus = NashEthereumConsensus(
       learning_rate=0.01,  # Lower learning rate
       max_iterations=2000  # More iterations
   )
   ```

3. **Connection Issues**
   ```python
   # Check RPC connection
   from web3 import Web3
   w3 = Web3(Web3.HTTPProvider(rpc_url))
   print(w3.is_connected())
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug smart contract calls
consensus = NashEthereumConsensus(debug=True)
```

## 10. Best Practices

1. **Gas Optimization**
   - Batch operations whenever possible
   - Use Layer 2 for non-critical operations
   - Monitor gas prices and schedule accordingly

2. **Security**
   - Never expose private keys
   - Use environment variables
   - Implement proper access controls
   - Regular security audits

3. **Monitoring**
   - Set up alerts for anomalies
   - Track gas usage trends
   - Monitor agent reputation
   - Log all consensus operations

4. **Testing**
   - Test on local network first
   - Use testnet before mainnet
   - Implement comprehensive test coverage
   - Stress test with multiple agents

## Additional Resources

- [Ethereum Development Documentation](https://ethereum.org/developers/)
- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts/)
- [Nash Equilibrium Theory](https://en.wikipedia.org/wiki/Nash_equilibrium)
- [AIQToolkit Documentation](https://aiqtoolkit.readthedocs.io/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/NVIDIA/aiqtoolkit/issues
- Documentation: https://aiqtoolkit.readthedocs.io/
- Community Forum: https://forums.nvidia.com/aiqtoolkit