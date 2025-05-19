# Digital Human Financial Advisor

A cutting-edge digital human financial advisor powered by AIQToolkit's neural supercomputer architecture, featuring real-time facial animation, natural conversation, and GPU-accelerated financial analysis.

## Features

- **Neural Supercomputer Reasoning**: Advanced AI reasoning with verification and self-correction
- **Real-time Avatar**: 60 FPS facial animation with emotion mapping and lip-sync
- **Financial Analysis**: GPU-accelerated Monte Carlo Tree Search for portfolio optimization
- **Natural Conversation**: Context-aware dialogue with emotional intelligence
- **Knowledge Graph**: Integrated RDF and graph database for financial knowledge
- **Production-Ready**: Dockerized deployment with monitoring and scaling

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Digital Human Financial Advisor                      │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│  │  Financial    │    │  Conversation │    │ Facial Anim.  │    │ Emotional     │  │
│  │  Analysis     │    │  Engine       │    │ System        │    │ Response      │  │
│  │  (MCTS)       │    │  (SgLang)     │    │  (Avatar)     │    │ (DSPy)        │  │
│  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘    └───────┬───────┘  │
│          │                    │                    │                    │          │
│  ┌───────┴───────────────────┴───────────────────┴───────────────────┴───────┐    │
│                           Core Orchestration Layer                            │    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- NVIDIA GPU (RTX 4090 or better recommended)
- Docker and Docker Compose
- Python 3.10+
- CUDA 12.2+

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/NVIDIA/AIQToolkit.git
cd AIQToolkit

# Install dependencies
pip install -r src/aiq/digital_human/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# - NVIDIA_API_KEY
# - ALPHA_VANTAGE_API_KEY (optional, for real market data)
# - POLYGON_API_KEY (optional, for real market data)
```

### 3. Run Locally

```bash
# Option 1: Run the demo
python examples/digital_human_demo/demo.py

# Option 2: Start the full system
python -m aiq.digital_human.launch_ui

# The system will be available at:
# - Digital Human UI: http://localhost:8080
# - API Documentation: http://localhost:8000/docs
# - Research Dashboard: http://localhost:8081
```

### 4. Deploy with Docker

```bash
# Build and start all services
./scripts/deploy_digital_human.sh development

# For production deployment
./scripts/deploy_digital_human.sh production
```

## API Usage

### Create a Session

```python
import requests

# Create a new session
response = requests.post('http://localhost:8000/sessions', json={
    'user_id': 'user123',
    'initial_context': {
        'interests': ['technology stocks', 'crypto'],
        'risk_tolerance': 'moderate'
    }
})
session_id = response.json()['session_id']
```

### Send a Message

```python
# Send a message to the digital human
response = requests.post('http://localhost:8000/messages', json={
    'session_id': session_id,
    'content': 'What do you think about my portfolio allocation?'
})

print(response.json()['response'])
print(f"Emotion: {response.json()['emotion']}")
```

### Perform Financial Analysis

```python
# Request portfolio optimization
response = requests.post('http://localhost:8000/analyze', json={
    'session_id': session_id,
    'analysis_type': 'portfolio_optimization',
    'parameters': {
        'portfolio_value': 100000,
        'holdings': {'AAPL': 100, 'GOOGL': 50},
        'cash_balance': 20000,
        'risk_tolerance': 0.6,
        'goal': 'maximize_return'
    }
})

analysis = response.json()['results']
print(f"Recommendation: {analysis['recommendation']['action']}")
print(f"Confidence: {analysis['recommendation']['confidence']}")
```

## Configuration

Edit `src/aiq/digital_human/config.yaml` to customize:

```yaml
# Model configuration
model:
  name: meta-llama/Llama-3.1-70B-Instruct
  device: cuda
  temperature: 0.7

# Avatar settings
avatar:
  resolution: [1920, 1080]
  target_fps: 60.0
  enable_gpu_skinning: true

# Performance settings
performance:
  enable_profiling: true
  gpu_memory_limit: 16384  # MB
```

## Components

### 1. Financial Analysis (MCTS)
- Monte Carlo Tree Search for portfolio optimization
- Real-time risk assessment
- GPU-accelerated simulations
- Market data integration

### 2. Conversation Engine (SgLang)
- Natural language understanding
- Context-aware responses
- Reasoning chain tracking
- Multi-turn conversation management

### 3. Avatar System
- Real-time facial animation
- Emotion-driven expressions
- Phoneme-based lip-sync
- GPU-accelerated rendering

### 4. Emotional Intelligence
- Sentiment analysis
- Contextual emotion mapping
- Empathetic response generation
- Non-verbal communication

## Monitoring

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Kibana: http://localhost:5601

## Development

### Running Tests

```bash
# Run all tests
pytest tests/aiq/digital_human/

# Run specific test
pytest tests/aiq/digital_human/test_digital_human_orchestrator.py
```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit PR

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
```

### API Connection Issues
```bash
# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs backend
```

### Performance Issues
- Reduce model size in config
- Enable profiling to identify bottlenecks
- Check GPU memory usage
- Adjust batch sizes

## License

See LICENSE.md for details.

## Contributing

See CONTRIBUTING.md for guidelines.

## Support

- Documentation: https://docs.aiqtoolkit.com
- Issues: https://github.com/NVIDIA/AIQToolkit/issues
- Discord: https://discord.gg/aiqtoolkit