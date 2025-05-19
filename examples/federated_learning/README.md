# Federated Learning Example

This example demonstrates how to use AIQToolkit's federated learning capabilities to train machine learning models across distributed edge devices while preserving privacy.

## Features

- Federated learning with multiple aggregation strategies
- Edge computing support for offline/online operation
- Differential privacy for data protection
- Secure aggregation protocols
- Adaptive client selection
- Model compression and efficient communication

## Setup

1. Install required dependencies:
```bash
uv sync --all-extras
```

2. Start the federated learning server:
```bash
python server.py --config configs/federated_server.yml
```

3. Start edge clients:
```bash
# On each edge device
python client.py --config configs/edge_client.yml --client-id edge_1
```

## Configuration

### Server Configuration

```yaml
federated_config:
  rounds: 100
  clients_per_round: 10
  min_clients: 5
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01
  aggregation_strategy: fedavg  # Options: fedavg, fedprox, fedyogi, scaffold
  client_selection: weighted    # Options: random, weighted, quality, resource
  differential_privacy: true
  privacy_budget: 10.0
  secure_aggregation: true
  checkpoint_interval: 10
```

### Edge Client Configuration

```yaml
edge_config:
  node_id: edge_1
  device_type: workstation  # Options: mobile, iot, embedded, workstation
  mode: intermittent       # Options: always_connected, intermittent, offline_first
  sync_interval: 3600
  cache_size_mb: 100
  power_mode: balanced     # Options: low_power, balanced, performance
```

## Running the Example

### Basic Federated Learning

```python
from aiq.distributed.federated.federated_learning import (
    FederatedLearningServer,
    FederatedConfig
)

# Configure federated learning
config = FederatedConfig(
    rounds=50,
    clients_per_round=5,
    aggregation_strategy=AggregationStrategy.FEDAVG,
    differential_privacy=True
)

# Create server
server = FederatedLearningServer(config)

# Initialize global model
model = YourModel()
await server.initialize_global_model(model)

# Run training
await server.run_training()
```

### Edge Client

```python
from aiq.distributed.edge.edge_node import EdgeNode, EdgeNodeConfig
from aiq.distributed.federated.federated_learning import FederatedLearningClient

# Configure edge node
edge_config = EdgeNodeConfig(
    node_id="edge_1",
    device_type="workstation",
    mode=EdgeMode.INTERMITTENT
)

# Create federated learning client
client = FederatedLearningClient(edge_config, fl_config)

# Register with server
await client.register_with_server("http://server:8080")

# Participate in training
while True:
    round_number = await client.wait_for_round()
    await client.participate_in_round(round_number)
```

## Advanced Features

### Personalized Federated Learning

```python
config = FederatedConfig(
    aggregation_strategy=AggregationStrategy.PERSONALIZED,
    # ... other config
)

# Server maintains personalized models for each client
```

### Secure Aggregation

```python
from aiq.distributed.security.privacy import SecureAggregator

# Enable secure aggregation
config = FederatedConfig(
    secure_aggregation=True,
    # ... other config
)

# Clients' updates are encrypted
```

### Differential Privacy

```python
# Configure privacy budget
config = FederatedConfig(
    differential_privacy=True,
    privacy_budget=10.0,  # Epsilon
    noise_multiplier=1.0
)

# Updates have calibrated noise added
```

## Monitoring

Monitor federated learning progress:

```python
# Access training metrics
metrics = server.metrics_history

# Check privacy budget spent
epsilon_spent, delta_spent = server.privacy_manager.get_privacy_spent()

# Monitor client participation
client_stats = server.get_client_statistics()
```

## Edge Computing Features

### Offline Operation

```python
# Edge node can run workflows offline
result = await edge_node.run_offline_workflow(workflow_config)

# Results are queued for sync when connection is restored
```

### Model Caching

```python
# Models are cached locally for offline use
edge_node.cache_manager.cache_model("model_id", model_data)

# Automatic cache management with LRU eviction
```

### Power Management

```python
# Configure for low power devices
edge_config = EdgeNodeConfig(
    power_mode="low_power",
    # ... other config
)

# Automatically adjusts resource usage
```

## Best Practices

1. **Privacy First**: Always enable differential privacy for sensitive data
2. **Adaptive Selection**: Use quality-based client selection for better convergence
3. **Checkpoint Regularly**: Save model checkpoints to handle failures
4. **Monitor Resources**: Track edge device resources and adjust accordingly
5. **Test Offline**: Ensure edge nodes can operate offline effectively

## Troubleshooting

### Common Issues

1. **Clients not connecting**:
   - Check firewall settings
   - Verify authentication tokens
   - Ensure server is reachable

2. **Slow convergence**:
   - Increase local epochs
   - Adjust learning rate
   - Use adaptive aggregation strategies

3. **Privacy budget exceeded**:
   - Reduce noise multiplier
   - Increase privacy budget
   - Use fewer training rounds

## References

- [Federated Learning Paper](https://arxiv.org/abs/1602.05629)
- [FedAvg Algorithm](https://arxiv.org/abs/1602.05629)
- [Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Secure Aggregation](https://arxiv.org/abs/1611.04482)