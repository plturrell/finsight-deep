# Front-End to Back-End Integration Guide

## Overview

The AIQToolkit now features a comprehensive front-end to back-end integration that connects the React UI with the Nash-Ethereum consensus system. This integration enables real-time multi-agent consensus visualization and interaction through WebSocket connections.

## Architecture

```
┌─────────────────┐     WebSocket      ┌────────────────────┐
│                 │ ←─────────────────→ │                    │
│  React UI       │                     │  FastAPI Server    │
│  - ConsensusPanel│     REST API       │  - WebSocket Handler│
│  - Chat         │ ←─────────────────→ │  - Consensus System │
│                 │                     │                    │
└─────────────────┘                     └────────────────────┘
         ↓                                        ↓
┌─────────────────┐                     ┌────────────────────┐
│  WebSocket Hook │                     │  Nash-Ethereum     │
│  - Auto-reconnect│                    │  Consensus         │
│  - Message queue │                    │  - Game Theory     │
│                 │                     │  - Smart Contracts │
└─────────────────┘                     └────────────────────┘
```

## Components

### Front-End Components

1. **ConsensusPanel** (`/components/Consensus/ConsensusPanel.tsx`)
   - Real-time consensus visualization
   - Agent position tracking
   - Nash equilibrium metrics
   - Gas optimization display
   - Interactive consensus requests

2. **useConsensusWebSocket Hook** (`/hooks/useConsensusWebSocket.ts`)
   - Manages WebSocket connections
   - Automatic reconnection logic
   - Message queueing when offline
   - Error handling

3. **Consensus Page** (`/pages/consensus.tsx`)
   - Split-view layout with chat and consensus panel
   - Toggle between views
   - Responsive design

### Back-End Components

1. **ConsensusWebSocketHandler** (`/src/aiq/digital_human/ui/consensus_websocket_handler.py`)
   - Manages WebSocket connections
   - Broadcasts consensus updates
   - Handles consensus requests
   - Real-time metrics streaming

2. **API Server** (`/src/aiq/digital_human/ui/api_server.py`)
   - REST endpoints for session management
   - WebSocket endpoints for real-time communication
   - Health checks and monitoring

3. **Nash-Ethereum Consensus** (`/src/aiq/neural/secure_nash_ethereum.py`)
   - Game theory implementation
   - Ethereum smart contract integration
   - Security features (signatures, staking)
   - Gas optimization

## API Endpoints

### REST Endpoints

- `POST /sessions` - Create new session
- `GET /sessions/{session_id}` - Get session details
- `DELETE /sessions/{session_id}` - End session
- `POST /messages` - Send message
- `POST /analyze` - Request analysis
- `GET /metrics/{session_id}` - Get current metrics
- `GET /health` - System health check

### WebSocket Endpoints

- `/ws/consensus` - Consensus operations
- `/ws/chat` - Real-time chat

## WebSocket Messages

### Consensus WebSocket Messages

#### Client → Server

```json
{
  "type": "request_consensus",
  "problemId": "problem_001",
  "description": "Content moderation decision",
  "agents": ["agent_1", "agent_2", "agent_3"],
  "maxIterations": 100,
  "targetConsensus": 0.95
}
```

#### Server → Client

```json
{
  "type": "agent_update",
  "agents": [
    {
      "id": "agent_1",
      "name": "Content Moderator",
      "position": [0.7, 0.3],
      "confidence": 0.85,
      "stake": 1.0
    }
  ]
}
```

```json
{
  "type": "consensus_metrics",
  "metrics": {
    "activeAgents": 3,
    "consensusProgress": 0.75,
    "gasEstimate": 50000,
    "nashDistance": 0.15,
    "convergenceRate": 0.05
  }
}
```

```json
{
  "type": "consensus_reached",
  "result": "approved",
  "confidence": 0.97,
  "txHash": "0x123...",
  "blockNumber": 12345
}
```

## Usage Example

### React Component Usage

```tsx
import { ConsensusPanel } from '@/components/Consensus/ConsensusPanel';
import { useConsensusWebSocket } from '@/hooks/useConsensusWebSocket';

function MyComponent() {
  const { isConnected, lastMessage, sendMessage } = useConsensusWebSocket();

  const requestConsensus = () => {
    sendMessage({
      type: 'request_consensus',
      problemId: `problem_${Date.now()}`,
      description: 'Should we approve this transaction?',
      agents: ['risk_analyzer', 'fraud_detector', 'compliance_checker'],
      maxIterations: 100,
      targetConsensus: 0.9
    });
  };

  return (
    <div>
      <ConsensusPanel />
      <button onClick={requestConsensus}>Request Consensus</button>
    </div>
  );
}
```

### Python Backend Usage

```python
from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus
from aiq.digital_human.ui.consensus_websocket_handler import ConsensusWebSocketHandler

# Initialize consensus system
consensus = SecureNashEthereumConsensus()
handler = ConsensusWebSocketHandler(consensus)

# In FastAPI app
@app.websocket("/ws/consensus")
async def consensus_websocket(websocket: WebSocket):
    await websocket.accept()
    await handler.handle_connection(websocket)
```

## Features

1. **Real-Time Updates**
   - Live agent position tracking
   - Consensus progress visualization
   - Nash equilibrium convergence metrics

2. **Security**
   - Cryptographic signature verification
   - Rate limiting
   - Staking mechanism
   - Smart contract verification

3. **Performance**
   - Gas optimization strategies
   - Layer 2 support
   - Batch transaction submission
   - WebSocket connection pooling

4. **Monitoring**
   - Prometheus metrics
   - Real-time charts
   - Error tracking
   - Performance analytics

## Deployment

1. Start the backend server:
```bash
cd src/aiq/digital_human/ui
python -m aiq.digital_human.ui.api_server
```

2. Start the frontend:
```bash
cd external/aiqtoolkit-opensource-ui
npm install
npm run dev
```

3. Access the consensus page:
```
http://localhost:3000/consensus
```

## Testing

Run the integration demo:
```bash
python examples/consensus/frontend_integration_demo.py
```

This will test:
- Session creation
- REST API endpoints
- Chat WebSocket
- Consensus WebSocket
- Real-time updates

## Best Practices

1. **Error Handling**
   - Use try-catch blocks for WebSocket operations
   - Implement reconnection logic
   - Queue messages when offline

2. **Performance**
   - Batch consensus requests
   - Use Layer 2 for gas optimization
   - Implement caching strategies

3. **Security**
   - Validate all inputs
   - Use secure WebSocket connections (wss://)
   - Implement authentication
   - Rate limit requests

4. **Monitoring**
   - Track WebSocket connection health
   - Monitor consensus performance
   - Log all errors
   - Set up alerts

## Troubleshooting

### WebSocket Connection Issues

1. Check server is running
2. Verify WebSocket URL
3. Check for CORS issues
4. Review browser console logs

### Consensus Not Reaching

1. Check agent configuration
2. Verify Nash parameters
3. Review gas settings
4. Check blockchain connection

### Performance Issues

1. Enable gas optimization
2. Use batch submissions
3. Configure Layer 2
4. Optimize agent count

## Future Enhancements

1. **Multi-chain Support**
   - Add support for other blockchains
   - Cross-chain consensus

2. **Advanced Visualizations**
   - 3D agent position rendering
   - Historical consensus replay
   - Predictive analytics

3. **Enhanced Security**
   - Multi-factor authentication
   - Advanced encryption
   - Decentralized identity

4. **Scalability**
   - Horizontal scaling
   - Load balancing
   - Edge computing