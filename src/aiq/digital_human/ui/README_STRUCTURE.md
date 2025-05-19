# Digital Human UI - Directory Structure

The Digital Human UI has been reorganized for better maintainability and clarity:

## Directory Organization

```
ui/
├── api/                    # API servers
│   ├── api_server.py       # Main production API server (improved)
│   ├── production_api_server.py  # Production-specific configuration
│   └── api_server_complete.py    # Alternative implementation
│
├── websocket/              # WebSocket handlers
│   ├── websocket_handler.py      # Base WebSocket handler
│   ├── consensus_websocket_handler.py  # Consensus operations
│   ├── websocket_server.py       # Standalone WebSocket server
│   └── simple_websocket_server.py # Simple implementation
│
├── frontend/               # Frontend UI files
│   ├── digital_human_interface.html  # Main interface
│   ├── digital_human_interface.js    # Core JavaScript
│   ├── elite_digital_human.js        # Enhanced features
│   └── research_dashboard.html       # Research interface
│
├── launchers/              # Startup scripts (Python)
│   ├── digital_human_launcher.py  # Full launcher
│   ├── launch_digital_human.py    # Automated launcher
│   ├── start.py                   # Simple launcher
│   └── auto_start.py              # Auto-start utility
│
├── scripts/                # Shell scripts
│   ├── start.sh            # Main startup script
│   ├── deploy.sh           # Deployment script
│   ├── run_all.sh          # Run all components
│   └── install_autostart.sh # Install auto-start
│
├── tests/                  # Test files
│   ├── test_connection.html
│   ├── test_websocket.html
│   └── test_elite.py
│
├── config/                 # Configuration files
│   ├── production_config.yaml
│   └── startup_config.json
│
├── logs/                   # Log files
├── pids/                   # Process IDs
├── backend/                # (deprecated - moved to websocket/)
└── venv/                   # Virtual environment
```

## Key Components

### API Server (`api/api_server.py`)
- Production-ready FastAPI server
- JWT authentication and RBAC
- Rate limiting and input validation
- Redis session storage
- Prometheus metrics
- WebSocket endpoints for real-time communication

### WebSocket Handlers
- `websocket_handler.py`: Base handler for real-time communication
- `consensus_websocket_handler.py`: Specialized for consensus operations
- `websocket_server.py`: Standalone server for development

### Frontend
- HTML/JS interface for user interaction
- WebSocket client for real-time updates
- Support for 2D/3D avatars
- Research dashboard for analytics

### Launchers
- Multiple options for starting the system
- Python scripts with proper error handling
- Integration with AIQ toolkit

## Quick Start

```bash
# From the ui/ directory
./scripts/start.sh
```

This will:
1. Start the API server (port 8000)
2. Start the frontend server (port 8080)
3. Open the browser to the interface

## Development

### Import Paths
With the new structure, use relative imports within modules:

```python
# In api/api_server.py
from ..websocket.consensus_websocket_handler import ConsensusWebSocketHandler

# In websocket/consensus_websocket_handler.py
from .websocket_handler import WebSocketHandler
```

### Configuration
- Environment variables in `.env`
- Production config in `config/production_config.yaml`
- Startup config in `config/startup_config.json`

## Security Features

The improved API server includes:
- JWT-based authentication
- Role-based access control (RBAC)
- Rate limiting per endpoint
- Input validation with Pydantic
- Secure session management (Redis)
- CORS configuration
- API documentation security

## Monitoring

- Prometheus metrics endpoint: `/metrics`
- Health check: `/health`
- System status: `/system/status` (admin only)
- Metrics summary: `/metrics/summary`

## API Documentation

When not in production:
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI schema: `/api/v1/openapi.json`