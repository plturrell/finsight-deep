# Digital Human UI

This is the web interface for the AIQ Digital Human system, featuring both 2D and 3D avatar modes with real-time WebSocket communication.

## Quick Start

### Automated Startup

The easiest way to start the Digital Human UI is to use the automated launcher:

```bash
./start.sh
```

This will:
1. Start the backend WebSocket server on port 8000
2. Start the frontend HTTP server on port 8080
3. Open your browser to http://localhost:8080/digital_human_interface.html

### Auto-Start on System Boot

To have the Digital Human UI start automatically when your system boots:

```bash
./install_autostart.sh
```

This installs:
- **macOS**: LaunchAgent that starts the service on login
- **Linux**: Systemd service that starts on boot

### Manual Startup

If you prefer to start services manually:

1. Start the backend server:
```bash
cd backend
source ../venv/bin/activate
python simple_websocket_server.py
```

2. Start the frontend server (in a new terminal):
```bash
cd frontend
python3 -m http.server 8080
```

3. Open http://localhost:8080/digital_human_interface.html

### Check Status

To check if services are running:

```bash
./check_status.sh
```

## Features

- **2D/3D Avatar Modes**: Toggle between 2D canvas-based and 3D WebGL avatars
- **Real-time Communication**: WebSocket connection for instant responses
- **Voice Control**: Speech recognition and synthesis
- **Financial Visualizations**: Interactive charts for portfolio data
- **Market Analysis**: Real-time market sentiment analysis

## Architecture

```
frontend/
├── digital_human_interface.html    # Main UI
├── digital_human_interface.js      # Core logic
└── styles.css                      # Styling

backend/
└── simple_websocket_server.py      # WebSocket server

scripts/
├── start.sh                        # Main launcher
├── launcher.sh                     # Process manager
├── auto_start.sh                   # Background startup
├── install_autostart.sh            # System integration
└── check_status.sh                 # Status checker
```

## Troubleshooting

### Connection Issues

If you see "Unable to connect":

1. Check that the backend is running: `./check_status.sh`
2. Check browser console for errors (F12)
3. Ensure ports 8000 and 8080 are not in use
4. Check firewall settings

### Performance Issues

- The 2D mode is lighter on resources than 3D mode
- Disable voice features if experiencing lag
- Check browser GPU acceleration settings

### Development

To modify the UI:
1. Edit files in `frontend/` directory
2. Refresh browser to see changes
3. Backend changes require server restart

## Requirements

- Python 3.8+
- Modern web browser with WebSocket support
- For 3D mode: WebGL-capable browser

## License

Part of the AIQ Toolkit project.