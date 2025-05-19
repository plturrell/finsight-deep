#!/bin/bash

# Install auto-start configuration for Digital Human UI

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Digital Human UI Auto-Start Installer${NC}"
echo "====================================="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Detected macOS${NC}"
    
    # Create LaunchAgent for macOS
    PLIST_FILE="$HOME/Library/LaunchAgents/com.aiq.digitalhuman.plist"
    CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aiq.digitalhuman</string>
    <key>ProgramArguments</key>
    <array>
        <string>$CURRENT_DIR/auto_start.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$CURRENT_DIR/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$CURRENT_DIR/logs/launchd.error.log</string>
    <key>WorkingDirectory</key>
    <string>$CURRENT_DIR</string>
</dict>
</plist>
EOF
    
    # Load the LaunchAgent
    launchctl load "$PLIST_FILE"
    echo -e "${GREEN}✓ LaunchAgent installed and loaded${NC}"
    echo -e "Location: $PLIST_FILE"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${YELLOW}Detected Linux${NC}"
    
    # Create systemd service for Linux
    SERVICE_FILE="/etc/systemd/system/digital-human-ui.service"
    CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    # Need sudo for systemd
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${RED}Please run with sudo for Linux installation${NC}"
        exit 1
    fi
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Digital Human UI Service
After=network.target

[Service]
Type=forking
ExecStart=$CURRENT_DIR/auto_start.sh
WorkingDirectory=$CURRENT_DIR
Restart=on-failure
User=$SUDO_USER
Group=$SUDO_USER

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable digital-human-ui.service
    systemctl start digital-human-ui.service
    
    echo -e "${GREEN}✓ Systemd service installed and started${NC}"
    echo -e "Service status: systemctl status digital-human-ui.service"
    
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

echo -e "\n${GREEN}Auto-start installation complete!${NC}"
echo -e "${YELLOW}The Digital Human UI will now start automatically on system boot.${NC}"
echo -e "\nTo uninstall auto-start:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "launchctl unload $PLIST_FILE"
    echo "rm $PLIST_FILE"
else
    echo "sudo systemctl disable digital-human-ui.service"
    echo "sudo systemctl stop digital-human-ui.service"
    echo "sudo rm $SERVICE_FILE"
fi