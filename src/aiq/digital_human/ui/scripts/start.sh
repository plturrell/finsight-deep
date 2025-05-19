#!/bin/bash

# Main Digital Human UI Startup Script

echo "🚀 Starting Digital Human UI..."

# Change to the UI directory (go up one level from scripts/)
cd "$(dirname "$0")/.."

# Check if launcher.sh exists and is executable
if [ ! -x "scripts/launcher.sh" ]; then
    echo "Making launcher.sh executable..."
    chmod +x scripts/launcher.sh
fi

# Run the launcher
scripts/launcher.sh