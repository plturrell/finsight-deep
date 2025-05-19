#!/bin/bash

# Open Elite Digital Human UI in browser

echo "🌐 Opening Elite Digital Human UI..."

# Check if servers are running
BACKEND_RUNNING=false
FRONTEND_RUNNING=false

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    BACKEND_RUNNING=true
    echo "✓ Backend server is running"
else
    echo "✗ Backend server is not running"
fi

if curl -s http://localhost:8080 > /dev/null 2>&1; then
    FRONTEND_RUNNING=true
    echo "✓ Frontend server is running"
else
    echo "✗ Frontend server is not running"
fi

if [ "$BACKEND_RUNNING" = true ] && [ "$FRONTEND_RUNNING" = true ]; then
    echo "✨ Opening Elite Digital Human UI..."
    
    # Open in default browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "http://localhost:8080/elite_interface.html"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "http://localhost:8080/elite_interface.html"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows
        start "http://localhost:8080/elite_interface.html"
    fi
    
    echo ""
    echo "🎉 Elite Digital Human UI is ready!"
    echo "🌐 URL: http://localhost:8080/elite_interface.html"
    echo ""
    echo "Features:"
    echo "  ✨ Photorealistic 3D avatar with PBR rendering"
    echo "  🎯 Real-time facial expressions and animations"
    echo "  🎤 Voice recognition and synthesis"
    echo "  📊 Interactive financial dashboards"
    echo "  🌙 Premium dark theme design"
    echo "  ⚡ GPU-accelerated performance"
    echo ""
    echo "Test the UI by:"
    echo "  1. Typing a message in the chat"
    echo "  2. Using voice commands (mic button)"
    echo "  3. Clicking quick action buttons"
    echo "  4. Watching the avatar respond"
else
    echo "❌ Please start the servers first:"
    echo "   ./deploy.sh"
fi