#!/usr/bin/env python3
"""
Quick Digital Human UI Launcher
Minimal configuration, immediate startup
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    
    print("🚀 Digital Human UI Quick Start")
    print("=" * 40)
    
    # Start backend in background
    print("Starting backend server...")
    backend_cmd = f'cd "{base_dir}/backend" && python3 websocket_server.py'
    subprocess.Popen(backend_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for backend
    time.sleep(3)
    
    # Start frontend in background
    print("Starting frontend server...")
    frontend_cmd = f'cd "{base_dir}/frontend" && python3 -m http.server 8080'
    subprocess.Popen(frontend_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for frontend
    time.sleep(2)
    
    # Open browser
    url = "http://localhost:8080/digital_human_interface.html"
    print(f"Opening browser: {url}")
    webbrowser.open(url)
    
    print("\n✅ Digital Human UI is running!")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:8080")
    print("\nPress Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        os.system("pkill -f websocket_server.py")
        os.system("pkill -f 'python3 -m http.server 8080'")
        print("Goodbye!")

if __name__ == "__main__":
    main()