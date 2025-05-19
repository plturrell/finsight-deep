#!/usr/bin/env python3
"""
One-click Digital Human UI Launcher
Simple automated startup with minimal configuration
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def start_digital_human():
    """Start the Digital Human UI with one click"""
    
    print("🚀 Starting Digital Human UI...")
    
    base_dir = Path(__file__).parent
    processes = []
    
    try:
        # Start backend
        print("📡 Starting backend server...")
        backend_cmd = [sys.executable, "websocket_server.py"]
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(base_dir / "backend"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(backend_proc)
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        print("🌐 Starting frontend server...")
        frontend_cmd = [sys.executable, "-m", "http.server", "8080"]
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(base_dir / "frontend"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(frontend_proc)
        time.sleep(2)  # Give frontend time to start
        
        # Open browser
        url = "http://localhost:8080/digital_human_interface.html"
        print(f"🌍 Opening browser: {url}")
        webbrowser.open(url)
        
        print("\n✅ Digital Human UI is running!")
        print("Press Ctrl+C to stop\n")
        
        # Keep running
        while True:
            time.sleep(1)
            # Check if processes are still running
            for proc in processes:
                if proc.poll() is not None:
                    raise Exception("A process has stopped unexpectedly")
                    
    except KeyboardInterrupt:
        print("\n🛑 Stopping Digital Human UI...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        print("👋 Digital Human UI stopped")

if __name__ == "__main__":
    start_digital_human()