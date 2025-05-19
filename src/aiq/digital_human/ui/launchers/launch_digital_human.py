#!/usr/bin/env python3
"""
Automated Digital Human UI Launcher
Starts both backend and frontend servers automatically
"""

import os
import sys
import time
import subprocess
import webbrowser
import signal
import atexit
from pathlib import Path

class DigitalHumanLauncher:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        self.backend_port = 8000
        self.frontend_port = 8080
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def cleanup(self):
        """Clean up all processes on exit"""
        print("\nShutting down Digital Human UI...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("Digital Human UI stopped.")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.cleanup()
        sys.exit(0)
    
    def check_port_available(self, port):
        """Check if a port is available"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def start_backend(self):
        """Start the backend WebSocket server"""
        print("Starting backend server...")
        
        backend_dir = self.base_dir / "backend"
        cmd = [sys.executable, "websocket_server.py"]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes.append(process)
            
            # Wait for backend to start
            time.sleep(2)
            
            if process.poll() is None:
                print(f"✓ Backend server started on port {self.backend_port}")
                return True
            else:
                print("✗ Backend server failed to start")
                return False
                
        except Exception as e:
            print(f"Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend HTTP server"""
        print("Starting frontend server...")
        
        frontend_dir = self.base_dir / "frontend"
        cmd = [sys.executable, "-m", "http.server", str(self.frontend_port)]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes.append(process)
            
            # Wait for frontend to start
            time.sleep(1)
            
            if process.poll() is None:
                print(f"✓ Frontend server started on port {self.frontend_port}")
                return True
            else:
                print("✗ Frontend server failed to start")
                return False
                
        except Exception as e:
            print(f"Error starting frontend: {e}")
            return False
    
    def open_browser(self):
        """Open the Digital Human UI in the default browser"""
        url = f"http://localhost:{self.frontend_port}/digital_human_interface.html"
        print(f"\nOpening Digital Human UI at: {url}")
        
        try:
            webbrowser.open(url)
            print("✓ Browser opened")
        except Exception as e:
            print(f"✗ Could not open browser automatically: {e}")
            print(f"Please open your browser and navigate to: {url}")
    
    def run(self):
        """Run the Digital Human UI"""
        print("=== Digital Human UI Launcher ===")
        print("Starting automated deployment...\n")
        
        # Check if ports are available
        if not self.check_port_available(self.backend_port):
            print(f"Port {self.backend_port} is already in use. Stopping existing process...")
            os.system(f"lsof -ti:{self.backend_port} | xargs kill -9 2>/dev/null")
            time.sleep(1)
        
        if not self.check_port_available(self.frontend_port):
            print(f"Port {self.frontend_port} is already in use. Stopping existing process...")
            os.system(f"lsof -ti:{self.frontend_port} | xargs kill -9 2>/dev/null")
            time.sleep(1)
        
        # Start servers
        if not self.start_backend():
            print("Failed to start backend server")
            return
        
        if not self.start_frontend():
            print("Failed to start frontend server")
            return
        
        # Open browser
        self.open_browser()
        
        print("\n=== Digital Human UI is running! ===")
        print(f"Backend: http://localhost:{self.backend_port}")
        print(f"Frontend: http://localhost:{self.frontend_port}")
        print("\nPress Ctrl+C to stop all servers")
        
        # Keep running
        try:
            while True:
                # Check if processes are still running
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        print(f"Process {i} has stopped unexpectedly")
                        self.cleanup()
                        return
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    launcher = DigitalHumanLauncher()
    launcher.run()