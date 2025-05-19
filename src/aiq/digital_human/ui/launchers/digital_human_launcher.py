#!/usr/bin/env python3
"""
Digital Human UI Integrated Launcher
Fully automated startup with AIQ toolkit integration
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add AIQ toolkit to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aiq.data_models.common import ServiceStatus
from aiq.utils.debugging_utils import setup_logging

class DigitalHumanLauncher:
    """Automated launcher for Digital Human UI with 2D/3D avatar support"""
    
    def __init__(self, config: Optional[Dict] = None, debug: bool = False):
        self.logger = setup_logging("DigitalHumanLauncher", level=logging.DEBUG if debug else logging.INFO)
        self.base_dir = Path(__file__).parent
        self.config = self._load_config(config)
        self.processes = {}
        self.status = ServiceStatus.INITIALIZING
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
        
    def _load_config(self, user_config: Optional[Dict] = None) -> Dict:
        """Load configuration with defaults"""
        config_file = self.base_dir / "startup_config.json"
        
        # Default configuration
        config = {
            "backend": {
                "port": 8000,
                "host": "localhost",
                "module": "websocket_server",
                "startup_timeout": 10,
                "health_endpoint": "/health"
            },
            "frontend": {
                "port": 8080,
                "host": "localhost",
                "startup_timeout": 5
            },
            "browser": {
                "auto_open": True,
                "delay": 3
            },
            "monitoring": {
                "enabled": True,
                "interval": 5,
                "restart_on_failure": True,
                "max_restart_attempts": 3
            },
            "ui_defaults": {
                "avatar_mode": "2d",
                "enable_voice": True,
                "enable_animations": True
            }
        }
        
        # Load from file if exists
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}")
        
        # Apply user config
        if user_config:
            config.update(user_config)
            
        return config
    
    def _check_port(self, port: int) -> bool:
        """Check if port is available"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return True
            except:
                return False
    
    def _kill_port_process(self, port: int):
        """Kill process using specified port"""
        try:
            if sys.platform in ["darwin", "linux"]:
                cmd = f"lsof -ti:{port} | xargs kill -9"
            else:  # Windows
                cmd = f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{port}') do taskkill /PID %a /F"
            
            subprocess.run(cmd, shell=True, capture_output=True)
            time.sleep(1)
        except Exception as e:
            self.logger.debug(f"Error killing process on port {port}: {e}")
    
    def _start_backend(self) -> Tuple[bool, Optional[subprocess.Popen]]:
        """Start backend WebSocket server"""
        config = self.config["backend"]
        port = config["port"]
        
        self.logger.info(f"Starting backend server on port {port}...")
        
        # Ensure port is available
        if not self._check_port(port):
            self.logger.warning(f"Port {port} is in use, attempting to free it...")
            self._kill_port_process(port)
            if not self._check_port(port):
                self.logger.error(f"Failed to free port {port}")
                return False, None
        
        # Start backend
        backend_script = self.base_dir / "backend" / f"{config['module']}.py"
        if not backend_script.exists():
            self.logger.error(f"Backend script not found: {backend_script}")
            return False, None
        
        env = os.environ.copy()
        env["DIGITAL_HUMAN_PORT"] = str(port)
        env["DIGITAL_HUMAN_HOST"] = config["host"]
        
        try:
            process = subprocess.Popen(
                [sys.executable, str(backend_script)],
                cwd=str(backend_script.parent),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for startup
            time.sleep(config.get("startup_timeout", 10))
            
            if process.poll() is None:
                self.logger.info("✅ Backend server started successfully")
                return True, process
            else:
                self.logger.error("❌ Backend server failed to start")
                stderr = process.stderr.read().decode() if process.stderr else ""
                if stderr:
                    self.logger.error(f"Backend error: {stderr}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Exception starting backend: {e}")
            return False, None
    
    def _start_frontend(self) -> Tuple[bool, Optional[subprocess.Popen]]:
        """Start frontend HTTP server"""
        config = self.config["frontend"]
        port = config["port"]
        
        self.logger.info(f"Starting frontend server on port {port}...")
        
        # Ensure port is available
        if not self._check_port(port):
            self.logger.warning(f"Port {port} is in use, attempting to free it...")
            self._kill_port_process(port)
            if not self._check_port(port):
                self.logger.error(f"Failed to free port {port}")
                return False, None
        
        # Start frontend
        frontend_dir = self.base_dir / "frontend"
        if not frontend_dir.exists():
            self.logger.error(f"Frontend directory not found: {frontend_dir}")
            return False, None
        
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(port)],
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for startup
            time.sleep(config.get("startup_timeout", 5))
            
            if process.poll() is None:
                self.logger.info("✅ Frontend server started successfully")
                return True, process
            else:
                self.logger.error("❌ Frontend server failed to start")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Exception starting frontend: {e}")
            return False, None
    
    def _open_browser(self):
        """Open Digital Human UI in browser"""
        if not self.config["browser"]["auto_open"]:
            return
            
        url = f"http://{self.config['frontend']['host']}:{self.config['frontend']['port']}/digital_human_interface.html"
        
        # Add default UI settings to URL
        params = []
        ui_config = self.config.get("ui_defaults", {})
        if ui_config.get("avatar_mode"):
            params.append(f"avatar={ui_config['avatar_mode']}")
        if ui_config.get("enable_voice"):
            params.append("voice=true")
            
        if params:
            url += "?" + "&".join(params)
        
        self.logger.info(f"Opening browser: {url}")
        
        try:
            webbrowser.open(url)
            self.logger.info("✅ Browser opened successfully")
        except Exception as e:
            self.logger.error(f"Failed to open browser: {e}")
            print(f"\n🌐 Please open your browser and navigate to: {url}")
    
    def _monitor_processes(self):
        """Monitor running processes"""
        if not self.config["monitoring"]["enabled"]:
            return
            
        restart_counts = {"backend": 0, "frontend": 0}
        max_restarts = self.config["monitoring"]["max_restart_attempts"]
        
        while self.status == ServiceStatus.RUNNING:
            time.sleep(self.config["monitoring"]["interval"])
            
            # Check each process
            for name, process in self.processes.items():
                if process and process.poll() is not None:
                    self.logger.warning(f"{name} process has stopped unexpectedly")
                    
                    if (self.config["monitoring"]["restart_on_failure"] and 
                        restart_counts[name] < max_restarts):
                        
                        self.logger.info(f"Attempting to restart {name}...")
                        restart_counts[name] += 1
                        
                        if name == "backend":
                            success, proc = self._start_backend()
                            if success:
                                self.processes[name] = proc
                                self.logger.info(f"Successfully restarted {name}")
                            else:
                                self.logger.error(f"Failed to restart {name}")
                                
                        elif name == "frontend":
                            success, proc = self._start_frontend()
                            if success:
                                self.processes[name] = proc
                                self.logger.info(f"Successfully restarted {name}")
                            else:
                                self.logger.error(f"Failed to restart {name}")
                    else:
                        self.logger.error(f"{name} exceeded max restart attempts")
                        self.status = ServiceStatus.ERROR
                        break
    
    def start(self):
        """Start the Digital Human UI"""
        print("\n🚀 Digital Human UI Launcher")
        print("=" * 40)
        
        try:
            self.status = ServiceStatus.STARTING
            
            # Start backend
            success, backend_proc = self._start_backend()
            if not success:
                self.status = ServiceStatus.ERROR
                return False
            self.processes["backend"] = backend_proc
            
            # Start frontend
            success, frontend_proc = self._start_frontend()
            if not success:
                self.status = ServiceStatus.ERROR
                return False
            self.processes["frontend"] = frontend_proc
            
            # Wait before opening browser
            if self.config["browser"]["delay"] > 0:
                self.logger.info(f"Waiting {self.config['browser']['delay']}s before opening browser...")
                time.sleep(self.config['browser']['delay"])
            
            # Open browser
            self._open_browser()
            
            self.status = ServiceStatus.RUNNING
            
            print("\n✅ Digital Human UI is running!")
            print("=" * 40)
            print(f"🔗 Backend: http://{self.config['backend']['host']}:{self.config['backend']['port']}")
            print(f"🌐 Frontend: http://{self.config['frontend']['host']}:{self.config['frontend']['port']}")
            print(f"🤖 UI: http://{self.config['frontend']['host']}:{self.config['frontend']['port']}/digital_human_interface.html")
            print("\n📝 Default avatar mode: " + self.config.get("ui_defaults", {}).get("avatar_mode", "2d"))
            print("\nPress Ctrl+C to stop all services")
            
            # Start monitoring
            self._monitor_processes()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.status = ServiceStatus.ERROR
        finally:
            self.shutdown()
            
        return False
    
    def shutdown(self):
        """Shutdown all services"""
        self.logger.info("Shutting down Digital Human UI...")
        self.status = ServiceStatus.STOPPING
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                self.logger.info(f"Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {name}")
                    process.kill()
                    process.wait()
                except Exception as e:
                    self.logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()
        self.status = ServiceStatus.STOPPED
        self.logger.info("Digital Human UI stopped successfully")
        print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Digital Human UI Launcher")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--backend-port", type=int, help="Backend server port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, help="Frontend server port (default: 8080)")
    parser.add_argument("--avatar-mode", choices=["2d", "3d"], help="Default avatar mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Build config from args
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Override with command line args
    if args.no_browser:
        config.setdefault("browser", {})["auto_open"] = False
    if args.backend_port:
        config.setdefault("backend", {})["port"] = args.backend_port
    if args.frontend_port:
        config.setdefault("frontend", {})["port"] = args.frontend_port
    if args.avatar_mode:
        config.setdefault("ui_defaults", {})["avatar_mode"] = args.avatar_mode
    
    # Create and run launcher
    launcher = DigitalHumanLauncher(config=config, debug=args.debug)
    success = launcher.start()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()