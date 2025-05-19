#!/usr/bin/env python3
"""
Advanced Digital Human UI Auto-Start System
Handles all dependencies, checks, and automated deployment
"""

import os
import sys
import time
import json
import logging
import asyncio
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DigitalHumanAutoStart:
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.config = self.load_config(config_path)
        self.processes = {}
        self.start_time = datetime.now()
        
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "backend": {
                "port": 8000,
                "host": "0.0.0.0",
                "module": "websocket_server",
                "timeout": 10
            },
            "frontend": {
                "port": 8080,
                "host": "0.0.0.0",
                "timeout": 5
            },
            "browser": {
                "auto_open": True,
                "delay": 3
            },
            "health_check": {
                "enabled": True,
                "interval": 5,
                "max_retries": 3
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        required_packages = [
            "fastapi",
            "uvicorn",
            "websockets",
            "pydantic"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            print(f"\nPlease install missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def kill_existing_processes(self, port: int):
        """Kill any existing processes on the specified port"""
        try:
            if sys.platform == "darwin" or sys.platform == "linux":
                subprocess.run(f"lsof -ti:{port} | xargs kill -9", 
                             shell=True, 
                             capture_output=True)
            elif sys.platform == "win32":
                subprocess.run(f"netstat -ano | findstr :{port}", 
                             shell=True, 
                             capture_output=True)
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"Error killing process on port {port}: {e}")
    
    async def start_backend(self) -> bool:
        """Start the backend WebSocket server asynchronously"""
        config = self.config["backend"]
        port = config["port"]
        
        logger.info(f"Starting backend server on port {port}...")
        
        # Kill existing processes
        self.kill_existing_processes(port)
        
        backend_script = self.base_dir / "backend" / f"{config['module']}.py"
        
        if not backend_script.exists():
            logger.error(f"Backend script not found: {backend_script}")
            return False
        
        cmd = [
            sys.executable,
            str(backend_script)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(backend_script.parent)
            )
            
            self.processes["backend"] = process
            
            # Wait for startup
            await asyncio.sleep(config.get("timeout", 10))
            
            if process.returncode is None:
                logger.info("✓ Backend server started successfully")
                return True
            else:
                logger.error("✗ Backend server failed to start")
                stdout, stderr = await process.communicate()
                if stderr:
                    logger.error(f"Backend error: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            return False
    
    async def start_frontend(self) -> bool:
        """Start the frontend HTTP server asynchronously"""
        config = self.config["frontend"]
        port = config["port"]
        
        logger.info(f"Starting frontend server on port {port}...")
        
        # Kill existing processes
        self.kill_existing_processes(port)
        
        frontend_dir = self.base_dir / "frontend"
        
        if not frontend_dir.exists():
            logger.error(f"Frontend directory not found: {frontend_dir}")
            return False
        
        cmd = [
            sys.executable,
            "-m", "http.server",
            str(port),
            "--bind", config["host"]
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(frontend_dir)
            )
            
            self.processes["frontend"] = process
            
            # Wait for startup
            await asyncio.sleep(config.get("timeout", 5))
            
            if process.returncode is None:
                logger.info("✓ Frontend server started successfully")
                return True
            else:
                logger.error("✗ Frontend server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start frontend: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Perform health checks on running services"""
        if not self.config["health_check"]["enabled"]:
            return True
        
        import aiohttp
        
        checks = [
            (f"http://localhost:{self.config['backend']['port']}/health", "Backend"),
            (f"http://localhost:{self.config['frontend']['port']}/", "Frontend")
        ]
        
        async with aiohttp.ClientSession() as session:
            for url, service in checks:
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"✓ {service} health check passed")
                        else:
                            logger.warning(f"✗ {service} health check failed: {response.status}")
                            return False
                except Exception as e:
                    logger.warning(f"✗ {service} health check failed: {e}")
                    # Frontend might not have health endpoint, so don't fail for it
                    if service == "Backend":
                        return False
        
        return True
    
    def open_browser(self):
        """Open the Digital Human UI in the default browser"""
        if not self.config["browser"]["auto_open"]:
            return
        
        url = f"http://localhost:{self.config['frontend']['port']}/digital_human_interface.html"
        logger.info(f"Opening browser: {url}")
        
        try:
            webbrowser.open(url)
            logger.info("✓ Browser opened successfully")
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            print(f"\nPlease open your browser and navigate to: {url}")
    
    async def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while True:
            await asyncio.sleep(self.config["health_check"]["interval"])
            
            for name, process in self.processes.items():
                if process.returncode is not None:
                    logger.warning(f"{name} process has stopped unexpectedly")
                    
                    # Attempt restart
                    if name == "backend":
                        if await self.start_backend():
                            logger.info(f"Successfully restarted {name}")
                    elif name == "frontend":
                        if await self.start_frontend():
                            logger.info(f"Successfully restarted {name}")
    
    async def run(self):
        """Main execution function"""
        print("\n=== Digital Human UI Auto-Start ===")
        print(f"Starting at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Check dependencies
        if not await self.check_dependencies():
            return
        
        # Start services
        try:
            # Start backend
            if not await self.start_backend():
                logger.error("Failed to start backend server")
                return
            
            # Start frontend
            if not await self.start_frontend():
                logger.error("Failed to start frontend server")
                return
            
            # Wait for services to be ready
            await asyncio.sleep(2)
            
            # Health check
            if not await self.health_check():
                logger.warning("Health checks failed, but continuing...")
            
            # Open browser
            if self.config["browser"]["auto_open"]:
                await asyncio.sleep(self.config["browser"]["delay"])
                self.open_browser()
            
            print("\n=== Digital Human UI is running! ===")
            print(f"Backend: http://localhost:{self.config['backend']['port']}")
            print(f"Frontend: http://localhost:{self.config['frontend']['port']}")
            print(f"UI: http://localhost:{self.config['frontend']['port']}/digital_human_interface.html")
            print("\nPress Ctrl+C to stop all services\n")
            
            # Monitor processes
            await self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up all processes"""
        logger.info("Cleaning up processes...")
        
        for name, process in self.processes.items():
            if process.returncode is None:
                logger.info(f"Stopping {name}...")
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Force killing {name}")
                    process.kill()
                    await process.wait()
        
        logger.info("All processes stopped")
        print("\nDigital Human UI stopped successfully")


def main():
    parser = argparse.ArgumentParser(description="Digital Human UI Auto-Start")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--backend-port", type=int, help="Backend server port")
    parser.add_argument("--frontend-port", type=int, help="Frontend server port")
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = DigitalHumanAutoStart(args.config)
    
    # Override config with command line args
    if args.no_browser:
        launcher.config["browser"]["auto_open"] = False
    if args.backend_port:
        launcher.config["backend"]["port"] = args.backend_port
    if args.frontend_port:
        launcher.config["frontend"]["port"] = args.frontend_port
    
    # Run
    try:
        asyncio.run(launcher.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()