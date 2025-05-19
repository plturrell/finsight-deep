#!/usr/bin/env python3
"""
Elite Digital Human UI Test Script
Verifies all components are working correctly
"""

import asyncio
import websockets
import json
import requests
import time
from datetime import datetime

class EliteUITester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000/ws"
        self.frontend_url = "http://localhost:8080"
        self.results = []
        
    def log_result(self, test_name, success, message=""):
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        # Print result
        status = "✓" if success else "✗"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {test_name}: {message}")
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.log_result("Backend Health", True, f"Healthy - Version {data.get('version')}")
            else:
                self.log_result("Backend Health", False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result("Backend Health", False, str(e))
    
    def test_frontend_access(self):
        """Test frontend accessibility"""
        try:
            response = requests.get(f"{self.frontend_url}/elite_interface.html")
            if response.status_code == 200:
                self.log_result("Frontend Access", True, "Elite interface accessible")
            else:
                self.log_result("Frontend Access", False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result("Frontend Access", False, str(e))
    
    async def test_websocket_connection(self):
        """Test WebSocket connection and messaging"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Test connection
                self.log_result("WebSocket Connection", True, "Connected successfully")
                
                # Send test message
                test_message = {
                    "type": "message",
                    "content": "Test message from Elite UI tester",
                    "sessionId": "test_session_123",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(test_message))
                self.log_result("WebSocket Send", True, "Message sent")
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    self.log_result("WebSocket Receive", True, f"Received: {data.get('type', 'unknown')}")
                except asyncio.TimeoutError:
                    self.log_result("WebSocket Receive", False, "No response received (timeout)")
                
        except Exception as e:
            self.log_result("WebSocket Connection", False, str(e))
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        endpoints = [
            ("/", "Root endpoint"),
            ("/health", "Health check"),
        ]
        
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{self.backend_url}{endpoint}")
                success = response.status_code == 200
                self.log_result(f"API {endpoint}", success, f"{description} - Status {response.status_code}")
            except Exception as e:
                self.log_result(f"API {endpoint}", False, str(e))
    
    def test_static_assets(self):
        """Test static asset loading"""
        assets = [
            "/elite_styles.css",
            "/elite_digital_human.js",
            "/photorealistic_avatar.js"
        ]
        
        for asset in assets:
            try:
                response = requests.get(f"{self.frontend_url}{asset}")
                success = response.status_code == 200
                self.log_result(f"Asset {asset}", success, f"Size: {len(response.content)} bytes")
            except Exception as e:
                self.log_result(f"Asset {asset}", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🔍 Elite Digital Human UI Test Suite")
        print("===================================\n")
        
        # Basic connectivity tests
        print("📡 Testing Connectivity...")
        self.test_backend_health()
        self.test_frontend_access()
        
        # API tests
        print("\n🔌 Testing API Endpoints...")
        self.test_api_endpoints()
        
        # Asset tests
        print("\n📦 Testing Static Assets...")
        self.test_static_assets()
        
        # WebSocket tests
        print("\n🔄 Testing WebSocket...")
        await self.test_websocket_connection()
        
        # Summary
        print("\n📊 Test Summary")
        print("==============")
        successful = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"Tests passed: {successful}/{total} ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("\n✅ All tests passed! Elite Digital Human UI is ready.")
        else:
            print("\n⚠️  Some tests failed. Check the results above.")
        
        return success_rate == 100

if __name__ == "__main__":
    tester = EliteUITester()
    success = asyncio.run(tester.run_all_tests())
    
    # Exit with appropriate code
    exit(0 if success else 1)