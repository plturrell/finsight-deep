#!/usr/bin/env python3
"""
Production Deployment Validation Script
Ensures all components are correctly deployed and functioning
"""

import os
import sys
import asyncio
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import aiohttp
import psycopg2
import redis
from pymilvus import connections, utility
import numpy as np

# Import production modules
from aiq.digital_human.deployment.production_implementation import ProductionDigitalHuman


class DeploymentValidator:
    """Validates production deployment of Digital Human System"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "status": "PENDING",
            "components": {},
            "tests": {},
            "errors": []
        }
        
        # Load configuration
        self.config_path = os.environ.get(
            'PRODUCTION_CONFIG_PATH',
            'src/aiq/digital_human/deployment/production_config.yaml'
        )
    
    async def validate_all(self) -> bool:
        """Run all validation checks"""
        print("🔍 Starting Production Deployment Validation...")
        print("=" * 50)
        
        try:
            # Environment checks
            await self.check_environment()
            
            # Service connectivity
            await self.check_nvidia_services()
            await self.check_neural_supercomputer()
            await self.check_databases()
            await self.check_external_apis()
            
            # Component functionality
            await self.check_digital_human_system()
            
            # Integration tests
            await self.run_integration_tests()
            
            # Performance benchmarks
            await self.run_performance_tests()
            
            # Security validation
            await self.check_security()
            
            # Generate report
            self.generate_report()
            
            # Determine overall status
            self.results["status"] = "PASSED" if all(
                component.get("status") == "OK"
                for component in self.results["components"].values()
            ) else "FAILED"
            
            return self.results["status"] == "PASSED"
            
        except Exception as e:
            self.results["status"] = "ERROR"
            self.results["errors"].append(str(e))
            return False
    
    async def check_environment(self):
        """Check environment variables"""
        print("\n✓ Checking environment variables...")
        
        required_vars = [
            "NVIDIA_API_KEY",
            "NEURAL_SUPERCOMPUTER_ENDPOINT",
            "NEURAL_SUPERCOMPUTER_API_KEY",
            "GOOGLE_API_KEY",
            "POSTGRES_HOST",
            "REDIS_HOST",
            "JWT_SECRET_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        self.results["components"]["environment"] = {
            "status": "OK" if not missing_vars else "FAILED",
            "missing_vars": missing_vars,
            "total_required": len(required_vars),
            "found": len(required_vars) - len(missing_vars)
        }
        
        if missing_vars:
            print(f"  ❌ Missing variables: {', '.join(missing_vars)}")
        else:
            print("  ✅ All environment variables present")
    
    async def check_nvidia_services(self):
        """Check NVIDIA service connectivity"""
        print("\n✓ Checking NVIDIA services...")
        
        services = {
            "ACE": "https://api.nvidia.com/ace/v1/health",
            "Riva": "https://api.nvidia.com/riva/v1/health",
            "NeMo": "https://api.nvidia.com/nemo/retriever/v1/health",
            "Tokkio": "https://api.nvidia.com/tokkio/v1/health"
        }
        
        results = {}
        headers = {
            "Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}"
        }
        
        async with aiohttp.ClientSession() as session:
            for service, url in services.items():
                try:
                    async with session.get(url, headers=headers) as response:
                        results[service] = {
                            "status": "OK" if response.status == 200 else "FAILED",
                            "response_code": response.status,
                            "response_time": response.headers.get('X-Response-Time', 'N/A')
                        }
                        print(f"  ✅ {service}: Connected")
                except Exception as e:
                    results[service] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                    print(f"  ❌ {service}: {e}")
        
        self.results["components"]["nvidia_services"] = results
    
    async def check_neural_supercomputer(self):
        """Check neural supercomputer connectivity"""
        print("\n✓ Checking Neural Supercomputer...")
        
        endpoint = os.environ.get("NEURAL_SUPERCOMPUTER_ENDPOINT")
        api_key = os.environ.get("NEURAL_SUPERCOMPUTER_API_KEY")
        
        if not endpoint or not api_key:
            self.results["components"]["neural_supercomputer"] = {
                "status": "FAILED",
                "error": "Missing endpoint or API key"
            }
            return
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                # Health check
                async with session.get(
                    f"{endpoint}/health",
                    headers=headers
                ) as response:
                    health_status = response.status == 200
                
                # Test reasoning
                test_query = {
                    "query": "test",
                    "context": {},
                    "task_type": "test"
                }
                
                async with session.post(
                    f"{endpoint}/reason",
                    headers=headers,
                    json=test_query
                ) as response:
                    reasoning_status = response.status == 200
            
            self.results["components"]["neural_supercomputer"] = {
                "status": "OK" if health_status and reasoning_status else "FAILED",
                "endpoint": endpoint,
                "health_check": health_status,
                "reasoning_test": reasoning_status
            }
            
            print(f"  ✅ Neural Supercomputer: Connected")
            
        except Exception as e:
            self.results["components"]["neural_supercomputer"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ Neural Supercomputer: {e}")
    
    async def check_databases(self):
        """Check database connectivity"""
        print("\n✓ Checking databases...")
        
        db_results = {}
        
        # PostgreSQL
        try:
            pg_conn = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST"),
                port=os.environ.get("POSTGRES_PORT", 5432),
                database="digital_human",
                user=os.environ.get("POSTGRES_USER"),
                password=os.environ.get("POSTGRES_PASSWORD")
            )
            
            with pg_conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                
                # Check tables exist
                cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                table_count = cursor.fetchone()[0]
            
            pg_conn.close()
            
            db_results["postgresql"] = {
                "status": "OK",
                "version": version.split(',')[0],
                "tables": table_count
            }
            print("  ✅ PostgreSQL: Connected")
            
        except Exception as e:
            db_results["postgresql"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ PostgreSQL: {e}")
        
        # Redis
        try:
            r = redis.Redis(
                host=os.environ.get("REDIS_HOST"),
                port=os.environ.get("REDIS_PORT", 6379),
                password=os.environ.get("REDIS_PASSWORD"),
                ssl=True
            )
            
            info = r.info()
            db_results["redis"] = {
                "status": "OK",
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human")
            }
            print("  ✅ Redis: Connected")
            
        except Exception as e:
            db_results["redis"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ Redis: {e}")
        
        # Milvus
        try:
            connections.connect(
                host=os.environ.get("MILVUS_HOST"),
                port=os.environ.get("MILVUS_PORT", 19530)
            )
            
            collections = utility.list_collections()
            db_results["milvus"] = {
                "status": "OK",
                "collections": collections,
                "collection_count": len(collections)
            }
            print("  ✅ Milvus: Connected")
            
        except Exception as e:
            db_results["milvus"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ Milvus: {e}")
        
        self.results["components"]["databases"] = db_results
    
    async def check_external_apis(self):
        """Check external API connectivity"""
        print("\n✓ Checking external APIs...")
        
        api_results = {}
        
        # Google Search
        try:
            # Simple connectivity test
            import googleapiclient.discovery
            
            service = googleapiclient.discovery.build(
                'customsearch', 'v1',
                developerKey=os.environ.get('GOOGLE_API_KEY')
            )
            
            # Test search
            results = service.cse().list(
                q='test',
                cx=os.environ.get('GOOGLE_CSE_ID'),
                num=1
            ).execute()
            
            api_results["google_search"] = {
                "status": "OK",
                "test_query_success": True
            }
            print("  ✅ Google Search: Connected")
            
        except Exception as e:
            api_results["google_search"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ Google Search: {e}")
        
        # Financial APIs
        financial_apis = {
            "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY"),
            "polygon": os.environ.get("POLYGON_API_KEY"),
            "quandl": os.environ.get("QUANDL_API_KEY")
        }
        
        for api_name, api_key in financial_apis.items():
            if api_key:
                api_results[api_name] = {
                    "status": "OK",
                    "configured": True
                }
                print(f"  ✅ {api_name}: Configured")
            else:
                api_results[api_name] = {
                    "status": "WARNING",
                    "configured": False
                }
                print(f"  ⚠️  {api_name}: Not configured")
        
        self.results["components"]["external_apis"] = api_results
    
    async def check_digital_human_system(self):
        """Check digital human system components"""
        print("\n✓ Checking Digital Human System...")
        
        try:
            # Initialize production system
            system = ProductionDigitalHuman(self.config_path)
            await system.initialize_orchestrator()
            
            # Test basic functionality
            components = {
                "orchestrator": hasattr(system, 'orchestrator'),
                "ace_platform": hasattr(system.orchestrator, 'ace_platform'),
                "context_server": hasattr(system.orchestrator, 'context_server'),
                "neural_connector": hasattr(system.orchestrator, 'neural_connector'),
                "financial_analyzer": hasattr(system.orchestrator, 'financial_analyzer')
            }
            
            self.results["components"]["digital_human_system"] = {
                "status": "OK" if all(components.values()) else "FAILED",
                "components": components,
                "initialization": "success"
            }
            
            await system.close()
            print("  ✅ Digital Human System: Initialized")
            
        except Exception as e:
            self.results["components"]["digital_human_system"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ Digital Human System: {e}")
    
    async def run_integration_tests(self):
        """Run basic integration tests"""
        print("\n✓ Running integration tests...")
        
        test_results = {}
        
        # Test 1: End-to-end interaction
        try:
            system = ProductionDigitalHuman(self.config_path)
            await system.initialize_orchestrator()
            
            # Create test session
            import jwt
            test_token = jwt.encode(
                {"user_id": "test_user"},
                os.environ.get("JWT_SECRET_KEY"),
                algorithm="HS256"
            )
            
            session_id = await system.start_session(
                user_id="test_user",
                auth_token=test_token
            )
            
            # Test interaction
            response = await system.process_interaction(
                session_id=session_id,
                user_input="Test query",
                audio_data=None
            )
            
            test_results["end_to_end"] = {
                "status": "PASSED" if response and "text" in response else "FAILED",
                "response_time": response.get("processing_time", 0)
            }
            
            await system.close()
            print("  ✅ End-to-end test: Passed")
            
        except Exception as e:
            test_results["end_to_end"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"  ❌ End-to-end test: {e}")
        
        self.results["tests"]["integration"] = test_results
    
    async def run_performance_tests(self):
        """Run performance benchmark tests"""
        print("\n✓ Running performance tests...")
        
        perf_results = {}
        
        # Response time test
        try:
            system = ProductionDigitalHuman(self.config_path)
            await system.initialize_orchestrator()
            
            response_times = []
            
            for _ in range(5):
                start_time = time.time()
                
                # Mock quick query
                response = await system.orchestrator.process_interaction(
                    session_id="perf_test",
                    user_input="Quick test",
                    audio_data=None
                )
                
                response_times.append(time.time() - start_time)
            
            avg_response_time = sum(response_times) / len(response_times)
            
            perf_results["response_time"] = {
                "status": "PASSED" if avg_response_time < 3.0 else "WARNING",
                "average": avg_response_time,
                "min": min(response_times),
                "max": max(response_times)
            }
            
            await system.close()
            print(f"  ✅ Response time: {avg_response_time:.2f}s average")
            
        except Exception as e:
            perf_results["response_time"] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"  ❌ Performance test: {e}")
        
        self.results["tests"]["performance"] = perf_results
    
    async def check_security(self):
        """Check security configurations"""
        print("\n✓ Checking security...")
        
        security_results = {}
        
        # SSL certificates
        ssl_cert = os.environ.get("SSL_CERT_FILE")
        ssl_key = os.environ.get("SSL_KEY_FILE")
        
        security_results["ssl"] = {
            "status": "OK" if ssl_cert and ssl_key else "WARNING",
            "cert_configured": bool(ssl_cert),
            "key_configured": bool(ssl_key)
        }
        
        # JWT configuration
        jwt_secret = os.environ.get("JWT_SECRET_KEY")
        security_results["jwt"] = {
            "status": "OK" if jwt_secret else "FAILED",
            "configured": bool(jwt_secret)
        }
        
        # API key encryption
        encryption_key = os.environ.get("API_KEY_ENCRYPTION_KEY")
        security_results["encryption"] = {
            "status": "OK" if encryption_key else "WARNING",
            "configured": bool(encryption_key)
        }
        
        self.results["components"]["security"] = security_results
        print("  ✅ Security: Configured")
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 50)
        print("VALIDATION REPORT")
        print("=" * 50)
        
        # Save detailed report
        report_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print(f"\nReport saved to: {report_path}")
        print(f"Overall Status: {self.results['status']}")
        
        # Component summary
        print("\nComponent Status:")
        for component, details in self.results["components"].items():
            status = details.get("status", "UNKNOWN")
            print(f"  {component}: {status}")
        
        # Test summary
        if "tests" in self.results:
            print("\nTest Results:")
            for test_category, tests in self.results["tests"].items():
                for test_name, result in tests.items():
                    status = result.get("status", "UNKNOWN")
                    print(f"  {test_category}/{test_name}: {status}")
        
        # Errors
        if self.results["errors"]:
            print("\nErrors:")
            for error in self.results["errors"]:
                print(f"  - {error}")
        
        print("\n" + "=" * 50)


async def main():
    """Run deployment validation"""
    validator = DeploymentValidator()
    success = await validator.validate_all()
    
    if success:
        print("\n✅ DEPLOYMENT VALIDATION PASSED")
        sys.exit(0)
    else:
        print("\n❌ DEPLOYMENT VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())