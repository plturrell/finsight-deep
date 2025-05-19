# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

log = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    latency: float
    last_check: float
    error: Optional[str] = None


class HealthMonitor:
    """Production health monitoring with k8s integration support"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.checks: Dict[str, HealthCheck] = {}
        self._running = False
        self._task = None
        
    async def start(self):
        """Start health monitoring"""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop health monitoring"""
        self._running = False
        if self._task:
            await self._task
    
    async def add_check(self, name: str, check_func):
        """Add a health check function"""
        async def wrapped_check():
            start = time.time()
            try:
                await check_func()
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    latency=time.time() - start,
                    last_check=time.time()
                )
            except Exception as e:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    latency=time.time() - start,
                    last_check=time.time(),
                    error=str(e)
                )
        
        # Run initial check
        self.checks[name] = await wrapped_check()
        
        # Store check function for monitoring
        self._check_funcs[name] = wrapped_check
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Run all health checks
                tasks = []
                for name, check_func in self._check_funcs.items():
                    tasks.append(check_func())
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update checks
                for i, (name, _) in enumerate(self._check_funcs.items()):
                    if isinstance(results[i], Exception):
                        self.checks[name] = HealthCheck(
                            name=name,
                            status=HealthStatus.UNHEALTHY,
                            latency=0,
                            last_check=time.time(),
                            error=str(results[i])
                        )
                    else:
                        self.checks[name] = results[i]
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                log.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status"""
        overall_status = HealthStatus.HEALTHY
        
        for check in self.checks.values():
            if check.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif check.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "checks": {
                name: {
                    "status": check.status.value,
                    "latency": check.latency,
                    "last_check": check.last_check,
                    "error": check.error
                }
                for name, check in self.checks.items()
            }
        }
    
    def is_healthy(self) -> bool:
        """Simple health check for k8s probes"""
        return all(check.status != HealthStatus.UNHEALTHY 
                  for check in self.checks.values())