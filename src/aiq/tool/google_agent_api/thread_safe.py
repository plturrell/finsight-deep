# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from typing import Any, Dict, Optional
import time
from contextlib import asynccontextmanager
from collections import defaultdict
import weakref

import logging
log = logging.getLogger(__name__)


class ThreadSafeCircuitBreaker:
    """Thread-safe circuit breaker implementation"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 half_open_max_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        
        self._lock = threading.RLock()
        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._half_open_requests = 0
    
    def get_state(self) -> str:
        """Get current state (thread-safe)"""
        with self._lock:
            # Check if we should transition from open to half-open
            if self._state == "open":
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "half-open"
                    self._half_open_requests = 0
            
            return self._state
    
    def record_success(self):
        """Record a successful request"""
        with self._lock:
            self._success_count += 1
            
            if self._state == "half-open":
                # Success in half-open state - check if we can close
                self._half_open_requests += 1
                if self._half_open_requests >= self.half_open_max_requests:
                    self._state = "closed"
                    self._failure_count = 0
                    log.info("Circuit breaker closed after successful half-open test")
    
    def record_failure(self):
        """Record a failed request"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == "closed":
                if self._failure_count >= self.failure_threshold:
                    self._state = "open"
                    log.warning("Circuit breaker opened due to failures")
            
            elif self._state == "half-open":
                # Failure in half-open state - go back to open
                self._state = "open"
                self._failure_count = 0
                log.warning("Circuit breaker re-opened after half-open failure")
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        state = self.get_state()
        
        if state == "closed":
            return True
        
        elif state == "open":
            return False
        
        else:  # half-open
            with self._lock:
                return self._half_open_requests < self.half_open_max_requests


class ThreadSafeCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        
        self._lock = threading.RLock()
        self._cache = {}
        self._access_times = {}
        self._creation_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check if expired
            if time.time() - self._creation_times[key] > self.ttl:
                self._evict(key)
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self._lock:
            # Evict if at capacity
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Set value
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._creation_times[key] = time.time()
    
    def delete(self, key: str):
        """Delete value from cache"""
        with self._lock:
            self._evict(key)
    
    def _evict(self, key: str):
        """Evict a key (internal use)"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._creation_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        self._evict(lru_key)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()


class ThreadSafeAsyncCache:
    """Thread-safe async cache for coroutines"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = ThreadSafeCache(max_size, ttl)
        self._pending = {}
        self._lock = asyncio.Lock()
    
    async def get_or_compute(self, key: str, compute_func):
        """Get from cache or compute if missing"""
        
        # Check cache first
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Avoid duplicate computation
        async with self._lock:
            # Check if computation is already pending
            if key in self._pending:
                return await self._pending[key]
            
            # Start computation
            future = asyncio.create_task(compute_func())
            self._pending[key] = future
        
        try:
            # Wait for computation
            value = await future
            
            # Cache the result
            self.cache.set(key, value)
            
            return value
            
        finally:
            # Remove from pending
            async with self._lock:
                self._pending.pop(key, None)


class ThreadSafeConnectionPool:
    """Thread-safe connection pool"""
    
    def __init__(self, 
                 create_func,
                 max_size: int = 100,
                 max_idle_time: int = 300):
        self.create_func = create_func
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        
        self._pool = []
        self._active = set()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._created_count = 0
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        conn = await self._acquire()
        try:
            yield conn
        finally:
            await self._release(conn)
    
    async def _acquire(self):
        """Get connection from pool"""
        while True:
            with self._lock:
                # Try to get from pool
                while self._pool:
                    conn, idle_since = self._pool.pop(0)
                    
                    # Check if connection is still valid
                    if time.time() - idle_since < self.max_idle_time:
                        self._active.add(id(conn))
                        return conn
                    else:
                        # Connection expired, close it
                        try:
                            await conn.close()
                        except:
                            pass
                        self._created_count -= 1
                
                # Check if we can create new connection
                if self._created_count < self.max_size:
                    self._created_count += 1
                    # Create outside lock to avoid blocking
                    break
                
                # Wait for available connection
                self._condition.wait()
        
        # Create new connection
        try:
            conn = await self.create_func()
            with self._lock:
                self._active.add(id(conn))
            return conn
        except:
            with self._lock:
                self._created_count -= 1
            raise
    
    async def _release(self, conn):
        """Return connection to pool"""
        with self._lock:
            self._active.discard(id(conn))
            
            # Return to pool if there's space
            if len(self._pool) < self.max_size:
                self._pool.append((conn, time.time()))
                self._condition.notify()
            else:
                # Close excess connection
                try:
                    await conn.close()
                except:
                    pass
                self._created_count -= 1
    
    async def close_all(self):
        """Close all connections"""
        with self._lock:
            # Get all connections
            all_conns = [conn for conn, _ in self._pool]
            self._pool.clear()
            
            # Note: we can't close active connections safely
            
        # Close idle connections
        for conn in all_conns:
            try:
                await conn.close()
            except:
                pass


class ThreadSafeMetrics:
    """Thread-safe metrics collection"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)
        self._histograms = defaultdict(list)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        with self._lock:
            self._counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value"""
        with self._lock:
            self._gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Record a histogram value"""
        with self._lock:
            self._histograms[name].append(value)
            
            # Keep only last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "mean": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0
                    }
                    for name, values in self._histograms.items()
                }
            }