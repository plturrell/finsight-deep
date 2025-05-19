# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Any, Optional
import redis.asyncio as redis
import pickle
import logging

log = logging.getLogger(__name__)


class DistributedCache:
    """Production distributed cache with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 ttl: int = 300, 
                 max_memory: str = "100mb"):
        self.redis_url = redis_url
        self.ttl = ttl
        self.max_memory = max_memory
        self._client = None
        self._local_cache = {}  # L1 cache
        self._local_cache_size = 100
        
    async def connect(self):
        """Initialize Redis connection"""
        if not self._client:
            self._client = await redis.from_url(
                self.redis_url,
                decode_responses=False
            )
            
            # Configure Redis memory policy
            await self._client.config_set("maxmemory", self.max_memory)
            await self._client.config_set("maxmemory-policy", "allkeys-lru")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with L1/L2 strategy"""
        # Check L1 cache first
        if key in self._local_cache:
            value, expiry = self._local_cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._local_cache[key]
        
        # Check L2 cache (Redis)
        try:
            if not self._client:
                await self.connect()
                
            data = await self._client.get(key)
            if data:
                value = pickle.loads(data)
                
                # Store in L1 cache
                self._update_local_cache(key, value)
                
                return value
        except Exception as e:
            log.warning(f"Cache get error: {e}")
            
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        ttl = ttl or self.ttl
        
        # Update L1 cache
        self._update_local_cache(key, value, ttl)
        
        # Update L2 cache
        try:
            if not self._client:
                await self.connect()
                
            data = pickle.dumps(value)
            await self._client.setex(key, ttl, data)
            
        except Exception as e:
            log.warning(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Remove from cache"""
        # Remove from L1
        self._local_cache.pop(key, None)
        
        # Remove from L2
        try:
            if self._client:
                await self._client.delete(key)
        except Exception as e:
            log.warning(f"Cache delete error: {e}")
    
    def _update_local_cache(self, key: str, value: Any, ttl: Optional[int] = None):
        """Update L1 cache with LRU eviction"""
        ttl = ttl or self.ttl
        expiry = time.time() + ttl
        
        # Evict oldest if at capacity
        if len(self._local_cache) >= self._local_cache_size:
            oldest_key = min(self._local_cache.keys(), 
                           key=lambda k: self._local_cache[k][1])
            del self._local_cache[oldest_key]
        
        self._local_cache[key] = (value, expiry)
    
    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()