# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import uuid
import time
import logging
from asyncio import Queue
import aio_pika

log = logging.getLogger(__name__)


@dataclass
class Message:
    id: str
    payload: Dict[str, Any]
    timestamp: float
    retries: int = 0
    max_retries: int = 3


class MessageQueue:
    """Production message queue with RabbitMQ backend"""
    
    def __init__(self, amqp_url: str = "amqp://guest:guest@localhost/"):
        self.amqp_url = amqp_url
        self.connection = None
        self.channel = None
        self.exchange = None
        
    async def connect(self):
        """Establish connection to RabbitMQ"""
        self.connection = await aio_pika.connect_robust(
            self.amqp_url,
            client_properties={"connection_name": "agent_queue"}
        )
        
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=10)
        
        # Declare exchange
        self.exchange = await self.channel.declare_exchange(
            "agent_exchange",
            type=aio_pika.ExchangeType.TOPIC,
            durable=True
        )
    
    async def publish(self, routing_key: str, message: Dict[str, Any]):
        """Publish message to queue"""
        if not self.channel:
            await self.connect()
        
        msg = Message(
            id=str(uuid.uuid4()),
            payload=message,
            timestamp=time.time()
        )
        
        await self.exchange.publish(
            aio_pika.Message(
                body=json.dumps(msg.__dict__).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                message_id=msg.id,
                timestamp=int(msg.timestamp)
            ),
            routing_key=routing_key
        )
    
    async def consume(
        self, 
        queue_name: str,
        routing_key: str,
        callback: Callable[[Message], None],
        error_queue: str = None
    ):
        """Consume messages from queue"""
        if not self.channel:
            await self.connect()
        
        # Declare queue
        queue = await self.channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": error_queue or f"{queue_name}.error"
            }
        )
        
        # Bind to exchange
        await queue.bind(self.exchange, routing_key)
        
        # Start consuming
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        msg_data = json.loads(message.body)
                        msg = Message(**msg_data)
                        
                        await callback(msg)
                        
                    except Exception as e:
                        log.error(f"Message processing error: {e}")
                        
                        # Retry logic
                        if msg.retries < msg.max_retries:
                            msg.retries += 1
                            await self.publish(routing_key, msg.payload)
                        else:
                            # Send to DLQ
                            if error_queue:
                                await self.publish(error_queue, {
                                    "original_message": msg.__dict__,
                                    "error": str(e),
                                    "timestamp": time.time()
                                })
                        
                        # Reject message
                        await message.reject(requeue=False)
    
    async def close(self):
        """Close connection"""
        if self.connection:
            await self.connection.close()


class InMemoryQueue:
    """Fallback in-memory queue for testing"""
    
    def __init__(self, max_size: int = 1000):
        self.queues: Dict[str, Queue] = {}
        self.max_size = max_size
        
    async def publish(self, routing_key: str, message: Dict[str, Any]):
        """Add message to queue"""
        if routing_key not in self.queues:
            self.queues[routing_key] = Queue(maxsize=self.max_size)
        
        msg = Message(
            id=str(uuid.uuid4()),
            payload=message,
            timestamp=time.time()
        )
        
        await self.queues[routing_key].put(msg)
    
    async def consume(
        self, 
        queue_name: str,
        routing_key: str,
        callback: Callable[[Message], None],
        error_queue: str = None
    ):
        """Consume messages from queue"""
        if routing_key not in self.queues:
            self.queues[routing_key] = Queue(maxsize=self.max_size)
        
        queue = self.queues[routing_key]
        
        while True:
            try:
                msg = await queue.get()
                await callback(msg)
            except Exception as e:
                log.error(f"Message processing error: {e}")
                
                if error_queue and error_queue not in self.queues:
                    self.queues[error_queue] = Queue(maxsize=self.max_size)
                
                if error_queue:
                    await self.queues[error_queue].put(Message(
                        id=str(uuid.uuid4()),
                        payload={
                            "original_message": msg.__dict__,
                            "error": str(e)
                        },
                        timestamp=time.time()
                    ))