# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example of running distributed inference with AIQToolkit"""

import os
import asyncio
import logging
from typing import List, Optional

import torch
from pydantic import BaseModel

from aiq.distributed.client import DistributedClient
from aiq.data_models.config import AIQConfig
from aiq.data_models.llm import LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    """Model for inference requests"""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    model: str = "nvidia/llama-3.1-nemotron-70b-instruct"

class InferenceResponse(BaseModel):
    """Model for inference responses"""
    text: str
    tokens_generated: int
    time_taken: float
    worker_id: str
    gpu_used: Optional[str] = None

class DistributedInferenceExample:
    """Example distributed inference client"""
    
    def __init__(self, manager_host: str = "localhost", manager_port: int = 50051):
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.client = DistributedClient()
    
    async def connect(self):
        """Connect to distributed system"""
        await self.client.connect(self.manager_host, self.manager_port)
        logger.info(f"Connected to distributed system at {self.manager_host}:{self.manager_port}")
    
    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run distributed inference"""
        logger.info(f"Running inference with model: {request.model}")
        logger.info(f"Prompt: {request.prompt[:50]}...")
        
        # Create task for distributed processing
        task = {
            "type": "inference",
            "model": request.model,
            "prompt": request.prompt,
            "parameters": {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
        }
        
        # Submit task and wait for result
        result = await self.client.submit_task(task)
        
        return InferenceResponse(
            text=result["text"],
            tokens_generated=result["tokens_generated"],
            time_taken=result["time_taken"],
            worker_id=result["worker_id"],
            gpu_used=result.get("gpu_used")
        )
    
    async def run_batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Run batch inference across multiple workers"""
        logger.info(f"Running batch inference for {len(requests)} requests")
        
        # Submit all tasks concurrently
        tasks = []
        for i, request in enumerate(requests):
            task = {
                "type": "inference",
                "model": request.model,
                "prompt": request.prompt,
                "parameters": {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                },
                "batch_id": f"batch-{i}"
            }
            tasks.append(self.client.submit_task(task))
        
        # Wait for all results
        results = await asyncio.gather(*tasks)
        
        responses = []
        for result in results:
            responses.append(InferenceResponse(
                text=result["text"],
                tokens_generated=result["tokens_generated"],
                time_taken=result["time_taken"],
                worker_id=result["worker_id"],
                gpu_used=result.get("gpu_used")
            ))
        
        return responses
    
    async def disconnect(self):
        """Disconnect from distributed system"""
        await self.client.disconnect()
        logger.info("Disconnected from distributed system")

async def main():
    """Main example runner"""
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the key principles of machine learning?",
        "Describe the architecture of a neural network.",
        "How does distributed computing improve performance?"
    ]
    
    # Create client
    client = DistributedInferenceExample(
        manager_host=os.environ.get("MANAGER_HOST", "localhost"),
        manager_port=int(os.environ.get("MANAGER_PORT", "50051"))
    )
    
    try:
        # Connect to distributed system
        await client.connect()
        
        # Single inference example
        print("\n=== Single Inference Example ===")
        single_request = InferenceRequest(
            prompt=prompts[0],
            max_tokens=150,
            temperature=0.7
        )
        response = await client.run_inference(single_request)
        print(f"Worker: {response.worker_id}")
        print(f"GPU: {response.gpu_used}")
        print(f"Time: {response.time_taken:.2f}s")
        print(f"Response: {response.text[:200]}...")
        
        # Batch inference example
        print("\n=== Batch Inference Example ===")
        batch_requests = [
            InferenceRequest(prompt=prompt, max_tokens=100, temperature=0.7)
            for prompt in prompts
        ]
        responses = await client.run_batch_inference(batch_requests)
        
        for i, response in enumerate(responses):
            print(f"\nRequest {i+1}:")
            print(f"  Worker: {response.worker_id}")
            print(f"  GPU: {response.gpu_used}")
            print(f"  Time: {response.time_taken:.2f}s")
            print(f"  Response: {response.text[:100]}...")
        
        # Calculate statistics
        total_time = sum(r.time_taken for r in responses)
        avg_time = total_time / len(responses)
        workers_used = set(r.worker_id for r in responses)
        
        print("\n=== Statistics ===")
        print(f"Total requests: {len(responses)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Workers used: {len(workers_used)}")
        print(f"Worker IDs: {', '.join(workers_used)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())