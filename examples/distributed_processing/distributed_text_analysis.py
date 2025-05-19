"""
Distributed Text Analysis Example
Demonstrates multi-GPU processing with AIQToolkit
"""

import asyncio
import time
from typing import Dict, List, Any
import torch

from aiq.builder.distributed_workflow_builder import (
    DistributedWorkflowBuilder, 
    DistributedSession
)
from aiq.data_models.workflow import WorkflowConfig
from aiq.data_models.function import FunctionConfig
from aiq.data_models.llm import LLMConfig
from aiq.builder.function import Function
from aiq.llm.nim_llm import NIMLLM


class ParallelTextAnalyzer(Function):
    """
    Example function that analyzes text in parallel across GPUs
    """
    
    def __init__(self, config: FunctionConfig):
        super().__init__(config)
        self.llm = NIMLLM(config.llm_config)
        
    def run(self, session: DistributedSession, inputs: Dict[str, Any]) -> Any:
        """Analyze text on assigned GPU"""
        text_batch = inputs.get('text_batch', [])
        analysis_type = inputs.get('analysis_type', 'sentiment')
        
        # Log which GPU we're using
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
        print(f"Processing {len(text_batch)} texts on GPU {device_id}")
        
        results = []
        for text in text_batch:
            prompt = f"Analyze the {analysis_type} of this text: {text}"
            
            # Use LLM for analysis
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                session=session
            )
            
            results.append({
                'text': text,
                'analysis': response,
                'device_id': device_id
            })
            
        return results


def create_distributed_text_workflow() -> WorkflowConfig:
    """Create a workflow configuration for distributed text analysis"""
    
    # LLM configuration
    llm_config = LLMConfig(
        type="nim",
        model="meta/llama3-8b-instruct",
        api_base="http://localhost:8000/v1",
        temperature=0.7
    )
    
    # Function configuration
    analyzer_config = FunctionConfig(
        type="custom",
        name="parallel_analyzer",
        llm_config=llm_config,
        inputs=["text_batch", "analysis_type"],
        outputs=["analysis_results"]
    )
    
    # Workflow configuration
    workflow_config = WorkflowConfig(
        name="distributed_text_analysis",
        description="Analyze text in parallel across multiple GPUs",
        functions=[analyzer_config],
        llm_config=llm_config,
        # Enable distributed processing
        metadata={
            "distributed": True,
            "max_gpus": 4
        }
    )
    
    return workflow_config


async def main():
    """Run distributed text analysis example"""
    
    # Sample texts to analyze
    texts = [
        "AIQToolkit makes distributed processing easy!",
        "The weather today is absolutely beautiful.",
        "I'm having issues with my computer again.",
        "Just finished an amazing book about AI.",
        "The new restaurant downtown has great food.",
        "Traffic was terrible this morning.",
        "Can't wait for the weekend!",
        "This project is taking longer than expected."
    ]
    
    # Create distributed workflow
    config = create_distributed_text_workflow()
    builder = DistributedWorkflowBuilder()
    workflow = builder.build(config)
    
    # Check available GPUs
    print(f"Available GPUs: {workflow.gpu_manager.device_count}")
    
    # Split texts into batches for parallel processing
    batch_size = 2
    text_batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # Create session
    session = DistributedSession()
    
    # Measure execution time
    start_time = time.time()
    
    # Run distributed analysis
    results = []
    for i, batch in enumerate(text_batches):
        inputs = {
            f"text_batch_{i}": batch,
            "analysis_type": "sentiment"
        }
        result = await workflow.run(session, inputs)
        results.extend(result)
    
    end_time = time.time()
    
    # Print results
    print("\n=== Analysis Results ===")
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Analysis: {result['analysis']}")
        print(f"Processed on GPU: {result['device_id']}")
    
    # Print performance summary
    print("\n=== Performance Summary ===")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Texts analyzed: {len(texts)}")
    print(f"Average time per text: {(end_time - start_time) / len(texts):.2f} seconds")
    
    # GPU usage summary
    session.record_gpu_usage(workflow.gpu_manager)
    perf_summary = session.get_performance_summary()
    
    print("\n=== GPU Usage ===")
    for device_id, usage in perf_summary['gpu_memory_usage'].items():
        print(f"GPU {device_id}: {usage['usage_percent']:.1f}% memory used")


if __name__ == "__main__":
    asyncio.run(main())