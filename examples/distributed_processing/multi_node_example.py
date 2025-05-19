"""
Multi-Node Distributed Processing Example
Demonstrates processing across multiple machines
"""

import asyncio
import argparse
from typing import Dict, List, Any
import logging

from aiq.distributed.node_manager import NodeManager
from aiq.distributed.worker_node import WorkerNode
from aiq.distributed.task_scheduler import TaskScheduler
from aiq.builder.distributed_workflow_builder import DistributedWorkflowBuilder
from aiq.data_models.workflow import WorkflowConfig
from aiq.data_models.function import FunctionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start_manager_node(port: int = 50051):
    """Start the manager node that coordinates distributed tasks"""
    logger.info("Starting manager node...")
    
    # Create node manager
    node_manager = NodeManager(port=port)
    
    # Create task scheduler
    task_scheduler = TaskScheduler(node_manager)
    
    # Start services
    await node_manager.run()
    await task_scheduler.start()
    
    logger.info(f"Manager node started on port {port}")
    
    # Keep running
    try:
        while True:
            # Print cluster status every 30 seconds
            await asyncio.sleep(30)
            status = node_manager.get_cluster_status()
            logger.info(f"Cluster status: {status['summary']}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down manager...")
    finally:
        await task_scheduler.stop()
        node_manager.stop()


async def start_worker_node(manager_host: str = "localhost", 
                          manager_port: int = 50051,
                          worker_port: int = 50052):
    """Start a worker node that executes tasks"""
    logger.info(f"Starting worker node on port {worker_port}...")
    
    # Create worker node
    worker = WorkerNode(
        manager_host=manager_host,
        manager_port=manager_port,
        worker_port=worker_port
    )
    
    # Register example functions
    from aiq.builder.function import Function
    from aiq.data_models.function import FunctionConfig
    
    # Example text analysis function
    class TextAnalysisFunction(Function):
        def run(self, session, inputs: Dict[str, Any]) -> Any:
            text = inputs.get("text", "")
            analysis_type = inputs.get("analysis_type", "word_count")
            
            if analysis_type == "word_count":
                words = text.split()
                return {
                    "text": text,
                    "word_count": len(words),
                    "char_count": len(text)
                }
            elif analysis_type == "sentiment":
                # Simplified sentiment analysis
                positive_words = ["good", "great", "excellent", "amazing"]
                negative_words = ["bad", "terrible", "awful", "horrible"]
                
                words = text.lower().split()
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                elif negative_count > positive_count:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                return {
                    "text": text,
                    "sentiment": sentiment,
                    "positive_words": positive_count,
                    "negative_words": negative_count
                }
            
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    # Register the function
    config = FunctionConfig(
        name="text_analysis",
        inputs=["text", "analysis_type"],
        outputs=["result"]
    )
    text_function = TextAnalysisFunction(config)
    worker.register_function("text_analysis", text_function)
    
    # Start the worker
    try:
        await worker.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
    finally:
        await worker.stop()


async def run_distributed_workflow(texts: List[str]):
    """Run a distributed workflow across multiple nodes"""
    logger.info("Running distributed workflow...")
    
    # Create distributed workflow builder with multi-node support
    builder = DistributedWorkflowBuilder(enable_multi_node=True)
    
    # Create workflow configuration
    config = WorkflowConfig(
        name="distributed_text_analysis",
        description="Analyze texts across multiple nodes",
        metadata={
            "distributed": True,
            "multi_node": True
        }
    )
    
    # Build workflow
    workflow = builder.build(config)
    
    # Start node manager and scheduler
    await workflow.node_manager.start_server()
    await workflow.task_scheduler.start()
    
    # Submit tasks
    tasks = []
    for i, text in enumerate(texts):
        task_id = await workflow.task_scheduler.submit_task(
            function_name="text_analysis",
            inputs={
                "text": text,
                "analysis_type": "sentiment" if i % 2 == 0 else "word_count"
            }
        )
        tasks.append(task_id)
    
    # Wait for all tasks to complete
    results = []
    for task_id in tasks:
        task = await workflow.task_scheduler.wait_for_task(task_id)
        results.append(task.result)
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {texts[i][:50]}...")
        print(f"Result: {result}")
    
    # Get cluster status
    status = workflow.node_manager.get_cluster_status()
    print(f"\nCluster status: {status['summary']}")
    
    # Cleanup
    await workflow.task_scheduler.stop()
    workflow.node_manager.stop()


async def main():
    parser = argparse.ArgumentParser(description="Multi-node distributed processing example")
    parser.add_argument("--mode", choices=["manager", "worker", "workflow"], 
                       required=True, help="Run mode")
    parser.add_argument("--manager-host", default="localhost",
                       help="Manager node hostname")
    parser.add_argument("--manager-port", type=int, default=50051,
                       help="Manager node port")
    parser.add_argument("--worker-port", type=int, default=50052,
                       help="Worker node port")
    
    args = parser.parse_args()
    
    if args.mode == "manager":
        await start_manager_node(port=args.manager_port)
    elif args.mode == "worker":
        await start_worker_node(
            manager_host=args.manager_host,
            manager_port=args.manager_port,
            worker_port=args.worker_port
        )
    elif args.mode == "workflow":
        # Example texts to analyze
        texts = [
            "This is a great example of distributed processing!",
            "The weather today is absolutely terrible.",
            "AIQToolkit makes distributed computing easy and efficient.",
            "I'm having issues with my computer, it's so frustrating.",
            "The new features are amazing and work perfectly.",
            "This is just a normal text without strong opinions.",
            "Excellent work on the multi-node implementation!",
            "The performance is awful and needs improvement."
        ]
        
        await run_distributed_workflow(texts)


if __name__ == "__main__":
    asyncio.run(main())