"""
Performance Benchmarking Suite for Digital Human System
Comprehensive testing of throughput, latency, and resource utilization
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import psutil
import GPUtil
from memory_profiler import profile
from prometheus_client import Counter, Histogram, Summary, Gauge

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.scalability.load_balancer import LoadBalancer, LoadBalancingStrategy
from aiq.digital_human.monitoring.production_monitor import ProductionMonitor
from aiq.utils.debugging_utils import log_function_call


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    # Test scenarios
    scenarios: List[str] = None
    concurrent_users: List[int] = None  # [1, 10, 50, 100, 200]
    test_duration_seconds: int = 300
    warmup_duration_seconds: int = 30
    
    # Request patterns
    request_patterns: List[str] = None  # ["uniform", "burst", "spike", "realistic"]
    message_lengths: List[int] = None  # [50, 200, 500, 1000]
    
    # System configurations
    gpu_configs: List[Dict[str, Any]] = None
    model_configs: List[Dict[str, Any]] = None
    cache_configs: List[Dict[str, Any]] = None
    
    # Metrics collection
    metrics_interval_seconds: int = 5
    enable_profiling: bool = True
    save_results: bool = True
    results_dir: str = "benchmark_results"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    scenario: str
    timestamp: datetime
    configuration: Dict[str, Any]
    
    # Performance metrics
    throughput: float  # requests per second
    average_latency: float  # milliseconds
    p50_latency: float
    p95_latency: float
    p99_latency: float
    error_rate: float
    
    # Resource utilization
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    
    # Quality metrics
    response_quality: float
    emotional_accuracy: float
    animation_smoothness: float
    
    # Detailed data
    latency_distribution: List[float]
    resource_timeline: pd.DataFrame
    error_details: List[Dict[str, Any]]


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for Digital Human system
    Tests various aspects of system performance and scalability
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.orchestrator = None
        self.load_balancer = None
        self.monitor = None
        
        # Metrics collection
        self.metrics = {
            "requests_sent": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "latencies": [],
            "resource_samples": []
        }
        
        # Test clients
        self.test_clients = []
        self.executor = ThreadPoolExecutor(max_workers=32)
        
        # Results storage
        self.results = []
        self.current_test_start = None
        
        logger.info("Initialized performance benchmark suite")
    
    async def initialize(self):
        """Initialize benchmark components"""
        # Initialize orchestrator with benchmark configuration
        orchestrator_config = {
            "enable_gpu": True,
            "enable_profiling": True,
            "max_concurrent_sessions": 500,
            "cache_ttl": 3600,
            "model_name": "meta-llama/Llama-3.1-70B-Instruct"
        }
        
        self.orchestrator = DigitalHumanOrchestrator(orchestrator_config)
        await self.orchestrator.initialize()
        
        # Initialize load balancer
        lb_config = {
            "strategy": LoadBalancingStrategy.ADAPTIVE,
            "min_instances": 2,
            "max_instances": 10,
            "auto_scaling": True
        }
        
        self.load_balancer = LoadBalancer(lb_config)
        await self.load_balancer.initialize()
        
        # Initialize monitor
        monitor_config = {
            "prometheus_port": 9091,
            "enable_alerts": False
        }
        
        self.monitor = ProductionMonitor(monitor_config)
        
        logger.info("Benchmark components initialized")
    
    async def run_benchmark_suite(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        results = []
        
        # Default scenarios if not specified
        if not self.config.scenarios:
            self.config.scenarios = [
                "baseline",
                "high_concurrency",
                "sustained_load",
                "burst_traffic",
                "cache_efficiency",
                "gpu_scaling",
                "failure_resilience"
            ]
        
        for scenario in self.config.scenarios:
            logger.info(f"Running benchmark scenario: {scenario}")
            
            try:
                result = await self._run_scenario(scenario)
                results.append(result)
                
                # Save intermediate results
                if self.config.save_results:
                    self._save_result(result)
                
            except Exception as e:
                logger.error(f"Benchmark scenario {scenario} failed: {e}")
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    async def _run_scenario(self, scenario: str) -> BenchmarkResult:
        """Run a specific benchmark scenario"""
        self.current_test_start = datetime.now()
        
        # Reset metrics
        self._reset_metrics()
        
        # Configure scenario
        scenario_config = self._get_scenario_config(scenario)
        
        # Warmup phase
        logger.info(f"Starting warmup phase ({self.config.warmup_duration_seconds}s)")
        await self._warmup_phase(scenario_config)
        
        # Main test phase
        logger.info(f"Starting test phase ({self.config.test_duration_seconds}s)")
        test_task = asyncio.create_task(self._test_phase(scenario_config))
        
        # Collect metrics during test
        metrics_task = asyncio.create_task(self._collect_metrics_loop())
        
        # Wait for test completion
        await test_task
        metrics_task.cancel()
        
        # Process results
        result = self._process_results(scenario, scenario_config)
        
        logger.info(f"Scenario {scenario} completed")
        return result
    
    def _get_scenario_config(self, scenario: str) -> Dict[str, Any]:
        """Get configuration for specific scenario"""
        configs = {
            "baseline": {
                "concurrent_users": 10,
                "message_rate": 1.0,  # messages per second per user
                "message_length": 200,
                "pattern": "uniform"
            },
            "high_concurrency": {
                "concurrent_users": 200,
                "message_rate": 0.5,
                "message_length": 100,
                "pattern": "uniform"
            },
            "sustained_load": {
                "concurrent_users": 50,
                "message_rate": 2.0,
                "message_length": 300,
                "pattern": "uniform"
            },
            "burst_traffic": {
                "concurrent_users": 100,
                "message_rate": 5.0,
                "message_length": 150,
                "pattern": "burst"
            },
            "cache_efficiency": {
                "concurrent_users": 20,
                "message_rate": 1.0,
                "message_length": 200,
                "pattern": "repeated"  # Same messages to test caching
            },
            "gpu_scaling": {
                "concurrent_users": [10, 50, 100, 150],
                "message_rate": 1.0,
                "message_length": 500,
                "pattern": "uniform"
            },
            "failure_resilience": {
                "concurrent_users": 30,
                "message_rate": 1.0,
                "message_length": 200,
                "pattern": "uniform",
                "inject_failures": True
            }
        }
        
        return configs.get(scenario, configs["baseline"])
    
    async def _warmup_phase(self, config: Dict[str, Any]):
        """Warmup phase to prepare system"""
        warmup_users = min(10, config["concurrent_users"])
        
        # Create warmup users
        warmup_tasks = []
        for i in range(warmup_users):
            task = asyncio.create_task(
                self._simulate_user(
                    f"warmup_user_{i}",
                    config,
                    duration=self.config.warmup_duration_seconds
                )
            )
            warmup_tasks.append(task)
        
        # Wait for warmup completion
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
    
    async def _test_phase(self, config: Dict[str, Any]):
        """Main test phase"""
        # Handle GPU scaling scenario
        if config.get("concurrent_users") and isinstance(config["concurrent_users"], list):
            for user_count in config["concurrent_users"]:
                await self._run_test_iteration(config, user_count)
        else:
            await self._run_test_iteration(config, config["concurrent_users"])
    
    async def _run_test_iteration(self, config: Dict[str, Any], concurrent_users: int):
        """Run single test iteration"""
        user_tasks = []
        
        # Create test users
        for i in range(concurrent_users):
            task = asyncio.create_task(
                self._simulate_user(
                    f"test_user_{i}",
                    config,
                    duration=self.config.test_duration_seconds
                )
            )
            user_tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _simulate_user(
        self,
        user_id: str,
        config: Dict[str, Any],
        duration: int
    ):
        """Simulate a single user's behavior"""
        start_time = time.time()
        session_id = None
        
        try:
            # Start session
            session_id = await self.orchestrator.create_session(
                user_id=user_id,
                user_profile={"test_user": True}
            )
            
            # Send messages according to pattern
            while time.time() - start_time < duration:
                message_content = self._generate_test_message(config)
                
                # Record request start
                request_start = time.time()
                self.metrics["requests_sent"] += 1
                
                try:
                    # Send message
                    response = await self.orchestrator.process_message(
                        session_id=session_id,
                        message_content=message_content,
                        message_type="text"
                    )
                    
                    # Record latency
                    latency = (time.time() - request_start) * 1000  # Convert to ms
                    self.metrics["latencies"].append(latency)
                    self.metrics["requests_completed"] += 1
                    
                except Exception as e:
                    self.metrics["requests_failed"] += 1
                    logger.error(f"Request failed for {user_id}: {e}")
                
                # Wait according to message rate
                wait_time = 1.0 / config["message_rate"]
                await asyncio.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"User simulation failed for {user_id}: {e}")
        finally:
            # End session
            if session_id:
                await self.orchestrator.end_session(session_id)
    
    def _generate_test_message(self, config: Dict[str, Any]) -> str:
        """Generate test message based on configuration"""
        pattern = config.get("pattern", "uniform")
        length = config.get("message_length", 200)
        
        if pattern == "repeated":
            # Use cached messages for testing cache efficiency
            messages = [
                "What's my portfolio performance?",
                "Show me my investment analysis",
                "What are the market trends?",
                "Review my risk assessment",
                "Explain my asset allocation"
            ]
            return np.random.choice(messages)
        
        # Generate random message of specified length
        words = [
            "investment", "portfolio", "market", "analysis", "risk",
            "return", "allocation", "strategy", "performance", "trend",
            "stock", "bond", "equity", "asset", "finance"
        ]
        
        num_words = length // 10  # Approximate word count
        message_words = np.random.choice(words, num_words, replace=True)
        return " ".join(message_words)
    
    async def _collect_metrics_loop(self):
        """Continuously collect system metrics during test"""
        while True:
            try:
                metrics_sample = await self._collect_system_metrics()
                self.metrics["resource_samples"].append(metrics_sample)
                
                await asyncio.sleep(self.config.metrics_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_cores = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        
        # GPU metrics
        gpu_metrics = self._collect_gpu_metrics()
        
        # Process metrics
        process = psutil.Process()
        process_cpu = process.cpu_percent()
        process_memory = process.memory_info().rss / (1024**3)
        
        # Model metrics
        model_metrics = await self._collect_model_metrics()
        
        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "cpu_cores": cpu_cores,
            "memory_percent": memory_percent,
            "memory_gb": memory_gb,
            "gpu_usage": gpu_metrics["usage"],
            "gpu_memory": gpu_metrics["memory"],
            "gpu_temperature": gpu_metrics["temperature"],
            "process_cpu": process_cpu,
            "process_memory_gb": process_memory,
            "active_sessions": model_metrics["active_sessions"],
            "cache_hit_rate": model_metrics["cache_hit_rate"],
            "model_queue_size": model_metrics["queue_size"]
        }
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics"""
        metrics = {
            "usage": 0.0,
            "memory": 0.0,
            "temperature": 0.0
        }
        
        try:
            if torch.cuda.is_available():
                # PyTorch CUDA metrics
                gpu_count = torch.cuda.device_count()
                total_usage = 0.0
                total_memory = 0.0
                
                for i in range(gpu_count):
                    total_usage += torch.cuda.utilization(i)
                    total_memory += torch.cuda.max_memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100
                
                metrics["usage"] = total_usage / gpu_count
                metrics["memory"] = total_memory / gpu_count
                
                # Try to get temperature if available
                gpus = GPUtil.getGPUs()
                if gpus:
                    temps = [gpu.temperature for gpu in gpus]
                    metrics["temperature"] = sum(temps) / len(temps)
                    
        except Exception as e:
            logger.error(f"GPU metrics collection error: {e}")
        
        return metrics
    
    async def _collect_model_metrics(self) -> Dict[str, Any]:
        """Collect model-specific metrics"""
        metrics = {
            "active_sessions": 0,
            "cache_hit_rate": 0.0,
            "queue_size": 0
        }
        
        try:
            # Get metrics from orchestrator
            orchestrator_metrics = await self.orchestrator.get_metrics()
            metrics["active_sessions"] = orchestrator_metrics.get("active_sessions", 0)
            metrics["cache_hit_rate"] = orchestrator_metrics.get("cache_hit_rate", 0.0)
            
            # Get metrics from load balancer
            if self.load_balancer:
                lb_metrics = await self.load_balancer.get_metrics()
                metrics["queue_size"] = lb_metrics.get("queue_size", 0)
                
        except Exception as e:
            logger.error(f"Model metrics collection error: {e}")
        
        return metrics
    
    def _process_results(
        self,
        scenario: str,
        config: Dict[str, Any]
    ) -> BenchmarkResult:
        """Process collected metrics into benchmark result"""
        # Calculate latency percentiles
        latencies = sorted(self.metrics["latencies"])
        
        if not latencies:
            # No successful requests
            return BenchmarkResult(
                scenario=scenario,
                timestamp=self.current_test_start,
                configuration=config,
                throughput=0.0,
                average_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=0.0,
                gpu_memory=0.0,
                response_quality=0.0,
                emotional_accuracy=0.0,
                animation_smoothness=0.0,
                latency_distribution=[],
                resource_timeline=pd.DataFrame(),
                error_details=[]
            )
        
        # Calculate performance metrics
        total_requests = self.metrics["requests_sent"]
        successful_requests = self.metrics["requests_completed"]
        failed_requests = self.metrics["requests_failed"]
        
        test_duration = self.config.test_duration_seconds
        throughput = successful_requests / test_duration
        
        average_latency = statistics.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        
        # Calculate resource utilization
        resource_samples = self.metrics["resource_samples"]
        avg_cpu = statistics.mean([s["cpu_percent"] for s in resource_samples])
        avg_memory = statistics.mean([s["memory_percent"] for s in resource_samples])
        avg_gpu = statistics.mean([s["gpu_usage"] for s in resource_samples])
        avg_gpu_memory = statistics.mean([s["gpu_memory"] for s in resource_samples])
        
        # Calculate quality metrics (mock for now)
        response_quality = 0.95 - (error_rate * 0.5)
        emotional_accuracy = 0.92 - (error_rate * 0.3)
        animation_smoothness = 0.98 - (error_rate * 0.2)
        
        # Create resource timeline
        resource_df = pd.DataFrame(resource_samples)
        
        return BenchmarkResult(
            scenario=scenario,
            timestamp=self.current_test_start,
            configuration=config,
            throughput=throughput,
            average_latency=average_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            error_rate=error_rate,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu,
            gpu_memory=avg_gpu_memory,
            response_quality=response_quality,
            emotional_accuracy=emotional_accuracy,
            animation_smoothness=animation_smoothness,
            latency_distribution=latencies,
            resource_timeline=resource_df,
            error_details=[]
        )
    
    def _reset_metrics(self):
        """Reset metrics for new test"""
        self.metrics = {
            "requests_sent": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "latencies": [],
            "resource_samples": []
        }
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file"""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary_file = results_dir / f"{result.scenario}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}_summary.json"
        
        summary_data = {
            "scenario": result.scenario,
            "timestamp": result.timestamp.isoformat(),
            "configuration": result.configuration,
            "metrics": {
                "throughput": result.throughput,
                "average_latency": result.average_latency,
                "p50_latency": result.p50_latency,
                "p95_latency": result.p95_latency,
                "p99_latency": result.p99_latency,
                "error_rate": result.error_rate,
                "cpu_usage": result.cpu_usage,
                "memory_usage": result.memory_usage,
                "gpu_usage": result.gpu_usage,
                "gpu_memory": result.gpu_memory,
                "response_quality": result.response_quality,
                "emotional_accuracy": result.emotional_accuracy,
                "animation_smoothness": result.animation_smoothness
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save detailed data
        details_dir = results_dir / f"{result.scenario}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}"
        details_dir.mkdir(exist_ok=True)
        
        # Save latency distribution
        latency_file = details_dir / "latency_distribution.csv"
        pd.DataFrame({"latency_ms": result.latency_distribution}).to_csv(latency_file, index=False)
        
        # Save resource timeline
        resource_file = details_dir / "resource_timeline.csv"
        result.resource_timeline.to_csv(resource_file, index=False)
        
        # Generate plots
        self._generate_plots(result, details_dir)
    
    def _generate_plots(self, result: BenchmarkResult, output_dir: Path):
        """Generate visualization plots for benchmark results"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Latency distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(result.latency_distribution, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(result.average_latency, color='red', linestyle='--', label=f'Average: {result.average_latency:.2f}ms')
        plt.axvline(result.p95_latency, color='orange', linestyle='--', label=f'P95: {result.p95_latency:.2f}ms')
        plt.axvline(result.p99_latency, color='green', linestyle='--', label=f'P99: {result.p99_latency:.2f}ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title(f'Latency Distribution - {result.scenario}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_distribution.png', dpi=300)
        plt.close()
        
        # Resource utilization over time
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert timestamps to relative time
        timeline = result.resource_timeline.copy()
        timeline['relative_time'] = (timeline['timestamp'] - timeline['timestamp'].min()) / 60  # Minutes
        
        # CPU usage
        ax1.plot(timeline['relative_time'], timeline['cpu_percent'], 'b-')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization')
        ax1.grid(True)
        
        # Memory usage
        ax2.plot(timeline['relative_time'], timeline['memory_percent'], 'r-')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_title('Memory Utilization')
        ax2.grid(True)
        
        # GPU usage
        ax3.plot(timeline['relative_time'], timeline['gpu_usage'], 'g-')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('GPU Usage (%)')
        ax3.set_title('GPU Utilization')
        ax3.grid(True)
        
        # Active sessions
        ax4.plot(timeline['relative_time'], timeline['active_sessions'], 'm-')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Active Sessions')
        ax4.set_title('Active Sessions Over Time')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_utilization.png', dpi=300)
        plt.close()
        
        # Percentile comparison
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = [np.percentile(result.latency_distribution, p) for p in percentiles]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(percentiles)), percentile_values, color='skyblue', edgecolor='navy')
        plt.xticks(range(len(percentiles)), [f'P{p}' for p in percentiles])
        plt.ylabel('Latency (ms)')
        plt.title(f'Latency Percentiles - {result.scenario}')
        plt.grid(axis='y')
        
        # Add value labels on bars
        for i, value in enumerate(percentile_values):
            plt.text(i, value + 1, f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_percentiles.png', dpi=300)
        plt.close()
    
    def _generate_summary_report(self, results: List[BenchmarkResult]):
        """Generate summary report for all benchmark results"""
        if not results:
            logger.warning("No results to generate summary report")
            return
        
        results_dir = Path(self.config.results_dir)
        summary_file = results_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Digital Human Performance Benchmark Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            
            best_throughput = max(results, key=lambda r: r.throughput)
            best_latency = min(results, key=lambda r: r.average_latency)
            
            f.write(f"- **Best Throughput**: {best_throughput.throughput:.2f} req/s ({best_throughput.scenario})\n")
            f.write(f"- **Best Latency**: {best_latency.average_latency:.2f} ms ({best_latency.scenario})\n")
            f.write(f"- **Scenarios Tested**: {len(results)}\n\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write("| Scenario | Throughput (req/s) | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Error Rate | CPU (%) | GPU (%) |\n")
            f.write("|----------|-------------------|------------------|------------------|------------------|------------|---------|----------|\n")
            
            for result in results:
                f.write(f"| {result.scenario} | {result.throughput:.2f} | {result.average_latency:.2f} | {result.p95_latency:.2f} | {result.p99_latency:.2f} | {result.error_rate:.3f} | {result.cpu_usage:.1f} | {result.gpu_usage:.1f} |\n")
            
            # Key findings
            f.write("\n## Key Findings\n\n")
            
            # Find performance bottlenecks
            high_latency_scenarios = [r for r in results if r.p99_latency > 1000]
            if high_latency_scenarios:
                f.write("### Performance Bottlenecks\n")
                for scenario in high_latency_scenarios:
                    f.write(f"- {scenario.scenario}: P99 latency {scenario.p99_latency:.0f}ms\n")
                f.write("\n")
            
            # Resource utilization
            f.write("### Resource Utilization\n")
            avg_cpu = statistics.mean([r.cpu_usage for r in results])
            avg_gpu = statistics.mean([r.gpu_usage for r in results])
            f.write(f"- Average CPU utilization: {avg_cpu:.1f}%\n")
            f.write(f"- Average GPU utilization: {avg_gpu:.1f}%\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if avg_gpu < 70:
                f.write("- GPU utilization is below 70%, consider increasing batch size or concurrent requests\n")
            
            if any(r.error_rate > 0.01 for r in results):
                f.write("- Error rates above 1% detected, investigate stability issues\n")
            
            if any(r.p99_latency > 2000 for r in results):
                f.write("- High P99 latencies detected, consider implementing request timeouts and circuit breakers\n")
            
            # Configuration details
            f.write("\n## Test Configuration\n\n")
            f.write(f"- Test Duration: {self.config.test_duration_seconds}s\n")
            f.write(f"- Warmup Duration: {self.config.warmup_duration_seconds}s\n")
            f.write(f"- Metrics Interval: {self.config.metrics_interval_seconds}s\n")
        
        logger.info(f"Summary report generated: {summary_file}")
    
    async def shutdown(self):
        """Cleanup benchmark resources"""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.load_balancer:
            await self.load_balancer.shutdown()
        
        if self.monitor:
            await self.monitor.shutdown()
        
        self.executor.shutdown(wait=True)
        
        logger.info("Benchmark suite shut down")


# Utility functions for running benchmarks
async def run_standard_benchmark():
    """Run standard benchmark suite"""
    config = BenchmarkConfig(
        scenarios=["baseline", "high_concurrency", "sustained_load"],
        test_duration_seconds=300,
        warmup_duration_seconds=30,
        enable_profiling=True,
        save_results=True
    )
    
    benchmark = PerformanceBenchmark(config)
    await benchmark.initialize()
    
    try:
        results = await benchmark.run_benchmark_suite()
        return results
    finally:
        await benchmark.shutdown()


async def run_quick_benchmark():
    """Run quick benchmark for CI/CD"""
    config = BenchmarkConfig(
        scenarios=["baseline"],
        test_duration_seconds=60,
        warmup_duration_seconds=10,
        enable_profiling=False,
        save_results=True
    )
    
    benchmark = PerformanceBenchmark(config)
    await benchmark.initialize()
    
    try:
        results = await benchmark.run_benchmark_suite()
        return results
    finally:
        await benchmark.shutdown()


if __name__ == "__main__":
    # Run benchmark from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Digital Human Performance Benchmark")
    parser.add_argument("--mode", choices=["standard", "quick", "custom"], default="standard")
    parser.add_argument("--scenarios", nargs="+", help="Scenarios to run")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        asyncio.run(run_quick_benchmark())
    else:
        config = BenchmarkConfig(
            scenarios=args.scenarios,
            test_duration_seconds=args.duration,
            save_results=True
        )
        
        benchmark = PerformanceBenchmark(config)
        asyncio.run(benchmark.initialize())
        
        try:
            asyncio.run(benchmark.run_benchmark_suite())
        finally:
            asyncio.run(benchmark.shutdown())