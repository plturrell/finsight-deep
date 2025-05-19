"""Performance tests for Digital Human system"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import statistics
import psutil
import gc

import numpy as np
import pandas as pd
import pytest
import torch

from aiq.digital_human.orchestrator import (
    DigitalHumanOrchestrator,
    OrchestratorConfig,
    ConversationContext
)
from aiq.digital_human.conversation_engine import ConversationEngine
from aiq.digital_human.financial.mcts_financial_analyzer import MCTSFinancialAnalyzer
from aiq.digital_human.financial.portfolio_optimizer import PortfolioOptimizer
from aiq.digital_human.financial.financial_data_processor import FinancialDataProcessor
from aiq.digital_human.financial.risk_assessment_engine import RiskAssessmentEngine


# Performance test configuration
PERFORMANCE_CONFIG = {
    "load_test": {
        "concurrent_users": [1, 5, 10, 20, 50],
        "requests_per_user": 10,
        "think_time_seconds": 0.5
    },
    "stress_test": {
        "max_concurrent_users": 100,
        "ramp_up_seconds": 30,
        "sustained_duration_seconds": 60
    },
    "endurance_test": {
        "concurrent_users": 20,
        "duration_minutes": 10,
        "request_interval_seconds": 1
    },
    "spike_test": {
        "base_users": 10,
        "spike_users": 50,
        "spike_duration_seconds": 5
    },
    "volume_test": {
        "data_sizes": [100, 1000, 10000, 100000],
        "concurrent_requests": 5
    }
}


# Test fixtures
@pytest.fixture
def performance_config():
    """Create performance test configuration"""
    return OrchestratorConfig(
        enable_gpu=torch.cuda.is_available(),
        max_concurrent_sessions=100,
        session_timeout_minutes=5,
        health_check_interval_seconds=10,
        enable_caching=True,
        cache_ttl_seconds=300,
        enable_logging=False,  # Disable for performance tests
        log_level="ERROR",
        enable_metrics=True,
        enable_tracing=False  # Disable for performance tests
    )


@pytest.fixture
async def orchestrator(performance_config):
    """Create Digital Human Orchestrator for performance testing"""
    orchestrator = DigitalHumanOrchestrator(performance_config)
    await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.shutdown()


# Performance metrics collection
class PerformanceMetrics:
    """Collect and analyze performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.throughput_samples = []
        self.error_count = 0
        self.success_count = 0
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start metrics collection"""
        self.start_time = time.time()
        self.response_times = []
        self.throughput_samples = []
        self.error_count = 0
        self.success_count = 0
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def record_system_metrics(self):
        """Record system resource usage"""
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            self.gpu_usage.append(gpu_memory * 100)
    
    def stop(self):
        """Stop metrics collection"""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        duration = self.end_time - self.start_time if self.end_time else 0
        
        return {
            "duration_seconds": duration,
            "total_requests": len(self.response_times),
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "error_rate": self.error_count / len(self.response_times) if self.response_times else 0,
            "response_times": {
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "mean": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
                "p95": np.percentile(self.response_times, 95) if self.response_times else 0,
                "p99": np.percentile(self.response_times, 99) if self.response_times else 0
            },
            "throughput": {
                "requests_per_second": len(self.response_times) / duration if duration > 0 else 0
            },
            "resource_usage": {
                "cpu": {
                    "mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                    "max": max(self.cpu_usage) if self.cpu_usage else 0
                },
                "memory": {
                    "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0,
                    "max": max(self.memory_usage) if self.memory_usage else 0
                },
                "gpu": {
                    "mean": statistics.mean(self.gpu_usage) if self.gpu_usage else 0,
                    "max": max(self.gpu_usage) if self.gpu_usage else 0
                } if self.gpu_usage else None
            }
        }


# Test scenarios
async def create_test_request(request_type: str = "mixed") -> Tuple[str, ConversationContext]:
    """Create test request based on type"""
    requests = {
        "simple": "What's my portfolio value?",
        "analysis": "Analyze my portfolio performance over the last month",
        "optimization": "Optimize my portfolio for maximum Sharpe ratio",
        "risk": "Run stress tests on my portfolio",
        "complex": "Show my performance, analyze risks, and suggest optimizations"
    }
    
    if request_type == "mixed":
        request_type = np.random.choice(list(requests.keys()))
    
    user_input = requests[request_type]
    
    context = ConversationContext(
        session_id=f"perf-test-{time.time_ns()}",
        user_id=f"perf-user-{np.random.randint(1000)}",
        conversation_history=[],
        user_profile={
            "portfolio_value": np.random.uniform(50000, 500000),
            "risk_tolerance": np.random.choice(["conservative", "moderate", "aggressive"])
        }
    )
    
    return user_input, context


# Performance test classes
class TestPerformanceLoad:
    """Load performance tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.parametrize("concurrent_users", PERFORMANCE_CONFIG["load_test"]["concurrent_users"])
    async def test_concurrent_user_load(self, orchestrator, concurrent_users):
        """Test system performance with varying concurrent user loads"""
        metrics = PerformanceMetrics()
        requests_per_user = PERFORMANCE_CONFIG["load_test"]["requests_per_user"]
        
        async def simulate_user():
            """Simulate a single user session"""
            for _ in range(requests_per_user):
                user_input, context = await create_test_request("mixed")
                
                start_time = time.time()
                try:
                    response = await orchestrator.process_user_input(user_input, context)
                    success = response is not None and response.error is None
                except Exception:
                    success = False
                
                response_time = time.time() - start_time
                metrics.record_request(response_time, success)
                
                # Think time
                await asyncio.sleep(PERFORMANCE_CONFIG["load_test"]["think_time_seconds"])
        
        # Start metrics collection
        metrics.start()
        
        # Create monitoring task
        async def monitor_resources():
            while metrics.end_time is None:
                metrics.record_system_metrics()
                await asyncio.sleep(1)
        
        monitor_task = asyncio.create_task(monitor_resources())
        
        # Simulate concurrent users
        user_tasks = [simulate_user() for _ in range(concurrent_users)]
        await asyncio.gather(*user_tasks)
        
        # Stop monitoring
        metrics.stop()
        monitor_task.cancel()
        
        # Analyze results
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["error_rate"] < 0.05  # Less than 5% error rate
        assert summary["response_times"]["p95"] < 2.0  # 95th percentile under 2 seconds
        assert summary["throughput"]["requests_per_second"] > concurrent_users  # At least 1 req/sec per user
        
        # Log performance results
        print(f"\nLoad Test Results - {concurrent_users} concurrent users:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Error Rate: {summary['error_rate']:.2%}")
        print(f"  Response Times (p95): {summary['response_times']['p95']:.3f}s")
        print(f"  Throughput: {summary['throughput']['requests_per_second']:.2f} req/s")
        print(f"  CPU Usage (max): {summary['resource_usage']['cpu']['max']:.1f}%")
        print(f"  Memory Usage (max): {summary['resource_usage']['memory']['max']:.1f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_stress_test(self, orchestrator):
        """Stress test with maximum concurrent users"""
        metrics = PerformanceMetrics()
        max_users = PERFORMANCE_CONFIG["stress_test"]["max_concurrent_users"]
        ramp_up_seconds = PERFORMANCE_CONFIG["stress_test"]["ramp_up_seconds"]
        sustained_duration = PERFORMANCE_CONFIG["stress_test"]["sustained_duration_seconds"]
        
        active_users = []
        stop_flag = False
        
        async def simulate_user(user_id: int):
            """Simulate a single user under stress"""
            while not stop_flag:
                user_input, context = await create_test_request("mixed")
                
                start_time = time.time()
                try:
                    response = await orchestrator.process_user_input(user_input, context)
                    success = response is not None and response.error is None
                except Exception:
                    success = False
                
                response_time = time.time() - start_time
                metrics.record_request(response_time, success)
                
                await asyncio.sleep(0.1)  # Minimal think time for stress test
        
        # Start metrics collection
        metrics.start()
        
        # Monitor resources
        monitor_task = asyncio.create_task(self._monitor_resources(metrics))
        
        # Ramp up users
        ramp_up_interval = ramp_up_seconds / max_users
        for i in range(max_users):
            user_task = asyncio.create_task(simulate_user(i))
            active_users.append(user_task)
            await asyncio.sleep(ramp_up_interval)
        
        # Sustain load
        await asyncio.sleep(sustained_duration)
        
        # Stop users
        stop_flag = True
        await asyncio.gather(*active_users, return_exceptions=True)
        
        # Stop monitoring
        metrics.stop()
        monitor_task.cancel()
        
        # Analyze results
        summary = metrics.get_summary()
        
        # Stress test assertions
        assert summary["error_rate"] < 0.10  # Less than 10% error rate under stress
        assert summary["response_times"]["p99"] < 5.0  # 99th percentile under 5 seconds
        
        print(f"\nStress Test Results - {max_users} max concurrent users:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Error Rate: {summary['error_rate']:.2%}")
        print(f"  Response Times (p99): {summary['response_times']['p99']:.3f}s")
        print(f"  Throughput: {summary['throughput']['requests_per_second']:.2f} req/s")
        print(f"  CPU Usage (max): {summary['resource_usage']['cpu']['max']:.1f}%")
        print(f"  Memory Usage (max): {summary['resource_usage']['memory']['max']:.1f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_endurance(self, orchestrator):
        """Endurance test over extended period"""
        metrics = PerformanceMetrics()
        concurrent_users = PERFORMANCE_CONFIG["endurance_test"]["concurrent_users"]
        duration_minutes = PERFORMANCE_CONFIG["endurance_test"]["duration_minutes"]
        request_interval = PERFORMANCE_CONFIG["endurance_test"]["request_interval_seconds"]
        
        stop_flag = False
        
        async def simulate_user(user_id: int):
            """Simulate a user over extended period"""
            while not stop_flag:
                user_input, context = await create_test_request("mixed")
                
                start_time = time.time()
                try:
                    response = await orchestrator.process_user_input(user_input, context)
                    success = response is not None and response.error is None
                except Exception:
                    success = False
                
                response_time = time.time() - start_time
                metrics.record_request(response_time, success)
                
                await asyncio.sleep(request_interval)
        
        # Start metrics collection
        metrics.start()
        
        # Monitor resources and performance degradation
        monitor_task = asyncio.create_task(self._monitor_resources(metrics))
        degradation_task = asyncio.create_task(self._monitor_degradation(metrics))
        
        # Run endurance test
        user_tasks = [asyncio.create_task(simulate_user(i)) for i in range(concurrent_users)]
        
        # Run for specified duration
        await asyncio.sleep(duration_minutes * 60)
        
        # Stop test
        stop_flag = True
        await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # Stop monitoring
        metrics.stop()
        monitor_task.cancel()
        degradation_task.cancel()
        
        # Analyze results
        summary = metrics.get_summary()
        
        # Check for performance degradation
        early_response_times = metrics.response_times[:100]
        late_response_times = metrics.response_times[-100:]
        
        early_mean = statistics.mean(early_response_times) if early_response_times else 0
        late_mean = statistics.mean(late_response_times) if late_response_times else 0
        degradation_rate = (late_mean - early_mean) / early_mean if early_mean > 0 else 0
        
        # Endurance test assertions
        assert summary["error_rate"] < 0.02  # Less than 2% error rate
        assert degradation_rate < 0.20  # Less than 20% performance degradation
        
        print(f"\nEndurance Test Results - {duration_minutes} minutes:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Error Rate: {summary['error_rate']:.2%}")
        print(f"  Response Times (mean): {summary['response_times']['mean']:.3f}s")
        print(f"  Performance Degradation: {degradation_rate:.2%}")
        print(f"  CPU Usage (mean): {summary['resource_usage']['cpu']['mean']:.1f}%")
        print(f"  Memory Usage (mean): {summary['resource_usage']['memory']['mean']:.1f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_spike_load(self, orchestrator):
        """Test system response to sudden load spikes"""
        metrics = PerformanceMetrics()
        base_users = PERFORMANCE_CONFIG["spike_test"]["base_users"]
        spike_users = PERFORMANCE_CONFIG["spike_test"]["spike_users"]
        spike_duration = PERFORMANCE_CONFIG["spike_test"]["spike_duration_seconds"]
        
        stop_flag = False
        
        async def simulate_user(user_id: int, is_spike: bool = False):
            """Simulate a user (base or spike)"""
            while not stop_flag:
                user_input, context = await create_test_request("mixed")
                
                start_time = time.time()
                try:
                    response = await orchestrator.process_user_input(user_input, context)
                    success = response is not None and response.error is None
                except Exception:
                    success = False
                
                response_time = time.time() - start_time
                metrics.record_request(response_time, success)
                
                # Different think times for base and spike users
                think_time = 0.1 if is_spike else 1.0
                await asyncio.sleep(think_time)
        
        # Start metrics collection
        metrics.start()
        
        # Monitor resources
        monitor_task = asyncio.create_task(self._monitor_resources(metrics))
        
        # Start base load
        base_tasks = [asyncio.create_task(simulate_user(i, False)) for i in range(base_users)]
        
        # Run base load for 30 seconds
        await asyncio.sleep(30)
        
        # Create spike
        spike_tasks = [asyncio.create_task(simulate_user(i + base_users, True)) for i in range(spike_users)]
        
        # Sustain spike
        await asyncio.sleep(spike_duration)
        
        # Remove spike (cancel spike tasks)
        for task in spike_tasks:
            task.cancel()
        
        # Continue base load for recovery observation
        await asyncio.sleep(30)
        
        # Stop test
        stop_flag = True
        await asyncio.gather(*base_tasks, return_exceptions=True)
        
        # Stop monitoring
        metrics.stop()
        monitor_task.cancel()
        
        # Analyze spike impact
        summary = metrics.get_summary()
        
        # Identify spike period in metrics
        spike_start_idx = int(len(metrics.response_times) * 0.3)
        spike_end_idx = int(len(metrics.response_times) * 0.6)
        
        pre_spike_times = metrics.response_times[:spike_start_idx]
        spike_times = metrics.response_times[spike_start_idx:spike_end_idx]
        post_spike_times = metrics.response_times[spike_end_idx:]
        
        pre_spike_mean = statistics.mean(pre_spike_times) if pre_spike_times else 0
        spike_mean = statistics.mean(spike_times) if spike_times else 0
        post_spike_mean = statistics.mean(post_spike_times) if post_spike_times else 0
        
        spike_impact = (spike_mean - pre_spike_mean) / pre_spike_mean if pre_spike_mean > 0 else 0
        recovery_rate = (spike_mean - post_spike_mean) / (spike_mean - pre_spike_mean) if spike_mean > pre_spike_mean else 1
        
        # Spike test assertions
        assert spike_impact < 3.0  # Response time increase less than 300%
        assert recovery_rate > 0.8  # System recovers to at least 80% of pre-spike performance
        
        print(f"\nSpike Test Results - {spike_users} spike users:")
        print(f"  Pre-spike Response Time: {pre_spike_mean:.3f}s")
        print(f"  During-spike Response Time: {spike_mean:.3f}s")
        print(f"  Post-spike Response Time: {post_spike_mean:.3f}s")
        print(f"  Spike Impact: {spike_impact:.2%}")
        print(f"  Recovery Rate: {recovery_rate:.2%}")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.parametrize("data_size", PERFORMANCE_CONFIG["volume_test"]["data_sizes"])
    async def test_data_volume_scaling(self, orchestrator, data_size):
        """Test performance with varying data volumes"""
        metrics = PerformanceMetrics()
        concurrent_requests = PERFORMANCE_CONFIG["volume_test"]["concurrent_requests"]
        
        async def process_large_portfolio():
            """Process request with large portfolio data"""
            # Create context with large portfolio
            assets = [f"ASSET{i}" for i in range(data_size)]
            portfolio = {asset: {"quantity": np.random.randint(1, 100), 
                               "avg_price": np.random.uniform(10, 1000)}
                        for asset in assets}
            
            context = ConversationContext(
                session_id=f"volume-test-{time.time_ns()}",
                user_id="volume-test-user",
                conversation_history=[],
                user_profile={
                    "portfolio": portfolio,
                    "portfolio_value": data_size * 50000
                }
            )
            
            user_input = "Optimize my entire portfolio for risk-adjusted returns"
            
            start_time = time.time()
            try:
                response = await orchestrator.process_user_input(user_input, context)
                success = response is not None and response.error is None
            except Exception:
                success = False
            
            response_time = time.time() - start_time
            metrics.record_request(response_time, success)
        
        # Start metrics collection
        metrics.start()
        
        # Process concurrent requests
        tasks = [process_large_portfolio() for _ in range(concurrent_requests)]
        await asyncio.gather(*tasks)
        
        # Stop metrics
        metrics.stop()
        
        # Analyze results
        summary = metrics.get_summary()
        
        # Volume test assertions (scaled by data size)
        max_acceptable_time = 0.5 + (data_size / 1000)  # Scale with data size
        assert summary["response_times"]["mean"] < max_acceptable_time
        assert summary["error_rate"] < 0.05
        
        print(f"\nVolume Test Results - {data_size} assets:")
        print(f"  Response Time (mean): {summary['response_times']['mean']:.3f}s")
        print(f"  Response Time (p95): {summary['response_times']['p95']:.3f}s")
        print(f"  Error Rate: {summary['error_rate']:.2%}")
        print(f"  Memory Usage (max): {summary['resource_usage']['memory']['max']:.1f}%")
    
    async def _monitor_resources(self, metrics: PerformanceMetrics):
        """Monitor system resources during test"""
        while metrics.end_time is None:
            metrics.record_system_metrics()
            await asyncio.sleep(1)
    
    async def _monitor_degradation(self, metrics: PerformanceMetrics):
        """Monitor performance degradation over time"""
        window_size = 100
        degradation_threshold = 0.5  # 50% degradation
        
        while metrics.end_time is None:
            if len(metrics.response_times) >= window_size * 2:
                early_window = metrics.response_times[:window_size]
                recent_window = metrics.response_times[-window_size:]
                
                early_mean = statistics.mean(early_window)
                recent_mean = statistics.mean(recent_window)
                
                if early_mean > 0:
                    degradation = (recent_mean - early_mean) / early_mean
                    if degradation > degradation_threshold:
                        logging.warning(f"Performance degradation detected: {degradation:.2%}")
            
            await asyncio.sleep(30)  # Check every 30 seconds


class TestComponentPerformance:
    """Test individual component performance"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_conversation_engine_performance(self, orchestrator):
        """Test conversation engine response times"""
        metrics = PerformanceMetrics()
        test_inputs = [
            "What's my portfolio value?",
            "How did AAPL perform today?",
            "Show me risk analysis",
            "Optimize my portfolio",
            "What are the market trends?"
        ]
        
        metrics.start()
        
        for _ in range(100):
            user_input = np.random.choice(test_inputs)
            context = ConversationContext(
                session_id=f"perf-{time.time_ns()}",
                user_id="perf-user",
                conversation_history=[]
            )
            
            start_time = time.time()
            try:
                response = await orchestrator.conversation_engine.process_input(
                    user_input,
                    context
                )
                success = response is not None
            except Exception:
                success = False
            
            response_time = time.time() - start_time
            metrics.record_request(response_time, success)
        
        metrics.stop()
        summary = metrics.get_summary()
        
        # Conversation engine should be fast
        assert summary["response_times"]["p95"] < 0.1  # 95th percentile under 100ms
        assert summary["error_rate"] < 0.01
        
        print(f"\nConversation Engine Performance:")
        print(f"  Response Time (p95): {summary['response_times']['p95']*1000:.1f}ms")
        print(f"  Throughput: {summary['throughput']['requests_per_second']:.1f} req/s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.gpu
    async def test_gpu_vs_cpu_performance(self, orchestrator):
        """Compare GPU vs CPU performance for financial calculations"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Test portfolio optimization
        portfolio_sizes = [10, 50, 100, 500]
        
        for size in portfolio_sizes:
            # Create test data
            assets = [f"ASSET{i}" for i in range(size)]
            returns = np.random.randn(252, size) * 0.01  # Daily returns
            
            # CPU performance
            cpu_start = time.time()
            _ = await orchestrator.portfolio_optimizer.optimize_portfolio(
                assets=assets,
                historical_returns=returns,
                optimization_method="mean_variance",
                use_gpu=False
            )
            cpu_time = time.time() - cpu_start
            
            # GPU performance
            gpu_start = time.time()
            _ = await orchestrator.portfolio_optimizer.optimize_portfolio(
                assets=assets,
                historical_returns=returns,
                optimization_method="mean_variance",
                use_gpu=True
            )
            gpu_time = time.time() - gpu_start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            print(f"\nGPU vs CPU Performance - {size} assets:")
            print(f"  CPU Time: {cpu_time:.3f}s")
            print(f"  GPU Time: {gpu_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            
            # GPU should provide speedup for larger portfolios
            if size >= 100:
                assert speedup > 1.5  # At least 1.5x speedup
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_caching_performance(self, orchestrator):
        """Test caching effectiveness"""
        metrics_nocache = PerformanceMetrics()
        metrics_cache = PerformanceMetrics()
        
        # Test without cache
        orchestrator.config.enable_caching = False
        
        user_input = "Analyze my portfolio performance"
        context = ConversationContext(
            session_id="cache-test",
            user_id="cache-user",
            conversation_history=[]
        )
        
        # First run - no cache
        metrics_nocache.start()
        for _ in range(10):
            start_time = time.time()
            response = await orchestrator.process_user_input(user_input, context)
            response_time = time.time() - start_time
            metrics_nocache.record_request(response_time, response is not None)
        metrics_nocache.stop()
        
        # Enable cache
        orchestrator.config.enable_caching = True
        await orchestrator._initialize_cache()
        
        # Second run - with cache
        metrics_cache.start()
        for _ in range(10):
            start_time = time.time()
            response = await orchestrator.process_user_input(user_input, context)
            response_time = time.time() - start_time
            metrics_cache.record_request(response_time, response is not None)
        metrics_cache.stop()
        
        # Compare results
        nocache_mean = statistics.mean(metrics_nocache.response_times)
        cache_mean = statistics.mean(metrics_cache.response_times)
        cache_improvement = (nocache_mean - cache_mean) / nocache_mean
        
        print(f"\nCaching Performance:")
        print(f"  No Cache (mean): {nocache_mean:.3f}s")
        print(f"  With Cache (mean): {cache_mean:.3f}s")
        print(f"  Improvement: {cache_improvement:.2%}")
        
        # Cache should provide improvement
        assert cache_improvement > 0.30  # At least 30% improvement


class TestMemoryLeaks:
    """Test for memory leaks and resource management"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_leak_detection(self, orchestrator):
        """Detect memory leaks during extended operation"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        
        # Run many requests
        for i in range(500):
            user_input, context = await create_test_request("mixed")
            
            try:
                response = await orchestrator.process_user_input(user_input, context)
                
                # Simulate session cleanup
                if i % 50 == 0:
                    await orchestrator.cleanup_old_sessions()
            except Exception:
                pass
            
            # Record memory usage
            if i % 10 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Force garbage collection
                if i % 100 == 0:
                    gc.collect()
        
        # Analyze memory growth
        memory_growth = memory_samples[-1] - memory_samples[0]
        growth_rate = memory_growth / initial_memory
        
        print(f"\nMemory Leak Test:")
        print(f"  Initial Memory: {initial_memory:.1f} MB")
        print(f"  Final Memory: {memory_samples[-1]:.1f} MB")
        print(f"  Memory Growth: {memory_growth:.1f} MB ({growth_rate:.2%})")
        
        # Memory growth should be minimal
        assert growth_rate < 0.20  # Less than 20% growth


class TestFailureRecovery:
    """Test system recovery from failures"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_recovery_time(self, orchestrator):
        """Test recovery time after component failure"""
        metrics_before = PerformanceMetrics()
        metrics_during = PerformanceMetrics()
        metrics_after = PerformanceMetrics()
        
        async def send_requests(metrics: PerformanceMetrics, duration: int):
            """Send requests for specified duration"""
            end_time = time.time() + duration
            while time.time() < end_time:
                user_input, context = await create_test_request("simple")
                
                start_time = time.time()
                try:
                    response = await orchestrator.process_user_input(user_input, context)
                    success = response is not None and response.error is None
                except Exception:
                    success = False
                
                response_time = time.time() - start_time
                metrics.record_request(response_time, success)
                await asyncio.sleep(0.1)
        
        # Normal operation
        metrics_before.start()
        await send_requests(metrics_before, 10)
        metrics_before.stop()
        
        # Simulate component failure
        original_analyzer = orchestrator.financial_analyzer
        orchestrator.financial_analyzer = None
        
        # Operation during failure
        metrics_during.start()
        await send_requests(metrics_during, 5)
        metrics_during.stop()
        
        # Restore component
        orchestrator.financial_analyzer = original_analyzer
        recovery_start = time.time()
        
        # Operation after recovery
        metrics_after.start()
        await send_requests(metrics_after, 10)
        metrics_after.stop()
        
        recovery_time = time.time() - recovery_start
        
        # Analyze recovery
        before_summary = metrics_before.get_summary()
        during_summary = metrics_during.get_summary()
        after_summary = metrics_after.get_summary()
        
        recovery_rate = (after_summary["error_rate"] - before_summary["error_rate"]) / before_summary["error_rate"] if before_summary["error_rate"] > 0 else 0
        
        print(f"\nFailure Recovery Test:")
        print(f"  Error Rate (before): {before_summary['error_rate']:.2%}")
        print(f"  Error Rate (during): {during_summary['error_rate']:.2%}")
        print(f"  Error Rate (after): {after_summary['error_rate']:.2%}")
        print(f"  Recovery Time: {recovery_time:.1f}s")
        
        # System should recover to normal operation
        assert after_summary["error_rate"] < 0.05
        assert recovery_time < 30  # Recovery within 30 seconds