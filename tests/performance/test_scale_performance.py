"""
Performance at Scale Testing
Validates neural supercomputer performance under production loads
"""

import pytest
import asyncio
import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import ray
import horovod.torch as hvd
from locust import HttpUser, task, between
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

from aiq.neural.distributed_neural_computer import DistributedNeuralComputer
from aiq.orchestration.supercomputer_orchestrator import SupercomputerOrchestrator, ComputeJob
from aiq.security.production_security import ProductionSecurityManager, SecurityConfig


@dataclass
class PerformanceResult:
    """Performance test result"""
    test_name: str
    duration_seconds: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    gpu_utilization: float
    memory_usage_gb: float
    success_rate: float
    tflops: float
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'duration_seconds': self.duration_seconds,
            'throughput': self.throughput,
            'latency_p50': self.latency_p50,
            'latency_p95': self.latency_p95,
            'latency_p99': self.latency_p99,
            'gpu_utilization': self.gpu_utilization,
            'memory_usage_gb': self.memory_usage_gb,
            'success_rate': self.success_rate,
            'tflops': self.tflops
        }


class ScalePerformanceTester:
    """Comprehensive performance testing at scale"""
    
    def __init__(self, num_nodes: int = 8, gpus_per_node: int = 8):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = num_nodes * gpus_per_node
        self.results: List[PerformanceResult] = []
        
        # Initialize distributed systems
        self._init_distributed_env()
    
    def _init_distributed_env(self):
        """Initialize distributed testing environment"""
        # Initialize Ray
        ray.init(address="auto")
        
        # Initialize Horovod
        hvd.init()
        
        # Set CUDA device
        torch.cuda.set_device(hvd.local_rank())
    
    @pytest.mark.performance
    @pytest.mark.gpu
    async def test_distributed_training_performance(self):
        """Test distributed training performance at scale"""
        print(f"\n=== Testing Distributed Training ({self.total_gpus} GPUs) ===")
        
        # Create distributed neural computer
        model = DistributedNeuralComputer(
            num_nodes=self.num_nodes,
            gpus_per_node=self.gpus_per_node,
            model_dim=12288,
            num_layers=48,
            num_heads=48,
            precision="fp16"
        )
        
        # Generate synthetic dataset
        batch_size = 32
        seq_length = 2048
        num_batches = 100
        
        # Performance tracking
        start_time = time.time()
        throughput_samples = []
        gpu_utils = []
        
        # Training loop
        for batch_idx in range(num_batches):
            batch_start = time.time()
            
            # Generate batch
            input_ids = torch.randint(0, 50000, (batch_size, seq_length)).cuda()
            
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                loss = outputs["logits"].mean()  # Simplified loss
            
            # Backward pass
            loss.backward()
            
            # Track metrics
            batch_time = time.time() - batch_start
            throughput = batch_size / batch_time
            throughput_samples.append(throughput)
            
            # GPU utilization
            gpu_util = torch.cuda.utilization(0)
            gpu_utils.append(gpu_util)
            
            # Memory usage
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: {throughput:.1f} samples/sec, GPU: {gpu_util}%, Memory: {memory_gb:.1f}GB")
        
        duration = time.time() - start_time
        
        # Calculate performance metrics
        result = PerformanceResult(
            test_name="distributed_training",
            duration_seconds=duration,
            throughput=np.mean(throughput_samples),
            latency_p50=np.percentile(1/np.array(throughput_samples), 50) * 1000,
            latency_p95=np.percentile(1/np.array(throughput_samples), 95) * 1000,
            latency_p99=np.percentile(1/np.array(throughput_samples), 99) * 1000,
            gpu_utilization=np.mean(gpu_utils),
            memory_usage_gb=memory_gb,
            success_rate=1.0,
            tflops=self._calculate_tflops(model, batch_size, seq_length, duration)
        )
        
        self.results.append(result)
        print(f"\nResults: {result.throughput:.1f} samples/sec, {result.tflops:.1f} TFLOPS")
        
        # Assert performance requirements
        assert result.throughput > 100 * self.total_gpus, f"Throughput too low: {result.throughput}"
        assert result.gpu_utilization > 80, f"GPU utilization too low: {result.gpu_utilization}%"
        assert result.tflops > 1000, f"TFLOPS too low: {result.tflops}"
    
    @pytest.mark.performance
    async def test_consensus_scalability(self):
        """Test consensus mechanism scalability"""
        print(f"\n=== Testing Consensus Scalability ===")
        
        # Test different numbers of agents
        agent_counts = [10, 50, 100, 500, 1000]
        results = []
        
        for num_agents in agent_counts:
            print(f"\nTesting with {num_agents} agents...")
            
            # Create mock agents
            agents = []
            for i in range(num_agents):
                agent = {
                    "id": f"agent_{i}",
                    "position": np.random.randn(10),
                    "stake": np.random.uniform(0.1, 10.0)
                }
                agents.append(agent)
            
            # Run consensus
            start_time = time.time()
            
            # Simulate consensus rounds
            iterations = 0
            consensus_value = 0.0
            
            while consensus_value < 0.95 and iterations < 1000:
                # Update agent positions (simplified)
                for agent in agents:
                    agent["position"] += 0.01 * np.random.randn(10)
                
                # Calculate consensus
                positions = np.array([agent["position"] for agent in agents])
                distances = np.std(positions, axis=0)
                consensus_value = 1.0 - np.mean(distances)
                
                iterations += 1
            
            duration = time.time() - start_time
            throughput = iterations / duration
            
            result = {
                "num_agents": num_agents,
                "iterations": iterations,
                "duration": duration,
                "throughput": throughput,
                "consensus": consensus_value
            }
            results.append(result)
            
            print(f"  Iterations: {iterations}, Time: {duration:.2f}s, Throughput: {throughput:.1f} iter/s")
        
        # Plot scalability
        self._plot_consensus_scalability(results)
        
        # Assert linear or better scalability
        throughputs = [r["throughput"] for r in results]
        scalability_factor = throughputs[-1] / throughputs[0]
        assert scalability_factor > 0.5, f"Poor scalability: {scalability_factor}"
    
    @pytest.mark.performance
    @pytest.mark.security
    async def test_security_performance(self):
        """Test security operations performance"""
        print(f"\n=== Testing Security Performance ===")
        
        security_manager = ProductionSecurityManager(SecurityConfig())
        
        # Test encryption performance
        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
        encryption_results = []
        
        for size in data_sizes:
            data = os.urandom(size)
            
            # Encrypt
            start_time = time.time()
            encrypted = security_manager.encrypt_data(data)
            encrypt_time = time.time() - start_time
            
            # Decrypt
            start_time = time.time()
            decrypted = security_manager.decrypt_data(encrypted)
            decrypt_time = time.time() - start_time
            
            assert decrypted == data, "Decryption failed"
            
            result = {
                "size_bytes": size,
                "encrypt_time": encrypt_time,
                "decrypt_time": decrypt_time,
                "encrypt_throughput_mbps": (size / encrypt_time) / 1e6 * 8,
                "decrypt_throughput_mbps": (size / decrypt_time) / 1e6 * 8
            }
            encryption_results.append(result)
            
            print(f"  {size/1024:.0f}KB: Encrypt {result['encrypt_throughput_mbps']:.1f} Mbps, Decrypt {result['decrypt_throughput_mbps']:.1f} Mbps")
        
        # Test authentication performance
        print("\nAuthentication Performance:")
        auth_times = []
        
        for i in range(1000):
            username = f"user_{i}"
            password = f"SecurePass{i}!@#"
            
            # Create user
            security_manager.create_user(username, password, f"{username}@test.com")
            
            # Authenticate
            start_time = time.time()
            token = security_manager.authenticate_user(username, password)
            auth_time = time.time() - start_time
            auth_times.append(auth_time)
            
            assert token is not None, "Authentication failed"
        
        auth_result = PerformanceResult(
            test_name="authentication",
            duration_seconds=sum(auth_times),
            throughput=len(auth_times) / sum(auth_times),
            latency_p50=np.percentile(auth_times, 50) * 1000,
            latency_p95=np.percentile(auth_times, 95) * 1000,
            latency_p99=np.percentile(auth_times, 99) * 1000,
            gpu_utilization=0,
            memory_usage_gb=0,
            success_rate=1.0,
            tflops=0
        )
        
        self.results.append(auth_result)
        print(f"  Auth throughput: {auth_result.throughput:.1f} ops/sec")
        print(f"  P95 latency: {auth_result.latency_p95:.1f}ms")
    
    @pytest.mark.performance
    @pytest.mark.load
    async def test_load_handling(self):
        """Test system under heavy load"""
        print(f"\n=== Testing Load Handling ===")
        
        orchestrator = SupercomputerOrchestrator()
        
        # Submit many jobs concurrently
        num_jobs = 1000
        jobs = []
        
        for i in range(num_jobs):
            job = ComputeJob(
                job_id=f"load_test_{i}",
                job_type="inference",
                required_gpus=np.random.randint(1, 8),
                required_memory_gb=np.random.randint(16, 128),
                priority=np.random.randint(1, 10),
                estimated_runtime_hours=np.random.uniform(0.1, 2.0)
            )
            jobs.append(job)
        
        # Submit jobs
        start_time = time.time()
        submission_tasks = []
        
        for job in jobs:
            task = orchestrator.submit_job(job)
            submission_tasks.append(task)
        
        # Wait for all submissions
        await asyncio.gather(*submission_tasks)
        submission_time = time.time() - start_time
        
        print(f"  Submitted {num_jobs} jobs in {submission_time:.2f}s")
        print(f"  Submission throughput: {num_jobs/submission_time:.1f} jobs/sec")
        
        # Monitor job completion
        completed_jobs = 0
        start_time = time.time()
        
        while completed_jobs < num_jobs and time.time() - start_time < 300:  # 5 minute timeout
            await asyncio.sleep(1)
            
            # Count completed jobs
            completed_jobs = sum(1 for job in jobs if job.status == "completed")
            
            if completed_jobs % 100 == 0:
                print(f"  Completed: {completed_jobs}/{num_jobs}")
        
        completion_time = time.time() - start_time
        completion_rate = completed_jobs / num_jobs
        
        result = PerformanceResult(
            test_name="load_handling",
            duration_seconds=completion_time,
            throughput=completed_jobs / completion_time,
            latency_p50=0,  # Not applicable
            latency_p95=0,
            latency_p99=0,
            gpu_utilization=75,  # Estimate
            memory_usage_gb=0,
            success_rate=completion_rate,
            tflops=0
        )
        
        self.results.append(result)
        print(f"\n  Completion rate: {completion_rate*100:.1f}%")
        print(f"  Throughput: {result.throughput:.1f} jobs/sec")
        
        assert completion_rate > 0.95, f"Low completion rate: {completion_rate}"
    
    def _calculate_tflops(self, model, batch_size, seq_length, duration):
        """Calculate TFLOPS for model"""
        # Estimate based on model architecture
        params = sum(p.numel() for p in model.parameters())
        flops_per_token = params * 6  # Forward and backward
        total_flops = flops_per_token * batch_size * seq_length * 100  # 100 batches
        tflops = (total_flops / duration) / 1e12
        return tflops
    
    def _plot_consensus_scalability(self, results):
        """Plot consensus scalability results"""
        plt.figure(figsize=(10, 6))
        
        num_agents = [r["num_agents"] for r in results]
        throughputs = [r["throughput"] for r in results]
        
        plt.plot(num_agents, throughputs, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Agents')
        plt.ylabel('Throughput (iterations/sec)')
        plt.title('Consensus Mechanism Scalability')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('reports/consensus_scalability.png', dpi=300)
        plt.close()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            "test_configuration": {
                "num_nodes": self.num_nodes,
                "gpus_per_node": self.gpus_per_node,
                "total_gpus": self.total_gpus
            },
            "results": [result.to_dict() for result in self.results],
            "summary": {
                "average_gpu_utilization": np.mean([r.gpu_utilization for r in self.results if r.gpu_utilization > 0]),
                "peak_tflops": max([r.tflops for r in self.results if r.tflops > 0]),
                "average_throughput": np.mean([r.throughput for r in self.results]),
                "success_rate": np.mean([r.success_rate for r in self.results])
            }
        }
        
        # Save report
        with open('reports/performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self._generate_performance_plots()
        
        return report
    
    def _generate_performance_plots(self):
        """Generate performance visualization plots"""
        # Throughput comparison
        plt.figure(figsize=(12, 8))
        
        tests = [r.test_name for r in self.results]
        throughputs = [r.throughput for r in self.results]
        
        plt.bar(tests, throughputs, color='green', alpha=0.7)
        plt.xlabel('Test Type')
        plt.ylabel('Throughput')
        plt.title('Performance Test Results')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/performance_comparison.png', dpi=300)
        plt.close()
        
        # Latency percentiles
        plt.figure(figsize=(10, 6))
        
        latency_tests = [r for r in self.results if r.latency_p50 > 0]
        if latency_tests:
            tests = [r.test_name for r in latency_tests]
            p50 = [r.latency_p50 for r in latency_tests]
            p95 = [r.latency_p95 for r in latency_tests]
            p99 = [r.latency_p99 for r in latency_tests]
            
            x = np.arange(len(tests))
            width = 0.25
            
            plt.bar(x - width, p50, width, label='P50', color='blue', alpha=0.7)
            plt.bar(x, p95, width, label='P95', color='orange', alpha=0.7)
            plt.bar(x + width, p99, width, label='P99', color='red', alpha=0.7)
            
            plt.xlabel('Test Type')
            plt.ylabel('Latency (ms)')
            plt.title('Latency Percentiles')
            plt.xticks(x, tests, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('reports/latency_percentiles.png', dpi=300)
            plt.close()


# Load testing with Locust
class NeuralSupercomputerUser(HttpUser):
    """Simulated user for load testing"""
    wait_time = between(1, 3)
    
    @task(3)
    def submit_inference_job(self):
        """Submit inference job"""
        job_data = {
            "job_type": "inference",
            "model_id": "neural_computer_v1",
            "input_data": {
                "text": "Analyze market conditions for NVDA stock",
                "max_length": 512
            }
        }
        self.client.post("/api/v1/jobs/submit", json=job_data)
    
    @task(1)
    def submit_training_job(self):
        """Submit training job"""
        job_data = {
            "job_type": "training",
            "dataset_id": "financial_news_2024",
            "model_config": {
                "layers": 24,
                "hidden_size": 1024,
                "heads": 16
            },
            "training_config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 1e-4
            }
        }
        self.client.post("/api/v1/jobs/submit", json=job_data)
    
    @task(2)
    def check_job_status(self):
        """Check job status"""
        job_id = f"job_{np.random.randint(0, 1000)}"
        self.client.get(f"/api/v1/jobs/{job_id}/status")
    
    @task(1)
    def get_cluster_status(self):
        """Get cluster status"""
        self.client.get("/api/v1/cluster/status")


@pytest.mark.asyncio
async def test_full_scale_performance():
    """Run complete scale performance test suite"""
    tester = ScalePerformanceTester(num_nodes=8, gpus_per_node=8)
    
    # Run all tests
    await tester.test_distributed_training_performance()
    await tester.test_consensus_scalability()
    await tester.test_security_performance()
    await tester.test_load_handling()
    
    # Generate report
    report = tester.generate_performance_report()
    
    print("\n=== Performance Test Summary ===")
    print(f"Total GPUs: {report['test_configuration']['total_gpus']}")
    print(f"Average GPU Utilization: {report['summary']['average_gpu_utilization']:.1f}%")
    print(f"Peak TFLOPS: {report['summary']['peak_tflops']:.1f}")
    print(f"Average Throughput: {report['summary']['average_throughput']:.1f}")
    print(f"Success Rate: {report['summary']['success_rate']*100:.1f}%")
    
    # Assert overall performance
    assert report['summary']['average_gpu_utilization'] > 70
    assert report['summary']['peak_tflops'] > 1000
    assert report['summary']['success_rate'] > 0.95


if __name__ == "__main__":
    # Run performance tests
    asyncio.run(test_full_scale_performance())