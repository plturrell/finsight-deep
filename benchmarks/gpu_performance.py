"""
GPU Performance Benchmarks for AIQToolkit
Demonstrates NVIDIA GPU acceleration benefits
"""

import torch
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

from aiq.cuda_kernels.cuda_similarity import CUDASimilarityCalculator
from aiq.neural.nash_ethereum_consensus import NashEthereumConsensus
from aiq.neural.advanced_architectures import FlashAttention, MixtureOfExperts
from aiq.digital_human.avatar.avatar_controller import AvatarController


@dataclass
class BenchmarkResult:
    task: str
    cpu_time: float
    gpu_time: float
    speedup: float
    gpu_memory_mb: float
    input_size: int


class GPUBenchmark:
    """Comprehensive GPU performance benchmarks"""
    
    def __init__(self, device: str = "cuda:0", verbose: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def benchmark_similarity_computation(self, sizes: List[int] = [1000, 5000, 10000]) -> List[BenchmarkResult]:
        """Benchmark CUDA similarity computation"""
        results = []
        
        for size in sizes:
            self.log(f"\nBenchmarking similarity computation (size={size})...")
            
            # Generate test data
            embeddings1 = np.random.randn(size, 768).astype(np.float32)
            embeddings2 = np.random.randn(size, 768).astype(np.float32)
            
            # CPU benchmark
            start_cpu = time.time()
            cpu_similarities = np.dot(embeddings1, embeddings2.T)
            cpu_time = time.time() - start_cpu
            
            # GPU benchmark
            cuda_calc = CUDASimilarityCalculator()
            torch.cuda.synchronize()
            start_gpu = time.time()
            gpu_similarities = cuda_calc.batch_cosine_similarity_cuda(
                torch.from_numpy(embeddings1).to(self.device),
                torch.from_numpy(embeddings2).to(self.device)
            )
            torch.cuda.synchronize()
            gpu_time = time.time() - start_gpu
            
            # Memory usage
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            result = BenchmarkResult(
                task=f"Similarity Matrix {size}x{size}",
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=cpu_time / gpu_time,
                gpu_memory_mb=gpu_memory,
                input_size=size
            )
            
            results.append(result)
            self.results.append(result)
            
            self.log(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {result.speedup:.1f}x")
        
        return results
    
    def benchmark_nash_equilibrium(self, agent_counts: List[int] = [10, 50, 100]) -> List[BenchmarkResult]:
        """Benchmark Nash equilibrium computation"""
        results = []
        
        for count in agent_counts:
            self.log(f"\nBenchmarking Nash equilibrium ({count} agents)...")
            
            # Create test scenario
            payoff_matrix = np.random.randn(count, count).astype(np.float32)
            
            # CPU benchmark
            start_cpu = time.time()
            # Simplified Nash computation for CPU
            cpu_strategies = np.ones(count) / count
            for _ in range(100):
                cpu_strategies = np.dot(payoff_matrix, cpu_strategies)
                cpu_strategies /= cpu_strategies.sum()
            cpu_time = time.time() - start_cpu
            
            # GPU benchmark
            gpu_matrix = torch.from_numpy(payoff_matrix).to(self.device)
            gpu_strategies = torch.ones(count, device=self.device) / count
            
            torch.cuda.synchronize()
            start_gpu = time.time()
            for _ in range(100):
                gpu_strategies = torch.matmul(gpu_matrix, gpu_strategies)
                gpu_strategies /= gpu_strategies.sum()
            torch.cuda.synchronize()
            gpu_time = time.time() - start_gpu
            
            # Memory usage
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            result = BenchmarkResult(
                task=f"Nash Equilibrium {count} agents",
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=cpu_time / gpu_time,
                gpu_memory_mb=gpu_memory,
                input_size=count
            )
            
            results.append(result)
            self.results.append(result)
            
            self.log(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {result.speedup:.1f}x")
        
        return results
    
    def benchmark_attention_mechanism(self, seq_lengths: List[int] = [512, 1024, 2048]) -> List[BenchmarkResult]:
        """Benchmark Flash Attention vs standard attention"""
        results = []
        
        for seq_len in seq_lengths:
            self.log(f"\nBenchmarking attention mechanism (seq_len={seq_len})...")
            
            # Create test data
            batch_size = 8
            num_heads = 12
            head_dim = 64
            
            q = torch.randn(batch_size, num_heads, seq_len, head_dim)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            
            # CPU benchmark (standard attention)
            q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
            start_cpu = time.time()
            scores = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_cpu)
            cpu_time = time.time() - start_cpu
            
            # GPU benchmark (Flash Attention)
            flash_attn = FlashAttention(dim=head_dim * num_heads, num_heads=num_heads).to(self.device)
            q_gpu, k_gpu, v_gpu = q.to(self.device), k.to(self.device), v.to(self.device)
            
            torch.cuda.synchronize()
            start_gpu = time.time()
            with torch.amp.autocast('cuda'):
                flash_output = flash_attn(
                    q_gpu.view(batch_size, seq_len, -1),
                    k_gpu.view(batch_size, seq_len, -1),
                    v_gpu.view(batch_size, seq_len, -1)
                )
            torch.cuda.synchronize()
            gpu_time = time.time() - start_gpu
            
            # Memory usage
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            result = BenchmarkResult(
                task=f"Attention seq_len={seq_len}",
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=cpu_time / gpu_time,
                gpu_memory_mb=gpu_memory,
                input_size=seq_len
            )
            
            results.append(result)
            self.results.append(result)
            
            self.log(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {result.speedup:.1f}x")
        
        return results
    
    def benchmark_mixture_of_experts(self, batch_sizes: List[int] = [32, 64, 128]) -> List[BenchmarkResult]:
        """Benchmark Mixture of Experts routing"""
        results = []
        
        for batch_size in batch_sizes:
            self.log(f"\nBenchmarking MoE routing (batch_size={batch_size})...")
            
            # Create test model
            input_dim = 768
            num_experts = 8
            expert_dim = 2048
            
            moe = MixtureOfExperts(
                input_dim=input_dim,
                num_experts=num_experts,
                expert_dim=expert_dim
            )
            
            # Test data
            x = torch.randn(batch_size, input_dim)
            
            # CPU benchmark
            moe_cpu = moe.cpu()
            x_cpu = x.cpu()
            start_cpu = time.time()
            with torch.no_grad():
                _ = moe_cpu(x_cpu)
            cpu_time = time.time() - start_cpu
            
            # GPU benchmark
            moe_gpu = moe.to(self.device)
            x_gpu = x.to(self.device)
            
            torch.cuda.synchronize()
            start_gpu = time.time()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    _ = moe_gpu(x_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_gpu
            
            # Memory usage
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            result = BenchmarkResult(
                task=f"MoE batch_size={batch_size}",
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                speedup=cpu_time / gpu_time,
                gpu_memory_mb=gpu_memory,
                input_size=batch_size
            )
            
            results.append(result)
            self.results.append(result)
            
            self.log(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {result.speedup:.1f}x")
        
        return results
    
    def benchmark_consensus_round(self) -> BenchmarkResult:
        """Benchmark full consensus round"""
        self.log("\nBenchmarking full consensus round...")
        
        # Setup consensus scenario
        num_agents = 100
        num_iterations = 50
        
        # CPU simulation
        start_cpu = time.time()
        agents_positions = np.random.randn(num_agents, 10)
        for _ in range(num_iterations):
            # Compute pairwise distances
            distances = np.linalg.norm(
                agents_positions[:, None] - agents_positions[None, :],
                axis=2
            )
            # Update positions (simplified)
            for i in range(num_agents):
                agents_positions[i] += 0.01 * np.mean(agents_positions - agents_positions[i], axis=0)
        cpu_time = time.time() - start_cpu
        
        # GPU simulation
        gpu_positions = torch.from_numpy(agents_positions).float().to(self.device)
        
        torch.cuda.synchronize()
        start_gpu = time.time()
        for _ in range(num_iterations):
            # Vectorized distance computation
            distances = torch.cdist(gpu_positions, gpu_positions)
            # Vectorized position update
            gpu_positions += 0.01 * (gpu_positions.mean(dim=0) - gpu_positions)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu
        
        # Memory usage
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        result = BenchmarkResult(
            task="Full Consensus Round",
            cpu_time=cpu_time,
            gpu_time=gpu_time,
            speedup=cpu_time / gpu_time,
            gpu_memory_mb=gpu_memory,
            input_size=num_agents
        )
        
        self.results.append(result)
        self.log(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {result.speedup:.1f}x")
        
        return result
    
    def generate_report(self, output_dir: str = "benchmarks/results"):
        """Generate comprehensive benchmark report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results dataframe
        df = pd.DataFrame([
            {
                'Task': r.task,
                'CPU Time (s)': r.cpu_time,
                'GPU Time (s)': r.gpu_time,
                'Speedup': r.speedup,
                'GPU Memory (MB)': r.gpu_memory_mb,
                'Input Size': r.input_size
            }
            for r in self.results
        ])
        
        # Save CSV
        df.to_csv(f"{output_dir}/gpu_benchmarks.csv", index=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AIQToolkit GPU Performance Benchmarks', fontsize=16)
        
        # Speedup by task
        ax1 = axes[0, 0]
        tasks = df['Task'].unique()
        speedups = [df[df['Task'] == task]['Speedup'].values[0] for task in tasks]
        bars = ax1.bar(range(len(tasks)), speedups, color='green')
        ax1.set_xticks(range(len(tasks)))
        ax1.set_xticklabels(tasks, rotation=45, ha='right')
        ax1.set_ylabel('Speedup (x)')
        ax1.set_title('GPU Speedup by Task')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add speedup values on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.1f}x',
                    ha='center', va='bottom')
        
        # Time comparison
        ax2 = axes[0, 1]
        x = np.arange(len(tasks))
        width = 0.35
        bars1 = ax2.bar(x - width/2, df.groupby('Task')['CPU Time (s)'].mean(), width, label='CPU', color='blue')
        bars2 = ax2.bar(x + width/2, df.groupby('Task')['GPU Time (s)'].mean(), width, label='GPU', color='orange')
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('CPU vs GPU Execution Time')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tasks, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Memory usage
        ax3 = axes[1, 0]
        ax3.bar(range(len(tasks)), 
                [df[df['Task'] == task]['GPU Memory (MB)'].values[0] for task in tasks],
                color='purple')
        ax3.set_xticks(range(len(tasks)))
        ax3.set_xticklabels(tasks, rotation=45, ha='right')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('GPU Memory Usage by Task')
        ax3.grid(axis='y', alpha=0.3)
        
        # Speedup vs Input Size (for similarity tasks)
        ax4 = axes[1, 1]
        similarity_tasks = df[df['Task'].str.contains('Similarity')]
        if not similarity_tasks.empty:
            ax4.scatter(similarity_tasks['Input Size'], similarity_tasks['Speedup'], s=100, color='red')
            ax4.set_xlabel('Input Size')
            ax4.set_ylabel('Speedup (x)')
            ax4.set_title('Speedup vs Input Size (Similarity Computation)')
            ax4.grid(True, alpha=0.3)
            
            # Fit trend line
            z = np.polyfit(similarity_tasks['Input Size'], similarity_tasks['Speedup'], 1)
            p = np.poly1d(z)
            ax4.plot(similarity_tasks['Input Size'], p(similarity_tasks['Input Size']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gpu_performance_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate markdown report
        report = f"""# GPU Performance Benchmark Report

## Summary

Total benchmarks run: {len(self.results)}
Average speedup: {df['Speedup'].mean():.1f}x
Maximum speedup: {df['Speedup'].max():.1f}x

## Results

"""
        report += df.to_markdown(index=False)
        
        report += f"""

## Key Findings

1. **Best GPU Acceleration**: {df.loc[df['Speedup'].idxmax(), 'Task']} with {df['Speedup'].max():.1f}x speedup
2. **Most Memory Efficient**: {df.loc[df['GPU Memory (MB)'].idxmin(), 'Task']} using only {df['GPU Memory (MB)'].min():.1f}MB
3. **Average GPU Time**: {df['GPU Time (s)'].mean():.3f} seconds
4. **Average CPU Time**: {df['CPU Time (s)'].mean():.3f} seconds

## GPU Configuration

- Device: {torch.cuda.get_device_name(0)}
- CUDA Version: {torch.version.cuda}
- Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB

![GPU Performance Chart](gpu_performance_chart.png)
"""
        
        with open(f"{output_dir}/benchmark_report.md", 'w') as f:
            f.write(report)
        
        self.log(f"\nReport generated in {output_dir}/")
        self.log(f"Average speedup: {df['Speedup'].mean():.1f}x")


def main():
    parser = argparse.ArgumentParser(description="GPU Performance Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--output", default="benchmarks/results", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    args = parser.parse_args()
    
    print("🚀 AIQToolkit GPU Performance Benchmarks")
    print("=" * 40)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ No GPU detected. These benchmarks require NVIDIA GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 40)
    
    benchmark = GPUBenchmark(device=args.device)
    
    # Run benchmarks
    if args.quick:
        # Quick benchmarks for demo
        benchmark.benchmark_similarity_computation([1000])
        benchmark.benchmark_nash_equilibrium([50])
        benchmark.benchmark_attention_mechanism([512])
    else:
        # Full benchmarks
        benchmark.benchmark_similarity_computation()
        benchmark.benchmark_nash_equilibrium()
        benchmark.benchmark_attention_mechanism()
        benchmark.benchmark_mixture_of_experts()
        benchmark.benchmark_consensus_round()
    
    # Generate report
    benchmark.generate_report(args.output)
    
    print("\n✅ Benchmarks complete!")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()