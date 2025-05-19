# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monitoring and observability for Nash-Ethereum consensus operations
"""

import torch
import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Info

from aiq.neural.nash_ethereum_consensus import NashEthereumState


logger = logging.getLogger(__name__)


# Prometheus metrics
consensus_operations = Counter(
    'consensus_operations_total',
    'Total number of consensus operations',
    ['operation_type', 'status']
)

consensus_duration = Histogram(
    'consensus_duration_seconds',
    'Duration of consensus operations',
    ['operation_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

gas_used_total = Counter(
    'consensus_gas_used_total',
    'Total gas used in consensus operations'
)

agent_reputation = Gauge(
    'agent_reputation_score',
    'Current reputation score of agents',
    ['agent_id']
)

consensus_convergence_rate = Gauge(
    'consensus_convergence_rate',
    'Rate of consensus convergence',
    ['network']
)

active_agents = Gauge(
    'active_agents_count',
    'Number of active agents in the system'
)

consensus_info = Info(
    'consensus_system_info',
    'Information about the consensus system'
)


@dataclass
class ConsensusMetrics:
    """Metrics for a consensus operation"""
    operation_id: str
    operation_type: str
    start_time: float
    end_time: float
    duration: float
    agents_count: int
    iterations: int
    converged: bool
    gas_used: int
    consensus_value: Optional[List[float]]
    error: Optional[str] = None


@dataclass
class AgentMetrics:
    """Metrics for individual agents"""
    agent_id: str
    total_participations: int
    successful_consensus: int
    failed_consensus: int
    average_confidence: float
    total_gas_used: int
    reputation_score: float
    last_active: float


class ConsensusMonitor:
    """
    Monitor and track consensus operations
    Provides real-time metrics and alerting
    """
    
    def __init__(
        self,
        prometheus_port: int = 8000,
        alert_webhook: Optional[str] = None,
        metrics_retention_days: int = 7
    ):
        self.prometheus_port = prometheus_port
        self.alert_webhook = alert_webhook
        self.metrics_retention_days = metrics_retention_days
        
        # Metrics storage
        self.consensus_history: List[ConsensusMetrics] = []
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # Alerting thresholds
        self.alert_thresholds = {
            "gas_spike_threshold": 1000000,  # 1M gas
            "convergence_failure_threshold": 0.2,  # 20% failure rate
            "agent_timeout_hours": 24,
            "reputation_drop_threshold": 0.3  # 30% drop
        }
        
        # Start Prometheus server
        prometheus_client.start_http_server(self.prometheus_port)
        
        # Update system info
        consensus_info.info({
            'version': '1.0.0',
            'network': 'ethereum',
            'monitoring_enabled': 'true'
        })
        
        # Start background tasks
        asyncio.create_task(self._cleanup_old_metrics())
        asyncio.create_task(self._check_alerts())
        
        logger.info(f"Consensus monitor started on port {self.prometheus_port}")
    
    async def record_consensus_operation(
        self,
        operation_id: str,
        operation_type: str,
        state: NashEthereumState,
        start_time: float,
        error: Optional[str] = None
    ):
        """Record metrics for a consensus operation"""
        end_time = time.time()
        duration = end_time - start_time
        
        # Create metrics record
        metrics = ConsensusMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            agents_count=len(state.participants),
            iterations=0,  # Would be extracted from state
            converged=state.converged,
            gas_used=state.gas_used,
            consensus_value=state.on_chain_consensus,
            error=error
        )
        
        # Store metrics
        self.consensus_history.append(metrics)
        
        # Update Prometheus metrics
        status = "success" if error is None else "failure"
        consensus_operations.labels(operation_type=operation_type, status=status).inc()
        consensus_duration.labels(operation_type=operation_type).observe(duration)
        gas_used_total.inc(state.gas_used)
        
        # Update convergence rate
        self._update_convergence_rate()
        
        # Update agent metrics
        for participant in state.participants:
            await self._update_agent_metrics(participant, metrics)
        
        # Check for alerts
        await self._check_operation_alerts(metrics)
    
    async def _update_agent_metrics(self, agent_id: str, consensus_metrics: ConsensusMetrics):
        """Update metrics for individual agent"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                total_participations=0,
                successful_consensus=0,
                failed_consensus=0,
                average_confidence=0.0,
                total_gas_used=0,
                reputation_score=100.0,
                last_active=time.time()
            )
        
        agent = self.agent_metrics[agent_id]
        agent.total_participations += 1
        
        if consensus_metrics.converged and consensus_metrics.error is None:
            agent.successful_consensus += 1
        else:
            agent.failed_consensus += 1
        
        agent.total_gas_used += consensus_metrics.gas_used // len(consensus_metrics.agents_count)
        agent.last_active = time.time()
        
        # Update Prometheus gauge
        agent_reputation.labels(agent_id=agent_id).set(agent.reputation_score)
    
    def _update_convergence_rate(self):
        """Update convergence rate metric"""
        recent_operations = [
            op for op in self.consensus_history
            if time.time() - op.end_time < 3600  # Last hour
        ]
        
        if recent_operations:
            converged = sum(1 for op in recent_operations if op.converged)
            rate = converged / len(recent_operations)
            consensus_convergence_rate.labels(network="ethereum").set(rate)
    
    async def _check_operation_alerts(self, metrics: ConsensusMetrics):
        """Check if operation metrics trigger any alerts"""
        alerts = []
        
        # Gas spike alert
        if metrics.gas_used > self.alert_thresholds["gas_spike_threshold"]:
            alerts.append({
                "type": "gas_spike",
                "severity": "warning",
                "message": f"High gas usage: {metrics.gas_used} in operation {metrics.operation_id}",
                "value": metrics.gas_used
            })
        
        # Convergence failure
        if not metrics.converged:
            alerts.append({
                "type": "convergence_failure",
                "severity": "error",
                "message": f"Consensus failed to converge: {metrics.operation_id}",
                "operation_id": metrics.operation_id
            })
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _check_alerts(self):
        """Background task to check system-wide alerts"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Check convergence rate
            recent_operations = [
                op for op in self.consensus_history
                if time.time() - op.end_time < 3600
            ]
            
            if recent_operations:
                failure_rate = sum(1 for op in recent_operations if not op.converged) / len(recent_operations)
                
                if failure_rate > self.alert_thresholds["convergence_failure_threshold"]:
                    await self._send_alert({
                        "type": "high_failure_rate",
                        "severity": "critical",
                        "message": f"High consensus failure rate: {failure_rate:.1%}",
                        "value": failure_rate
                    })
            
            # Check inactive agents
            inactive_threshold = time.time() - (self.alert_thresholds["agent_timeout_hours"] * 3600)
            
            for agent_id, agent in self.agent_metrics.items():
                if agent.last_active < inactive_threshold:
                    await self._send_alert({
                        "type": "inactive_agent",
                        "severity": "warning",
                        "message": f"Agent {agent_id} inactive for >24 hours",
                        "agent_id": agent_id
                    })
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to webhook"""
        logger.warning(f"Alert: {alert}")
        
        if self.alert_webhook:
            # Send to webhook (implementation depends on alert service)
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                await session.post(self.alert_webhook, json=alert)
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        while True:
            await asyncio.sleep(3600)  # Run hourly
            
            cutoff_time = time.time() - (self.metrics_retention_days * 86400)
            
            # Clean consensus history
            self.consensus_history = [
                m for m in self.consensus_history
                if m.end_time > cutoff_time
            ]
            
            logger.info(f"Cleaned up metrics older than {self.metrics_retention_days} days")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.consensus_history:
            return {}
        
        recent_history = [
            m for m in self.consensus_history
            if time.time() - m.end_time < 86400  # Last 24 hours
        ]
        
        if not recent_history:
            return {}
        
        return {
            "total_operations": len(recent_history),
            "success_rate": sum(1 for m in recent_history if m.converged) / len(recent_history),
            "average_duration": np.mean([m.duration for m in recent_history]),
            "total_gas_used": sum(m.gas_used for m in recent_history),
            "active_agents": len(self.agent_metrics),
            "average_agents_per_operation": np.mean([m.agents_count for m in recent_history])
        }
    
    def get_agent_leaderboard(self) -> List[Dict[str, Any]]:
        """Get agent performance leaderboard"""
        leaderboard = []
        
        for agent_id, metrics in self.agent_metrics.items():
            success_rate = (
                metrics.successful_consensus / metrics.total_participations
                if metrics.total_participations > 0 else 0
            )
            
            leaderboard.append({
                "agent_id": agent_id,
                "reputation": metrics.reputation_score,
                "success_rate": success_rate,
                "total_participations": metrics.total_participations,
                "average_gas": metrics.total_gas_used / max(metrics.total_participations, 1)
            })
        
        # Sort by reputation
        leaderboard.sort(key=lambda x: x["reputation"], reverse=True)
        
        return leaderboard[:10]  # Top 10


class ConsensusAnalytics:
    """
    Advanced analytics for consensus operations
    """
    
    def __init__(self, monitor: ConsensusMonitor):
        self.monitor = monitor
    
    def analyze_gas_efficiency(self) -> Dict[str, Any]:
        """Analyze gas usage patterns"""
        if not self.monitor.consensus_history:
            return {}
        
        gas_by_type = {}
        gas_by_hour = {}
        
        for metrics in self.monitor.consensus_history:
            # By operation type
            op_type = metrics.operation_type
            if op_type not in gas_by_type:
                gas_by_type[op_type] = []
            gas_by_type[op_type].append(metrics.gas_used)
            
            # By hour of day
            hour = datetime.fromtimestamp(metrics.start_time).hour
            if hour not in gas_by_hour:
                gas_by_hour[hour] = []
            gas_by_hour[hour].append(metrics.gas_used)
        
        return {
            "by_operation_type": {
                op_type: {
                    "average": np.mean(gas_list),
                    "median": np.median(gas_list),
                    "std": np.std(gas_list)
                }
                for op_type, gas_list in gas_by_type.items()
            },
            "by_hour": {
                hour: {
                    "average": np.mean(gas_list),
                    "count": len(gas_list)
                }
                for hour, gas_list in gas_by_hour.items()
            },
            "optimal_hours": self._find_optimal_hours(gas_by_hour)
        }
    
    def _find_optimal_hours(self, gas_by_hour: Dict[int, List[int]]) -> List[int]:
        """Find hours with lowest gas usage"""
        hour_averages = {
            hour: np.mean(gas_list)
            for hour, gas_list in gas_by_hour.items()
        }
        
        sorted_hours = sorted(hour_averages.items(), key=lambda x: x[1])
        return [hour for hour, _ in sorted_hours[:3]]  # Top 3 cheapest hours
    
    def analyze_convergence_patterns(self) -> Dict[str, Any]:
        """Analyze convergence patterns"""
        convergence_times = []
        convergence_by_size = {}
        
        for metrics in self.monitor.consensus_history:
            if metrics.converged:
                convergence_times.append(metrics.duration)
                
                size = metrics.agents_count
                if size not in convergence_by_size:
                    convergence_by_size[size] = []
                convergence_by_size[size].append(metrics.duration)
        
        return {
            "average_convergence_time": np.mean(convergence_times) if convergence_times else 0,
            "convergence_by_agent_count": {
                size: {
                    "average_time": np.mean(times),
                    "success_rate": len(times) / len([
                        m for m in self.monitor.consensus_history
                        if m.agents_count == size
                    ])
                }
                for size, times in convergence_by_size.items()
            }
        }
    
    def predict_gas_cost(self, operation_type: str, agents_count: int) -> float:
        """Predict gas cost for an operation"""
        # Simple prediction based on historical data
        relevant_operations = [
            m for m in self.monitor.consensus_history
            if m.operation_type == operation_type and 
            abs(m.agents_count - agents_count) <= 2
        ]
        
        if not relevant_operations:
            # Fallback to general estimate
            return 200000 * agents_count
        
        gas_values = [m.gas_used for m in relevant_operations]
        return np.mean(gas_values)


# Example monitoring dashboard
class ConsensusDemo:
    """Demonstration of monitoring in action"""
    
    def __init__(self):
        self.monitor = ConsensusMonitor()
        self.analytics = ConsensusAnalytics(self.monitor)
    
    async def simulate_consensus_operations(self):
        """Simulate consensus operations for monitoring demo"""
        operation_types = ["content_ranking", "model_evaluation", "data_validation"]
        
        for i in range(20):
            # Create mock consensus state
            state = NashEthereumState(
                task_hash=f"task_{i}",
                contract_address="0x" + "0" * 40,
                participants=[f"agent_{j}" for j in range(np.random.randint(3, 8))],
                positions={},
                nash_equilibrium=torch.randn(10),
                on_chain_consensus=[float(x) for x in torch.randn(10)],
                gas_used=np.random.randint(100000, 500000),
                block_number=12345 + i,
                converged=np.random.random() > 0.1  # 90% success rate
            )
            
            # Record operation
            await self.monitor.record_consensus_operation(
                operation_id=f"op_{i}",
                operation_type=np.random.choice(operation_types),
                state=state,
                start_time=time.time() - np.random.uniform(1, 10)
            )
            
            await asyncio.sleep(0.1)
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        print("\n=== Consensus Monitoring Dashboard ===\n")
        
        # Summary stats
        stats = self.monitor.get_summary_stats()
        print("Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Agent leaderboard
        print("\nAgent Leaderboard:")
        for i, agent in enumerate(self.monitor.get_agent_leaderboard()):
            print(f"  {i+1}. {agent['agent_id']}: {agent['reputation']:.1f} reputation")
        
        # Gas analytics
        print("\nGas Usage Analytics:")
        gas_analysis = self.analytics.analyze_gas_efficiency()
        if "optimal_hours" in gas_analysis:
            print(f"  Optimal hours for submission: {gas_analysis['optimal_hours']}")
        
        # Convergence patterns
        print("\nConvergence Analysis:")
        convergence = self.analytics.analyze_convergence_patterns()
        print(f"  Average convergence time: {convergence.get('average_convergence_time', 0):.2f}s")


if __name__ == "__main__":
    # Run monitoring demo
    async def main():
        demo = ConsensusDemo()
        
        # Start simulation
        await demo.simulate_consensus_operations()
        
        # Display dashboard
        demo.display_dashboard()
        
        # Keep monitoring server running
        print("\nPrometheus metrics available at http://localhost:8000")
        await asyncio.sleep(3600)
    
    asyncio.run(main())