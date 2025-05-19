# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Predictive autoscaling for distributed AIQToolkit
Uses ML models to predict resource requirements
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WorkloadMetrics:
    """Metrics for workload prediction"""
    timestamp: datetime
    task_count: int
    avg_task_duration: float
    queue_depth: int
    gpu_utilization: float
    memory_usage: float
    pending_tasks: int
    failed_tasks: int
    
    def to_features(self) -> np.ndarray:
        """Convert to ML features"""
        return np.array([
            self.timestamp.hour,  # Hour of day
            self.timestamp.weekday(),  # Day of week
            self.task_count,
            self.avg_task_duration,
            self.queue_depth,
            self.gpu_utilization,
            self.memory_usage,
            self.pending_tasks,
            self.failed_tasks
        ])


@dataclass
class ScalingDecision:
    """Autoscaling decision"""
    timestamp: datetime
    current_nodes: int
    target_nodes: int
    confidence: float
    reason: str
    cost_estimate: float


class PredictiveScaler:
    """ML-based predictive autoscaler"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.history: List[WorkloadMetrics] = []
        self.predictions: List[ScalingDecision] = []
        self.model_trained = False
        self.update_interval = 60  # seconds
        
    def train_model(self, historical_data: pd.DataFrame):
        """Train the prediction model on historical data"""
        logger.info("Training predictive model...")
        
        # Prepare features
        features = []
        targets = []
        
        for i in range(len(historical_data) - 1):
            row = historical_data.iloc[i]
            next_row = historical_data.iloc[i + 1]
            
            feature = [
                row['timestamp'].hour,
                row['timestamp'].weekday(),
                row['task_count'],
                row['avg_task_duration'],
                row['queue_depth'],
                row['gpu_utilization'],
                row['memory_usage'],
                row['pending_tasks'],
                row['failed_tasks']
            ]
            
            # Target is next period's resource requirement
            target = next_row['required_nodes']
            
            features.append(feature)
            targets.append(target)
        
        X = np.array(features)
        y = np.array(targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.model_trained = True
        
        # Calculate feature importance
        importance = self.model.feature_importances_
        feature_names = [
            'hour', 'weekday', 'task_count', 'avg_duration',
            'queue_depth', 'gpu_util', 'memory', 'pending', 'failed'
        ]
        
        for name, imp in zip(feature_names, importance):
            logger.info(f"Feature importance - {name}: {imp:.3f}")
        
        logger.info("Model training completed")
    
    def predict_requirements(self, 
                           current_metrics: WorkloadMetrics,
                           lookahead_minutes: int = 15) -> Tuple[int, float]:
        """
        Predict resource requirements
        
        Returns:
            Tuple of (required_nodes, confidence)
        """
        if not self.model_trained:
            logger.warning("Model not trained, using rule-based scaling")
            return self._rule_based_scaling(current_metrics), 0.5
        
        # Prepare features
        features = current_metrics.to_features()
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on model's prediction variance
        if hasattr(self.model, 'predict_proba'):
            confidence = self.model.predict_proba(features_scaled).max()
        else:
            # Use prediction variance from trees
            predictions = [tree.predict(features_scaled)[0] 
                          for tree in self.model.estimators_]
            variance = np.var(predictions)
            confidence = 1.0 / (1.0 + variance)
        
        required_nodes = max(1, int(np.ceil(prediction)))
        
        return required_nodes, confidence
    
    def _rule_based_scaling(self, metrics: WorkloadMetrics) -> int:
        """Fallback rule-based scaling"""
        # Simple rules
        if metrics.queue_depth > 100:
            return max(5, metrics.task_count // 20)
        elif metrics.gpu_utilization > 0.8:
            return max(3, metrics.task_count // 30)
        elif metrics.pending_tasks > 50:
            return max(2, metrics.task_count // 40)
        else:
            return max(1, metrics.task_count // 50)
    
    def make_scaling_decision(self,
                            current_metrics: WorkloadMetrics,
                            current_nodes: int,
                            cost_per_node: float = 0.10) -> ScalingDecision:
        """Make a scaling decision based on predictions"""
        # Predict requirements
        required_nodes, confidence = self.predict_requirements(current_metrics)
        
        # Apply confidence threshold
        if confidence < 0.7:
            logger.info(f"Low confidence ({confidence:.2f}), maintaining current scale")
            required_nodes = current_nodes
        
        # Apply hysteresis to prevent flapping
        if abs(required_nodes - current_nodes) <= 1:
            required_nodes = current_nodes
        
        # Cost estimation
        cost_estimate = required_nodes * cost_per_node
        
        # Determine reason
        if required_nodes > current_nodes:
            reason = f"Scale up: High load (queue={current_metrics.queue_depth}, GPU={current_metrics.gpu_utilization:.2f})"
        elif required_nodes < current_nodes:
            reason = f"Scale down: Low utilization (GPU={current_metrics.gpu_utilization:.2f})"
        else:
            reason = "Maintain: Optimal scale"
        
        decision = ScalingDecision(
            timestamp=datetime.now(),
            current_nodes=current_nodes,
            target_nodes=required_nodes,
            confidence=confidence,
            reason=reason,
            cost_estimate=cost_estimate
        )
        
        self.predictions.append(decision)
        
        return decision
    
    def update_history(self, metrics: WorkloadMetrics):
        """Update historical metrics"""
        self.history.append(metrics)
        
        # Keep only recent history (e.g., last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.history = [m for m in self.history if m.timestamp > cutoff]
    
    def save_model(self, path: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'history': self.history
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.history = model_data.get('history', [])
        self.model_trained = True
        logger.info(f"Model loaded from {path}")


class CostAwareScaler:
    """Cost-aware autoscaling with spot instance support"""
    
    def __init__(self):
        self.spot_prices: Dict[str, float] = {
            "g4dn.xlarge": 0.0526,  # per hour
            "g4dn.2xlarge": 0.1052,
            "g4dn.4xlarge": 0.2104,
            "p3.2xlarge": 0.3825,
            "p3.8xlarge": 1.5300
        }
        
        self.on_demand_prices: Dict[str, float] = {
            "g4dn.xlarge": 0.526,
            "g4dn.2xlarge": 0.752,
            "g4dn.4xlarge": 1.204,
            "p3.2xlarge": 3.825,
            "p3.8xlarge": 15.300
        }
        
        self.instance_specs: Dict[str, Dict[str, int]] = {
            "g4dn.xlarge": {"gpus": 1, "memory": 16, "vcpus": 4},
            "g4dn.2xlarge": {"gpus": 1, "memory": 32, "vcpus": 8},
            "g4dn.4xlarge": {"gpus": 1, "memory": 64, "vcpus": 16},
            "p3.2xlarge": {"gpus": 1, "memory": 61, "vcpus": 8},
            "p3.8xlarge": {"gpus": 4, "memory": 244, "vcpus": 32}
        }
    
    def select_instance_mix(self,
                          required_gpus: int,
                          required_memory: int,
                          budget_per_hour: float,
                          spot_percentage: float = 0.7) -> Dict[str, int]:
        """
        Select optimal mix of instance types
        
        Args:
            required_gpus: Number of GPUs needed
            required_memory: Memory needed (GB)
            budget_per_hour: Budget constraint
            spot_percentage: Percentage of spot instances
            
        Returns:
            Dictionary of instance_type -> count
        """
        best_mix = {}
        best_cost = float('inf')
        
        # Try different combinations
        for instance_type, specs in self.instance_specs.items():
            gpus_per_instance = specs['gpus']
            memory_per_instance = specs['memory']
            
            # Calculate required instances
            instances_for_gpus = np.ceil(required_gpus / gpus_per_instance)
            instances_for_memory = np.ceil(required_memory / memory_per_instance)
            total_instances = int(max(instances_for_gpus, instances_for_memory))
            
            if total_instances == 0:
                continue
            
            # Calculate cost with spot/on-demand mix
            spot_instances = int(total_instances * spot_percentage)
            on_demand_instances = total_instances - spot_instances
            
            spot_cost = spot_instances * self.spot_prices[instance_type]
            on_demand_cost = on_demand_instances * self.on_demand_prices[instance_type]
            total_cost = spot_cost + on_demand_cost
            
            # Check budget constraint
            if total_cost <= budget_per_hour and total_cost < best_cost:
                best_cost = total_cost
                best_mix = {
                    f"{instance_type}_spot": spot_instances,
                    f"{instance_type}_ondemand": on_demand_instances
                }
        
        return best_mix
    
    def estimate_savings(self, 
                        instance_mix: Dict[str, int],
                        duration_hours: float = 1.0) -> Dict[str, float]:
        """Estimate cost savings from spot instances"""
        total_spot_cost = 0
        total_on_demand_cost = 0
        
        for instance_key, count in instance_mix.items():
            if count == 0:
                continue
                
            instance_type = instance_key.replace("_spot", "").replace("_ondemand", "")
            
            if "_spot" in instance_key:
                total_spot_cost += count * self.spot_prices[instance_type] * duration_hours
            else:
                total_on_demand_cost += count * self.on_demand_prices[instance_type] * duration_hours
        
        # Calculate what it would cost if all on-demand
        all_on_demand_cost = 0
        for instance_key, count in instance_mix.items():
            if count == 0:
                continue
            instance_type = instance_key.replace("_spot", "").replace("_ondemand", "")
            all_on_demand_cost += count * self.on_demand_prices[instance_type] * duration_hours
        
        savings = all_on_demand_cost - (total_spot_cost + total_on_demand_cost)
        savings_percentage = (savings / all_on_demand_cost) * 100 if all_on_demand_cost > 0 else 0
        
        return {
            "total_cost": total_spot_cost + total_on_demand_cost,
            "on_demand_equivalent": all_on_demand_cost,
            "savings": savings,
            "savings_percentage": savings_percentage,
            "spot_cost": total_spot_cost,
            "on_demand_cost": total_on_demand_cost
        }


# Example usage
if __name__ == "__main__":
    # Create scaler
    scaler = PredictiveScaler()
    
    # Create sample metrics
    metrics = WorkloadMetrics(
        timestamp=datetime.now(),
        task_count=100,
        avg_task_duration=5.5,
        queue_depth=150,
        gpu_utilization=0.85,
        memory_usage=0.75,
        pending_tasks=75,
        failed_tasks=2
    )
    
    # Make scaling decision
    decision = scaler.make_scaling_decision(
        current_metrics=metrics,
        current_nodes=5
    )
    
    print(f"Scaling Decision: {decision.current_nodes} -> {decision.target_nodes}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reason: {decision.reason}")
    print(f"Cost estimate: ${decision.cost_estimate:.2f}")
    
    # Test cost-aware scaler
    cost_scaler = CostAwareScaler()
    instance_mix = cost_scaler.select_instance_mix(
        required_gpus=10,
        required_memory=200,
        budget_per_hour=5.0,
        spot_percentage=0.7
    )
    
    print(f"\nInstance mix: {instance_mix}")
    
    savings = cost_scaler.estimate_savings(instance_mix)
    print(f"Total cost: ${savings['total_cost']:.2f}")
    print(f"Savings: ${savings['savings']:.2f} ({savings['savings_percentage']:.1f}%)")