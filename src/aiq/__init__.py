# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA Agent Intelligence Toolkit (AIQ Toolkit)

A flexible, lightweight, and unifying library that allows you to easily connect
existing enterprise agents to data sources and tools across any framework.
"""

# Import core components
from aiq.builder import Builder
from aiq.runtime import Runner, Session

# Import new advanced features
from aiq.correction import (
    SelfCorrectingResearchSystem,
    CorrectionStrategy,
    ContentType,
    CorrectionResult,
    create_self_correcting_system
)

from aiq.research import (
    ResearchTaskExecutor,
    ResearchTask,
    TaskType,
    TaskPriority,
    TaskResult
)

from aiq.verification import (
    VerificationSystem,
    VerificationResult,
    ConfidenceMethod,
    SourceType,
    Source,
    ProvenanceRecord,
    create_verification_system
)

from aiq.hardware import (
    TensorCoreOptimizer,
    ResourcePredictor,
    OptimizationConfig,
    ResourceRequirements
)

from aiq.memory import (
    ResearchEntity,
    ResearchRelation,
    ResearchContext,
    ResearchContextManager,
    CrossFrameworkMemory,
    MemoryScope
)

from aiq.visualization import (
    GPUAcceleratedVisualizer,
    VisualizationType,
    create_knowledge_graph_visualization,
    create_research_dashboard,
    visualize_embeddings
)

# Version info
__version__ = "2.0.0"

__all__ = [
    # Core components
    'Builder',
    'Runner',
    'Session',
    
    # Self-correcting AI system
    'SelfCorrectingResearchSystem',
    'CorrectionStrategy',
    'ContentType',
    'CorrectionResult',
    'create_self_correcting_system',
    
    # Research task execution
    'ResearchTaskExecutor',
    'ResearchTask',
    'TaskType',
    'TaskPriority',
    'TaskResult',
    
    # Verification system
    'VerificationSystem',
    'VerificationResult',
    'ConfidenceMethod',
    'SourceType',
    'Source',
    'ProvenanceRecord',
    'create_verification_system',
    
    # Hardware optimization
    'TensorCoreOptimizer',
    'ResourcePredictor',
    'OptimizationConfig',
    'ResourceRequirements',
    
    # Memory system
    'ResearchEntity',
    'ResearchRelation',
    'ResearchContext',
    'ResearchContextManager',
    'CrossFrameworkMemory',
    'MemoryScope',
    
    # Visualization
    'GPUAcceleratedVisualizer',
    'VisualizationType',
    'create_knowledge_graph_visualization',
    'create_research_dashboard',
    'visualize_embeddings',
]