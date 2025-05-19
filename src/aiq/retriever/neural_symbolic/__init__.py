# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIQ Toolkit Neural-Symbolic Retriever

This module provides a neural-symbolic retriever that combines neural embeddings
with symbolic reasoning for enhanced retrieval capabilities.
"""

from aiq.retriever.neural_symbolic.neural_symbolic_retriever import (
    NeuralSymbolicRetriever,
    ModelType,
    ReasoningType,
    KnowledgeEntity,
    KnowledgeRelation,
    KnowledgeGraph,
    RetrievalCandidate,
    create_neural_symbolic_retriever
)

try:
    from aiq.retriever.neural_symbolic.dspy_retriever import (
        DSPyNeuralSymbolicRetriever,
        create_dspy_neural_symbolic_retriever
    )
except ImportError:
    # DSPy not available
    DSPyNeuralSymbolicRetriever = None
    create_dspy_neural_symbolic_retriever = None

__all__ = [
    'NeuralSymbolicRetriever',
    'ModelType',
    'ReasoningType',
    'KnowledgeEntity',
    'KnowledgeRelation',
    'KnowledgeGraph',
    'RetrievalCandidate',
    'create_neural_symbolic_retriever',
    'DSPyNeuralSymbolicRetriever',
    'create_dspy_neural_symbolic_retriever'
]