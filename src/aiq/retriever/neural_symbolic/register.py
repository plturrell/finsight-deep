# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

from pydantic import Field

from aiq.builder import Builder
from aiq.cli.register_workflow import register_retriever_client
from aiq.cli.register_workflow import register_retriever_provider
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.retriever.neural_symbolic import (
    NeuralSymbolicRetriever,
    create_neural_symbolic_retriever,
    create_dspy_neural_symbolic_retriever
)

logger = logging.getLogger(__name__)


class NeuralSymbolicRetrieverConfig(RetrieverBaseConfig, name="neural_symbolic_retriever"):
    """
    Configuration for a Neural-Symbolic Retriever which combines neural embeddings with symbolic reasoning.
    """
    dataset_name: str = Field(
        description="Name of the dataset to search"
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return",
        gt=0,
        le=50
    )
    model_type: str = Field(
        default="neuro",
        description="Type of embedding model to use ('basic' or 'neuro')"
    )
    knowledge_fusion: bool = Field(
        default=True,
        description="Whether to use knowledge fusion"
    )
    reasoning_weight: float = Field(
        default=0.3,
        description="Weight of reasoning in results (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    device: str | None = Field(
        default=None,
        description="Device to use for inference ('cpu', 'cuda:0', etc.)"
    )
    dspy_compatible: bool = Field(
        default=False,
        description="Whether to create a DSPy-compatible retriever"
    )
    use_cuda_kernels: bool = Field(
        default=True,
        description="Whether to use CUDA kernels for acceleration"
    )
    embedding_dim: int = Field(
        default=768,
        description="Dimension of embeddings"
    )


@register_retriever_provider(config_type=NeuralSymbolicRetrieverConfig)
async def neural_symbolic_retriever(retriever_config: NeuralSymbolicRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(
        config=retriever_config,
        description="A Neural-Symbolic Retriever combining neural embeddings with symbolic reasoning"
    )


@register_retriever_client(config_type=NeuralSymbolicRetrieverConfig, wrapper_type=None)
async def neural_symbolic_retriever_client(config: NeuralSymbolicRetrieverConfig, builder: Builder):
    # Create appropriate retriever based on DSPy compatibility
    if config.dspy_compatible and create_dspy_neural_symbolic_retriever is not None:
        retriever = create_dspy_neural_symbolic_retriever(
            dataset_name=config.dataset_name,
            model_type=config.model_type,
            knowledge_fusion=config.knowledge_fusion,
            device=config.device,
            reasoning_weight=config.reasoning_weight,
            k=config.top_k,
            use_cuda_kernels=config.use_cuda_kernels,
            embedding_dim=config.embedding_dim
        )
    else:
        retriever = create_neural_symbolic_retriever(
            dataset_name=config.dataset_name,
            model_type=config.model_type,
            knowledge_fusion=config.knowledge_fusion,
            device=config.device,
            reasoning_weight=config.reasoning_weight,
            use_cuda_kernels=config.use_cuda_kernels,
            embedding_dim=config.embedding_dim
        )
    
    yield retriever