# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Optional, List

from pydantic import Field
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.retriever import RetrieverProviderInfo
from aiq.cli.register_workflow import register_retriever_client
from aiq.cli.register_workflow import register_retriever_provider
from aiq.data_models.retriever import RetrieverBaseConfig

logger = logging.getLogger(__name__)


class NemoRetrieverConfig(RetrieverBaseConfig, name="nemo_retriever"):
    """
    Configuration for a Retriever which pulls data from a Nemo Retriever service.
    """
    uri: HttpUrl = Field(description="The uri of the Nemo Retriever service.")
    collection_name: str | None = Field(description="The name of the collection to search", default=None)
    top_k: int | None = Field(description="The number of results to return", gt=0, le=50, default=None)
    output_fields: list[str] | None = Field(
        default=None,
        description="A list of fields to return from the datastore. If 'None', all fields but the vector are returned.")
    timeout: int = Field(default=60, description="Maximum time to wait for results to be returned from the service.")
    nvidia_api_key: str | None = Field(
        description="API key used to authenticate with the service. If 'None', will use ENV Variable 'NVIDIA_API_KEY'",
        default=None,
    )


class JenaVectorRetrieverConfig(RetrieverBaseConfig, name="jena_vector_retriever"):
    """
    Configuration for a Retriever which combines NeMo Retriever with Jena Vector capabilities.
    This enhanced retriever supports GPU acceleration, knowledge graph integration, and neural-symbolic retrieval.
    """
    uri: HttpUrl = Field(description="The uri of the Nemo Retriever service.")
    collection_name: str | None = Field(description="The name of the collection to search", default=None)
    top_k: int | None = Field(description="The number of results to return", gt=0, le=50, default=10)
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU acceleration for vector operations"
    )
    device_id: int = Field(
        default=0,
        description="GPU device ID to use (only applicable if use_gpu is True)"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache embeddings and retrieval results"
    )
    tensor_core_optimization: bool = Field(
        default=True,
        description="Whether to use Tensor Core optimizations when available"
    )
    nvidia_api_key: str | None = Field(
        description="API key used to authenticate with the service. If 'None', will use ENV Variable 'NVIDIA_API_KEY'",
        default=None,
    )
    model_type: str = Field(
        default="neuro",
        description="Type of embedding model to use ('basic' or 'neuro')"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embeddings"
    )
    knowledge_fusion: bool = Field(
        default=True,
        description="Whether to use knowledge fusion"
    )
    timeout: int = Field(
        default=60,
        description="Maximum time to wait for results to be returned from the service."
    )


class NeuralSymbolicRetrieverConfig(RetrieverBaseConfig, name="neural_symbolic_retriever"):
    """
    Configuration for a Neural-Symbolic Retriever which combines neural embeddings with symbolic reasoning.
    This retriever provides knowledge graph-enhanced search with reasoning capabilities.
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


@register_retriever_provider(config_type=NemoRetrieverConfig)
async def nemo_retriever(retriever_config: NemoRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(config=retriever_config,
                                description="An adapter for a Nemo data store for use with a Retriever Client")


@register_retriever_client(config_type=NemoRetrieverConfig, wrapper_type=None)
async def nemo_retriever_client(config: NemoRetrieverConfig, builder: Builder):
    from aiq.retriever.nemo_retriever.retriever import NemoRetriever

    retriever = NemoRetriever(**config.model_dump(exclude={"type", "top_k", "collection_name"}))
    optional_fields = ["collection_name", "top_k", "output_fields"]
    model_dict = config.model_dump()
    optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}

    retriever.bind(**optional_args)

    yield retriever


@register_retriever_provider(config_type=JenaVectorRetrieverConfig)
async def jena_vector_retriever(retriever_config: JenaVectorRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(
        config=retriever_config,
        description="An enhanced Nemo Retriever with Jena Vector capabilities including GPU acceleration and knowledge graph integration"
    )


@register_retriever_client(config_type=JenaVectorRetrieverConfig, wrapper_type=None)
async def jena_vector_retriever_client(config: JenaVectorRetrieverConfig, builder: Builder):
    # Try to import JenaVectorRetriever
    try:
        from aiq.jena.nemo_vector import JenaVectorRetriever, create_jena_vector_retriever
        
        # Try to import knowledge graph functionality
        try:
            from aiq.jena.knowledge_graph import KnowledgeGraph, create_knowledge_graph
            # Create knowledge graph if needed for integration
            knowledge_graph = create_knowledge_graph()
        except ImportError:
            logger.warning("Knowledge graph integration not available. Some features will be limited.")
            knowledge_graph = None
            
        # Convert cache_dir to Path if provided
        cache_dir = Path(config.cache_dir) if config.cache_dir else None
            
        # Create JenaVectorRetriever using the factory function
        retriever = create_jena_vector_retriever(
            nemo_uri=str(config.uri),
            use_gpu=config.use_gpu,
            device_id=config.device_id,
            knowledge_graph=knowledge_graph,
            tensor_core_optimization=config.tensor_core_optimization,
            nvidia_api_key=config.nvidia_api_key,
            collection_name=config.collection_name,
            model_type=config.model_type
        )
        
        # Create an async wrapper for search since the AIQToolkit expects async functions
        class AsyncJenaVectorRetriever:
            def __init__(self, retriever):
                self.retriever = retriever
                
            async def search(self, query: str, **kwargs):
                return await self.retriever.search(query, **kwargs)
                
            async def batch_search(self, queries: List[str], **kwargs):
                return await self.retriever.batch_search(queries, **kwargs)
                
            def get_embeddings(self, texts: List[str]):
                return self.retriever.get_embeddings(texts)
                
            def create_collection(self, collection_name: str, documents, embed_field: str = "content"):
                return self.retriever.create_collection(collection_name, documents, embed_field)
                
            def add_documents(self, collection_id: str, documents, embed_field: str = "content"):
                return self.retriever.add_documents(collection_id, documents, embed_field)
                
            def delete_collection(self, collection_id: str):
                return self.retriever.delete_collection(collection_id)
                
        yield AsyncJenaVectorRetriever(retriever)
        
    except ImportError as e:
        logger.error(f"Failed to import JenaVectorRetriever: {e}")
        # Fall back to standard NemoRetriever
        from aiq.retriever.nemo_retriever.retriever import NemoRetriever
        
        logger.warning("Falling back to standard NemoRetriever")
        retriever = NemoRetriever(**config.model_dump(exclude={"type", "top_k", "collection_name", 
                                                               "use_gpu", "device_id", "cache_dir", 
                                                               "tensor_core_optimization", "model_type",
                                                               "embedding_dimension", "knowledge_fusion"}))
        optional_fields = ["collection_name", "top_k", "output_fields"]
        model_dict = config.model_dump()
        optional_args = {field: model_dict[field] for field in optional_fields if model_dict[field] is not None}
        
        retriever.bind(**optional_args)
        
        yield retriever


@register_retriever_provider(config_type=NeuralSymbolicRetrieverConfig)
async def neural_symbolic_retriever(retriever_config: NeuralSymbolicRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(
        config=retriever_config,
        description="A Neural-Symbolic Retriever combining neural embeddings with symbolic reasoning"
    )


@register_retriever_client(config_type=NeuralSymbolicRetrieverConfig, wrapper_type=None)
async def neural_symbolic_retriever_client(config: NeuralSymbolicRetrieverConfig, builder: Builder):
    # Try to import NeuralSymbolicRetriever
    try:
        from aiq.retriever.nemo_retriever.neural_symbolic import create_neural_symbolic_retriever
        
        # Create NeuralSymbolicRetriever using the factory function
        retriever = create_neural_symbolic_retriever(
            dataset_name=config.dataset_name,
            model_type=config.model_type,
            knowledge_fusion=config.knowledge_fusion,
            dspy_compatible=config.dspy_compatible,
            device=config.device
        )
        
        # Create an async wrapper for search
        class AsyncNeuralSymbolicRetriever:
            def __init__(self, retriever):
                self.retriever = retriever
                
            async def search(self, query: str, **kwargs):
                k = kwargs.get("top_k", config.top_k)
                threshold = kwargs.get("threshold", 0.0)
                use_reasoning = kwargs.get("use_reasoning", True)
                
                results = self.retriever.search(
                    query=query,
                    k=k,
                    threshold=threshold,
                    use_reasoning=use_reasoning
                )
                
                # Format results to match AIQToolkit expected format
                formatted_results = {
                    "results": [],
                    "metadata": {
                        "query": query,
                        "neural_symbolic": True,
                        "knowledge_fusion": config.knowledge_fusion
                    }
                }
                
                for result in results:
                    formatted_result = {
                        "content": result.get("text", ""),
                        "metadata": {
                            "entity": result.get("entity", ""),
                            "name": result.get("name", ""),
                            "score": result.get("score", 0.0)
                        }
                    }
                    
                    # Add properties if available
                    if "properties" in result:
                        formatted_result["metadata"]["properties"] = result["properties"]
                        
                    formatted_results["results"].append(formatted_result)
                
                return formatted_results
                
            async def search_and_infer(self, query: str, **kwargs):
                k = kwargs.get("top_k", config.top_k)
                threshold = kwargs.get("threshold", 0.0)
                include_connections = kwargs.get("include_connections", True)
                
                return self.retriever.search_and_infer(
                    query=query,
                    k=k,
                    threshold=threshold,
                    include_connections=include_connections
                )
                
            def set_dataset(self, dataset_name: str):
                return self.retriever.set_dataset(dataset_name)
                
            def create_dataset(self, name: str = None, dimension: int = None, apply_reasoning: bool = True):
                return self.retriever.create_dataset(name, dimension, apply_reasoning)
                
            def index_texts(self, texts: List[str], entity_uris: List[str], generate_knowledge: bool = True):
                return self.retriever.index_texts(texts, entity_uris, generate_knowledge)
                
            def apply_reasoning(self, reasoner_type: str = "rdfs"):
                return self.retriever.apply_reasoning(reasoner_type)
                
        yield AsyncNeuralSymbolicRetriever(retriever)
        
    except ImportError as e:
        logger.error(f"Failed to import NeuralSymbolicRetriever: {e}")
        # Cannot fall back to another retriever as this is specialized
        raise ImportError(f"Neural-Symbolic Retriever components not available: {e}")
