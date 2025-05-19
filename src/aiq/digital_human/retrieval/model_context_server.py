"""
Model Context Server with RAG and Web Search Integration
Implements NVIDIA RAG blueprint with external data retrieval
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import requests
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from aiq.retriever.interface import RetrieverInterface
from aiq.embedder.nim_embedder import NVIDIANIMEmbedder
from aiq.utils.optional_imports import optional_import

# Optional imports
googlesearch = optional_import("googlesearch")
yahoo_news = optional_import("yfinance")
milvus = optional_import("pymilvus")
nemo_retriever = optional_import("nemo.collections.retriever")

logger = logging.getLogger(__name__)


@dataclass
class ContextServerConfig:
    """Configuration for Model Context Server"""
    # API Keys
    google_api_key: Optional[str] = None
    yahoo_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    
    # Vector Database
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "financial_context"
    
    # NeMo Retriever
    embedding_model: str = "nvidia/nemo-retriever-embedding-v1"
    retrieval_model: str = "nvidia/nemo-retriever-reranking-v1"
    
    # Web Search
    max_search_results: int = 10
    search_timeout: int = 30
    
    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    
    # Cache
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour


class ModelContextServer:
    """
    Model Context Server implementing NVIDIA RAG blueprint
    with web search capabilities for financial data.
    """
    
    def __init__(self, config: ContextServerConfig):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self._init_embedder()
        self._init_vector_db()
        self._init_web_search()
        self._init_retriever()
        
        # Cache for search results
        self.cache = {} if config.enable_cache else None
        
    def _init_embedder(self):
        """Initialize NeMo embedding service"""
        self.embedder = NVIDIANIMEmbedder(
            model_name=self.config.embedding_model,
            api_key=self.config.nvidia_api_key
        )
        
    def _init_vector_db(self):
        """Initialize Milvus vector database"""
        if milvus:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            
            # Connect to Milvus
            connections.connect(
                host=self.config.milvus_host,
                port=self.config.milvus_port
            )
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Financial context embeddings"
            )
            
            # Create or load collection
            try:
                self.collection = Collection(
                    name=self.config.collection_name,
                    schema=schema
                )
            except Exception:
                self.collection = Collection(self.config.collection_name)
                
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            self.collection.load()
        else:
            self.collection = None
            self.logger.warning("Milvus not available, using in-memory storage")
            
    def _init_web_search(self):
        """Initialize web search capabilities"""
        self.search_providers = {}
        
        # Google Search
        if googlesearch and self.config.google_api_key:
            self.search_providers["google"] = GoogleSearchProvider(
                api_key=self.config.google_api_key
            )
        else:
            self.search_providers["google"] = FallbackSearchProvider()
            
        # Yahoo News
        if yahoo_news and self.config.yahoo_api_key:
            self.search_providers["yahoo"] = YahooNewsProvider(
                api_key=self.config.yahoo_api_key
            )
        else:
            self.search_providers["yahoo"] = FallbackSearchProvider()
            
        # Financial data sources
        self.search_providers["financial"] = FinancialDataProvider()
        
    def _init_retriever(self):
        """Initialize NeMo retriever"""
        if nemo_retriever:
            self.retriever = nemo_retriever.NeMoRetriever(
                retrieval_model=self.config.retrieval_model,
                reranking_model=self.config.retrieval_model
            )
        else:
            self.retriever = None
            
    async def retrieve_context(
        self,
        query: str,
        sources: List[str] = ["all"],
        context_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            sources: List of sources to search
            context_type: Type of context (general, financial, news)
            
        Returns:
            Retrieved context with sources
        """
        # Check cache
        cache_key = f"{query}:{':'.join(sources)}:{context_type}"
        if self.cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if (datetime.now().timestamp() - cached["timestamp"]) < self.config.cache_ttl:
                return cached["result"]
                
        results = {
            "query": query,
            "sources": [],
            "context": [],
            "metadata": {}
        }
        
        # Search vector database
        if self.collection:
            vector_results = await self._search_vector_db(query)
            results["sources"].extend(vector_results)
            
        # Search web sources
        if "all" in sources or "web" in sources:
            web_results = await self._search_web(query, context_type)
            results["sources"].extend(web_results)
            
        # Aggregate and rank results
        ranked_results = await self._rank_results(
            query,
            results["sources"]
        )
        
        # Extract context
        context = self._extract_context(ranked_results)
        results["context"] = context
        
        # Add metadata
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "num_sources": len(results["sources"]),
            "context_type": context_type
        }
        
        # Cache results
        if self.cache:
            self.cache[cache_key] = {
                "result": results,
                "timestamp": datetime.now().timestamp()
            }
            
        return results
        
    async def _search_vector_db(self, query: str) -> List[Dict[str, Any]]:
        """Search vector database for relevant content"""
        if not self.collection:
            return []
            
        # Generate query embedding
        query_embedding = await self.embedder.get_embeddings([query])
        query_vector = query_embedding[0]
        
        # Search
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=self.config.top_k,
            output_fields=["text", "source", "metadata"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score": hit.score,
                    "metadata": hit.entity.get("metadata", {})
                })
                
        return formatted_results
        
    async def _search_web(
        self,
        query: str,
        context_type: str
    ) -> List[Dict[str, Any]]:
        """Search web sources for relevant content"""
        results = []
        
        # Determine which providers to use
        providers = []
        if context_type == "financial":
            providers = ["financial", "yahoo"]
        elif context_type == "news":
            providers = ["yahoo", "google"]
        else:
            providers = ["google", "yahoo", "financial"]
            
        # Search each provider
        tasks = []
        for provider_name in providers:
            if provider_name in self.search_providers:
                provider = self.search_providers[provider_name]
                tasks.append(provider.search(query))
                
        # Gather results
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in provider_results:
            if isinstance(result, Exception):
                self.logger.error(f"Search error: {result}")
                continue
            results.extend(result)
            
        return results
        
    async def _rank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank results by relevance"""
        if self.retriever:
            # Use NeMo reranker
            texts = [r["text"] for r in results]
            scores = await self.retriever.rerank(query, texts)
            
            for i, score in enumerate(scores):
                results[i]["rerank_score"] = score
                
            # Sort by rerank score
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        else:
            # Simple scoring based on original scores
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
        return results[:self.config.top_k]
        
    def _extract_context(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract structured context from results"""
        context = []
        
        for result in results:
            context_item = {
                "content": result["text"],
                "source": result["source"],
                "relevance": result.get("rerank_score", result.get("score", 0)),
                "metadata": result.get("metadata", {})
            }
            context.append(context_item)
            
        return context
        
    async def add_to_knowledge_base(
        self,
        text: str,
        source: str,
        metadata: Dict[str, Any] = None
    ):
        """Add new content to the knowledge base"""
        if not self.collection:
            return
            
        # Generate embedding
        embeddings = await self.embedder.get_embeddings([text])
        embedding = embeddings[0]
        
        # Prepare data
        data = {
            "embedding": embedding,
            "text": text,
            "source": source,
            "timestamp": int(datetime.now().timestamp()),
            "metadata": metadata or {}
        }
        
        # Insert into collection
        self.collection.insert([data])
        

class GoogleSearchProvider:
    """Google Search API provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search Google for content"""
        results = []
        
        params = {
            "key": self.api_key,
            "q": query,
            "num": 10
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data.get("items", []):
                        results.append({
                            "text": f"{item.get('title', '')} {item.get('snippet', '')}",
                            "source": item.get("link", ""),
                            "score": 1.0,
                            "metadata": {
                                "provider": "google",
                                "title": item.get("title"),
                                "snippet": item.get("snippet")
                            }
                        })
                        
        return results


class YahooNewsProvider:
    """Yahoo News API provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search Yahoo News for content"""
        results = []
        
        # Implementation would use Yahoo News API
        # This is a placeholder
        
        return results


class FinancialDataProvider:
    """Financial data provider for market information"""
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search financial data sources"""
        results = []
        
        # Extract tickers from query
        import re
        tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
        
        for ticker in tickers:
            # Get financial data (placeholder)
            results.append({
                "text": f"Financial data for {ticker}: Price $100, Volume 1M",
                "source": f"finance/{ticker}",
                "score": 0.9,
                "metadata": {
                    "provider": "financial",
                    "ticker": ticker,
                    "type": "quote"
                }
            })
            
        return results


class FallbackSearchProvider:
    """Fallback search provider when APIs are not available"""
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search implementation"""
        return [{
            "text": f"Sample result for query: {query}",
            "source": "fallback",
            "score": 0.5,
            "metadata": {"provider": "fallback"}
        }]


# Utility function for easy creation
async def create_context_server(config: Dict[str, Any]) -> ModelContextServer:
    """Create and initialize Model Context Server"""
    server_config = ContextServerConfig(**config)
    server = ModelContextServer(server_config)
    return server