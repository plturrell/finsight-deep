# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework adapters for cross-framework memory integration
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from aiq.memory.models import MemoryItem

logger = logging.getLogger(__name__)


class FrameworkAdapter(ABC):
    """Base class for framework memory adapters"""
    
    @abstractmethod
    def get_memories(self) -> List[Dict[str, Any]]:
        """Get all memories from the framework"""
        pass
    
    @abstractmethod
    def add_memory(self, memory_item: MemoryItem) -> bool:
        """Add a memory to the framework"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search memories in the framework"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the framework"""
        pass


class HaystackAdapter(FrameworkAdapter):
    """Adapter for Haystack framework"""
    
    def __init__(self, document_store: Optional[Any] = None):
        self.document_store = document_store
        try:
            from haystack.document_stores import InMemoryDocumentStore
            from haystack.nodes import EmbeddingRetriever
            
            if self.document_store is None:
                self.document_store = InMemoryDocumentStore()
            
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model="sentence-transformers/all-mpnet-base-v2"
            )
        except ImportError:
            logger.warning("Haystack not installed. Install with: pip install farm-haystack")
            self.retriever = None
    
    def get_memories(self) -> List[Dict[str, Any]]:
        """Get all documents from Haystack"""
        if not self.document_store:
            return []
        
        documents = self.document_store.get_all_documents()
        memories = []
        
        for doc in documents:
            memories.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.meta or {}
            })
        
        return memories
    
    def add_memory(self, memory_item: MemoryItem) -> bool:
        """Add a document to Haystack"""
        if not self.document_store:
            return False
        
        try:
            from haystack import Document
            
            doc = Document(
                content=memory_item.content,
                id=memory_item.id,
                meta=memory_item.metadata
            )
            
            self.document_store.write_documents([doc])
            
            # Update embeddings if retriever is available
            if self.retriever:
                self.document_store.update_embeddings(self.retriever)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add memory to Haystack: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search documents in Haystack"""
        if not self.document_store or not self.retriever:
            return []
        
        try:
            results = self.retriever.retrieve(
                query=query,
                top_k=limit
            )
            
            memory_items = []
            for doc in results:
                memory_items.append(MemoryItem(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.meta or {}
                ))
            
            return memory_items
        except Exception as e:
            logger.error(f"Failed to search in Haystack: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a document from Haystack"""
        if not self.document_store:
            return False
        
        try:
            self.document_store.delete_documents([memory_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory from Haystack: {e}")
            return False


class LlamaIndexAdapter(FrameworkAdapter):
    """Adapter for LlamaIndex framework"""
    
    def __init__(self, index: Optional[Any] = None):
        self.index = index
        try:
            from llama_index import VectorStoreIndex, Document
            from llama_index.vector_stores import SimpleVectorStore
            
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [],
                    vector_store=SimpleVectorStore()
                )
            
            self.Document = Document
        except ImportError:
            logger.warning("LlamaIndex not installed. Install with: pip install llama-index")
            self.Document = None
    
    def get_memories(self) -> List[Dict[str, Any]]:
        """Get all documents from LlamaIndex"""
        if not self.index:
            return []
        
        memories = []
        
        # Access documents through the docstore
        docstore = self.index.docstore
        for doc_id, doc in docstore.docs.items():
            memories.append({
                "id": doc_id,
                "content": doc.text,
                "metadata": doc.metadata or {}
            })
        
        return memories
    
    def add_memory(self, memory_item: MemoryItem) -> bool:
        """Add a document to LlamaIndex"""
        if not self.index or not self.Document:
            return False
        
        try:
            doc = self.Document(
                text=memory_item.content,
                id_=memory_item.id,
                metadata=memory_item.metadata
            )
            
            self.index.insert(doc)
            return True
        except Exception as e:
            logger.error(f"Failed to add memory to LlamaIndex: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search documents in LlamaIndex"""
        if not self.index:
            return []
        
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=limit
            )
            
            response = query_engine.query(query)
            
            memory_items = []
            for node in response.source_nodes:
                memory_items.append(MemoryItem(
                    id=node.node.id_,
                    content=node.node.text,
                    metadata=node.node.metadata or {}
                ))
            
            return memory_items
        except Exception as e:
            logger.error(f"Failed to search in LlamaIndex: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a document from LlamaIndex"""
        if not self.index:
            return False
        
        try:
            self.index.delete_ref_doc(memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory from LlamaIndex: {e}")
            return False


class LangChainAdapter(FrameworkAdapter):
    """Adapter for LangChain framework"""
    
    def __init__(self, vectorstore: Optional[Any] = None):
        self.vectorstore = vectorstore
        try:
            from langchain.vectorstores import Chroma
            from langchain.embeddings import OpenAIEmbeddings
            
            if self.vectorstore is None:
                self.vectorstore = Chroma(
                    embedding_function=OpenAIEmbeddings()
                )
        except ImportError:
            logger.warning("LangChain not installed. Install with: pip install langchain")
    
    def get_memories(self) -> List[Dict[str, Any]]:
        """Get all documents from LangChain"""
        if not self.vectorstore:
            return []
        
        # Get all documents (this is vectorstore-specific)
        # For Chroma:
        try:
            results = self.vectorstore.get()
            memories = []
            
            for i, doc_id in enumerate(results.get('ids', [])):
                memories.append({
                    "id": doc_id,
                    "content": results['documents'][i] if 'documents' in results else "",
                    "metadata": results['metadatas'][i] if 'metadatas' in results else {}
                })
            
            return memories
        except Exception as e:
            logger.error(f"Failed to get memories from LangChain: {e}")
            return []
    
    def add_memory(self, memory_item: MemoryItem) -> bool:
        """Add a document to LangChain"""
        if not self.vectorstore:
            return False
        
        try:
            self.vectorstore.add_texts(
                texts=[memory_item.content],
                metadatas=[memory_item.metadata],
                ids=[memory_item.id]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add memory to LangChain: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search documents in LangChain"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=limit
            )
            
            memory_items = []
            for doc in results:
                memory_items.append(MemoryItem(
                    id=doc.metadata.get('id', str(hash(doc.page_content))),
                    content=doc.page_content,
                    metadata=doc.metadata
                ))
            
            return memory_items
        except Exception as e:
            logger.error(f"Failed to search in LangChain: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a document from LangChain"""
        if not self.vectorstore:
            return False
        
        try:
            # This is vectorstore-specific
            # For Chroma:
            self.vectorstore.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory from LangChain: {e}")
            return False


class SLLangAdapter(FrameworkAdapter):
    """Adapter for SLLang (SGLang) framework"""
    
    def __init__(self, cache_manager: Optional[Any] = None):
        self.cache_manager = cache_manager
        self.memories = {}  # Simple in-memory storage for now
    
    def get_memories(self) -> List[Dict[str, Any]]:
        """Get all memories from SLLang cache"""
        memories = []
        
        for memory_id, memory in self.memories.items():
            memories.append({
                "id": memory_id,
                "content": memory["content"],
                "metadata": memory.get("metadata", {})
            })
        
        return memories
    
    def add_memory(self, memory_item: MemoryItem) -> bool:
        """Add a memory to SLLang cache"""
        try:
            self.memories[memory_item.id] = {
                "content": memory_item.content,
                "metadata": memory_item.metadata
            }
            return True
        except Exception as e:
            logger.error(f"Failed to add memory to SLLang: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search memories in SLLang"""
        # Simple text-based search for now
        results = []
        query_lower = query.lower()
        
        for memory_id, memory in self.memories.items():
            if query_lower in memory["content"].lower():
                results.append(MemoryItem(
                    id=memory_id,
                    content=memory["content"],
                    metadata=memory.get("metadata", {})
                ))
                
                if len(results) >= limit:
                    break
        
        return results
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from SLLang"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False


# Factory functions for creating adapters
def create_haystack_adapter(document_store: Optional[Any] = None) -> HaystackAdapter:
    """Create a Haystack adapter"""
    return HaystackAdapter(document_store)


def create_llama_index_adapter(index: Optional[Any] = None) -> LlamaIndexAdapter:
    """Create a LlamaIndex adapter"""
    return LlamaIndexAdapter(index)


def create_langchain_adapter(vectorstore: Optional[Any] = None) -> LangChainAdapter:
    """Create a LangChain adapter"""
    return LangChainAdapter(vectorstore)


def create_sllang_adapter(cache_manager: Optional[Any] = None) -> SLLangAdapter:
    """Create an SLLang adapter"""
    return SLLangAdapter(cache_manager)