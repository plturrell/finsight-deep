"""
Comprehensive test suite for Memory interfaces and implementations
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import tempfile
import json
import numpy as np
from typing import Dict, List, Any, Optional

from aiq.memory.interfaces import (
    MemoryInterface,
    ConversationMemory,
    VectorMemory,
    PersistentMemory,
    HierarchicalMemory,
    CachedMemory
)
from aiq.memory.models import (
    MemoryItem,
    Message,
    MemoryMetadata,
    ConversationContext
)


class TestMemoryInterface:
    """Test suite for base MemoryInterface"""
    
    @pytest.mark.asyncio
    async def test_interface_methods(self):
        """Test that interface methods are defined"""
        # This is a basic implementation for testing
        class TestMemory(MemoryInterface):
            async def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
                pass
            
            async def retrieve(self, key: str) -> Optional[Any]:
                return None
            
            async def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[MemoryItem]:
                return []
            
            async def delete(self, key: str) -> bool:
                return True
            
            async def clear(self) -> None:
                pass
            
            async def get_stats(self) -> Dict[str, Any]:
                return {}
        
        memory = TestMemory()
        
        # Test all interface methods exist
        await memory.store("test", "value")
        result = await memory.retrieve("test")
        results = await memory.search("query")
        deleted = await memory.delete("test")
        await memory.clear()
        stats = await memory.get_stats()
        
        assert result is None or isinstance(result, (str, dict, list))
        assert isinstance(results, list)
        assert isinstance(deleted, bool)
        assert isinstance(stats, dict)


class TestConversationMemory:
    """Test suite for ConversationMemory"""
    
    @pytest.fixture
    def conversation_memory(self):
        """Create conversation memory instance"""
        return ConversationMemory(
            max_history=10,
            summarization_threshold=5
        )
    
    @pytest.mark.asyncio
    async def test_add_message(self, conversation_memory):
        """Test adding messages to conversation history"""
        await conversation_memory.add_message("user", "Hello AI!")
        await conversation_memory.add_message("assistant", "Hello! How can I help you?")
        
        history = await conversation_memory.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello AI!"
        assert history[1].role == "assistant"
    
    @pytest.mark.asyncio
    async def test_message_metadata(self, conversation_memory):
        """Test message metadata storage"""
        metadata = {"intent": "greeting", "confidence": 0.95}
        await conversation_memory.add_message(
            "user", 
            "Hello", 
            metadata=metadata
        )
        
        history = await conversation_memory.get_history()
        assert history[0].metadata == metadata
    
    @pytest.mark.asyncio
    async def test_history_limit(self, conversation_memory):
        """Test conversation history limit"""
        # Add more messages than the limit
        for i in range(15):
            await conversation_memory.add_message("user", f"Message {i}")
        
        history = await conversation_memory.get_history()
        assert len(history) == 10  # max_history limit
        assert history[0].content == "Message 5"  # Oldest kept message
        assert history[-1].content == "Message 14"  # Newest message
    
    @pytest.mark.asyncio
    async def test_conversation_summary(self, conversation_memory):
        """Test conversation summarization"""
        # Add messages to trigger summarization
        for i in range(6):
            await conversation_memory.add_message("user", f"Query about topic {i}")
            await conversation_memory.add_message("assistant", f"Response about topic {i}")
        
        summary = await conversation_memory.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_clear_history(self, conversation_memory):
        """Test clearing conversation history"""
        await conversation_memory.add_message("user", "Test message")
        history = await conversation_memory.get_history()
        assert len(history) == 1
        
        await conversation_memory.clear_history()
        history = await conversation_memory.get_history()
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_search_messages(self, conversation_memory):
        """Test searching conversation history"""
        await conversation_memory.add_message("user", "Tell me about Python")
        await conversation_memory.add_message("assistant", "Python is a programming language")
        await conversation_memory.add_message("user", "What about JavaScript?")
        
        results = await conversation_memory.search("Python")
        assert len(results) == 2
        assert any("Python" in r.content for r in results)
    
    @pytest.mark.asyncio
    async def test_get_context(self, conversation_memory):
        """Test getting conversation context"""
        await conversation_memory.add_message("user", "Let's talk about AI")
        await conversation_memory.add_message("assistant", "Sure, AI is fascinating")
        
        context = await conversation_memory.get_context(num_messages=2)
        assert isinstance(context, ConversationContext)
        assert len(context.messages) == 2
        assert context.summary is not None


class TestVectorMemory:
    """Test suite for VectorMemory"""
    
    @pytest.fixture
    def vector_memory(self):
        """Create vector memory instance"""
        with patch('aiq.memory.interfaces.EmbeddingModel') as mock_embedding:
            mock_embedding.return_value.embed.return_value = np.random.rand(384).tolist()
            return VectorMemory(
                embedding_model="test-embedding",
                dimension=384,
                index_type="cosine"
            )
    
    @pytest.mark.asyncio
    async def test_store_document(self, vector_memory):
        """Test storing document with embeddings"""
        doc_id = await vector_memory.store_document(
            "Python is a versatile programming language",
            metadata={"category": "programming"}
        )
        
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_memory):
        """Test similarity search"""
        # Store some documents
        await vector_memory.store_document("Python is great for data science")
        await vector_memory.store_document("JavaScript is used for web development")
        await vector_memory.store_document("Machine learning requires mathematics")
        
        # Search for similar documents
        results = await vector_memory.similarity_search(
            "What programming language is good for data analysis?",
            top_k=2
        )
        
        assert len(results) <= 2
        assert all(hasattr(r, 'score') for r in results)
        assert all(0 <= r.score <= 1 for r in results)
    
    @pytest.mark.asyncio
    async def test_threshold_search(self, vector_memory):
        """Test similarity search with threshold"""
        await vector_memory.store_document("AI is transforming industries")
        await vector_memory.store_document("Weather is nice today")
        
        results = await vector_memory.similarity_search(
            "Artificial intelligence applications",
            top_k=5,
            threshold=0.7
        )
        
        # Only highly similar documents should be returned
        assert all(r.score >= 0.7 for r in results)
    
    @pytest.mark.asyncio
    async def test_update_embeddings(self, vector_memory):
        """Test updating document embeddings"""
        doc_id = await vector_memory.store_document("Initial content")
        
        # Update the document
        await vector_memory.update_embeddings(doc_id, "Updated content with more details")
        
        # Search should find updated content
        results = await vector_memory.similarity_search("Updated content")
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_delete_document(self, vector_memory):
        """Test deleting documents from vector store"""
        doc_id = await vector_memory.store_document("Temporary document")
        
        # Verify it exists
        results = await vector_memory.similarity_search("Temporary document")
        assert len(results) > 0
        
        # Delete it
        deleted = await vector_memory.delete(doc_id)
        assert deleted
        
        # Verify it's gone
        results = await vector_memory.similarity_search("Temporary document")
        assert len(results) == 0


class TestPersistentMemory:
    """Test suite for PersistentMemory"""
    
    @pytest.fixture
    def persistent_memory(self):
        """Create persistent memory instance with temp directory"""
        temp_dir = tempfile.mkdtemp()
        return PersistentMemory(
            storage_path=temp_dir,
            format="json",
            compression=False
        )
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, persistent_memory):
        """Test storing and retrieving data"""
        data = {"user": "john", "preferences": {"theme": "dark"}}
        await persistent_memory.store("user_123", data)
        
        retrieved = await persistent_memory.retrieve("user_123")
        assert retrieved == data
    
    @pytest.mark.asyncio
    async def test_metadata_storage(self, persistent_memory):
        """Test metadata storage with items"""
        data = "Important information"
        metadata = {"timestamp": datetime.now().isoformat(), "priority": "high"}
        
        await persistent_memory.store("item_456", data, metadata=metadata)
        
        item = await persistent_memory.retrieve_with_metadata("item_456")
        assert item.content == data
        assert item.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_compression(self):
        """Test storage with compression"""
        temp_dir = tempfile.mkdtemp()
        memory = PersistentMemory(
            storage_path=temp_dir,
            format="json",
            compression=True
        )
        
        large_data = {"data": ["x" * 1000 for _ in range(100)]}
        await memory.store("large_item", large_data)
        
        retrieved = await memory.retrieve("large_item")
        assert retrieved == large_data
    
    @pytest.mark.asyncio
    async def test_different_formats(self):
        """Test different storage formats"""
        temp_dir = tempfile.mkdtemp()
        
        # Test JSON format
        json_memory = PersistentMemory(storage_path=temp_dir, format="json")
        await json_memory.store("json_item", {"key": "value"})
        
        # Test pickle format (if implemented)
        pickle_memory = PersistentMemory(storage_path=temp_dir, format="pickle")
        await pickle_memory.store("pickle_item", {"key": "value"})
        
        json_result = await json_memory.retrieve("json_item")
        pickle_result = await pickle_memory.retrieve("pickle_item")
        
        assert json_result == {"key": "value"}
        assert pickle_result == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_list_items(self, persistent_memory):
        """Test listing all stored items"""
        await persistent_memory.store("item1", "data1")
        await persistent_memory.store("item2", "data2")
        await persistent_memory.store("item3", "data3")
        
        items = await persistent_memory.list_items()
        assert len(items) == 3
        assert "item1" in items
        assert "item2" in items
        assert "item3" in items


class TestHierarchicalMemory:
    """Test suite for HierarchicalMemory"""
    
    @pytest.fixture
    def hierarchical_memory(self):
        """Create hierarchical memory instance"""
        return HierarchicalMemory(levels=[
            {"name": "hot", "ttl": 60, "max_size": 10},
            {"name": "warm", "ttl": 300, "max_size": 50},
            {"name": "cold", "ttl": None, "max_size": None}
        ])
    
    @pytest.mark.asyncio
    async def test_store_retrieval_promotion(self, hierarchical_memory):
        """Test item storage and promotion on retrieval"""
        await hierarchical_memory.store("key1", "value1")
        
        # First retrieval
        value = await hierarchical_memory.retrieve("key1")
        assert value == "value1"
        
        # Check item is promoted to hot cache
        stats = await hierarchical_memory.get_level_stats()
        assert stats["hot"]["count"] > 0
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, hierarchical_memory):
        """Test TTL expiration in cache levels"""
        # Store with short TTL
        memory = HierarchicalMemory(levels=[
            {"name": "hot", "ttl": 0.1, "max_size": 10},  # 0.1 second TTL
            {"name": "cold", "ttl": None, "max_size": None}
        ])
        
        await memory.store("expire_test", "value")
        value = await memory.retrieve("expire_test")
        assert value == "value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should still be available in cold storage
        value = await memory.retrieve("expire_test")
        assert value == "value"
    
    @pytest.mark.asyncio
    async def test_level_overflow(self, hierarchical_memory):
        """Test overflow from one level to another"""
        # Fill hot cache beyond capacity
        for i in range(15):  # More than max_size of 10
            await hierarchical_memory.store(f"key{i}", f"value{i}")
        
        stats = await hierarchical_memory.get_level_stats()
        assert stats["hot"]["count"] <= 10
        assert stats["warm"]["count"] > 0
    
    @pytest.mark.asyncio
    async def test_search_across_levels(self, hierarchical_memory):
        """Test searching across all hierarchy levels"""
        await hierarchical_memory.store("hot_item", "hot data")
        await hierarchical_memory.store("warm_item", "warm data")
        await hierarchical_memory.store("cold_item", "cold data")
        
        # Force some items to different levels
        for i in range(12):
            await hierarchical_memory.store(f"filler{i}", f"data{i}")
        
        results = await hierarchical_memory.search("data")
        assert len(results) > 3  # Should find items across levels


class TestCachedMemory:
    """Test suite for CachedMemory"""
    
    @pytest.fixture
    def cached_memory(self):
        """Create cached memory instance"""
        backend = Mock(spec=MemoryInterface)
        backend.retrieve = AsyncMock(return_value="backend_value")
        backend.store = AsyncMock()
        backend.search = AsyncMock(return_value=[])
        
        return CachedMemory(
            backend=backend,
            cache_size=100,
            ttl=3600
        )
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, cached_memory):
        """Test cache hit scenario"""
        # First call - goes to backend
        value1 = await cached_memory.retrieve("test_key")
        assert value1 == "backend_value"
        cached_memory.backend.retrieve.assert_called_once()
        
        # Second call - should hit cache
        cached_memory.backend.retrieve.reset_mock()
        value2 = await cached_memory.retrieve("test_key")
        assert value2 == "backend_value"
        cached_memory.backend.retrieve.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cached_memory):
        """Test cache miss scenario"""
        value = await cached_memory.retrieve("new_key")
        assert value == "backend_value"
        cached_memory.backend.retrieve.assert_called_once_with("new_key")
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cached_memory):
        """Test cache invalidation on update"""
        # Cache a value
        await cached_memory.retrieve("test_key")
        
        # Update the value
        await cached_memory.store("test_key", "new_value")
        
        # Cache should be invalidated
        cached_memory.backend.retrieve.reset_mock()
        cached_memory.backend.retrieve.return_value = "new_value"
        
        value = await cached_memory.retrieve("test_key")
        assert value == "new_value"
        cached_memory.backend.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL expiration"""
        backend = Mock(spec=MemoryInterface)
        backend.retrieve = AsyncMock(side_effect=["value1", "value2"])
        
        # Very short TTL for testing
        memory = CachedMemory(backend=backend, cache_size=10, ttl=0.1)
        
        # First retrieval
        value1 = await memory.retrieve("test_key")
        assert value1 == "value1"
        
        # Wait for TTL expiration
        await asyncio.sleep(0.2)
        
        # Should fetch from backend again
        value2 = await memory.retrieve("test_key")
        assert value2 == "value2"
        assert backend.retrieve.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_size_limit(self):
        """Test cache size limitation"""
        backend = Mock(spec=MemoryInterface)
        backend.retrieve = AsyncMock(side_effect=lambda k: f"value_{k}")
        
        memory = CachedMemory(backend=backend, cache_size=3, ttl=3600)
        
        # Fill cache beyond limit
        for i in range(5):
            await memory.retrieve(f"key{i}")
        
        # Cache should only contain last 3 items
        cache_stats = await memory.get_cache_stats()
        assert cache_stats["size"] <= 3
        assert cache_stats["hit_rate"] >= 0


class TestMemoryIntegration:
    """Integration tests for memory systems"""
    
    @pytest.mark.asyncio
    async def test_conversation_with_vector_memory(self):
        """Test integration of conversation and vector memory"""
        conversation = ConversationMemory(max_history=10)
        vector = VectorMemory(dimension=384)
        
        # Add conversation messages
        await conversation.add_message("user", "Tell me about quantum computing")
        await conversation.add_message("assistant", "Quantum computing uses quantum mechanics")
        
        # Store conversation in vector memory for semantic search
        history = await conversation.get_history()
        for msg in history:
            await vector.store_document(
                f"{msg.role}: {msg.content}",
                metadata={"role": msg.role, "timestamp": msg.timestamp}
            )
        
        # Search for related conversations
        results = await vector.similarity_search("quantum physics")
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_persistent_conversation_memory(self):
        """Test persisting conversation memory"""
        temp_dir = tempfile.mkdtemp()
        
        # Create conversation with persistence
        conversation = ConversationMemory(max_history=10)
        persistent = PersistentMemory(storage_path=temp_dir)
        
        # Add messages
        await conversation.add_message("user", "Hello")
        await conversation.add_message("assistant", "Hi there!")
        
        # Save conversation
        history = await conversation.get_history()
        await persistent.store("conversation_123", {
            "messages": [msg.to_dict() for msg in history],
            "metadata": {"session_id": "123"}
        })
        
        # Restore conversation
        saved = await persistent.retrieve("conversation_123")
        assert len(saved["messages"]) == 2
        assert saved["messages"][0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_hierarchical_vector_memory(self):
        """Test hierarchical caching with vector memory"""
        vector_backend = VectorMemory(dimension=384)
        hierarchical = HierarchicalMemory(
            levels=[
                {"name": "hot", "ttl": 60, "max_size": 5},
                {"name": "cold", "ttl": None, "max_size": None}
            ],
            backend=vector_backend
        )
        
        # Store documents
        for i in range(10):
            await hierarchical.store_document(
                f"doc{i}",
                f"Document {i} about AI and machine learning"
            )
        
        # Search should work across levels
        results = await hierarchical.similarity_search("artificial intelligence")
        assert len(results) > 0