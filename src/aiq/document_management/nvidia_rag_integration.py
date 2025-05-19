"""NVIDIA RAG integration for document management with advanced features."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime
import json

from ..tool.nvidia_rag import NVIDIARAGToolConfig, nvidia_rag_tool
from ..retriever.interface import AIQRetriever
from ..retriever.models import RetrieverOutput, RetrieverResult
from ..builder.builder import Builder
from ..data_models.config import WorkflowConfig
from .models import Document, DocumentVector, ResearchOutput
from .embedder import NVIDIADocumentEmbedder
from .jena_integration import JenaTripleStore


logger = logging.getLogger(__name__)


class NVIDIARAGDocumentManager:
    """Enhanced document manager with NVIDIA RAG integration."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None, builder: Optional[Builder] = None):
        """Initialize NVIDIA RAG document manager."""
        self.config = config
        self.builder = builder
        
        # Initialize components
        self.embedder = NVIDIADocumentEmbedder(config)
        self.jena_store = JenaTripleStore(config)
        
        # NVIDIA RAG configuration
        self.rag_config = NVIDIARAGToolConfig(
            base_url=self._get_rag_base_url(),
            timeout=120,
            top_k=10,
            collection_name="aiq_documents"
        )
        
        # Document collections
        self.collections = {
            "documents": "aiq_documents",
            "research": "aiq_research_outputs",
            "web_content": "aiq_web_content",
            "hybrid": "aiq_hybrid_collection"
        }
        
        # Initialize chunking and processing strategies
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.max_context_length = 8192
    
    def _get_rag_base_url(self) -> str:
        """Get NVIDIA RAG base URL from config or environment."""
        if self.config and hasattr(self.config, 'nvidia_rag_url'):
            return self.config.nvidia_rag_url
        return "http://localhost:8080/v1/retrieval"
    
    async def ingest_document_with_rag(self, document: Document) -> Dict[str, Any]:
        """Ingest document using NVIDIA RAG pipeline."""
        try:
            # 1. Generate embeddings with NVIDIA embedder
            embeddings = self.embedder.embed_document(document)
            document.vector = embeddings
            
            # 2. Smart chunking with context awareness
            chunks = await self._intelligent_chunking(document)
            
            # 3. Process each chunk with RAG pipeline
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "document_id": str(document.id),
                    "chunk_id": f"{document.id}_chunk_{i}",
                    "document_type": document.type.value,
                    "title": document.metadata.title,
                    "creator": document.metadata.creator,
                    "date": document.metadata.date.isoformat() if document.metadata.date else None,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                
                # Embed chunk
                chunk_embedding = self.embedder.embed_text(chunk["content"])
                
                # Store in NVIDIA RAG
                rag_response = await self._store_in_nvidia_rag(
                    content=chunk["content"],
                    embedding=chunk_embedding,
                    metadata=chunk_metadata,
                    collection=self.collections["documents"]
                )
                
                processed_chunks.append({
                    "chunk_id": chunk_metadata["chunk_id"],
                    "rag_response": rag_response,
                    "metadata": chunk_metadata
                })
            
            # 4. Store complete document in Jena with OWL
            jena_uri = self.jena_store.store_document(document)
            document.jena_uri = jena_uri
            
            # 5. Create hybrid index for cross-collection search
            await self._create_hybrid_index(document, processed_chunks)
            
            return {
                "document_id": str(document.id),
                "chunks_processed": len(processed_chunks),
                "jena_uri": jena_uri,
                "embeddings_generated": True,
                "rag_collections": [self.collections["documents"], self.collections["hybrid"]]
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document with RAG: {e}")
            raise
    
    async def _intelligent_chunking(self, document: Document) -> List[Dict[str, Any]]:
        """Perform intelligent chunking with semantic boundaries."""
        chunks = []
        content = document.content or ""
        
        # Use different chunking strategies based on document type
        if document.type.value == "pdf":
            chunks = self._chunk_pdf_content(content)
        elif document.type.value == "excel":
            chunks = self._chunk_structured_data(content)
        elif document.type.value == "web_page":
            chunks = self._chunk_html_content(content)
        else:
            chunks = self._chunk_text_content(content)
        
        # Add semantic context to chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            context = self._generate_chunk_context(chunk, i, chunks)
            enhanced_chunks.append({
                "content": chunk,
                "context": context,
                "semantic_type": self._detect_semantic_type(chunk)
            })
        
        return enhanced_chunks
    
    def _chunk_text_content(self, content: str) -> List[str]:
        """Chunk text content with overlap."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def _chunk_pdf_content(self, content: str) -> List[str]:
        """Chunk PDF content by sections and paragraphs."""
        # Split by common PDF section markers
        sections = content.split("\n\n")
        chunks = []
        
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) > self.chunk_size * 4:  # Approximate token count
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = section
            else:
                current_chunk += "\n\n" + section
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_structured_data(self, content: str) -> List[str]:
        """Chunk structured data (Excel, CSV) by rows or semantic groups."""
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        
        for line in lines:
            current_chunk.append(line)
            if len("\n".join(current_chunk)) > self.chunk_size * 4:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks
    
    def _chunk_html_content(self, content: str) -> List[str]:
        """Chunk HTML content by semantic elements."""
        # Simple implementation - in production use proper HTML parser
        return self._chunk_text_content(content)
    
    def _generate_chunk_context(self, chunk: str, index: int, all_chunks: List[str]) -> str:
        """Generate context for a chunk based on surrounding chunks."""
        context_parts = []
        
        # Add previous chunk summary if available
        if index > 0:
            prev_summary = self._summarize_chunk(all_chunks[index - 1])
            context_parts.append(f"Previous context: {prev_summary}")
        
        # Add next chunk preview if available
        if index < len(all_chunks) - 1:
            next_preview = all_chunks[index + 1][:100]
            context_parts.append(f"Following content: {next_preview}...")
        
        return " | ".join(context_parts)
    
    def _summarize_chunk(self, chunk: str) -> str:
        """Generate a brief summary of a chunk."""
        # Simple implementation - in production use LLM for summarization
        return chunk[:100] + "..."
    
    def _detect_semantic_type(self, chunk: str) -> str:
        """Detect the semantic type of content."""
        # Simple heuristic - in production use ML classifier
        if any(keyword in chunk.lower() for keyword in ["table", "figure", "chart"]):
            return "visual"
        elif any(keyword in chunk.lower() for keyword in ["conclusion", "summary", "abstract"]):
            return "summary"
        elif any(keyword in chunk.lower() for keyword in ["method", "procedure", "algorithm"]):
            return "technical"
        else:
            return "general"
    
    async def _store_in_nvidia_rag(self, content: str, embedding: List[float], 
                                  metadata: Dict[str, Any], collection: str) -> Dict[str, Any]:
        """Store content in NVIDIA RAG system."""
        async with httpx.AsyncClient(timeout=self.rag_config.timeout) as client:
            payload = {
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "collection": collection
            }
            
            try:
                response = await client.post(
                    f"{self.rag_config.base_url}/ingest",
                    json=payload
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error storing in NVIDIA RAG: {e}")
                raise
    
    async def _create_hybrid_index(self, document: Document, chunks: List[Dict[str, Any]]):
        """Create hybrid index for multi-modal search."""
        # Combine document-level and chunk-level information
        hybrid_entry = {
            "document_id": str(document.id),
            "title": document.metadata.title,
            "full_embedding": document.vector.embeddings if document.vector else None,
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "metadata": chunk["metadata"]
                }
                for chunk in chunks
            ],
            "jena_uri": document.jena_uri,
            "created_at": datetime.utcnow().isoformat()
        }
        
        async with httpx.AsyncClient(timeout=self.rag_config.timeout) as client:
            try:
                response = await client.post(
                    f"{self.rag_config.base_url}/hybrid",
                    json={
                        "entry": hybrid_entry,
                        "collection": self.collections["hybrid"]
                    }
                )
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Error creating hybrid index: {e}")
    
    async def advanced_search(self, query: str, search_type: str = "hybrid",
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform advanced search using NVIDIA RAG."""
        try:
            # 1. Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # 2. Determine search strategy
            if search_type == "hybrid":
                results = await self._hybrid_search(query, query_embedding, filters)
            elif search_type == "semantic":
                results = await self._semantic_search(query, query_embedding, filters)
            elif search_type == "keyword":
                results = await self._keyword_search(query, filters)
            else:
                results = await self._multi_collection_search(query, query_embedding, filters)
            
            # 3. Post-process results with reranking
            reranked_results = await self._rerank_results(results, query)
            
            # 4. Enrich with metadata from Jena
            enriched_results = await self._enrich_with_metadata(reranked_results)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return []
    
    async def _hybrid_search(self, query: str, query_embedding: List[float],
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        async with httpx.AsyncClient(timeout=self.rag_config.timeout) as client:
            payload = {
                "query": query,
                "embedding": query_embedding,
                "collection": self.collections["hybrid"],
                "top_k": self.rag_config.top_k,
                "filters": filters or {},
                "search_mode": "hybrid"
            }
            
            response = await client.post(
                f"{self.rag_config.base_url}/search/hybrid",
                json=payload
            )
            response.raise_for_status()
            return response.json()["results"]
    
    async def _semantic_search(self, query: str, query_embedding: List[float],
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform pure semantic search."""
        # Use NVIDIA RAG tool for semantic search
        tool_response = await nvidia_rag_tool(self.rag_config, self.builder)
        
        # Convert tool response to our format
        results = []
        async for function_info in tool_response:
            result = await function_info.fn(query)
            results.append({"content": result, "type": "semantic"})
        
        return results
    
    async def _keyword_search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        # This would integrate with a text search engine
        return []
    
    async def _multi_collection_search(self, query: str, query_embedding: List[float],
                                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search across multiple collections."""
        tasks = []
        
        for collection_name in self.collections.values():
            task = self._search_collection(query, query_embedding, collection_name, filters)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Merge and deduplicate results
        merged_results = []
        seen_ids = set()
        
        for collection_results in results:
            for result in collection_results:
                doc_id = result.get("document_id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_results.append(result)
        
        return merged_results
    
    async def _search_collection(self, query: str, query_embedding: List[float],
                               collection: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search a specific collection."""
        async with httpx.AsyncClient(timeout=self.rag_config.timeout) as client:
            payload = {
                "query": query,
                "embedding": query_embedding,
                "collection": collection,
                "top_k": self.rag_config.top_k,
                "filters": filters or {}
            }
            
            try:
                response = await client.post(
                    f"{self.rag_config.base_url}/search",
                    json=payload
                )
                response.raise_for_status()
                return response.json()["results"]
            except Exception as e:
                logger.error(f"Error searching collection {collection}: {e}")
                return []
    
    async def _rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank results using NVIDIA reranker."""
        if not results:
            return results
        
        async with httpx.AsyncClient(timeout=self.rag_config.timeout) as client:
            payload = {
                "query": query,
                "documents": [
                    {
                        "id": r.get("document_id", ""),
                        "content": r.get("content", ""),
                        "metadata": r.get("metadata", {})
                    }
                    for r in results
                ],
                "model": "nvidia-rerank-qa-mistral-4b"
            }
            
            try:
                response = await client.post(
                    f"{self.rag_config.base_url}/rerank",
                    json=payload
                )
                response.raise_for_status()
                return response.json()["reranked_results"]
            except Exception as e:
                logger.error(f"Error reranking results: {e}")
                return results
    
    async def _enrich_with_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich results with metadata from Jena."""
        enriched_results = []
        
        for result in results:
            doc_id = result.get("document_id")
            if doc_id:
                metadata = self.jena_store._get_document_metadata(doc_id)
                result["dublin_core_metadata"] = metadata
            
            enriched_results.append(result)
        
        return enriched_results
    
    async def generate_rag_based_summary(self, document_id: str, prompt: Optional[str] = None) -> str:
        """Generate a summary using NVIDIA RAG and LLM."""
        # Retrieve document chunks
        chunks = await self._get_document_chunks(document_id)
        
        # Create context from chunks
        context = "\n\n".join([chunk["content"] for chunk in chunks[:5]])  # Top 5 chunks
        
        # Use LLM to generate summary
        summary_prompt = prompt or f"Please provide a comprehensive summary of the following document:\n\n{context}"
        
        # This would integrate with NVIDIA LLM service
        summary = f"Summary for document {document_id}: [Generated summary would go here]"
        
        # Store summary as research output
        research_output = ResearchOutput(
            source_document_id=document_id,
            agent_name="nvidia_rag_summarizer",
            output_type="summary",
            content={"summary": summary, "context_chunks": len(chunks)}
        )
        
        self.jena_store.store_research_output(research_output)
        
        return summary
    
    async def _get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document."""
        filters = {"document_id": document_id}
        return await self._multi_collection_search("", [], filters)