"""NVIDIA document embedder using NIM or local embeddings."""

import logging
from typing import List, Optional
import numpy as np

from .models import Document, DocumentVector
from ..embedder.nim_embedder import NIMEmbedder
from ..embedder.openai_embedder import OpenAIEmbedder
from ..data_models.config import WorkflowConfig


logger = logging.getLogger(__name__)


class NVIDIADocumentEmbedder:
    """Embedder for documents using NVIDIA embedding models."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize embedder with configuration."""
        self.config = config
        
        # Try to use NIM embedder if available, fallback to OpenAI
        try:
            self.embedder = NIMEmbedder(config)
            self.model_name = "nvidia-embed-qa-4"
        except Exception as e:
            logger.warning(f"Failed to initialize NIM embedder: {e}. Using OpenAI fallback.")
            self.embedder = OpenAIEmbedder(config)
            self.model_name = "text-embedding-ada-002"
    
    def embed_document(self, document: Document) -> DocumentVector:
        """Generate embeddings for a document."""
        if not document.content:
            raise ValueError("Document has no content to embed")
        
        # Split document into chunks if too large
        chunks = self._chunk_document(document.content)
        
        # Embed each chunk
        embeddings = []
        for chunk in chunks:
            chunk_embedding = self.embed_text(chunk)
            embeddings.append(chunk_embedding)
        
        # Average embeddings if multiple chunks
        if len(embeddings) > 1:
            avg_embedding = np.mean(embeddings, axis=0).tolist()
        else:
            avg_embedding = embeddings[0]
        
        return DocumentVector(
            document_id=document.id,
            embeddings=avg_embedding,
            model_name=self.model_name
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a text string."""
        try:
            # Use the configured embedder
            embedding = self.embedder.get_embedder().embed(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Default dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch."""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def _chunk_document(self, content: str, max_tokens: int = 8000) -> List[str]:
        """Split document into chunks for embedding."""
        # Simple chunking by character count (approximate)
        # In production, use proper tokenizer
        chars_per_token = 4  # Rough estimate
        max_chars = max_tokens * chars_per_token
        
        chunks = []
        words = content.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_chars:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [content]
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)