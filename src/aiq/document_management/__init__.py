"""Document Management System for AIQToolkit.

Provides comprehensive document upload, storage, and management with 
vector embeddings, Jena integration, and Dublin Core metadata support.
"""

from .models import (
    Document,
    DocumentStatus,
    DocumentType,
    DublinCoreMetadata,
    DocumentVector,
    ResearchOutput
)
from .manager import DocumentManager
from .embedder import NVIDIADocumentEmbedder
from .crawler import ResearchWebCrawler
from .jena_integration import JenaTripleStore
from .metadata_catalog import MetadataCataloguer

__all__ = [
    "Document",
    "DocumentStatus",
    "DocumentType",
    "DublinCoreMetadata",
    "DocumentVector",
    "ResearchOutput",
    "DocumentManager",
    "NVIDIADocumentEmbedder",
    "ResearchWebCrawler",
    "JenaTripleStore",
    "MetadataCataloguer"
]