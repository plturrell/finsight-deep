"""Document manager for handling upload, storage, and processing."""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import PyPDF2
import pandas as pd
import requests
from datetime import datetime
import json
import logging

from .models import Document, DocumentType, DocumentStatus, DublinCoreMetadata, ResearchOutput
from .embedder import NVIDIADocumentEmbedder
from .jena_integration import JenaTripleStore
from .metadata_catalog import MetadataCataloguer
from ..data_models.config import WorkflowConfig


logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document lifecycle including upload, processing, and storage."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize document manager with configuration."""
        self.config = config
        self.storage_path = Path(os.getenv("DOCUMENT_STORAGE_PATH", "/tmp/aiq_documents"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedder = NVIDIADocumentEmbedder(config)
        self.jena_store = JenaTripleStore(config)
        self.cataloguer = MetadataCataloguer()
        
        # Document registry
        self.documents: Dict[uuid.UUID, Document] = {}
        
    def upload_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Upload a document from file path."""
        doc_type = self._detect_document_type(file_path)
        
        # Create document instance
        doc = Document(
            type=doc_type,
            file_path=file_path,
            title=os.path.basename(file_path),
            status=DocumentStatus.UPLOADED
        )
        
        # Extract content based on type
        if doc_type == DocumentType.PDF:
            doc.content = self._extract_pdf_content(file_path)
        elif doc_type == DocumentType.EXCEL:
            doc.content = self._extract_excel_content(file_path)
        elif doc_type == DocumentType.TEXT:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc.content = f.read()
        
        # Generate metadata
        dublin_core = self.cataloguer.generate_metadata(doc, metadata)
        doc.metadata = dublin_core
        
        # Store document
        self.documents[doc.id] = doc
        self._store_document_file(doc, file_path)
        
        # Process document asynchronously
        self._process_document(doc)
        
        return doc
    
    def upload_web_content(self, url: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Upload web content."""
        doc = Document(
            type=DocumentType.WEB_PAGE,
            url=url,
            content=content,
            title=metadata.get('title', url) if metadata else url,
            status=DocumentStatus.UPLOADED
        )
        
        # Generate metadata
        dublin_core = self.cataloguer.generate_metadata(doc, metadata)
        doc.metadata = dublin_core
        
        # Store document
        self.documents[doc.id] = doc
        
        # Process document
        self._process_document(doc)
        
        return doc
    
    def store_research_output(self, agent_name: str, output_data: Dict[str, Any],
                           source_doc_id: Optional[uuid.UUID] = None) -> ResearchOutput:
        """Store research output from AI agent."""
        research_output = ResearchOutput(
            agent_name=agent_name,
            output_type=output_data.get('type', 'text'),
            content=output_data,
            source_document_id=source_doc_id
        )
        
        # Generate metadata for research output
        metadata = {
            'title': f"Research Output - {agent_name}",
            'creator': agent_name,
            'type': 'research_output',
            'date': datetime.utcnow()
        }
        dublin_core = self.cataloguer.generate_metadata_from_dict(metadata)
        research_output.metadata = dublin_core
        
        # Generate embeddings
        if 'content' in output_data and isinstance(output_data['content'], str):
            embeddings = self.embedder.embed_text(output_data['content'])
            research_output.embeddings = embeddings
        
        # Store in Jena
        self.jena_store.store_research_output(research_output)
        
        return research_output
    
    def _process_document(self, doc: Document):
        """Process document with embeddings and Jena storage."""
        try:
            doc.update_status(DocumentStatus.PROCESSING)
            
            # Generate embeddings
            if doc.content:
                embeddings = self.embedder.embed_document(doc)
                doc.vector = embeddings
                doc.update_status(DocumentStatus.EMBEDDED)
            
            # Store in Jena with both OWL and vector
            jena_uri = self.jena_store.store_document(doc)
            doc.jena_uri = jena_uri
            doc.update_status(DocumentStatus.INDEXED)
            
        except Exception as e:
            logger.error(f"Error processing document {doc.id}: {e}")
            doc.update_status(DocumentStatus.FAILED)
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext in ['.xlsx', '.xls']:
            return DocumentType.EXCEL
        elif ext in ['.txt', '.md']:
            return DocumentType.TEXT
        elif ext == '.json':
            return DocumentType.JSON
        else:
            return DocumentType.TEXT
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """Extract text content from PDF."""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
        return text
    
    def _extract_excel_content(self, file_path: str) -> str:
        """Extract content from Excel file."""
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error extracting Excel content: {e}")
            return ""
    
    def _store_document_file(self, doc: Document, source_path: str):
        """Store document file in storage directory."""
        dest_path = self.storage_path / f"{doc.id}{Path(source_path).suffix}"
        try:
            Path(source_path).rename(dest_path)
            doc.file_path = str(dest_path)
        except Exception as e:
            logger.error(f"Error storing document file: {e}")
    
    def get_document(self, doc_id: uuid.UUID) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def search_documents(self, query: str, limit: int = 10) -> List[Document]:
        """Search documents using vector similarity."""
        query_embedding = self.embedder.embed_text(query)
        results = self.jena_store.vector_search(query_embedding, limit)
        
        # Convert results to documents
        documents = []
        for result in results:
            doc_id = result.get('document_id')
            if doc_id and doc_id in self.documents:
                documents.append(self.documents[doc_id])
        
        return documents