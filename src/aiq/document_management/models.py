"""Data models for the document management system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4


class DocumentType(Enum):
    """Types of documents supported by the system."""
    PDF = "pdf"
    EXCEL = "excel"
    WEB_PAGE = "web_page"
    RESEARCH_OUTPUT = "research_output"
    TEXT = "text"
    JSON = "json"


class DocumentStatus(Enum):
    """Status of document processing."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass
class DublinCoreMetadata:
    """Dublin Core metadata schema."""
    title: str
    creator: Optional[str] = None
    subject: Optional[List[str]] = None
    description: Optional[str] = None
    publisher: Optional[str] = None
    contributor: Optional[List[str]] = None
    date: Optional[datetime] = None
    type: Optional[str] = None
    format: Optional[str] = None
    identifier: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = "en"
    relation: Optional[List[str]] = None
    coverage: Optional[str] = None
    rights: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }
    
    def to_owl(self) -> str:
        """Convert metadata to OWL/RDF format."""
        # This would generate proper OWL/RDF representation
        owl_template = f"""
        <rdf:Description rdf:about="{self.identifier or uuid4()}">
            <dc:title>{self.title}</dc:title>
            {"".join(f"<dc:creator>{c}</dc:creator>" for c in [self.creator] if self.creator)}
            {"".join(f"<dc:subject>{s}</dc:subject>" for s in self.subject or [])}
            {f"<dc:description>{self.description}</dc:description>" if self.description else ""}
            {f"<dc:date>{self.date.isoformat()}</dc:date>" if self.date else ""}
            {f"<dc:type>{self.type}</dc:type>" if self.type else ""}
            {f"<dc:format>{self.format}</dc:format>" if self.format else ""}
            {f"<dc:language>{self.language}</dc:language>"}
        </rdf:Description>
        """
        return owl_template


@dataclass
class DocumentVector:
    """Vector representation of a document."""
    document_id: UUID
    embeddings: List[float]
    model_name: str = "nvidia-embed-qa-4"
    dimensions: int = field(init=False)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        self.dimensions = len(self.embeddings)


@dataclass
class Document:
    """Core document model."""
    id: UUID = field(default_factory=uuid4)
    type: DocumentType = DocumentType.TEXT
    title: str = ""
    content: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    metadata: DublinCoreMetadata = field(default_factory=lambda: DublinCoreMetadata(title="Untitled"))
    vector: Optional[DocumentVector] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    jena_uri: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    
    def update_status(self, status: DocumentStatus):
        """Update document status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()


@dataclass
class ResearchOutput:
    """Research output from AI/ML processing."""
    id: UUID = field(default_factory=uuid4)
    source_document_id: Optional[UUID] = None
    agent_name: str = ""
    output_type: str = "text"
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: DublinCoreMetadata = field(default_factory=lambda: DublinCoreMetadata(title="Research Output"))
    created_at: datetime = field(default_factory=datetime.utcnow)
    embeddings: Optional[List[float]] = None