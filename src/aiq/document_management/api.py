"""API endpoints for document management system."""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

from .manager import DocumentManager
from .nvidia_rag_integration import NVIDIARAGDocumentManager
from .crawler import ResearchWebCrawler
from .models import Document, DocumentStatus, DocumentType, DublinCoreMetadata
from .auth import get_current_user, User, Permissions, auth_manager, login, refresh, logout
from .validation import validate_document_upload, validate_search_query, DocumentUploadRequest
from ..data_models.config import WorkflowConfig
from ..builder.builder import Builder


# Create FastAPI app
app = FastAPI(
    title="AIQ Document Management API", 
    version="1.0.0",
    description="Production-ready document management system with NVIDIA RAG integration"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
document_manager = None
rag_manager = None
web_crawler = None


def get_document_manager() -> DocumentManager:
    """Get document manager instance."""
    global document_manager
    if not document_manager:
        document_manager = DocumentManager()
    return document_manager


def get_rag_manager() -> NVIDIARAGDocumentManager:
    """Get NVIDIA RAG manager instance."""
    global rag_manager
    if not rag_manager:
        rag_manager = NVIDIARAGDocumentManager()
    return rag_manager


def get_web_crawler() -> ResearchWebCrawler:
    """Get web crawler instance."""
    global web_crawler
    if not web_crawler:
        web_crawler = ResearchWebCrawler()
    return web_crawler


@app.post("/api/documents/upload", dependencies=[Depends(get_current_user)])
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    creator: Optional[str] = None,
    description: Optional[str] = None,
    subject: Optional[List[str]] = Query(None),
    doc_manager: DocumentManager = Depends(get_document_manager),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload a document and process it."""
    try:
        # Save uploaded file temporarily
        temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
        
        with temp_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "creator": creator,
            "description": description,
            "subject": subject
        }
        
        # Upload document
        document = doc_manager.upload_document(str(temp_path), metadata)
        
        # Clean up temp file
        temp_path.unlink()
        
        return {
            "id": str(document.id),
            "status": document.status.value,
            "title": document.metadata.title,
            "type": document.type.value,
            "created_at": document.created_at.isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload-batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Upload multiple documents at once."""
    results = []
    errors = []
    
    for file in files:
        try:
            # Save uploaded file temporarily
            temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
            
            with temp_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Upload document
            document = doc_manager.upload_document(
                str(temp_path), 
                {"title": file.filename}
            )
            
            # Clean up temp file
            temp_path.unlink()
            
            results.append({
                "id": str(document.id),
                "filename": file.filename,
                "status": "success"
            })
        
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "uploaded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }


@app.post("/api/documents/crawl")
async def crawl_website(
    url: str,
    max_depth: Optional[int] = 3,
    max_pages: Optional[int] = 100,
    keywords: Optional[List[str]] = Query(None),
    crawler: ResearchWebCrawler = Depends(get_web_crawler),
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Crawl a website and collect research documents."""
    try:
        # Configure crawler
        crawler.max_depth = max_depth
        crawler.max_pages = max_pages
        
        # Perform crawl
        documents = await crawler.crawl_site(url, keywords)
        
        # Upload collected documents
        uploaded_docs = []
        for doc in documents:
            # Convert to document manager format
            metadata = doc.metadata.to_dict()
            uploaded = doc_manager.upload_web_content(
                doc.url,
                doc.content,
                metadata
            )
            uploaded_docs.append(uploaded)
        
        return {
            "pages_crawled": len(crawler.visited_urls),
            "documents_collected": len(documents),
            "documents_uploaded": len(uploaded_docs),
            "document_ids": [str(doc.id) for doc in uploaded_docs]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}")
async def get_document(
    document_id: str,
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Get document details by ID."""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = doc_manager.get_document(doc_uuid)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": str(document.id),
            "type": document.type.value,
            "title": document.metadata.title,
            "status": document.status.value,
            "metadata": document.metadata.to_dict(),
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "jena_uri": document.jena_uri,
            "has_embeddings": document.vector is not None
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/search")
async def search_documents(
    query: str,
    limit: Optional[int] = 10,
    search_type: Optional[str] = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    rag_manager: NVIDIARAGDocumentManager = Depends(get_rag_manager)
) -> Dict[str, Any]:
    """Search documents using advanced NVIDIA RAG."""
    try:
        results = await rag_manager.advanced_search(
            query=query,
            search_type=search_type,
            filters=filters
        )
        
        # Limit results
        results = results[:limit]
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": search_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/{document_id}/ingest-rag")
async def ingest_with_rag(
    document_id: str,
    doc_manager: DocumentManager = Depends(get_document_manager),
    rag_manager: NVIDIARAGDocumentManager = Depends(get_rag_manager)
) -> Dict[str, Any]:
    """Ingest document into NVIDIA RAG system."""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = doc_manager.get_document(doc_uuid)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Ingest with RAG
        result = await rag_manager.ingest_document_with_rag(document)
        
        return result
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/store")
async def store_research_output(
    agent_name: str,
    output_data: Dict[str, Any],
    source_document_id: Optional[str] = None,
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Store research output from AI agents."""
    try:
        source_uuid = None
        if source_document_id:
            source_uuid = uuid.UUID(source_document_id)
        
        research_output = doc_manager.store_research_output(
            agent_name=agent_name,
            output_data=output_data,
            source_doc_id=source_uuid
        )
        
        return {
            "id": str(research_output.id),
            "agent_name": research_output.agent_name,
            "output_type": research_output.output_type,
            "created_at": research_output.created_at.isoformat(),
            "has_embeddings": research_output.embeddings is not None
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source document ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}/summary")
async def generate_summary(
    document_id: str,
    prompt: Optional[str] = None,
    rag_manager: NVIDIARAGDocumentManager = Depends(get_rag_manager)
) -> Dict[str, Any]:
    """Generate a summary for a document using NVIDIA RAG."""
    try:
        summary = await rag_manager.generate_rag_based_summary(
            document_id=document_id,
            prompt=prompt
        )
        
        return {
            "document_id": document_id,
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metadata/dublin-core/{document_id}")
async def get_dublin_core_metadata(
    document_id: str,
    format: Optional[str] = "json",
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Any:
    """Get Dublin Core metadata for a document."""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = doc_manager.get_document(doc_uuid)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if format == "xml":
            from .metadata_catalog import MetadataCataloguer
            cataloguer = MetadataCataloguer()
            xml_content = cataloguer.export_to_xml(document.metadata)
            return JSONResponse(content={"xml": xml_content})
        else:
            return document.metadata.to_dict()
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sparql/query")
async def execute_sparql_query(
    query: str,
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Execute SPARQL query against Jena triple store."""
    try:
        results = doc_manager.jena_store.sparql_query(query)
        return {
            "query": query,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics(
    doc_manager: DocumentManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Get document management system statistics."""
    try:
        total_documents = len(doc_manager.documents)
        
        # Count by type
        type_counts = {}
        status_counts = {}
        
        for doc in doc_manager.documents.values():
            doc_type = doc.type.value
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            doc_status = doc.status.value
            status_counts[doc_status] = status_counts.get(doc_status, 0) + 1
        
        return {
            "total_documents": total_documents,
            "documents_by_type": type_counts,
            "documents_by_status": status_counts,
            "storage_path": str(doc_manager.storage_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Check API health status."""
    return {"status": "healthy", "service": "document_management"}


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )