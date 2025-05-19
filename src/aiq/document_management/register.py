"""Register document management components with AIQ."""

from ..cli.register_workflow import register_workflow, register_function
from ..data_models.function import FunctionBaseConfig
from ..builder.builder import Builder
from ..builder.function_info import FunctionInfo
from pydantic import Field
from typing import Optional, List, Dict, Any


class DocumentUploadConfig(FunctionBaseConfig, name="document_upload"):
    """Configuration for document upload function."""
    storage_path: str = Field(default="/tmp/aiq_documents", description="Path to store documents")
    enable_rag: bool = Field(default=True, description="Enable NVIDIA RAG integration")
    auto_metadata: bool = Field(default=True, description="Auto-generate metadata")


class DocumentSearchConfig(FunctionBaseConfig, name="document_search"):
    """Configuration for document search function."""
    search_type: str = Field(default="hybrid", description="Search type: hybrid, semantic, keyword")
    top_k: int = Field(default=10, description="Number of results to return")
    use_reranking: bool = Field(default=True, description="Use NVIDIA reranking")


class WebCrawlerConfig(FunctionBaseConfig, name="web_crawler"):
    """Configuration for web crawler function."""
    max_depth: int = Field(default=3, description="Maximum crawl depth")
    max_pages: int = Field(default=100, description="Maximum pages to crawl")
    research_focus: bool = Field(default=True, description="Focus on research content")


@register_function(config_type=DocumentUploadConfig)
async def document_upload_function(config: DocumentUploadConfig, builder: Builder):
    """Function to upload documents to the management system."""
    from .manager import DocumentManager
    
    manager = DocumentManager()
    
    async def upload_document(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload a document with optional metadata."""
        try:
            document = manager.upload_document(file_path, metadata)
            return {
                "id": str(document.id),
                "title": document.metadata.title,
                "status": document.status.value,
                "type": document.type.value
            }
        except Exception as e:
            return {"error": str(e)}
    
    yield FunctionInfo.from_fn(
        upload_document,
        description="Upload a document to the AIQ document management system"
    )


@register_function(config_type=DocumentSearchConfig)
async def document_search_function(config: DocumentSearchConfig, builder: Builder):
    """Function to search documents using NVIDIA RAG."""
    from .nvidia_rag_integration import NVIDIARAGDocumentManager
    
    rag_manager = NVIDIARAGDocumentManager(builder=builder)
    
    async def search_documents(query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using advanced NVIDIA RAG."""
        try:
            results = await rag_manager.advanced_search(
                query=query,
                search_type=config.search_type,
                filters=filters
            )
            return results[:config.top_k]
        except Exception as e:
            return [{"error": str(e)}]
    
    yield FunctionInfo.from_fn(
        search_documents,
        description=f"Search documents using {config.search_type} search with NVIDIA RAG"
    )


@register_function(config_type=WebCrawlerConfig)
async def web_crawler_function(config: WebCrawlerConfig, builder: Builder):
    """Function to crawl websites for research content."""
    from .crawler import ResearchWebCrawler
    from .manager import DocumentManager
    
    crawler = ResearchWebCrawler(max_depth=config.max_depth, max_pages=config.max_pages)
    manager = DocumentManager()
    
    async def crawl_website(url: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Crawl a website and collect research documents."""
        try:
            documents = await crawler.crawl_site(url, keywords)
            
            uploaded_ids = []
            for doc in documents:
                metadata = doc.metadata.to_dict()
                uploaded = manager.upload_web_content(doc.url, doc.content, metadata)
                uploaded_ids.append(str(uploaded.id))
            
            return {
                "pages_crawled": len(crawler.visited_urls),
                "documents_collected": len(documents),
                "document_ids": uploaded_ids
            }
        except Exception as e:
            return {"error": str(e)}
    
    yield FunctionInfo.from_fn(
        crawl_website,
        description="Crawl websites to collect research documents"
    )


# Register workflow configuration
workflow_config = {
    "name": "document_management_workflow",
    "description": "Workflow for document management with NVIDIA RAG integration",
    "version": "1.0.0",
    "components": {
        "functions": [
            {
                "name": "document_upload",
                "type": "document_upload",
                "config": {
                    "storage_path": "/tmp/aiq_documents",
                    "enable_rag": True,
                    "auto_metadata": True
                }
            },
            {
                "name": "document_search",
                "type": "document_search",
                "config": {
                    "search_type": "hybrid",
                    "top_k": 10,
                    "use_reranking": True
                }
            },
            {
                "name": "web_crawler",
                "type": "web_crawler",
                "config": {
                    "max_depth": 3,
                    "max_pages": 100,
                    "research_focus": True
                }
            }
        ],
        "llms": [
            {
                "name": "default_llm",
                "type": "openai",
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            }
        ],
        "embedders": [
            {
                "name": "nvidia_embedder",
                "type": "nim",
                "config": {
                    "model": "nvidia-embed-qa-4"
                }
            }
        ]
    }
}


@register_workflow(name="document_management", config=workflow_config)
async def document_management_workflow(builder: Builder):
    """Document management workflow with NVIDIA RAG integration."""
    # This workflow integrates all document management components
    # and provides a unified interface for document operations
    pass