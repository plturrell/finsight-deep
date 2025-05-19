#!/usr/bin/env python3
"""Test document management with Jena integration."""

import asyncio
import uuid
from pathlib import Path
from aiq.document_management.models import Document, DocumentType, DublinCoreMetadata
from aiq.document_management.manager import DocumentManager
from aiq.document_management.jena_integration import JenaTripleStore

async def test_jena_integration():
    """Test the document management system with Jena."""
    print("Testing Document Management with Jena Integration...")
    
    # Create test document
    test_content = """
    This is a test document for the AIQ Toolkit document management system.
    It demonstrates integration with Apache Jena for semantic storage.
    """
    
    # Initialize document manager
    manager = DocumentManager()
    
    # Create a test document
    doc = Document(
        type=DocumentType.TEXT,
        title="Test Document for Jena",
        content=test_content,
        metadata=DublinCoreMetadata(
            title="Test Document for Jena Integration",
            creator="AIQ Test Suite",
            subject=["testing", "jena", "document management"],
            description="A test document to verify Jena integration"
        )
    )
    
    # Process the document (this triggers Jena storage)
    manager.documents[doc.id] = doc
    manager._process_document(doc)
    
    print(f"Document created with ID: {doc.id}")
    print(f"Document status: {doc.status.value}")
    print(f"Jena URI: {doc.jena_uri}")
    
    # Test SPARQL query
    jena_store = JenaTripleStore()
    query = """
    PREFIX aiq: <http://aiqtoolkit.org/ontology#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?doc ?title ?type
    WHERE {
        ?doc a aiq:Document ;
             dc:title ?title ;
             aiq:documentType ?type .
    }
    LIMIT 10
    """
    
    try:
        results = jena_store.sparql_query(query)
        print("\nSPARQL Query Results:")
        print(results)
    except Exception as e:
        print(f"SPARQL query failed: {e}")
    
    # Test document search
    search_results = manager.search_documents("test document")
    print(f"\nSearch results found: {len(search_results)}")
    for result in search_results:
        print(f"- {result.title} (ID: {result.id})")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_jena_integration())