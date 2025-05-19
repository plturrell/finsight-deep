#!/usr/bin/env python3
"""
Example: Using Document Management System with Apache Jena

This example demonstrates how to:
1. Upload documents to the system
2. Store them in Apache Jena triple store
3. Query documents using SPARQL
4. Perform vector search
5. Store AI/ML research outputs
"""

import asyncio
from pathlib import Path
from src.aiq.document_management.manager import DocumentManager
from src.aiq.document_management.models import DocumentType, DublinCoreMetadata


async def main():
    """Main example function."""
    
    # Initialize the document manager
    print("Initializing Document Manager with Jena support...")
    doc_manager = DocumentManager()
    
    # Example 1: Upload a PDF document
    print("\n1. Uploading a PDF document...")
    pdf_path = "research_paper.pdf"  # Replace with actual path
    
    if Path(pdf_path).exists():
        pdf_metadata = {
            'title': 'Advanced Neural Networks Research',
            'creator': 'Dr. Jane Smith',
            'subject': ['AI', 'Neural Networks', 'Deep Learning'],
            'description': 'A comprehensive study on neural network architectures',
            'type': 'Research Paper',
            'language': 'en'
        }
        
        pdf_doc = doc_manager.upload_document(pdf_path, pdf_metadata)
        print(f"✓ PDF uploaded with ID: {pdf_doc.id}")
        print(f"✓ Status: {pdf_doc.status.value}")
        print(f"✓ Jena URI: {pdf_doc.jena_uri}")
    
    # Example 2: Upload web content
    print("\n2. Uploading web content...")
    web_content = """
    Artificial Intelligence in Healthcare
    
    AI is revolutionizing healthcare through:
    - Diagnostic imaging analysis
    - Drug discovery acceleration
    - Personalized treatment plans
    - Predictive analytics for patient outcomes
    """
    
    web_doc = doc_manager.upload_web_content(
        url="https://example.com/ai-healthcare",
        content=web_content,
        metadata={
            'title': 'AI in Healthcare',
            'creator': 'Tech Blog',
            'subject': ['AI', 'Healthcare', 'Medical Technology'],
            'type': 'Blog Post'
        }
    )
    print(f"✓ Web content uploaded with ID: {web_doc.id}")
    
    # Example 3: SPARQL Query
    print("\n3. Executing SPARQL query...")
    sparql_query = """
    PREFIX aiq: <http://aiqtoolkit.org/ontology#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?doc ?title ?creator ?subject
    WHERE {
        ?doc a aiq:Document ;
             dc:title ?title ;
             dc:creator ?creator .
        OPTIONAL { ?doc dc:subject ?subject }
    }
    LIMIT 10
    """
    
    results = doc_manager.jena_store.sparql_query(sparql_query)
    if results and "results" in results:
        bindings = results["results"]["bindings"]
        print(f"✓ Found {len(bindings)} documents")
        for binding in bindings[:3]:  # Show first 3
            print(f"  - Title: {binding.get('title', {}).get('value', 'N/A')}")
            print(f"    Creator: {binding.get('creator', {}).get('value', 'N/A')}")
    
    # Example 4: Vector Search
    print("\n4. Performing vector search...")
    search_query = "neural networks and deep learning applications"
    search_results = doc_manager.search_documents(search_query, limit=5)
    
    print(f"✓ Found {len(search_results)} similar documents")
    for i, result in enumerate(search_results[:3]):
        print(f"  {i+1}. {result.metadata.title}")
        print(f"     Type: {result.type.value}")
        print(f"     Status: {result.status.value}")
    
    # Example 5: Store AI Research Output
    print("\n5. Storing AI research output...")
    research_output = doc_manager.store_research_output(
        agent_name="DocumentAnalyzer",
        output_data={
            'type': 'document_analysis',
            'content': 'Key findings from document analysis',
            'entities': ['Neural Networks', 'Machine Learning', 'AI'],
            'sentiment': 'positive',
            'summary': 'The document discusses advanced AI techniques...',
            'confidence_score': 0.92
        },
        source_doc_id=web_doc.id  # Link to source document
    )
    print(f"✓ Research output stored with ID: {research_output.id}")
    
    # Example 6: Advanced SPARQL with Research Outputs
    print("\n6. Querying research outputs...")
    research_query = """
    PREFIX aiq: <http://aiqtoolkit.org/ontology#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?output ?agent ?source ?date
    WHERE {
        ?output a aiq:ResearchOutput ;
                aiq:agentName ?agent ;
                aiq:sourceDocument ?source ;
                dc:date ?date .
    }
    ORDER BY DESC(?date)
    LIMIT 5
    """
    
    research_results = doc_manager.jena_store.sparql_query(research_query)
    if research_results and "results" in research_results:
        bindings = research_results["results"]["bindings"]
        print(f"✓ Found {len(bindings)} research outputs")
    
    print("\n=== Example completed successfully! ===")
    print("\nKey Features Demonstrated:")
    print("1. Document upload (PDF, web content)")
    print("2. Automatic Jena triple store integration")
    print("3. SPARQL querying capabilities")
    print("4. Vector similarity search")
    print("5. AI research output storage")
    print("6. Dublin Core metadata support")


if __name__ == "__main__":
    asyncio.run(main())