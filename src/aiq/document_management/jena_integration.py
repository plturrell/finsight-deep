"""Apache Jena integration for OWL and vector storage."""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional
import requests
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import DC, DCTERMS, OWL
import numpy as np

from .models import Document, ResearchOutput, DublinCoreMetadata
from ..data_models.config import WorkflowConfig
from ..settings.global_settings import GlobalSettings


logger = logging.getLogger(__name__)


class JenaTripleStore:
    """Integration with Apache Jena for triple store and vector operations."""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize Jena connection."""
        self.config = config
        settings = GlobalSettings()
        
        # Jena Fuseki endpoint
        self.sparql_endpoint = settings.get("JENA_SPARQL_ENDPOINT", "http://localhost:3030/dataset/sparql")
        self.update_endpoint = settings.get("JENA_UPDATE_ENDPOINT", "http://localhost:3030/dataset/update")
        self.data_endpoint = settings.get("JENA_DATA_ENDPOINT", "http://localhost:3030/dataset/data")
        
        # Vector store endpoint (could be Jena with GeoSPARQL or separate)
        self.vector_endpoint = settings.get("VECTOR_STORE_ENDPOINT", "http://localhost:8001/vectors")
        
        # Namespaces
        self.aiq = Namespace("http://aiqtoolkit.org/ontology#")
        self.dc = DC
        self.owl = OWL
        
        # Initialize graph
        self.graph = Graph()
        self.graph.bind("aiq", self.aiq)
        self.graph.bind("dc", self.dc)
        self.graph.bind("owl", self.owl)
    
    def store_document(self, document: Document) -> str:
        """Store document in Jena with OWL representation and vectors."""
        doc_uri = URIRef(f"http://aiqtoolkit.org/documents/{document.id}")
        
        # Create OWL representation
        self.graph.add((doc_uri, RDF.type, self.aiq.Document))
        self.graph.add((doc_uri, self.aiq.documentId, Literal(str(document.id))))
        self.graph.add((doc_uri, self.aiq.documentType, Literal(document.type.value)))
        self.graph.add((doc_uri, self.aiq.status, Literal(document.status.value)))
        
        # Add Dublin Core metadata
        self._add_dublin_core_metadata(doc_uri, document.metadata)
        
        # Add content if available
        if document.content:
            self.graph.add((doc_uri, self.aiq.content, Literal(document.content[:1000])))  # Store preview
        
        # Store vector separately
        if document.vector:
            self._store_vector(document.id, document.vector.embeddings)
        
        # Upload to Jena
        self._upload_graph()
        
        return str(doc_uri)
    
    def store_research_output(self, output: ResearchOutput) -> str:
        """Store research output in Jena."""
        output_uri = URIRef(f"http://aiqtoolkit.org/research/{output.id}")
        
        # Create OWL representation
        self.graph.add((output_uri, RDF.type, self.aiq.ResearchOutput))
        self.graph.add((output_uri, self.aiq.outputId, Literal(str(output.id))))
        self.graph.add((output_uri, self.aiq.agentName, Literal(output.agent_name)))
        self.graph.add((output_uri, self.aiq.outputType, Literal(output.output_type)))
        
        # Link to source document if available
        if output.source_document_id:
            source_uri = URIRef(f"http://aiqtoolkit.org/documents/{output.source_document_id}")
            self.graph.add((output_uri, self.aiq.sourceDocument, source_uri))
        
        # Add metadata
        self._add_dublin_core_metadata(output_uri, output.metadata)
        
        # Store content as JSON
        content_json = json.dumps(output.content)
        self.graph.add((output_uri, self.aiq.content, Literal(content_json)))
        
        # Store embeddings if available
        if output.embeddings:
            self._store_vector(output.id, output.embeddings)
        
        # Upload to Jena
        self._upload_graph()
        
        return str(output_uri)
    
    def vector_search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            response = requests.post(
                f"{self.vector_endpoint}/search",
                json={
                    "query_vector": query_embedding,
                    "limit": limit,
                    "threshold": 0.7
                }
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Enrich results with document metadata
                enriched_results = []
                for result in results:
                    doc_id = result.get("document_id")
                    metadata = self._get_document_metadata(doc_id)
                    enriched_results.append({
                        **result,
                        **metadata
                    })
                
                return enriched_results
            else:
                logger.error(f"Vector search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against Jena."""
        try:
            response = requests.post(
                self.sparql_endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"SPARQL query failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return []
    
    def _add_dublin_core_metadata(self, subject: URIRef, metadata: DublinCoreMetadata):
        """Add Dublin Core metadata to graph."""
        self.graph.add((subject, DC.title, Literal(metadata.title)))
        
        if metadata.creator:
            self.graph.add((subject, DC.creator, Literal(metadata.creator)))
        
        if metadata.subject:
            for subj in metadata.subject:
                self.graph.add((subject, DC.subject, Literal(subj)))
        
        if metadata.description:
            self.graph.add((subject, DC.description, Literal(metadata.description)))
        
        if metadata.date:
            self.graph.add((subject, DC.date, Literal(metadata.date.isoformat())))
        
        if metadata.type:
            self.graph.add((subject, DC.type, Literal(metadata.type)))
        
        if metadata.format:
            self.graph.add((subject, DC.format, Literal(metadata.format)))
        
        if metadata.language:
            self.graph.add((subject, DC.language, Literal(metadata.language)))
    
    def _store_vector(self, doc_id: uuid.UUID, embeddings: List[float]):
        """Store vector embeddings in vector store."""
        try:
            response = requests.post(
                f"{self.vector_endpoint}/store",
                json={
                    "document_id": str(doc_id),
                    "embeddings": embeddings,
                    "dimensions": len(embeddings)
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to store vector: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
    
    def _upload_graph(self):
        """Upload RDF graph to Jena."""
        try:
            # Serialize graph to Turtle format
            turtle_data = self.graph.serialize(format="turtle")
            
            response = requests.post(
                self.data_endpoint,
                data=turtle_data,
                headers={"Content-Type": "text/turtle"}
            )
            
            if response.status_code not in [200, 201, 204]:
                logger.error(f"Failed to upload to Jena: {response.status_code}")
            
            # Clear graph after successful upload
            self.graph = Graph()
            self.graph.bind("aiq", self.aiq)
            self.graph.bind("dc", self.dc)
            self.graph.bind("owl", self.owl)
            
        except Exception as e:
            logger.error(f"Error uploading to Jena: {e}")
    
    def _get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve document metadata from Jena."""
        query = f"""
        PREFIX aiq: <http://aiqtoolkit.org/ontology#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        
        SELECT ?title ?creator ?type ?date ?description
        WHERE {{
            ?doc aiq:documentId "{doc_id}" .
            ?doc dc:title ?title .
            OPTIONAL {{ ?doc dc:creator ?creator }}
            OPTIONAL {{ ?doc dc:type ?type }}
            OPTIONAL {{ ?doc dc:date ?date }}
            OPTIONAL {{ ?doc dc:description ?description }}
        }}
        """
        
        results = self.sparql_query(query)
        
        if results and "results" in results and "bindings" in results["results"]:
            bindings = results["results"]["bindings"]
            if bindings:
                return {
                    "title": bindings[0].get("title", {}).get("value", ""),
                    "creator": bindings[0].get("creator", {}).get("value", ""),
                    "type": bindings[0].get("type", {}).get("value", ""),
                    "date": bindings[0].get("date", {}).get("value", ""),
                    "description": bindings[0].get("description", {}).get("value", "")
                }
        
        return {}