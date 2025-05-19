"""
Knowledge Graph Builder for Financial Domain
Constructs and manages financial knowledge graphs
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

import networkx as nx
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL
import spacy
from transformers import pipeline

from aiq.utils.debugging_utils import log_function_call


logger = logging.getLogger(__name__)


# Define financial ontology namespaces
FIN = Namespace("http://finance.aiqtoolkit.com/")
ORG = Namespace("http://organization.aiqtoolkit.com/")
TIME = Namespace("http://time.aiqtoolkit.com/")


@dataclass
class Entity:
    """Knowledge graph entity"""
    entity_id: str
    entity_type: str
    name: str
    properties: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Relationship:
    """Knowledge graph relationship"""
    relationship_id: str
    relationship_type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class KnowledgeGraph:
    """Financial knowledge graph"""
    graph_id: str
    entities: Dict[str, Entity]
    relationships: List[Relationship]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class KnowledgeGraphBuilder:
    """
    Builds and manages financial knowledge graphs
    Extracts entities, relationships, and insights from financial data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize NLP models
        self.nlp = spacy.load(config.get("spacy_model", "en_core_web_lg"))
        self.ner_model = pipeline(
            "ner",
            model=config.get("ner_model", "dslim/bert-base-NER")
        )
        
        # Initialize graph
        self.nx_graph = nx.DiGraph()
        self.rdf_graph = Graph()
        self._init_namespaces()
        
        # Entity types
        self.entity_types = {
            'COMPANY': ['ORG', 'CORPORATION'],
            'PERSON': ['PERSON', 'EXECUTIVE'],
            'PRODUCT': ['PRODUCT', 'SERVICE'],
            'MONEY': ['MONEY', 'CURRENCY'],
            'PERCENTAGE': ['PERCENT', 'RATE'],
            'DATE': ['DATE', 'TIME'],
            'LOCATION': ['GPE', 'LOC'],
            'METRIC': ['METRIC', 'KPI']
        }
        
        # Relationship types
        self.relationship_types = {
            'HAS_CEO': {'source': 'COMPANY', 'target': 'PERSON'},
            'REPORTS_REVENUE': {'source': 'COMPANY', 'target': 'MONEY'},
            'HAS_METRIC': {'source': 'COMPANY', 'target': 'METRIC'},
            'LOCATED_IN': {'source': 'COMPANY', 'target': 'LOCATION'},
            'OWNS': {'source': 'COMPANY', 'target': 'COMPANY'},
            'COMPETES_WITH': {'source': 'COMPANY', 'target': 'COMPANY'},
            'SUPPLIES_TO': {'source': 'COMPANY', 'target': 'COMPANY'}
        }
        
        # Financial metrics patterns
        self.metric_patterns = {
            'revenue': ['revenue', 'sales', 'income'],
            'profit': ['profit', 'earnings', 'net income'],
            'margin': ['margin', 'profitability'],
            'growth': ['growth', 'increase', 'expansion'],
            'debt': ['debt', 'liabilities', 'obligations'],
            'assets': ['assets', 'holdings', 'resources']
        }
        
        # Graph statistics
        self.stats = {
            'entities_created': 0,
            'relationships_created': 0,
            'graphs_built': 0
        }
        
        logger.info("Initialized Knowledge Graph Builder")
    
    def _init_namespaces(self):
        """Initialize RDF namespaces"""
        self.rdf_graph.bind("fin", FIN)
        self.rdf_graph.bind("org", ORG)
        self.rdf_graph.bind("time", TIME)
        self.rdf_graph.bind("rdf", RDF)
        self.rdf_graph.bind("rdfs", RDFS)
        self.rdf_graph.bind("owl", OWL)
    
    async def build_graph(
        self,
        text_content: str,
        existing_entities: Optional[Dict[str, Entity]] = None
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from text content
        
        Args:
            text_content: Text to extract knowledge from
            existing_entities: Existing entities to link to
            
        Returns:
            Knowledge graph
        """
        try:
            # Extract entities
            entities = await self._extract_entities(text_content)
            
            # Merge with existing entities
            if existing_entities:
                entities = self._merge_entities(entities, existing_entities)
            
            # Extract relationships
            relationships = await self._extract_relationships(text_content, entities)
            
            # Build graph structure
            self._build_graph_structure(entities, relationships)
            
            # Enhance graph with inferences
            enhanced_relationships = self._infer_relationships(entities, relationships)
            
            # Calculate graph metrics
            metadata = self._calculate_graph_metrics()
            
            # Create knowledge graph
            knowledge_graph = KnowledgeGraph(
                graph_id=f"kg_{datetime.now().timestamp()}",
                entities=entities,
                relationships=enhanced_relationships,
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Update statistics
            self.stats['graphs_built'] += 1
            
            logger.info(f"Built knowledge graph with {len(entities)} entities")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            raise
    
    async def _extract_entities(self, text: str) -> Dict[str, Entity]:
        """Extract entities from text"""
        entities = {}
        
        # SpaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            entity_type = self._map_entity_type(ent.label_)
            if entity_type:
                entity_id = f"{entity_type}_{ent.text.replace(' ', '_')}"
                
                entity = Entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=ent.text,
                    properties={
                        'source': 'spacy',
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'confidence': 0.8
                    },
                    confidence=0.8
                )
                
                entities[entity_id] = entity
        
        # Transformer NER for additional entities
        ner_results = self.ner_model(text)
        for result in ner_results:
            entity_type = self._map_entity_type(result['entity'])
            if entity_type and result['score'] > 0.7:
                entity_text = result['word'].replace('##', '')
                entity_id = f"{entity_type}_{entity_text.replace(' ', '_')}"
                
                if entity_id not in entities:
                    entity = Entity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        name=entity_text,
                        properties={
                            'source': 'transformer',
                            'score': result['score'],
                            'confidence': result['score']
                        },
                        confidence=result['score']
                    )
                    
                    entities[entity_id] = entity
        
        # Extract financial metrics
        metric_entities = self._extract_financial_metrics(text)
        entities.update(metric_entities)
        
        self.stats['entities_created'] += len(entities)
        
        return entities
    
    def _extract_financial_metrics(self, text: str) -> Dict[str, Entity]:
        """Extract financial metrics from text"""
        metrics = {}
        text_lower = text.lower()
        
        for metric_type, patterns in self.metric_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Find associated values
                    import re
                    value_pattern = rf'{pattern}[:\s]+\$?([\d,]+(?:\.\d+)?)'
                    matches = re.finditer(value_pattern, text_lower)
                    
                    for match in matches:
                        value = match.group(1)
                        entity_id = f"METRIC_{metric_type}_{value.replace(',', '')}"
                        
                        entity = Entity(
                            entity_id=entity_id,
                            entity_type='METRIC',
                            name=f"{metric_type}: {value}",
                            properties={
                                'metric_type': metric_type,
                                'value': value,
                                'pattern': pattern
                            },
                            confidence=0.9
                        )
                        
                        metrics[entity_id] = entity
        
        return metrics
    
    async def _extract_relationships(
        self,
        text: str,
        entities: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []
        
        # Dependency parsing for relationships
        doc = self.nlp(text)
        
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                # Check if token and head are entities
                subject_entity = self._find_entity_by_text(token.text, entities)
                object_entity = self._find_entity_by_text(token.head.text, entities)
                
                if subject_entity and object_entity:
                    relationship = Relationship(
                        relationship_id=f"rel_{len(relationships)}",
                        relationship_type=token.dep_,
                        source_id=subject_entity.entity_id,
                        target_id=object_entity.entity_id,
                        properties={
                            'verb': token.head.text,
                            'dependency': token.dep_
                        },
                        confidence=0.7
                    )
                    
                    relationships.append(relationship)
        
        # Pattern-based relationship extraction
        pattern_relationships = self._extract_pattern_relationships(text, entities)
        relationships.extend(pattern_relationships)
        
        self.stats['relationships_created'] += len(relationships)
        
        return relationships
    
    def _extract_pattern_relationships(
        self,
        text: str,
        entities: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract relationships using patterns"""
        relationships = []
        
        # CEO relationship pattern
        ceo_pattern = r'(\w+(?:\s+\w+)*)\s+(?:CEO|chief executive|president)\s+(?:of|at)\s+(\w+(?:\s+\w+)*)'
        
        import re
        for match in re.finditer(ceo_pattern, text, re.IGNORECASE):
            person_name = match.group(1)
            company_name = match.group(2)
            
            person_entity = self._find_entity_by_text(person_name, entities)
            company_entity = self._find_entity_by_text(company_name, entities)
            
            if person_entity and company_entity:
                relationship = Relationship(
                    relationship_id=f"rel_{len(relationships)}",
                    relationship_type='HAS_CEO',
                    source_id=company_entity.entity_id,
                    target_id=person_entity.entity_id,
                    properties={
                        'pattern': 'ceo_pattern',
                        'text': match.group(0)
                    },
                    confidence=0.9
                )
                
                relationships.append(relationship)
        
        # Revenue relationship pattern
        revenue_pattern = r'(\w+(?:\s+\w+)*)\s+(?:reported|posted|generated)\s+(?:revenue|sales)\s+(?:of|at)\s+\$?([\d,]+(?:\.\d+)?)'
        
        for match in re.finditer(revenue_pattern, text, re.IGNORECASE):
            company_name = match.group(1)
            revenue_value = match.group(2)
            
            company_entity = self._find_entity_by_text(company_name, entities)
            
            if company_entity:
                # Create revenue entity
                revenue_entity_id = f"MONEY_revenue_{revenue_value.replace(',', '')}"
                revenue_entity = Entity(
                    entity_id=revenue_entity_id,
                    entity_type='MONEY',
                    name=f"Revenue: ${revenue_value}",
                    properties={'value': revenue_value},
                    confidence=0.9
                )
                
                entities[revenue_entity_id] = revenue_entity
                
                relationship = Relationship(
                    relationship_id=f"rel_{len(relationships)}",
                    relationship_type='REPORTS_REVENUE',
                    source_id=company_entity.entity_id,
                    target_id=revenue_entity_id,
                    properties={
                        'pattern': 'revenue_pattern',
                        'text': match.group(0)
                    },
                    confidence=0.9
                )
                
                relationships.append(relationship)
        
        return relationships
    
    def _find_entity_by_text(
        self,
        text: str,
        entities: Dict[str, Entity]
    ) -> Optional[Entity]:
        """Find entity by text match"""
        text_lower = text.lower()
        
        for entity in entities.values():
            if entity.name.lower() == text_lower:
                return entity
            
            # Partial match
            if text_lower in entity.name.lower() or entity.name.lower() in text_lower:
                return entity
        
        return None
    
    def _map_entity_type(self, label: str) -> Optional[str]:
        """Map NER label to entity type"""
        for entity_type, labels in self.entity_types.items():
            if label in labels:
                return entity_type
        
        return None
    
    def _merge_entities(
        self,
        new_entities: Dict[str, Entity],
        existing_entities: Dict[str, Entity]
    ) -> Dict[str, Entity]:
        """Merge new entities with existing ones"""
        merged = existing_entities.copy()
        
        for entity_id, entity in new_entities.items():
            if entity_id in merged:
                # Update confidence
                existing = merged[entity_id]
                existing.confidence = max(existing.confidence, entity.confidence)
                
                # Merge properties
                existing.properties.update(entity.properties)
            else:
                merged[entity_id] = entity
        
        return merged
    
    def _build_graph_structure(
        self,
        entities: Dict[str, Entity],
        relationships: List[Relationship]
    ):
        """Build NetworkX and RDF graphs"""
        # Clear existing graphs
        self.nx_graph.clear()
        
        # Add entities to NetworkX
        for entity_id, entity in entities.items():
            self.nx_graph.add_node(
                entity_id,
                entity_type=entity.entity_type,
                name=entity.name,
                **entity.properties
            )
            
            # Add to RDF
            entity_uri = FIN[entity_id]
            self.rdf_graph.add((entity_uri, RDF.type, FIN[entity.entity_type]))
            self.rdf_graph.add((entity_uri, RDFS.label, Literal(entity.name)))
        
        # Add relationships
        for relationship in relationships:
            self.nx_graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relationship_type=relationship.relationship_type,
                **relationship.properties
            )
            
            # Add to RDF
            source_uri = FIN[relationship.source_id]
            target_uri = FIN[relationship.target_id]
            predicate_uri = FIN[relationship.relationship_type]
            
            self.rdf_graph.add((source_uri, predicate_uri, target_uri))
    
    def _infer_relationships(
        self,
        entities: Dict[str, Entity],
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """Infer additional relationships"""
        inferred = relationships.copy()
        
        # Infer competition relationships
        companies = [e for e in entities.values() if e.entity_type == 'COMPANY']
        
        for i, company1 in enumerate(companies):
            for company2 in companies[i+1:]:
                # Check if they share similar metrics
                if self._are_competitors(company1, company2, relationships):
                    relationship = Relationship(
                        relationship_id=f"inferred_{len(inferred)}",
                        relationship_type='COMPETES_WITH',
                        source_id=company1.entity_id,
                        target_id=company2.entity_id,
                        properties={
                            'inferred': True,
                            'confidence': 0.6
                        },
                        confidence=0.6
                    )
                    
                    inferred.append(relationship)
        
        return inferred
    
    def _are_competitors(
        self,
        company1: Entity,
        company2: Entity,
        relationships: List[Relationship]
    ) -> bool:
        """Check if two companies are competitors"""
        # Simple heuristic: companies in same industry with similar metrics
        # This is a simplified implementation
        
        return False  # Placeholder
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate graph metrics"""
        metrics = {
            'node_count': self.nx_graph.number_of_nodes(),
            'edge_count': self.nx_graph.number_of_edges(),
            'density': nx.density(self.nx_graph),
            'is_connected': nx.is_weakly_connected(self.nx_graph),
            'average_degree': sum(dict(self.nx_graph.degree()).values()) / self.nx_graph.number_of_nodes() if self.nx_graph.number_of_nodes() > 0 else 0
        }
        
        # Centrality measures
        if self.nx_graph.number_of_nodes() > 0:
            metrics['degree_centrality'] = nx.degree_centrality(self.nx_graph)
            
            if self.nx_graph.number_of_edges() > 0:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(self.nx_graph)
        
        return metrics
    
    async def expand_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        additional_text: str
    ) -> KnowledgeGraph:
        """Expand existing knowledge graph with new information"""
        # Extract new entities and relationships
        new_entities = await self._extract_entities(additional_text)
        merged_entities = self._merge_entities(new_entities, knowledge_graph.entities)
        
        new_relationships = await self._extract_relationships(additional_text, merged_entities)
        all_relationships = knowledge_graph.relationships + new_relationships
        
        # Rebuild graph structure
        self._build_graph_structure(merged_entities, all_relationships)
        
        # Update metadata
        metadata = self._calculate_graph_metrics()
        
        # Create updated graph
        updated_graph = KnowledgeGraph(
            graph_id=knowledge_graph.graph_id,
            entities=merged_entities,
            relationships=all_relationships,
            metadata=metadata,
            created_at=knowledge_graph.created_at,
            updated_at=datetime.now()
        )
        
        return updated_graph
    
    def query_graph(self, query: str, knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
        """Query knowledge graph using natural language"""
        # Parse query
        doc = self.nlp(query)
        
        # Extract query entities
        query_entities = []
        for ent in doc.ents:
            entity = self._find_entity_by_text(ent.text, knowledge_graph.entities)
            if entity:
                query_entities.append(entity)
        
        # Find relevant subgraph
        if query_entities:
            # Get neighbors of query entities
            relevant_nodes = set()
            for entity in query_entities:
                if entity.entity_id in self.nx_graph:
                    neighbors = nx.single_source_shortest_path_length(
                        self.nx_graph,
                        entity.entity_id,
                        cutoff=2
                    )
                    relevant_nodes.update(neighbors.keys())
            
            # Extract subgraph
            subgraph = self.nx_graph.subgraph(relevant_nodes)
            
            return {
                'query': query,
                'entities_found': [e.name for e in query_entities],
                'subgraph_nodes': list(subgraph.nodes()),
                'subgraph_edges': list(subgraph.edges()),
                'node_count': subgraph.number_of_nodes(),
                'edge_count': subgraph.number_of_edges()
            }
        
        return {
            'query': query,
            'entities_found': [],
            'message': 'No relevant entities found in query'
        }
    
    def export_graph(self, knowledge_graph: KnowledgeGraph, format: str = 'json') -> str:
        """Export knowledge graph in specified format"""
        if format == 'json':
            return json.dumps({
                'graph_id': knowledge_graph.graph_id,
                'entities': {k: v.__dict__ for k, v in knowledge_graph.entities.items()},
                'relationships': [r.__dict__ for r in knowledge_graph.relationships],
                'metadata': knowledge_graph.metadata,
                'created_at': knowledge_graph.created_at.isoformat(),
                'updated_at': knowledge_graph.updated_at.isoformat()
            }, indent=2)
        
        elif format == 'rdf':
            return self.rdf_graph.serialize(format='turtle')
        
        elif format == 'graphml':
            from io import StringIO
            output = StringIO()
            nx.write_graphml(self.nx_graph, output)
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get builder statistics"""
        return {
            'entities_created': self.stats['entities_created'],
            'relationships_created': self.stats['relationships_created'],
            'graphs_built': self.stats['graphs_built'],
            'current_graph_nodes': self.nx_graph.number_of_nodes(),
            'current_graph_edges': self.nx_graph.number_of_edges()
        }