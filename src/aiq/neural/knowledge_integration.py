# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration layer between Neural components and Jena knowledge graph
Provides bridges between neural computation and semantic knowledge
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging

from aiq.neural.advanced_architectures import (
    HybridNeuralSymbolicLayer, NeuralMemoryBank
)
from aiq.digital_human.persistence.jena_database import JenaPersistenceManager
from aiq.retriever.neural_symbolic.neural_symbolic_retriever import (
    NeuralSymbolicRetriever, KnowledgeGraph, KnowledgeEntity
)


logger = logging.getLogger(__name__)


@dataclass
class NeuralJenaConfig:
    """Configuration for Neural-Jena integration"""
    jena_endpoint: str = "http://localhost:3030"
    dataset_name: str = "neural_knowledge"
    embedding_dim: int = 768
    use_gpu: bool = True
    cache_embeddings: bool = True


class NeuralJenaIntegration:
    """
    Integration layer between neural models and Jena knowledge graphs
    Enables neural models to query and update semantic knowledge
    """
    
    def __init__(self, config: NeuralJenaConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize Jena persistence
        jena_config = {
            "fuseki_url": config.jena_endpoint,
            "dataset_name": config.dataset_name
        }
        self.jena_manager = JenaPersistenceManager(jena_config)
        
        # Initialize neural-symbolic retriever
        self.neural_retriever = NeuralSymbolicRetriever(
            dataset_name=config.dataset_name,
            embedding_dim=config.embedding_dim,
            device=str(self.device)
        )
        
        # Cache for embeddings
        self.embedding_cache = {} if config.cache_embeddings else None
        
        logger.info("Initialized Neural-Jena integration layer")
    
    async def embed_knowledge_entity(
        self,
        entity_uri: str,
        entity_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Convert a Jena knowledge entity to neural embedding
        
        Args:
            entity_uri: URI of the entity in Jena
            entity_data: Entity properties and relations
            
        Returns:
            Neural embedding of the entity
        """
        # Check cache
        if self.embedding_cache and entity_uri in self.embedding_cache:
            return self.embedding_cache[entity_uri]
        
        # Create text representation of entity
        text_repr = self._entity_to_text(entity_uri, entity_data)
        
        # Generate embedding
        embedding = self.neural_retriever.neural_encoder.encode([text_repr])[0]
        
        # Cache embedding
        if self.embedding_cache:
            self.embedding_cache[entity_uri] = embedding
        
        return embedding
    
    def _entity_to_text(self, uri: str, data: Dict[str, Any]) -> str:
        """Convert entity data to text for embedding"""
        parts = [f"Entity: {uri}"]
        
        # Add properties
        for key, value in data.get("properties", {}).items():
            parts.append(f"{key}: {value}")
        
        # Add relations
        for relation in data.get("relations", []):
            parts.append(f"{relation['predicate']}: {relation['object']}")
        
        return " | ".join(parts)
    
    async def query_with_neural_context(
        self,
        sparql_query: str,
        neural_context: torch.Tensor,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query enhanced with neural context
        
        Args:
            sparql_query: Base SPARQL query
            neural_context: Neural embedding for context
            top_k: Number of results to return
            
        Returns:
            Query results ranked by neural similarity
        """
        # Execute base SPARQL query
        results = await self.jena_manager.execute_sparql_query(sparql_query)
        
        # Extract entities from results
        entities = []
        for binding in results.get("results", {}).get("bindings", []):
            entity_uri = binding.get("entity", {}).get("value")
            if entity_uri:
                entities.append(entity_uri)
        
        # Get embeddings for entities
        entity_embeddings = []
        for uri in entities:
            # Fetch entity data from Jena
            entity_query = f"""
            SELECT ?p ?o
            WHERE {{
                <{uri}> ?p ?o .
            }}
            """
            entity_data = await self.jena_manager.execute_sparql_query(entity_query)
            
            # Convert to embedding
            embedding = await self.embed_knowledge_entity(uri, entity_data)
            entity_embeddings.append(embedding)
        
        if not entity_embeddings:
            return []
        
        # Compute similarities
        entity_embeddings = torch.stack(entity_embeddings)
        similarities = torch.cosine_similarity(
            neural_context.unsqueeze(0),
            entity_embeddings,
            dim=1
        )
        
        # Rank results
        ranked_indices = similarities.argsort(descending=True)[:top_k]
        
        # Return ranked results
        ranked_results = []
        for idx in ranked_indices:
            result = {
                "entity": entities[idx],
                "similarity": similarities[idx].item(),
                "data": results["results"]["bindings"][idx]
            }
            ranked_results.append(result)
        
        return ranked_results
    
    async def update_knowledge_from_neural(
        self,
        entity_uri: str,
        neural_output: torch.Tensor,
        threshold: float = 0.8
    ):
        """
        Update Jena knowledge graph based on neural model output
        
        Args:
            entity_uri: Entity to update
            neural_output: Neural model predictions
            threshold: Confidence threshold for updates
        """
        # Convert neural output to knowledge updates
        updates = self._neural_to_knowledge_updates(neural_output, threshold)
        
        # Apply updates to Jena
        for update in updates:
            update_query = f"""
            PREFIX dh: <http://aiqtoolkit.com/digital-human/>
            
            INSERT DATA {{
                <{entity_uri}> dh:{update['predicate']} "{update['value']}" .
            }}
            """
            
            await self.jena_manager.sparql_update.setQuery(update_query)
            await self.jena_manager.sparql_update.query()
        
        logger.info(f"Updated {len(updates)} properties for entity {entity_uri}")
    
    def _neural_to_knowledge_updates(
        self,
        neural_output: torch.Tensor,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Convert neural predictions to knowledge updates"""
        updates = []
        
        # Example: Convert classification outputs to properties
        if neural_output.dim() == 1:  # Classification vector
            probs = torch.softmax(neural_output, dim=0)
            max_prob, max_idx = probs.max(dim=0)
            
            if max_prob > threshold:
                updates.append({
                    "predicate": "predictedClass",
                    "value": f"class_{max_idx.item()}",
                    "confidence": max_prob.item()
                })
        
        return updates
    
    async def create_neural_backed_knowledge_graph(
        self,
        documents: List[Dict[str, Any]]
    ) -> KnowledgeGraph:
        """
        Create a knowledge graph with neural embeddings from documents
        
        Args:
            documents: Documents to process
            
        Returns:
            Knowledge graph with entities and embeddings
        """
        entities = {}
        relations = []
        embeddings = []
        
        for i, doc in enumerate(documents):
            # Create entity
            entity_uri = f"doc:{i}"
            entity = KnowledgeEntity(
                uri=entity_uri,
                entity_type="document",
                label=doc.get("title", f"Document {i}"),
                properties=doc.get("metadata", {})
            )
            
            # Generate embedding
            text = doc.get("content", "")
            embedding = self.neural_retriever.neural_encoder.encode([text])[0]
            entity.embedding = embedding
            
            entities[entity_uri] = entity
            embeddings.append(embedding)
            
            # Store in Jena
            await self._store_entity_in_jena(entity)
        
        # Stack embeddings
        if embeddings:
            embeddings_tensor = torch.stack(embeddings)
        else:
            embeddings_tensor = None
        
        return KnowledgeGraph(
            entities=entities,
            relations=relations,
            embeddings=embeddings_tensor,
            index_mapping={uri: i for i, uri in enumerate(entities.keys())}
        )
    
    async def _store_entity_in_jena(self, entity: KnowledgeEntity):
        """Store entity in Jena triple store"""
        insert_query = f"""
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        INSERT DATA {{
            <{entity.uri}> rdf:type dh:{entity.entity_type} ;
                dh:label "{entity.label}" .
        """
        
        # Add properties
        for key, value in entity.properties.items():
            insert_query += f"""
                <{entity.uri}> dh:{key} "{value}" .
            """
        
        insert_query += "}"
        
        await self.jena_manager.sparql_update.setQuery(insert_query)
        await self.jena_manager.sparql_update.query()
    
    async def neural_reasoning_with_jena(
        self,
        query: str,
        neural_model: nn.Module,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Perform iterative neural reasoning with Jena knowledge base
        
        Args:
            query: Natural language query
            neural_model: Neural model for reasoning
            max_iterations: Maximum reasoning iterations
            
        Returns:
            Reasoning results with explanation
        """
        # Initialize reasoning state
        context = self.neural_retriever.neural_encoder.encode([query])[0]
        reasoning_chain = []
        
        for iteration in range(max_iterations):
            # Query Jena for relevant knowledge
            sparql_query = self._generate_sparql_from_context(context)
            knowledge = await self.query_with_neural_context(
                sparql_query, context, top_k=5
            )
            
            # Neural reasoning step
            if knowledge:
                knowledge_embeddings = torch.stack([
                    await self.embed_knowledge_entity(k["entity"], k["data"])
                    for k in knowledge
                ])
                
                # Apply neural model
                reasoning_output = neural_model(knowledge_embeddings)
                
                # Update context
                context = (context + reasoning_output.mean(dim=0)) / 2
                
                reasoning_chain.append({
                    "iteration": iteration,
                    "knowledge": knowledge,
                    "output": reasoning_output
                })
                
                # Check for convergence
                if iteration > 0:
                    prev_output = reasoning_chain[-2]["output"]
                    similarity = torch.cosine_similarity(
                        reasoning_output.mean(dim=0),
                        prev_output.mean(dim=0),
                        dim=0
                    )
                    
                    if similarity > 0.95:  # Converged
                        break
        
        return {
            "final_output": reasoning_chain[-1]["output"] if reasoning_chain else None,
            "reasoning_chain": reasoning_chain,
            "iterations": len(reasoning_chain)
        }
    
    def _generate_sparql_from_context(self, context: torch.Tensor) -> str:
        """Generate SPARQL query from neural context"""
        # This is a simplified example - in practice, you'd use a
        # neural model to generate the query
        return """
        SELECT ?entity ?type ?label
        WHERE {
            ?entity rdf:type ?type ;
                   rdfs:label ?label .
        }
        LIMIT 100
        """
    
    async def close(self):
        """Cleanup resources"""
        # Clear caches
        if self.embedding_cache:
            self.embedding_cache.clear()
        
        # Close connections
        if hasattr(self.jena_manager, 'close'):
            await self.jena_manager.close()


# Example usage for Digital Human integration
class DigitalHumanNeuralInterface:
    """
    Specialized interface for Digital Human neural-knowledge integration
    """
    
    def __init__(self, config: NeuralJenaConfig):
        self.integration = NeuralJenaIntegration(config)
        
        # Initialize hybrid neural-symbolic layer
        self.hybrid_layer = HybridNeuralSymbolicLayer(
            neural_dim=config.embedding_dim,
            symbolic_dim=256,
            num_rules=100
        )
    
    async def process_user_query(
        self,
        user_id: str,
        query: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process user query with neural-symbolic reasoning
        """
        # Get user context from Jena
        user_query = f"""
        PREFIX user: <http://aiqtoolkit.com/user/>
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        
        SELECT ?property ?value
        WHERE {{
            <{user_id}> ?property ?value .
        }}
        """
        
        user_context = await self.integration.jena_manager.execute_sparql_query(user_query)
        
        # Encode query and context
        query_embedding = self.integration.neural_retriever.neural_encoder.encode([query])[0]
        
        # Apply hybrid reasoning
        neural_output = self.hybrid_layer(query_embedding.unsqueeze(0))
        
        # Query knowledge base with neural context
        relevant_knowledge = await self.integration.query_with_neural_context(
            self._generate_financial_sparql(query),
            neural_output.squeeze(0),
            top_k=5
        )
        
        # Generate response
        response = {
            "answer": self._generate_answer(relevant_knowledge),
            "confidence": self._calculate_confidence(neural_output),
            "knowledge_used": relevant_knowledge,
            "session_id": session_id
        }
        
        # Store interaction in Jena
        await self._store_interaction(user_id, session_id, query, response)
        
        return response
    
    def _generate_financial_sparql(self, query: str) -> str:
        """Generate financial knowledge SPARQL query"""
        return """
        PREFIX finance: <http://aiqtoolkit.com/finance/>
        
        SELECT ?entity ?type ?value
        WHERE {
            ?entity rdf:type finance:FinancialInstrument ;
                   ?property ?value .
        }
        LIMIT 20
        """
    
    def _generate_answer(self, knowledge: List[Dict[str, Any]]) -> str:
        """Generate natural language answer from knowledge"""
        if not knowledge:
            return "I couldn't find relevant information to answer your query."
        
        # Simple template-based generation
        top_result = knowledge[0]
        return f"Based on the available information: {top_result['data']}"
    
    def _calculate_confidence(self, neural_output: torch.Tensor) -> float:
        """Calculate confidence score from neural output"""
        # Use entropy as inverse confidence measure
        probs = torch.softmax(neural_output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        confidence = 1.0 - (entropy / torch.log(torch.tensor(neural_output.shape[-1])))
        return confidence.item()
    
    async def _store_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: Dict[str, Any]
    ):
        """Store interaction in Jena"""
        interaction_uri = f"interaction:{session_id}:{hash(query)}"
        
        insert_query = f"""
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        PREFIX user: <http://aiqtoolkit.com/user/>
        
        INSERT DATA {{
            <{interaction_uri}> rdf:type dh:Interaction ;
                dh:user <{user_id}> ;
                dh:session "{session_id}" ;
                dh:query "{query}" ;
                dh:response "{response['answer']}" ;
                dh:confidence {response['confidence']} ;
                dh:timestamp "{torch.tensor([]).new_zeros(1).item()}" .
        }}
        """
        
        await self.integration.jena_manager.sparql_update.setQuery(insert_query)
        await self.integration.jena_manager.sparql_update.query()