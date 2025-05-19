# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
import torch
import json
import pickle
from pathlib import Path
import logging
from enum import Enum

from aiq.memory.interfaces import MemoryEditor, MemoryReader, MemoryWriter, MemoryManager
from aiq.memory.models import MemoryItem
from aiq.cuda_kernels import cosine_similarity

logger = logging.getLogger(__name__)


class MemoryScope(Enum):
    """Scope of memory persistence"""
    SESSION = "session"
    PROJECT = "project"
    GLOBAL = "global"


@dataclass
class ResearchEntity:
    """An entity in research context"""
    entity_id: str
    entity_type: str
    content: str
    embeddings: Optional[torch.Tensor] = None
    properties: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    scope: MemoryScope = MemoryScope.SESSION
    framework_source: Optional[str] = None  # Which framework created this
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ResearchRelation:
    """A relation between entities"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    evidence: Optional[List[str]] = None
    created_at: datetime = None
    framework_source: Optional[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ResearchContext:
    """Research context with entities and relations"""
    context_id: str
    entities: Dict[str, ResearchEntity]
    relations: List[ResearchRelation]
    metadata: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def add_entity(self, entity: ResearchEntity):
        """Add or update an entity"""
        self.entities[entity.entity_id] = entity
        self.updated_at = datetime.now()
    
    def add_relation(self, relation: ResearchRelation):
        """Add a relation"""
        self.relations.append(relation)
        self.updated_at = datetime.now()
    
    def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> Set[str]:
        """Get entities related to a given entity"""
        related = set()
        queue = [(entity_id, 0)]
        visited = set()
        
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth >= max_depth:
                continue
            
            visited.add(current_id)
            
            for relation in self.relations:
                if relation.source_id == current_id:
                    if relation_type is None or relation.relation_type == relation_type:
                        related.add(relation.target_id)
                        queue.append((relation.target_id, depth + 1))
        
        return related
    
    def merge_context(self, other: 'ResearchContext', overwrite: bool = False):
        """Merge another context into this one"""
        # Merge entities
        for entity_id, entity in other.entities.items():
            if entity_id not in self.entities or overwrite:
                self.entities[entity_id] = entity
            else:
                # Merge properties
                existing = self.entities[entity_id]
                existing.properties.update(entity.properties)
                existing.updated_at = datetime.now()
        
        # Merge relations
        existing_relations = {
            (r.source_id, r.target_id, r.relation_type)
            for r in self.relations
        }
        
        for relation in other.relations:
            key = (relation.source_id, relation.target_id, relation.relation_type)
            if key not in existing_relations:
                self.relations.append(relation)
        
        self.updated_at = datetime.now()


class ResearchContextMemoryEditor(MemoryEditor):
    """Editor for research context memory"""
    
    def __init__(self, context: ResearchContext):
        self.context = context
    
    def edit(self, memory_id: str, new_content: str) -> bool:
        """Edit an entity's content"""
        if memory_id in self.context.entities:
            entity = self.context.entities[memory_id]
            entity.content = new_content
            entity.updated_at = datetime.now()
            return True
        return False
    
    def add_property(self, entity_id: str, key: str, value: Any) -> bool:
        """Add or update a property on an entity"""
        if entity_id in self.context.entities:
            entity = self.context.entities[entity_id]
            entity.properties[key] = value
            entity.updated_at = datetime.now()
            return True
        return False


class ResearchContextReader(MemoryReader):
    """Reader for research context memory"""
    
    def __init__(self, context: ResearchContext):
        self.context = context
    
    def read(self, memory_id: str) -> Optional[MemoryItem]:
        """Read an entity as memory item"""
        if memory_id in self.context.entities:
            entity = self.context.entities[memory_id]
            return MemoryItem(
                id=entity.entity_id,
                content=entity.content,
                metadata={
                    "entity_type": entity.entity_type,
                    "properties": entity.properties,
                    "created_at": entity.created_at.isoformat(),
                    "updated_at": entity.updated_at.isoformat(),
                    "scope": entity.scope.value,
                    "framework_source": entity.framework_source
                }
            )
        return None
    
    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[MemoryItem]:
        """Search entities by content similarity"""
        results = []
        
        # If embeddings are available, use them
        entities_with_embeddings = [
            e for e in self.context.entities.values()
            if e.embeddings is not None
        ]
        
        if entities_with_embeddings:
            # TODO: Implement embedding-based search
            pass
        
        # Fallback to simple text matching
        query_lower = query.lower()
        for entity in self.context.entities.values():
            if query_lower in entity.content.lower():
                results.append(self.read(entity.entity_id))
                if len(results) >= limit:
                    break
        
        return results
    
    def get_related(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Get related entities"""
        related_ids = self.context.get_related_entities(
            entity_id,
            relation_type,
            max_depth=2
        )
        
        results = []
        for rel_id in list(related_ids)[:limit]:
            item = self.read(rel_id)
            if item:
                results.append(item)
        
        return results


class ResearchContextWriter(MemoryWriter):
    """Writer for research context memory"""
    
    def __init__(self, context: ResearchContext):
        self.context = context
    
    def write(self, memory_item: MemoryItem) -> bool:
        """Write a memory item as an entity"""
        entity = ResearchEntity(
            entity_id=memory_item.id,
            entity_type=memory_item.metadata.get("entity_type", "generic"),
            content=memory_item.content,
            properties=memory_item.metadata.get("properties", {}),
            scope=MemoryScope(memory_item.metadata.get("scope", "session")),
            framework_source=memory_item.metadata.get("framework_source")
        )
        
        self.context.add_entity(entity)
        return True
    
    def write_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 1.0,
        evidence: Optional[List[str]] = None
    ) -> bool:
        """Write a relation between entities"""
        if source_id in self.context.entities and target_id in self.context.entities:
            relation = ResearchRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=confidence,
                evidence=evidence
            )
            
            self.context.add_relation(relation)
            return True
        return False
    
    def delete(self, memory_id: str) -> bool:
        """Delete an entity and its relations"""
        if memory_id in self.context.entities:
            # Remove entity
            del self.context.entities[memory_id]
            
            # Remove related relations
            self.context.relations = [
                r for r in self.context.relations
                if r.source_id != memory_id and r.target_id != memory_id
            ]
            
            self.context.updated_at = datetime.now()
            return True
        return False


class ResearchContextManager(MemoryManager):
    """Manager for research contexts"""
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        device: str = 'cuda'
    ):
        self.storage_path = storage_path or Path.home() / ".aiq" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.contexts: Dict[str, ResearchContext] = {}
        self.current_context_id: Optional[str] = None
    
    def create_context(
        self,
        context_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ResearchContext:
        """Create a new research context"""
        context = ResearchContext(
            context_id=context_id,
            entities={},
            relations=[],
            metadata=metadata or {}
        )
        
        self.contexts[context_id] = context
        self.current_context_id = context_id
        return context
    
    def get_context(self, context_id: str) -> Optional[ResearchContext]:
        """Get a research context"""
        return self.contexts.get(context_id)
    
    def set_current_context(self, context_id: str) -> bool:
        """Set the current active context"""
        if context_id in self.contexts:
            self.current_context_id = context_id
            return True
        return False
    
    def get_reader(self, context_id: Optional[str] = None) -> ResearchContextReader:
        """Get reader for a context"""
        context_id = context_id or self.current_context_id
        if context_id and context_id in self.contexts:
            return ResearchContextReader(self.contexts[context_id])
        raise ValueError(f"Context {context_id} not found")
    
    def get_writer(self, context_id: Optional[str] = None) -> ResearchContextWriter:
        """Get writer for a context"""
        context_id = context_id or self.current_context_id
        if context_id and context_id in self.contexts:
            return ResearchContextWriter(self.contexts[context_id])
        raise ValueError(f"Context {context_id} not found")
    
    def get_editor(self, context_id: Optional[str] = None) -> ResearchContextMemoryEditor:
        """Get editor for a context"""
        context_id = context_id or self.current_context_id
        if context_id and context_id in self.contexts:
            return ResearchContextMemoryEditor(self.contexts[context_id])
        raise ValueError(f"Context {context_id} not found")
    
    def save_context(self, context_id: str) -> bool:
        """Save a context to disk"""
        if context_id not in self.contexts:
            return False
        
        context = self.contexts[context_id]
        file_path = self.storage_path / f"{context_id}.pkl"
        
        # Convert tensors to numpy for serialization
        serializable_context = self._make_serializable(context)
        
        with open(file_path, 'wb') as f:
            pickle.dump(serializable_context, f)
        
        logger.info(f"Saved context {context_id} to {file_path}")
        return True
    
    def load_context(self, context_id: str) -> bool:
        """Load a context from disk"""
        file_path = self.storage_path / f"{context_id}.pkl"
        
        if not file_path.exists():
            return False
        
        with open(file_path, 'rb') as f:
            serializable_context = pickle.load(f)
        
        # Convert numpy arrays back to tensors
        context = self._restore_tensors(serializable_context)
        self.contexts[context_id] = context
        
        logger.info(f"Loaded context {context_id} from {file_path}")
        return True
    
    def _make_serializable(self, context: ResearchContext) -> ResearchContext:
        """Convert tensors to numpy for serialization"""
        # Deep copy to avoid modifying original
        import copy
        context_copy = copy.deepcopy(context)
        
        for entity in context_copy.entities.values():
            if entity.embeddings is not None:
                entity.embeddings = entity.embeddings.cpu().numpy()
        
        return context_copy
    
    def _restore_tensors(self, context: ResearchContext) -> ResearchContext:
        """Restore tensors from numpy arrays"""
        for entity in context.entities.values():
            if entity.embeddings is not None and isinstance(entity.embeddings, np.ndarray):
                entity.embeddings = torch.tensor(
                    entity.embeddings,
                    device=self.device
                )
        
        return context


class CrossFrameworkMemory:
    """
    Memory system that persists research context across different frameworks
    """
    def __init__(self, manager: ResearchContextManager):
        self.manager = manager
        self.framework_adapters = {}
    
    def register_framework_adapter(self, framework_name: str, adapter: Any):
        """Register an adapter for a specific framework"""
        self.framework_adapters[framework_name] = adapter
    
    def sync_from_framework(
        self,
        framework_name: str,
        context_id: Optional[str] = None
    ) -> bool:
        """Sync memory from a framework to research context"""
        if framework_name not in self.framework_adapters:
            logger.warning(f"No adapter registered for {framework_name}")
            return False
        
        adapter = self.framework_adapters[framework_name]
        context = self.manager.get_context(context_id or self.manager.current_context_id)
        
        if not context:
            logger.error(f"Context {context_id} not found")
            return False
        
        # Get memories from framework
        framework_memories = adapter.get_memories()
        
        # Convert to research entities
        writer = self.manager.get_writer(context.context_id)
        for memory in framework_memories:
            entity_item = MemoryItem(
                id=memory.get("id", f"{framework_name}_{hash(memory['content'])}"),
                content=memory["content"],
                metadata={
                    "framework_source": framework_name,
                    **memory.get("metadata", {})
                }
            )
            writer.write(entity_item)
        
        return True
    
    def sync_to_framework(
        self,
        framework_name: str,
        context_id: Optional[str] = None
    ) -> bool:
        """Sync research context to a framework"""
        if framework_name not in self.framework_adapters:
            logger.warning(f"No adapter registered for {framework_name}")
            return False
        
        adapter = self.framework_adapters[framework_name]
        context = self.manager.get_context(context_id or self.manager.current_context_id)
        
        if not context:
            logger.error(f"Context {context_id} not found")
            return False
        
        # Get entities for this framework
        reader = self.manager.get_reader(context.context_id)
        
        for entity in context.entities.values():
            if entity.framework_source == framework_name or entity.framework_source is None:
                memory_item = reader.read(entity.entity_id)
                adapter.add_memory(memory_item)
        
        return True
    
    def get_unified_memory(
        self,
        query: str,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Get memories from all frameworks and research context"""
        results = []
        
        # Get from research context
        if self.manager.current_context_id:
            reader = self.manager.get_reader()
            results.extend(reader.search(query, limit=limit))
        
        # Get from all frameworks
        for framework_name, adapter in self.framework_adapters.items():
            framework_results = adapter.search(query, limit=limit)
            for result in framework_results:
                # Add framework source if not present
                if "framework_source" not in result.metadata:
                    result.metadata["framework_source"] = framework_name
                results.append(result)
        
        # Deduplicate and sort by relevance
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        return unique_results[:limit]