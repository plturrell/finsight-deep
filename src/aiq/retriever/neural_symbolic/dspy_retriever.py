# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DSPy-compatible wrapper for Neural-Symbolic Retriever
"""

from typing import List, Optional, Union, Any
import dspy
from aiq.retriever.neural_symbolic.neural_symbolic_retriever import (
    NeuralSymbolicRetriever,
    create_neural_symbolic_retriever
)
from aiq.retriever.models import Document

class DSPyNeuralSymbolicRetriever(dspy.Retrieve):
    """
    DSPy-compatible retriever that combines neural and symbolic search
    """
    def __init__(
        self,
        dataset_name: str,
        model_type: str = "neuro",
        knowledge_fusion: bool = True,
        device: Optional[str] = None,
        reasoning_weight: float = 0.3,
        k: int = 10,
        **kwargs
    ):
        super().__init__(k=k)
        
        # Initialize the neural-symbolic retriever
        self.retriever = create_neural_symbolic_retriever(
            dataset_name=dataset_name,
            model_type=model_type,
            knowledge_fusion=knowledge_fusion,
            device=device,
            reasoning_weight=reasoning_weight,
            **kwargs
        )
        
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.knowledge_fusion = knowledge_fusion
    
    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
        **kwargs
    ) -> Union[List[dspy.Example], List[List[dspy.Example]]]:
        """
        Forward pass for DSPy compatibility
        
        Args:
            query_or_queries: Single query or list of queries
            k: Number of passages to retrieve
            **kwargs: Additional arguments for retrieval
        
        Returns:
            DSPy Examples with retrieved passages
        """
        k = k or self.k
        
        # Handle single query
        if isinstance(query_or_queries, str):
            return self._retrieve_single(query_or_queries, k, **kwargs)
        
        # Handle multiple queries
        return [self._retrieve_single(q, k, **kwargs) for q in query_or_queries]
    
    async def _retrieve_single(
        self,
        query: str,
        k: int,
        use_reasoning: bool = True,
        **kwargs
    ) -> List[dspy.Example]:
        """Retrieve for a single query"""
        # Perform neural-symbolic retrieval
        result = await self.retriever.search(
            query=query,
            top_k=k,
            use_reasoning=use_reasoning,
            **kwargs
        )
        
        # Convert to DSPy Examples
        examples = []
        for doc in result.documents:
            example = dspy.Example(
                query=query,
                passage=doc.content,
                score=doc.metadata.get("combined_score", 1.0),
                neural_score=doc.metadata.get("neural_score", 1.0),
                symbolic_score=doc.metadata.get("symbolic_score", 0.0),
                explanations=doc.metadata.get("explanations", {})
            )
            
            # Add additional metadata
            for key, value in doc.metadata.items():
                if key not in ["combined_score", "neural_score", "symbolic_score", "explanations"]:
                    setattr(example, key, value)
            
            examples.append(example)
        
        return examples
    
    def add_documents(self, documents: List[Union[str, Document]], **kwargs):
        """Add documents to the retriever"""
        # Convert strings to Document objects if needed
        doc_objects = []
        for doc in documents:
            if isinstance(doc, str):
                doc_objects.append(Document(content=doc, metadata={}))
            else:
                doc_objects.append(doc)
        
        self.retriever.add_documents(doc_objects, **kwargs)
    
    def set_reasoning_rules(self, rules: List[dict]):
        """Set custom reasoning rules"""
        self.retriever.set_reasoning_rules(rules)
    
    def save(self, path: str):
        """Save retriever state"""
        self.retriever.save(path)
    
    def load(self, path: str):
        """Load retriever state"""
        self.retriever.load(path)
    
    @property
    def corpus_size(self) -> int:
        """Get size of document corpus"""
        return len(self.retriever.documents)

# Example usage with DSPy
class ExampleDSPySystem(dspy.Module):
    """Example system using neural-symbolic retriever"""
    
    def __init__(self, retriever: DSPyNeuralSymbolicRetriever):
        super().__init__()
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question, k=5)
        
        # Use the passages for generation
        context = "\n\n".join([p.passage for p in passages])
        
        # Generate answer
        answer = self.generate(context=context, question=question)
        
        return dspy.Prediction(
            answer=answer.answer,
            passages=passages,
            reasoning_explanations=[p.explanations for p in passages]
        )

# Factory function for DSPy compatibility
def create_dspy_neural_symbolic_retriever(
    dataset_name: str,
    model_type: str = "neuro",
    knowledge_fusion: bool = True,
    device: Optional[str] = None,
    k: int = 10,
    **kwargs
) -> DSPyNeuralSymbolicRetriever:
    """Create a DSPy-compatible neural-symbolic retriever"""
    return DSPyNeuralSymbolicRetriever(
        dataset_name=dataset_name,
        model_type=model_type,
        knowledge_fusion=knowledge_fusion,
        device=device,
        k=k,
        **kwargs
    )