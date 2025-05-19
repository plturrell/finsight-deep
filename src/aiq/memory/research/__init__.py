# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIQ Toolkit Research Context Memory System

This module provides cross-framework memory persistence for research contexts,
enabling seamless integration between different AI frameworks.
"""

from aiq.memory.research.research_context import (
    ResearchEntity,
    ResearchRelation,
    ResearchContext,
    ResearchContextMemoryEditor,
    ResearchContextReader,
    ResearchContextWriter,
    ResearchContextManager,
    CrossFrameworkMemory,
    MemoryScope
)

from aiq.memory.research.framework_adapters import (
    FrameworkAdapter,
    HaystackAdapter,
    LlamaIndexAdapter,
    LangChainAdapter,
    SLLangAdapter,
    create_haystack_adapter,
    create_llama_index_adapter,
    create_langchain_adapter,
    create_sllang_adapter
)

__all__ = [
    # Core classes
    'ResearchEntity',
    'ResearchRelation',
    'ResearchContext',
    'ResearchContextMemoryEditor',
    'ResearchContextReader',
    'ResearchContextWriter',
    'ResearchContextManager',
    'CrossFrameworkMemory',
    'MemoryScope',
    
    # Framework adapters
    'FrameworkAdapter',
    'HaystackAdapter',
    'LlamaIndexAdapter',
    'LangChainAdapter',
    'SLLangAdapter',
    'create_haystack_adapter',
    'create_llama_index_adapter',
    'create_langchain_adapter',
    'create_sllang_adapter'
]