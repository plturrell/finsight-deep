# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AIQ Toolkit Memory Module

This package provides foundational classes and interfaces
for managing text-based memory in AIQ Toolkit's LLM-based agents.

It also includes a Cross-Framework Memory System for persisting research 
context across different frameworks like Haystack, LlamaIndex, LangChain, 
and SGLang.
"""

from aiq.memory.interfaces import (
    MemoryEditor, 
    MemoryIOBase, 
    MemoryReader, 
    MemoryWriter, 
    MemoryManager
)
from aiq.memory.models import MemoryItem, SearchMemoryInput, DeleteMemoryInput
from aiq.memory.research import (
    ResearchEntity,
    ResearchRelation,
    ResearchContext,
    ResearchContextMemoryEditor,
    ResearchContextReader,
    ResearchContextWriter,
    ResearchContextManager,
    CrossFrameworkMemory,
    MemoryScope,
    FrameworkAdapter,
    HaystackAdapter,
    LlamaIndexAdapter,
    LangChainAdapter,
    SLLangAdapter
)

__all__ = [
    # Base memory interfaces and models
    "MemoryEditor",
    "MemoryIOBase",
    "MemoryReader",
    "MemoryWriter",
    "MemoryManager",
    "MemoryItem",
    "SearchMemoryInput",
    "DeleteMemoryInput",
    
    # Research context memory system
    "ResearchEntity",
    "ResearchRelation",
    "ResearchContext",
    "ResearchContextMemoryEditor",
    "ResearchContextReader",
    "ResearchContextWriter",
    "ResearchContextManager",
    "CrossFrameworkMemory",
    "MemoryScope",
    
    # Framework adapters
    "FrameworkAdapter",
    "HaystackAdapter",
    "LlamaIndexAdapter",
    "LangChainAdapter",
    "SLLangAdapter"
]