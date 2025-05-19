# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIQ Toolkit Digital Human System

This module provides components for creating realistic digital humans with
GPU-accelerated processing, natural conversation, emotional intelligence,
and advanced reasoning capabilities powered by neural supercomputer.
"""


from aiq.digital_human.conversation.sglang_engine import SgLangConversationEngine
from aiq.digital_human.conversation.emotional_mapper import EmotionalResponseMapper
from aiq.digital_human.conversation.context_manager import FinancialContextManager
from aiq.digital_human.conversation.conversation_orchestrator import ConversationOrchestrator

from aiq.digital_human.avatar.facial_animator import FacialAnimationSystem
from aiq.digital_human.avatar.emotion_renderer import EmotionRenderer
from aiq.digital_human.avatar.avatar_controller import AvatarController
from aiq.digital_human.avatar.expression_library import ExpressionLibrary

from aiq.digital_human.orchestrator.digital_human_orchestrator import (
    DigitalHumanOrchestrator,
    StateManager,
    ResponseGenerator,
    PerformanceMonitor
)

from aiq.digital_human.financial import (
    MCTSFinancialAnalyzer,
    FinancialDataProcessor,
    PortfolioOptimizer,
    RiskAssessmentEngine
)

__all__ = [
    # Financial Analysis
    'MCTSFinancialAnalyzer',
    'FinancialDataProcessor',
    'PortfolioOptimizer',
    'RiskAssessmentEngine',
    
    # Conversation
    'SgLangConversationEngine',
    'EmotionalResponseMapper',
    'FinancialContextManager',
    'ConversationOrchestrator',
    
    # Avatar
    'FacialAnimationSystem',
    'EmotionRenderer',
    'AvatarController',
    'ExpressionLibrary',
    
    # Orchestration
    'DigitalHumanOrchestrator',
    'StateManager',
    'ResponseGenerator',
    'PerformanceMonitor'
]