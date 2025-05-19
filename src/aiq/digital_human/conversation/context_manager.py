"""
Financial Context Manager for digital human conversations.

Manages conversation context, user preferences, and reasoning states.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime
import numpy as np

from aiq.memory.research.research_context import ResearchContext
from aiq.data_models.intermediate_step import IntermediateStep


@dataclass
class UserProfile:
    """User profile and preferences"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    knowledge_level: str = "intermediate"  # beginner, intermediate, expert
    communication_style: str = "balanced"  # formal, casual, balanced
    topics_of_interest: List[str] = field(default_factory=list)


@dataclass
class ConversationState:
    """Current conversation state"""
    active_topic: str
    depth_level: int  # 1-5, how deep into the topic
    clarity_score: float  # 0-1, how clear the conversation is
    engagement_score: float  # 0-1, user engagement level
    confusion_indicators: List[str]
    key_concepts: List[str]
    pending_clarifications: List[str]


class FinancialContextManager:
    """
    Manages conversation context for digital human interactions.
    Integrates with research context and reasoning systems.
    """
    
    def __init__(
        self,
        max_history_size: int = 100,
        context_window: int = 10
    ):
        self.max_history_size = max_history_size
        self.context_window = context_window
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Active conversation states
        self.conversation_states: Dict[str, ConversationState] = {}
        
        # Research contexts per session
        self.research_contexts: Dict[str, ResearchContext] = {}
        
        # Topic knowledge graph
        self.topic_graph = self._initialize_topic_graph()
    
    def _initialize_topic_graph(self) -> Dict[str, List[str]]:
        """Initialize topic relationship graph"""
        return {
            "science": ["physics", "chemistry", "biology", "astronomy"],
            "physics": ["quantum", "mechanics", "thermodynamics", "relativity"],
            "technology": ["ai", "software", "hardware", "networking"],
            "ai": ["machine learning", "deep learning", "nlp", "computer vision"],
            "history": ["ancient", "medieval", "modern", "contemporary"],
            "economics": ["microeconomics", "macroeconomics", "finance", "trade"],
            "philosophy": ["ethics", "metaphysics", "epistemology", "logic"],
            "mathematics": ["algebra", "calculus", "statistics", "geometry"]
        }
    
    def create_user_profile(
        self,
        user_id: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """Create or update user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences=preferences or {},
                interaction_history=[]
            )
        elif preferences:
            self.user_profiles[user_id].preferences.update(preferences)
        
        return self.user_profiles[user_id]
    
    def initialize_conversation(
        self,
        session_id: str,
        user_id: str,
        initial_topic: Optional[str] = None
    ) -> ConversationState:
        """Initialize conversation state for a session"""
        # Create research context
        self.research_contexts[session_id] = ResearchContext()
        
        # Create conversation state
        self.conversation_states[session_id] = ConversationState(
            active_topic=initial_topic or "general",
            depth_level=1,
            clarity_score=1.0,
            engagement_score=0.5,
            confusion_indicators=[],
            key_concepts=[],
            pending_clarifications=[]
        )
        
        return self.conversation_states[session_id]
    
    def update_context(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Dict[str, Any]
    ):
        """Update conversation context with new interaction"""
        if session_id not in self.conversation_states:
            return
        
        state = self.conversation_states[session_id]
        research_context = self.research_contexts.get(session_id)
        
        # Update topic if changed
        new_topic = metadata.get("detected_topic")
        if new_topic and new_topic != state.active_topic:
            state.active_topic = new_topic
            state.depth_level = 1
        
        # Update depth level based on complexity
        complexity = metadata.get("complexity", "medium")
        if complexity == "high":
            state.depth_level = min(5, state.depth_level + 1)
        
        # Update clarity score
        state.clarity_score = self._calculate_clarity_score(
            user_message,
            assistant_response,
            metadata
        )
        
        # Update engagement score
        state.engagement_score = self._calculate_engagement_score(
            user_message,
            metadata
        )
        
        # Detect confusion indicators
        confusion_indicators = self._detect_confusion(user_message)
        state.confusion_indicators.extend(confusion_indicators)
        
        # Extract key concepts
        key_concepts = metadata.get("key_concepts", [])
        state.key_concepts.extend(key_concepts)
        state.key_concepts = list(set(state.key_concepts))  # Remove duplicates
        
        # Update research context if available
        if research_context:
            research_context.add_finding(assistant_response, metadata)
            
            # Add entities and relations from metadata
            entities = metadata.get("entities", [])
            for entity in entities:
                research_context.add_entity(entity["name"], entity["type"])
            
            relations = metadata.get("relations", [])
            for relation in relations:
                research_context.add_relation(
                    relation["source"],
                    relation["relation"],
                    relation["target"]
                )
    
    def _calculate_clarity_score(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate conversation clarity score"""
        score = 1.0
        
        # Check for confusion indicators in user message
        confusion_words = ["confused", "don't understand", "what", "unclear", "explain"]
        for word in confusion_words:
            if word in user_message.lower():
                score -= 0.1
        
        # Check response confidence
        confidence = metadata.get("confidence_score", 0.8)
        score *= confidence
        
        # Check for clarification requests
        if "?" in user_message:
            question_count = user_message.count("?")
            score -= question_count * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_engagement_score(
        self,
        user_message: str,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate user engagement score"""
        score = 0.5
        
        # Message length indicates engagement
        message_length = len(user_message.split())
        if message_length > 20:
            score += 0.2
        elif message_length < 5:
            score -= 0.1
        
        # Questions indicate engagement
        if "?" in user_message:
            score += 0.1
        
        # Emotional indicators
        positive_words = ["interesting", "fascinating", "great", "amazing", "cool"]
        negative_words = ["boring", "tired", "stop", "enough"]
        
        for word in positive_words:
            if word in user_message.lower():
                score += 0.1
        
        for word in negative_words:
            if word in user_message.lower():
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _detect_confusion(self, user_message: str) -> List[str]:
        """Detect confusion indicators in user message"""
        indicators = []
        message_lower = user_message.lower()
        
        confusion_patterns = {
            "clarification_request": ["what do you mean", "can you explain", "i don't get"],
            "uncertainty": ["not sure", "confused", "lost", "don't understand"],
            "repetition_request": ["again", "repeat", "say that again"],
            "example_request": ["example", "for instance", "such as"],
            "definition_request": ["what is", "what does", "meaning of"]
        }
        
        for category, patterns in confusion_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    indicators.append(category)
                    break
        
        return indicators
    
    def get_conversation_context(
        self,
        session_id: str,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get current conversation context"""
        if session_id not in self.conversation_states:
            return {}
        
        state = self.conversation_states[session_id]
        research_context = self.research_contexts.get(session_id)
        
        # Get recent history window
        window_size = window_size or self.context_window
        
        context = {
            "active_topic": state.active_topic,
            "depth_level": state.depth_level,
            "clarity_score": state.clarity_score,
            "engagement_score": state.engagement_score,
            "key_concepts": state.key_concepts[-window_size:],
            "confusion_indicators": state.confusion_indicators[-window_size:],
            "pending_clarifications": state.pending_clarifications
        }
        
        # Add research context summary
        if research_context:
            context["research_summary"] = research_context.get_summary()
            context["entities"] = research_context.entities[-window_size:]
            context["relations"] = research_context.relations[-window_size:]
        
        return context
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences for personalization"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        return {
            "knowledge_level": profile.knowledge_level,
            "communication_style": profile.communication_style,
            "topics_of_interest": profile.topics_of_interest,
            "preferences": profile.preferences
        }
    
    def suggest_next_topic(
        self,
        session_id: str,
        user_id: str
    ) -> List[str]:
        """Suggest relevant next topics based on context"""
        if session_id not in self.conversation_states:
            return []
        
        state = self.conversation_states[session_id]
        current_topic = state.active_topic
        
        # Get related topics from graph
        related_topics = self.topic_graph.get(current_topic, [])
        
        # Consider user interests
        if user_id in self.user_profiles:
            user_interests = self.user_profiles[user_id].topics_of_interest
            # Prioritize topics that match user interests
            related_topics = sorted(
                related_topics,
                key=lambda t: t in user_interests,
                reverse=True
            )
        
        # Consider conversation depth
        if state.depth_level > 3:
            # Suggest broader topics if too deep
            parent_topics = [
                topic for topic, children in self.topic_graph.items()
                if current_topic in children
            ]
            related_topics = parent_topics + related_topics
        
        return related_topics[:5]
    
    def should_clarify(self, session_id: str) -> bool:
        """Determine if clarification is needed"""
        if session_id not in self.conversation_states:
            return False
        
        state = self.conversation_states[session_id]
        
        # Check clarity score
        if state.clarity_score < 0.6:
            return True
        
        # Check confusion indicators
        if len(state.confusion_indicators) > 2:
            return True
        
        # Check pending clarifications
        if state.pending_clarifications:
            return True
        
        return False
    
    def get_clarification_suggestions(
        self,
        session_id: str
    ) -> List[str]:
        """Get suggestions for clarification"""
        if session_id not in self.conversation_states:
            return []
        
        state = self.conversation_states[session_id]
        suggestions = []
        
        # Based on confusion indicators
        if "definition_request" in state.confusion_indicators:
            suggestions.append(f"Let me define the key terms related to {state.active_topic}")
        
        if "example_request" in state.confusion_indicators:
            suggestions.append(f"Here's a concrete example of {state.active_topic}")
        
        if "clarification_request" in state.confusion_indicators:
            suggestions.append("Let me break this down into simpler concepts")
        
        # Based on key concepts
        for concept in state.key_concepts[-3:]:
            suggestions.append(f"Would you like me to explain more about {concept}?")
        
        return suggestions[:3]
    
    def update_user_preferences(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Update user preferences based on interaction"""
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Update topics of interest
        topic = interaction_data.get("topic")
        engagement = interaction_data.get("engagement_score", 0.5)
        
        if topic and engagement > 0.7:
            if topic not in profile.topics_of_interest:
                profile.topics_of_interest.append(topic)
        
        # Update knowledge level based on depth
        depth = interaction_data.get("depth_level", 1)
        if depth > 3:
            profile.knowledge_level = "expert"
        elif depth > 2:
            profile.knowledge_level = "intermediate"
        
        # Update communication style based on language
        formality = interaction_data.get("formality_score", 0.5)
        if formality > 0.7:
            profile.communication_style = "formal"
        elif formality < 0.3:
            profile.communication_style = "casual"
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session"""
        if session_id not in self.conversation_states:
            return {}
        
        state = self.conversation_states[session_id]
        research_context = self.research_contexts.get(session_id)
        
        summary = {
            "topics_covered": list(set([state.active_topic] + 
                                     [e["type"] for e in research_context.entities if "type" in e])),
            "key_concepts": state.key_concepts,
            "depth_achieved": state.depth_level,
            "clarity_score": state.clarity_score,
            "engagement_score": state.engagement_score,
            "confusion_points": state.confusion_indicators,
            "clarifications_provided": []  # Would track these
        }
        
        if research_context:
            summary["research_findings"] = len(research_context.findings)
            summary["entities_discovered"] = len(research_context.entities)
            summary["relations_identified"] = len(research_context.relations)
        
        return summary