"""
Conversation Orchestrator for digital human system.

Coordinates conversation flow, manages dialogue state, and handles
multi-turn interactions with context awareness.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json

from aiq.digital_human.conversation.sglang_engine import (
    SgLangConversationEngine,
    ConversationContext
)
from aiq.digital_human.conversation.emotional_mapper import EmotionalResponseMapper
from aiq.digital_human.conversation.context_manager import FinancialContextManager


@dataclass
class DialogueState:
    """Represents current dialogue state"""
    turn_number: int
    current_topic: str
    previous_topics: List[str]
    open_questions: List[str]
    clarification_needed: bool
    dialogue_acts: List[str]
    sentiment_history: List[float]


@dataclass
class ConversationStrategy:
    """Defines conversation strategy parameters"""
    proactivity_level: float  # 0-1, how proactive to be
    clarification_threshold: float  # When to ask for clarification
    topic_switching_threshold: float  # When to suggest new topics
    empathy_level: float  # How much emotional support to provide
    teaching_mode: bool  # Whether to explain concepts in detail


class ConversationOrchestrator:
    """
    Orchestrates multi-turn conversations for digital humans,
    managing flow, context, and interaction strategies.
    """
    
    def __init__(
        self,
        conversation_engine: SgLangConversationEngine,
        emotion_mapper: EmotionalResponseMapper,
        context_manager: FinancialContextManager,
        strategy: Optional[ConversationStrategy] = None
    ):
        self.conversation_engine = conversation_engine
        self.emotion_mapper = emotion_mapper
        self.context_manager = context_manager
        
        # Default conversation strategy
        self.strategy = strategy or ConversationStrategy(
            proactivity_level=0.5,
            clarification_threshold=0.6,
            topic_switching_threshold=0.3,
            empathy_level=0.7,
            teaching_mode=True
        )
        
        # Active dialogue states
        self.dialogue_states: Dict[str, DialogueState] = {}
        
        # Conversation patterns
        self.dialogue_patterns = self._initialize_dialogue_patterns()
        
        # Topic transition rules
        self.topic_transitions = self._initialize_topic_transitions()
    
    def _initialize_dialogue_patterns(self) -> Dict[str, List[str]]:
        """Initialize common dialogue act patterns"""
        return {
            "greeting": ["acknowledge", "reciprocate", "inquire_purpose"],
            "question": ["acknowledge", "answer", "clarify_if_needed", "follow_up"],
            "statement": ["acknowledge", "respond", "elaborate", "relate"],
            "confusion": ["clarify", "simplify", "provide_example", "check_understanding"],
            "request": ["acknowledge", "fulfill", "explain_process", "confirm_completion"],
            "feedback": ["acknowledge", "thank", "incorporate", "improve"],
            "farewell": ["acknowledge", "summarize", "offer_future_help", "close"]
        }
    
    def _initialize_topic_transitions(self) -> Dict[str, List[str]]:
        """Initialize natural topic transition paths"""
        return {
            "science": ["technology", "research", "discovery", "innovation"],
            "technology": ["ai", "software", "hardware", "future"],
            "history": ["culture", "events", "people", "timeline"],
            "philosophy": ["ethics", "logic", "existence", "knowledge"],
            "current_events": ["politics", "economy", "society", "environment"],
            "personal": ["interests", "experiences", "goals", "challenges"]
        }
    
    async def start_conversation(
        self,
        session_id: str,
        user_id: str,
        initial_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a new conversation session"""
        # Initialize context
        context = self.context_manager.initialize_conversation(
            session_id,
            user_id,
            initial_topic="general"
        )
        
        # Initialize dialogue state
        self.dialogue_states[session_id] = DialogueState(
            turn_number=0,
            current_topic="general",
            previous_topics=[],
            open_questions=[],
            clarification_needed=False,
            dialogue_acts=[],
            sentiment_history=[]
        )
        
        # Generate initial response if no message provided
        if not initial_message:
            initial_message = "Hello! I'm here to help you explore any topic you're interested in."
            dialogue_act = "greeting"
        else:
            dialogue_act = await self._classify_dialogue_act(initial_message)
        
        # Process initial message
        response = await self.process_turn(
            session_id,
            initial_message,
            dialogue_act
        )
        
        return response
    
    async def process_turn(
        self,
        session_id: str,
        user_message: str,
        dialogue_act: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a conversation turn"""
        if session_id not in self.dialogue_states:
            raise ValueError(f"Unknown session: {session_id}")
        
        state = self.dialogue_states[session_id]
        state.turn_number += 1
        
        # Classify dialogue act if not provided
        if not dialogue_act:
            dialogue_act = await self._classify_dialogue_act(user_message)
        
        state.dialogue_acts.append(dialogue_act)
        
        # Get conversation context
        context = self.context_manager.get_conversation_context(session_id)
        
        # Analyze user intent and emotion
        user_emotion = await self.emotion_mapper.analyze_emotional_context(
            user_message,
            [],  # Conversation history from context
            context
        )
        
        # Determine response strategy
        response_strategy = await self._determine_response_strategy(
            dialogue_act,
            user_emotion,
            state,
            context
        )
        
        # Generate response based on strategy
        response_data = await self._generate_strategic_response(
            session_id,
            user_message,
            response_strategy,
            state,
            context
        )
        
        # Update dialogue state
        self._update_dialogue_state(
            state,
            user_message,
            response_data,
            user_emotion.intensity
        )
        
        # Update context manager
        self.context_manager.update_context(
            session_id,
            user_message,
            response_data["text"],
            response_data["metadata"]
        )
        
        return response_data
    
    async def _classify_dialogue_act(self, message: str) -> str:
        """Classify the dialogue act of a message"""
        message_lower = message.lower()
        
        # Simple rule-based classification
        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif "?" in message:
            return "question"
        elif any(word in message_lower for word in ["confused", "don't understand", "unclear"]):
            return "confusion"
        elif any(word in message_lower for word in ["please", "could you", "can you", "would you"]):
            return "request"
        elif any(word in message_lower for word in ["thank", "thanks", "good", "great", "helpful"]):
            return "feedback"
        elif any(word in message_lower for word in ["bye", "goodbye", "see you", "farewell"]):
            return "farewell"
        else:
            return "statement"
    
    async def _determine_response_strategy(
        self,
        dialogue_act: str,
        user_emotion: Any,
        state: DialogueState,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine appropriate response strategy"""
        strategy = {
            "dialogue_pattern": self.dialogue_patterns.get(dialogue_act, ["acknowledge", "respond"]),
            "emotional_tone": self._select_emotional_tone(user_emotion),
            "clarification_needed": context.get("clarity_score", 1.0) < self.strategy.clarification_threshold,
            "topic_transition": self._should_transition_topic(state, context),
            "teaching_level": self._determine_teaching_level(context),
            "proactivity": self._calculate_proactivity(state, context)
        }
        
        return strategy
    
    def _select_emotional_tone(self, user_emotion: Any) -> str:
        """Select appropriate emotional tone for response"""
        if user_emotion.primary_emotion.value in ["sad", "angry", "fearful"]:
            return "empathetic"
        elif user_emotion.primary_emotion.value in ["happy", "excited"]:
            return "enthusiastic"
        elif user_emotion.primary_emotion.value == "confused":
            return "patient"
        else:
            return "professional"
    
    def _should_transition_topic(
        self,
        state: DialogueState,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Determine if topic should be transitioned"""
        engagement_score = context.get("engagement_score", 0.5)
        
        if engagement_score < self.strategy.topic_switching_threshold:
            # Suggest related topic
            current_topic = state.current_topic
            related_topics = self.topic_transitions.get(current_topic, [])
            
            if related_topics:
                # Choose topic not recently discussed
                for topic in related_topics:
                    if topic not in state.previous_topics[-3:]:
                        return topic
        
        return None
    
    def _determine_teaching_level(self, context: Dict[str, Any]) -> str:
        """Determine appropriate teaching/explanation level"""
        knowledge_level = context.get("knowledge_level", "intermediate")
        
        if self.strategy.teaching_mode:
            if knowledge_level == "beginner":
                return "detailed"
            elif knowledge_level == "intermediate":
                return "moderate"
            else:
                return "concise"
        else:
            return "minimal"
    
    def _calculate_proactivity(
        self,
        state: DialogueState,
        context: Dict[str, Any]
    ) -> float:
        """Calculate how proactive to be in response"""
        base_proactivity = self.strategy.proactivity_level
        
        # Adjust based on engagement
        engagement = context.get("engagement_score", 0.5)
        if engagement < 0.3:
            proactivity = min(1.0, base_proactivity + 0.3)
        elif engagement > 0.8:
            proactivity = max(0.2, base_proactivity - 0.2)
        else:
            proactivity = base_proactivity
        
        # Adjust based on turn number
        if state.turn_number < 3:
            proactivity *= 0.7  # Less proactive at start
        
        return proactivity
    
    async def _generate_strategic_response(
        self,
        session_id: str,
        user_message: str,
        strategy: Dict[str, Any],
        state: DialogueState,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response based on strategy"""
        # Build response components based on dialogue pattern
        response_components = []
        
        for act in strategy["dialogue_pattern"]:
            if act == "acknowledge":
                response_components.append(self._generate_acknowledgment(user_message))
            elif act == "answer":
                answer = await self._generate_answer(user_message, context)
                response_components.append(answer)
            elif act == "clarify":
                clarification = self._generate_clarification(context)
                response_components.append(clarification)
            elif act == "elaborate":
                elaboration = self._generate_elaboration(state.current_topic, strategy["teaching_level"])
                response_components.append(elaboration)
            elif act == "follow_up":
                follow_up = self._generate_follow_up(state)
                response_components.append(follow_up)
        
        # Combine components
        base_response = " ".join(response_components)
        
        # Apply emotional tone
        toned_response = await self._apply_emotional_tone(
            base_response,
            strategy["emotional_tone"]
        )
        
        # Add proactive elements
        if strategy["proactivity"] > 0.5:
            proactive_element = self._generate_proactive_element(state, context)
            toned_response += f" {proactive_element}"
        
        # Handle topic transition
        if strategy["topic_transition"]:
            transition = self._generate_topic_transition(
                state.current_topic,
                strategy["topic_transition"]
            )
            toned_response += f" {transition}"
        
        # Generate complete response through conversation engine
        conv_context = ConversationContext(
            user_id=session_id.split("_")[1],  # Extract user_id
            session_id=session_id,
            research_context=context,
            conversation_history=[],
            reasoning_chain=[],
            verification_results=[],
            emotional_state=strategy["emotional_tone"],
            topic_domain=state.current_topic
        )
        
        final_response, metadata = await self.conversation_engine.process_message(
            user_message,
            conv_context
        )
        
        # Enhance with strategic elements
        enhanced_response = self._enhance_response(
            final_response,
            toned_response,
            strategy
        )
        
        return {
            "text": enhanced_response,
            "metadata": metadata,
            "strategy": strategy,
            "dialogue_act": state.dialogue_acts[-1],
            "turn_number": state.turn_number
        }
    
    def _generate_acknowledgment(self, message: str) -> str:
        """Generate acknowledgment phrase"""
        if "?" in message:
            return "That's a great question."
        elif any(word in message.lower() for word in ["interesting", "cool", "amazing"]):
            return "I share your enthusiasm!"
        else:
            return "I understand."
    
    async def _generate_answer(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate answer to question"""
        # This would use the conversation engine
        return "Based on my analysis..."
    
    def _generate_clarification(self, context: Dict[str, Any]) -> str:
        """Generate clarification based on context"""
        confusion_indicators = context.get("confusion_indicators", [])
        
        if "definition_request" in confusion_indicators:
            return "Let me define that more clearly:"
        elif "example_request" in confusion_indicators:
            return "Here's a concrete example:"
        else:
            return "To clarify:"
    
    def _generate_elaboration(self, topic: str, teaching_level: str) -> str:
        """Generate topic elaboration"""
        if teaching_level == "detailed":
            return f"Let me explain more about {topic} in detail."
        elif teaching_level == "moderate":
            return f"To expand on {topic}:"
        else:
            return ""
    
    def _generate_follow_up(self, state: DialogueState) -> str:
        """Generate follow-up question"""
        if state.open_questions:
            return f"Regarding your earlier question about {state.open_questions[-1]}..."
        else:
            return "Is there anything specific you'd like to explore further?"
    
    async def _apply_emotional_tone(
        self,
        response: str,
        tone: str
    ) -> str:
        """Apply emotional tone to response"""
        tone_prefixes = {
            "empathetic": "I understand this might be challenging. ",
            "enthusiastic": "This is exciting! ",
            "patient": "Let's take this step by step. ",
            "professional": ""
        }
        
        prefix = tone_prefixes.get(tone, "")
        return prefix + response
    
    def _generate_proactive_element(
        self,
        state: DialogueState,
        context: Dict[str, Any]
    ) -> str:
        """Generate proactive conversation element"""
        if context.get("key_concepts"):
            concept = context["key_concepts"][-1]
            return f"Would you like to explore {concept} further?"
        else:
            return "What aspect interests you most?"
    
    def _generate_topic_transition(
        self,
        current_topic: str,
        new_topic: str
    ) -> str:
        """Generate smooth topic transition"""
        return f"Speaking of {current_topic}, this relates to {new_topic} in interesting ways."
    
    def _enhance_response(
        self,
        base_response: str,
        strategic_elements: str,
        strategy: Dict[str, Any]
    ) -> str:
        """Enhance response with strategic elements"""
        # Blend base response with strategic elements
        if strategy["clarification_needed"]:
            base_response = strategic_elements + " " + base_response
        
        return base_response
    
    def _update_dialogue_state(
        self,
        state: DialogueState,
        user_message: str,
        response_data: Dict[str, Any],
        sentiment: float
    ):
        """Update dialogue state after turn"""
        # Update sentiment history
        state.sentiment_history.append(sentiment)
        if len(state.sentiment_history) > 10:
            state.sentiment_history.pop(0)
        
        # Extract questions from user message
        if "?" in user_message:
            state.open_questions.append(user_message)
        
        # Update topic if changed
        new_topic = response_data["metadata"].get("detected_topic")
        if new_topic and new_topic != state.current_topic:
            state.previous_topics.append(state.current_topic)
            state.current_topic = new_topic
        
        # Update clarification status
        state.clarification_needed = response_data["strategy"].get("clarification_needed", False)
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        if session_id not in self.dialogue_states:
            return {}
        
        state = self.dialogue_states[session_id]
        context_summary = self.context_manager.get_session_summary(session_id)
        
        return {
            "session_id": session_id,
            "total_turns": state.turn_number,
            "topics_discussed": [state.current_topic] + state.previous_topics,
            "dialogue_acts": state.dialogue_acts,
            "average_sentiment": np.mean(state.sentiment_history) if state.sentiment_history else 0.5,
            "open_questions": state.open_questions,
            **context_summary
        }
    
    def save_conversation(self, session_id: str, filepath: str):
        """Save conversation to file"""
        summary = self.get_conversation_summary(session_id)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_conversation(self, filepath: str) -> str:
        """Load conversation from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session_id = data["session_id"]
        # Reconstruct dialogue state
        # This would rebuild the conversation context
        
        return session_id