"""
Emotional response mapping for digital human interactions.

Maps user emotions and generates appropriate emotional responses using
neural supercomputer capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from enum import Enum

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from aiq.verification.verification_system import VerificationSystem
from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer


class EmotionType(Enum):
    """Emotion categories for digital human responses"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EMPATHETIC = "empathetic"
    CONCERNED = "concerned"
    EXCITED = "excited"
    THOUGHTFUL = "thoughtful"
    ENCOURAGING = "encouraging"
    PROFESSIONAL = "professional"


@dataclass
class EmotionalState:
    """Represents the current emotional state"""
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    secondary_emotions: List[Tuple[EmotionType, float]]
    context_factors: Dict[str, Any]
    transition_duration: float = 1.0  # seconds


class EmotionalResponseMapper:
    """
    Maps conversation context to appropriate emotional responses.
    Uses GPU acceleration for real-time emotion analysis.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "bert-base-uncased",
        enable_gpu_optimization: bool = True
    ):
        self.device = device
        self.model_name = model_name
        
        # Initialize emotion detection pipeline
        if pipeline is not None:
            self.emotion_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if device == "cuda" else -1
            )
        else:
            self.emotion_pipeline = None
        
        # GPU optimization
        if enable_gpu_optimization:
            self.tensor_optimizer = TensorCoreOptimizer()
        else:
            self.tensor_optimizer = None
        
        # Emotion mapping rules
        self.emotion_rules = self._initialize_emotion_rules()
        
        # Emotional memory for context
        self.emotional_memory = []
        self.max_memory_size = 50
    
    def _initialize_emotion_rules(self) -> Dict[str, List[EmotionType]]:
        """Initialize rule-based emotion mappings"""
        return {
            "greeting": [EmotionType.HAPPY, EmotionType.PROFESSIONAL],
            "research": [EmotionType.THOUGHTFUL, EmotionType.PROFESSIONAL],
            "problem": [EmotionType.CONCERNED, EmotionType.EMPATHETIC],
            "success": [EmotionType.EXCITED, EmotionType.ENCOURAGING],
            "confusion": [EmotionType.EMPATHETIC, EmotionType.THOUGHTFUL],
            "curiosity": [EmotionType.EXCITED, EmotionType.THOUGHTFUL],
            "frustration": [EmotionType.EMPATHETIC, EmotionType.ENCOURAGING],
            "appreciation": [EmotionType.HAPPY, EmotionType.PROFESSIONAL]
        }
    
    async def analyze_emotional_context(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        reasoning_context: Dict[str, Any]
    ) -> EmotionalState:
        """
        Analyze the emotional context of the conversation.
        
        Args:
            message: Current user message
            conversation_history: Recent conversation history
            reasoning_context: Context from reasoning engine
            
        Returns:
            EmotionalState object with appropriate emotions
        """
        # Step 1: Analyze user emotion
        user_emotion = await self._detect_user_emotion(message)
        
        # Step 2: Analyze conversation trajectory
        trajectory_emotion = await self._analyze_conversation_trajectory(
            conversation_history
        )
        
        # Step 3: Consider reasoning context
        context_emotion = await self._analyze_reasoning_context(
            reasoning_context
        )
        
        # Step 4: Synthesize emotional state
        emotional_state = await self._synthesize_emotional_state(
            user_emotion,
            trajectory_emotion,
            context_emotion,
            message
        )
        
        # Step 5: Update emotional memory
        self._update_emotional_memory(emotional_state)
        
        return emotional_state
    
    async def _detect_user_emotion(self, message: str) -> Dict[str, float]:
        """Detect emotion from user message"""
        if self.emotion_pipeline:
            try:
                results = self.emotion_pipeline(message)
                emotion_scores = {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 0.0
                }
                
                for result in results:
                    label = result["label"].lower()
                    score = result["score"]
                    
                    if label in ["positive", "joy", "happy"]:
                        emotion_scores["positive"] = max(emotion_scores["positive"], score)
                    elif label in ["negative", "sadness", "anger"]:
                        emotion_scores["negative"] = max(emotion_scores["negative"], score)
                    else:
                        emotion_scores["neutral"] = max(emotion_scores["neutral"], score)
                
                return emotion_scores
            except Exception as e:
                print(f"Emotion detection error: {e}")
        
        # Fallback to keyword analysis
        return await self._keyword_emotion_analysis(message)
    
    async def _keyword_emotion_analysis(self, message: str) -> Dict[str, float]:
        """Fallback keyword-based emotion analysis"""
        message_lower = message.lower()
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.5}
        
        # Positive keywords
        positive_keywords = ["happy", "great", "excellent", "wonderful", "thanks", "appreciate"]
        negative_keywords = ["sad", "angry", "frustrated", "confused", "problem", "issue"]
        
        for keyword in positive_keywords:
            if keyword in message_lower:
                scores["positive"] += 0.2
        
        for keyword in negative_keywords:
            if keyword in message_lower:
                scores["negative"] += 0.2
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    async def _analyze_conversation_trajectory(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> EmotionType:
        """Analyze emotional trajectory of conversation"""
        if not conversation_history:
            return EmotionType.NEUTRAL
        
        # Look at recent messages
        recent_emotions = []
        for entry in conversation_history[-5:]:
            user_msg = entry.get("user", "")
            emotion = await self._detect_user_emotion(user_msg)
            if emotion["positive"] > emotion["negative"]:
                recent_emotions.append("positive")
            elif emotion["negative"] > emotion["positive"]:
                recent_emotions.append("negative")
            else:
                recent_emotions.append("neutral")
        
        # Determine trajectory
        positive_count = recent_emotions.count("positive")
        negative_count = recent_emotions.count("negative")
        
        if positive_count > negative_count * 2:
            return EmotionType.HAPPY
        elif negative_count > positive_count * 2:
            return EmotionType.EMPATHETIC
        else:
            return EmotionType.THOUGHTFUL
    
    async def _analyze_reasoning_context(
        self,
        reasoning_context: Dict[str, Any]
    ) -> EmotionType:
        """Analyze emotion based on reasoning context"""
        confidence = reasoning_context.get("confidence_score", 0.8)
        complexity = reasoning_context.get("complexity", "medium")
        success = reasoning_context.get("success", True)
        
        if confidence > 0.9 and success:
            return EmotionType.EXCITED
        elif confidence < 0.5 or not success:
            return EmotionType.CONCERNED
        elif complexity == "high":
            return EmotionType.THOUGHTFUL
        else:
            return EmotionType.PROFESSIONAL
    
    async def _synthesize_emotional_state(
        self,
        user_emotion: Dict[str, float],
        trajectory_emotion: EmotionType,
        context_emotion: EmotionType,
        message: str
    ) -> EmotionalState:
        """Synthesize final emotional state from multiple inputs"""
        # Determine primary emotion
        if user_emotion["negative"] > 0.7:
            primary_emotion = EmotionType.EMPATHETIC
            intensity = user_emotion["negative"]
        elif user_emotion["positive"] > 0.7:
            primary_emotion = EmotionType.HAPPY
            intensity = user_emotion["positive"]
        else:
            primary_emotion = trajectory_emotion
            intensity = 0.6
        
        # Add secondary emotions
        secondary_emotions = []
        if context_emotion != primary_emotion:
            secondary_emotions.append((context_emotion, 0.4))
        
        # Check for specific intents
        message_lower = message.lower()
        if any(word in message_lower for word in ["help", "confused", "don't understand"]):
            secondary_emotions.append((EmotionType.ENCOURAGING, 0.5))
        if any(word in message_lower for word in ["amazing", "brilliant", "excellent"]):
            secondary_emotions.append((EmotionType.EXCITED, 0.6))
        
        # Context factors
        context_factors = {
            "user_sentiment": max(user_emotion.items(), key=lambda x: x[1])[0],
            "conversation_trend": trajectory_emotion.value,
            "reasoning_confidence": context_emotion.value,
            "message_length": len(message),
            "exclamation_marks": message.count("!"),
            "question_marks": message.count("?")
        }
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            secondary_emotions=secondary_emotions,
            context_factors=context_factors
        )
    
    def _update_emotional_memory(self, emotional_state: EmotionalState):
        """Update emotional memory with current state"""
        self.emotional_memory.append({
            "state": emotional_state,
            "timestamp": np.datetime64('now')
        })
        
        # Maintain memory size limit
        if len(self.emotional_memory) > self.max_memory_size:
            self.emotional_memory.pop(0)
    
    def map_emotion_to_expression(
        self,
        emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """
        Map emotional state to facial expression parameters.
        
        Returns:
            Dictionary with expression parameters for avatar system
        """
        expression_map = {
            EmotionType.NEUTRAL: {
                "eyebrow_raise": 0.0,
                "smile": 0.0,
                "eye_openness": 1.0,
                "head_tilt": 0.0,
                "blink_rate": "normal"
            },
            EmotionType.HAPPY: {
                "eyebrow_raise": 0.2,
                "smile": 0.8,
                "eye_openness": 0.9,
                "head_tilt": 0.1,
                "blink_rate": "normal"
            },
            EmotionType.EMPATHETIC: {
                "eyebrow_raise": -0.1,
                "smile": 0.2,
                "eye_openness": 0.8,
                "head_tilt": 0.15,
                "blink_rate": "slow"
            },
            EmotionType.CONCERNED: {
                "eyebrow_raise": -0.3,
                "smile": -0.1,
                "eye_openness": 0.9,
                "head_tilt": 0.1,
                "blink_rate": "slow"
            },
            EmotionType.EXCITED: {
                "eyebrow_raise": 0.4,
                "smile": 0.9,
                "eye_openness": 1.1,
                "head_tilt": 0.0,
                "blink_rate": "fast"
            },
            EmotionType.THOUGHTFUL: {
                "eyebrow_raise": 0.1,
                "smile": 0.1,
                "eye_openness": 0.7,
                "head_tilt": 0.2,
                "blink_rate": "very_slow"
            },
            EmotionType.ENCOURAGING: {
                "eyebrow_raise": 0.3,
                "smile": 0.6,
                "eye_openness": 1.0,
                "head_tilt": -0.1,
                "blink_rate": "normal"
            },
            EmotionType.PROFESSIONAL: {
                "eyebrow_raise": 0.0,
                "smile": 0.3,
                "eye_openness": 0.95,
                "head_tilt": 0.0,
                "blink_rate": "normal"
            }
        }
        
        # Get base expression
        base_expression = expression_map.get(
            emotional_state.primary_emotion,
            expression_map[EmotionType.NEUTRAL]
        ).copy()
        
        # Apply intensity scaling
        intensity = emotional_state.intensity
        for key in ["eyebrow_raise", "smile", "head_tilt"]:
            base_expression[key] *= intensity
        
        # Blend with secondary emotions
        for emotion, weight in emotional_state.secondary_emotions:
            secondary_exp = expression_map.get(emotion, {})
            for key in base_expression:
                if key in secondary_exp and key != "blink_rate":
                    base_expression[key] += secondary_exp[key] * weight * 0.5
        
        # Add transition information
        base_expression["transition_duration"] = emotional_state.transition_duration
        base_expression["emotion_label"] = emotional_state.primary_emotion.value
        
        return base_expression
    
    def map_emotion_to_voice_params(
        self,
        emotional_state: EmotionalState
    ) -> Dict[str, float]:
        """
        Map emotional state to voice synthesis parameters.
        
        Returns:
            Dictionary with voice parameters
        """
        voice_map = {
            EmotionType.NEUTRAL: {
                "pitch": 1.0,
                "speed": 1.0,
                "volume": 1.0,
                "tone": "neutral"
            },
            EmotionType.HAPPY: {
                "pitch": 1.1,
                "speed": 1.05,
                "volume": 1.05,
                "tone": "warm"
            },
            EmotionType.EMPATHETIC: {
                "pitch": 0.95,
                "speed": 0.95,
                "volume": 0.95,
                "tone": "gentle"
            },
            EmotionType.CONCERNED: {
                "pitch": 0.9,
                "speed": 0.9,
                "volume": 0.9,
                "tone": "serious"
            },
            EmotionType.EXCITED: {
                "pitch": 1.15,
                "speed": 1.1,
                "volume": 1.1,
                "tone": "energetic"
            },
            EmotionType.THOUGHTFUL: {
                "pitch": 0.98,
                "speed": 0.93,
                "volume": 0.95,
                "tone": "contemplative"
            },
            EmotionType.ENCOURAGING: {
                "pitch": 1.05,
                "speed": 1.0,
                "volume": 1.05,
                "tone": "supportive"
            },
            EmotionType.PROFESSIONAL: {
                "pitch": 1.0,
                "speed": 0.98,
                "volume": 1.0,
                "tone": "formal"
            }
        }
        
        voice_params = voice_map.get(
            emotional_state.primary_emotion,
            voice_map[EmotionType.NEUTRAL]
        ).copy()
        
        # Apply intensity modulation
        intensity = emotional_state.intensity
        voice_params["pitch"] = 1.0 + (voice_params["pitch"] - 1.0) * intensity
        voice_params["volume"] = 1.0 + (voice_params["volume"] - 1.0) * intensity
        
        return voice_params
    
    def get_emotional_phrases(
        self,
        emotional_state: EmotionalState
    ) -> List[str]:
        """Get emotion-appropriate phrases for response enhancement"""
        phrase_map = {
            EmotionType.HAPPY: [
                "That's wonderful!",
                "I'm delighted to help with that!",
                "How exciting!"
            ],
            EmotionType.EMPATHETIC: [
                "I understand your concern.",
                "Let me help you with that.",
                "I can see why that might be challenging."
            ],
            EmotionType.CONCERNED: [
                "I see this is important.",
                "Let's work through this carefully.",
                "I want to make sure we get this right."
            ],
            EmotionType.EXCITED: [
                "This is fascinating!",
                "Great question!",
                "I love exploring topics like this!"
            ],
            EmotionType.THOUGHTFUL: [
                "That's an interesting perspective.",
                "Let me think about this...",
                "This requires careful consideration."
            ],
            EmotionType.ENCOURAGING: [
                "You're on the right track!",
                "Keep up the great work!",
                "That's a smart approach!"
            ],
            EmotionType.PROFESSIONAL: [
                "I'll analyze this for you.",
                "Based on the available data...",
                "Let me provide you with the information."
            ]
        }
        
        return phrase_map.get(emotional_state.primary_emotion, [])
    
    async def adapt_response_tone(
        self,
        response: str,
        emotional_state: EmotionalState
    ) -> str:
        """Adapt response tone based on emotional state"""
        # Add emotional prefix if appropriate
        if emotional_state.intensity > 0.7:
            phrases = self.get_emotional_phrases(emotional_state)
            if phrases:
                prefix = np.random.choice(phrases)
                response = f"{prefix} {response}"
        
        # Adjust punctuation based on emotion
        if emotional_state.primary_emotion == EmotionType.EXCITED:
            if not response.endswith("!") and response.endswith("."):
                response = response[:-1] + "!"
        elif emotional_state.primary_emotion == EmotionType.THOUGHTFUL:
            if response.endswith("!"):
                response = response[:-1] + "."
        
        return response