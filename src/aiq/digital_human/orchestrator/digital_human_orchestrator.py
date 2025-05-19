"""
Digital Human Orchestrator - coordinates all components for a complete
digital human experience with neural supercomputer reasoning.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time
import json
from datetime import datetime
from enum import Enum

import numpy as np
import torch

from aiq.digital_human.conversation.sglang_engine import (
    SgLangConversationEngine,
    ConversationContext
)
from aiq.digital_human.conversation.emotional_mapper import (
    EmotionalResponseMapper,
    EmotionalState,
    EmotionType
)
from aiq.digital_human.avatar.facial_animator import FacialAnimationSystem
from aiq.digital_human.avatar.emotion_renderer import EmotionRenderer
from aiq.profiler.inference_metrics_model import InferenceMetrics


class SystemState(Enum):
    """Digital human system states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    THINKING = "thinking"
    ERROR = "error"


@dataclass
class InteractionSession:
    """Tracks a complete interaction session"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_interactions: int = 0
    conversation_context: Optional[ConversationContext] = None
    performance_metrics: Dict[str, Any] = None


class DigitalHumanOrchestrator:
    """
    Main orchestrator for digital human interactions.
    Coordinates conversation, emotion, reasoning, and rendering.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        enable_profiling: bool = True,
        enable_gpu_acceleration: bool = True
    ):
        self.config = config
        self.enable_profiling = enable_profiling
        self.device = "cuda" if enable_gpu_acceleration and torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self.current_state = SystemState.IDLE
        self.active_session: Optional[InteractionSession] = None
        
        # Performance tracking
        self.performance_metrics = InferenceMetrics() if enable_profiling else None
        
        # Event handlers
        self.event_handlers = {
            "on_user_input": [],
            "on_response_generated": [],
            "on_emotion_changed": [],
            "on_error": []
        }
    
    def _initialize_components(self):
        """Initialize all digital human subsystems"""
        # Conversation engine with neural supercomputer
        self.conversation_engine = SgLangConversationEngine(
            model_name=self.config.get("model_name", "meta-llama/Llama-3.1-70B-Instruct"),
            device=self.device,
            temperature=self.config.get("temperature", 0.7),
            enable_research=True,
            enable_verification=True
        )
        
        # Emotional response mapper
        self.emotion_mapper = EmotionalResponseMapper(
            device=self.device,
            enable_gpu_optimization=True
        )
        
        # Avatar systems
        self.facial_animator = FacialAnimationSystem(
            device=self.device,
            fps=60.0,
            enable_gpu_skinning=True
        )
        
        self.emotion_renderer = EmotionRenderer(
            device=self.device,
            render_resolution=self.config.get("resolution", (1920, 1080)),
            enable_effects=True,
            use_gpu_rendering=True
        )
        
        # Research and correction systems are embedded in the conversation engine
        
        # State manager
        self.state_manager = StateManager()
        
        # Response generator
        self.response_generator = ResponseGenerator(
            conversation_engine=self.conversation_engine,
            emotion_mapper=self.emotion_mapper,
            facial_animator=self.facial_animator
        )
    
    async def start_session(
        self,
        user_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new interaction session"""
        session_id = f"session_{user_id}_{int(time.time())}"
        
        # Create conversation context
        conversation_context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            research_context=initial_context or {},
            conversation_history=[],
            reasoning_chain=[],
            verification_results=[],
            emotional_state="neutral",
            topic_domain="general"
        )
        
        # Create session
        self.active_session = InteractionSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            conversation_context=conversation_context,
            performance_metrics={}
        )
        
        # Set initial state
        self.current_state = SystemState.IDLE
        
        # Initialize avatar
        self.facial_animator.set_expression("neutral")
        
        return session_id
    
    async def process_user_input(
        self,
        user_input: str,
        audio_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process user input and generate complete response.
        
        Args:
            user_input: Text input from user
            audio_data: Optional audio for lip-sync
            
        Returns:
            Response containing text, emotion, avatar state, etc.
        """
        if not self.active_session:
            raise ValueError("No active session. Call start_session first.")
        
        start_time = time.time()
        self.current_state = SystemState.PROCESSING
        
        try:
            # Step 1: Analyze emotional context
            emotional_state = await self._analyze_emotion(
                user_input,
                self.active_session.conversation_context
            )
            
            # Step 2: Process through conversation engine
            response_text, reasoning_metadata = await self._generate_response(
                user_input,
                self.active_session.conversation_context
            )
            
            # Step 3: Apply emotional adaptation
            adapted_response = await self.emotion_mapper.adapt_response_tone(
                response_text,
                emotional_state
            )
            
            # Step 4: Generate avatar animation
            animation_data = await self._generate_avatar_animation(
                adapted_response,
                emotional_state,
                audio_data
            )
            
            # Step 5: Render emotion
            render_data = self.emotion_renderer.render_emotion(
                emotional_state,
                adapted_response
            )
            
            # Step 6: Update session
            self.active_session.total_interactions += 1
            
            # Track performance
            if self.performance_metrics:
                self.performance_metrics.record_inference(
                    duration=time.time() - start_time,
                    model_name=self.config.get("model_name"),
                    input_tokens=len(user_input.split()),
                    output_tokens=len(adapted_response.split())
                )
            
            # Prepare complete response
            response = {
                "session_id": self.active_session.session_id,
                "text": adapted_response,
                "original_text": response_text,
                "emotion": emotional_state.primary_emotion.value,
                "emotion_intensity": emotional_state.intensity,
                "reasoning": reasoning_metadata,
                "animation": animation_data,
                "render": render_data,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time
            }
            
            # Fire event
            await self._fire_event("on_response_generated", response)
            
            self.current_state = SystemState.SPEAKING
            return response
            
        except Exception as e:
            self.current_state = SystemState.ERROR
            await self._fire_event("on_error", {"error": str(e)})
            raise
    
    async def _analyze_emotion(
        self,
        user_input: str,
        context: ConversationContext
    ) -> EmotionalState:
        """Analyze emotional context of interaction"""
        emotional_state = await self.emotion_mapper.analyze_emotional_context(
            message=user_input,
            conversation_history=context.conversation_history,
            reasoning_context=context.research_context
        )
        
        # Update context
        context.emotional_state = emotional_state.primary_emotion.value
        
        await self._fire_event("on_emotion_changed", {
            "emotion": emotional_state.primary_emotion.value,
            "intensity": emotional_state.intensity
        })
        
        return emotional_state
    
    async def _generate_response(
        self,
        user_input: str,
        context: ConversationContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using neural supercomputer reasoning"""
        response_text, metadata = await self.conversation_engine.process_message(
            message=user_input,
            context=context
        )
        
        return response_text, metadata
    
    async def _generate_avatar_animation(
        self,
        response_text: str,
        emotional_state: EmotionalState,
        audio_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate avatar animation data"""
        # Set emotional expression
        expression_params = self.emotion_mapper.map_emotion_to_expression(
            emotional_state
        )
        self.facial_animator.set_emotion_expression(expression_params)
        
        # Generate lip-sync if audio provided
        if audio_data is not None:
            # Extract phonemes from audio (placeholder)
            phonemes = await self._extract_phonemes(audio_data)
            self.facial_animator.process_lip_sync(phonemes)
        else:
            # Generate synthetic lip-sync from text
            phonemes = await self._text_to_phonemes(response_text)
            self.facial_animator.process_lip_sync(phonemes)
        
        # Get voice parameters
        voice_params = self.emotion_mapper.map_emotion_to_voice_params(
            emotional_state
        )
        
        return {
            "expression_weights": self.facial_animator.get_expression_weights(),
            "phonemes": phonemes,
            "voice_params": voice_params,
            "emotion_label": emotional_state.primary_emotion.value
        }
    
    async def _extract_phonemes(
        self,
        audio_data: np.ndarray
    ) -> List[Tuple[str, float, float]]:
        """Extract phonemes from audio for lip-sync using speech recognition"""
        try:
            # Check if we have a speech recognition module available
            if hasattr(self, 'speech_recognizer'):
                phonemes = await self.speech_recognizer.extract_phonemes(audio_data)
                return phonemes
            
            # Try using available speech recognition libraries
            try:
                import speech_recognition as sr
                import phonemizer
                
                # Convert audio to text first
                recognizer = sr.Recognizer()
                audio = sr.AudioData(audio_data.tobytes(), sample_rate=16000, sample_width=2)
                text = recognizer.recognize_google(audio)
                
                # Convert text to phonemes
                phonemes = phonemizer.phonemize(text, language='en-us', backend='espeak')
                
                # Simple timing estimation based on phoneme count
                phoneme_list = []
                current_time = 0.0
                for phoneme in phonemes.split():
                    duration = 0.05 + np.random.uniform(0, 0.03)  # Variable duration
                    phoneme_list.append((phoneme, current_time, duration))
                    current_time += duration + 0.01  # Small gap
                
                return phoneme_list
                
            except ImportError:
                # Use basic phoneme generation with improved mapping
                return await self._generate_basic_phonemes_from_audio(audio_data)
                
        except Exception as e:
            self.logger.warning(f"Phoneme extraction failed: {e}")
            # Fallback to basic phonemes
            return [
                ("AA", 0.0, 0.1),
                ("R", 0.1, 0.05),
                ("T", 0.15, 0.05)
            ]
    
    async def _generate_basic_phonemes_from_audio(
        self,
        audio_data: np.ndarray
    ) -> List[Tuple[str, float, float]]:
        """Generate basic phonemes from audio analysis"""
        # Analyze audio amplitude patterns
        window_size = int(0.02 * 16000)  # 20ms windows at 16kHz
        num_windows = len(audio_data) // window_size
        
        phonemes = []
        current_time = 0.0
        
        for i in range(num_windows):
            window = audio_data[i * window_size:(i + 1) * window_size]
            amplitude = np.mean(np.abs(window))
            
            # Map amplitude patterns to basic phonemes
            if amplitude > 0.1:
                # Voiced sounds
                if i % 3 == 0:
                    phoneme = "AA"  # Open vowel
                elif i % 3 == 1:
                    phoneme = "IY"  # Close vowel
                else:
                    phoneme = "UW"  # Round vowel
            else:
                # Unvoiced sounds
                if i % 2 == 0:
                    phoneme = "S"  # Fricative
                else:
                    phoneme = "T"  # Stop
            
            duration = window_size / 16000.0
            phonemes.append((phoneme, current_time, duration))
            current_time += duration
        
        return phonemes
    
    async def _text_to_phonemes(
        self,
        text: str
    ) -> List[Tuple[str, float, float]]:
        """Convert text to phonemes for lip-sync"""
        try:
            # Try to use proper text-to-phoneme library if available
            try:
                import phonemizer
                phoneme_text = phonemizer.phonemize(
                    text,
                    language='en-us',
                    backend='espeak',
                    with_stress=True,
                    preserve_punctuation=True
                )
                
                # Parse phoneme text into timed segments
                phonemes = []
                current_time = 0.0
                
                for phoneme in phoneme_text.split():
                    # Estimate duration based on phoneme type
                    if phoneme in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']:
                        duration = 0.08  # Vowels are longer
                    elif phoneme in ['B', 'D', 'G', 'K', 'P', 'T']:
                        duration = 0.04  # Stops are shorter
                    elif phoneme in ['CH', 'DH', 'F', 'JH', 'S', 'SH', 'TH', 'V', 'Z', 'ZH']:
                        duration = 0.06  # Fricatives are medium
                    else:
                        duration = 0.05  # Default duration
                    
                    phonemes.append((phoneme, current_time, duration))
                    current_time += duration + 0.01  # Small gap between phonemes
                
                return phonemes
                
            except ImportError:
                # Fallback to rule-based phoneme generation
                return await self._rule_based_text_to_phonemes(text)
                
        except Exception as e:
            self.logger.warning(f"Text to phoneme conversion failed: {e}")
            # Final fallback
            return [("N", 0.0, 0.5)]  # Neutral mouth position
    
    async def _rule_based_text_to_phonemes(
        self,
        text: str
    ) -> List[Tuple[str, float, float]]:
        """Rule-based text to phoneme conversion"""
        # Common word to phoneme mappings
        phoneme_map = {
            # Common words
            "the": [("DH", 0.04), ("IY", 0.06)],
            "and": [("AE", 0.06), ("N", 0.04), ("D", 0.04)],
            "is": [("IH", 0.05), ("Z", 0.05)],
            "are": [("AA", 0.06), ("R", 0.04)],
            "to": [("T", 0.04), ("UW", 0.06)],
            "in": [("IH", 0.05), ("N", 0.05)],
            "it": [("IH", 0.05), ("T", 0.04)],
            "you": [("Y", 0.04), ("UW", 0.06)],
            "that": [("DH", 0.04), ("AE", 0.06), ("T", 0.04)],
            "was": [("W", 0.04), ("AA", 0.06), ("Z", 0.04)],
        }
        
        words = text.lower().split()
        phonemes = []
        current_time = 0.0
        
        for word in words:
            if word in phoneme_map:
                # Use known mapping
                for phoneme, duration in phoneme_map[word]:
                    phonemes.append((phoneme, current_time, duration))
                    current_time += duration
            else:
                # Generate phonemes based on spelling patterns
                word_phonemes = self._spell_to_phonemes(word)
                for phoneme, duration in word_phonemes:
                    phonemes.append((phoneme, current_time, duration))
                    current_time += duration
            
            current_time += 0.05  # Pause between words
        
        return phonemes
    
    def _spell_to_phonemes(self, word: str) -> List[Tuple[str, float]]:
        """Convert spelling to approximate phonemes"""
        phonemes = []
        i = 0
        
        while i < len(word):
            # Check for common patterns
            if i < len(word) - 1:
                digraph = word[i:i+2]
                
                # Common digraphs
                if digraph in ['th', 'ch', 'sh', 'ph', 'wh']:
                    if digraph == 'th':
                        phonemes.append(("TH", 0.06))
                    elif digraph == 'ch':
                        phonemes.append(("CH", 0.06))
                    elif digraph == 'sh':
                        phonemes.append(("SH", 0.06))
                    elif digraph == 'ph':
                        phonemes.append(("F", 0.06))
                    elif digraph == 'wh':
                        phonemes.append(("W", 0.04))
                    i += 2
                    continue
                
                # Vowel combinations
                if digraph in ['oo', 'ee', 'ea', 'ou', 'ai', 'ey']:
                    if digraph in ['oo', 'ou']:
                        phonemes.append(("UW", 0.08))
                    elif digraph in ['ee', 'ea']:
                        phonemes.append(("IY", 0.08))
                    elif digraph == 'ai':
                        phonemes.append(("EY", 0.08))
                    elif digraph == 'ey':
                        phonemes.append(("EY", 0.08))
                    i += 2
                    continue
            
            # Single characters
            char = word[i]
            
            # Vowels
            if char in 'aeiou':
                if char == 'a':
                    phonemes.append(("AE", 0.08))
                elif char == 'e':
                    phonemes.append(("EH", 0.08))
                elif char == 'i':
                    phonemes.append(("IH", 0.08))
                elif char == 'o':
                    phonemes.append(("AA", 0.08))
                elif char == 'u':
                    phonemes.append(("AH", 0.08))
            
            # Consonants
            elif char in 'bcdfghjklmnpqrstvwxyz':
                if char in 'bp':
                    phonemes.append(("B" if char == 'b' else "P", 0.04))
                elif char in 'td':
                    phonemes.append(("T" if char == 't' else "D", 0.04))
                elif char in 'kg':
                    phonemes.append(("K" if char == 'k' else "G", 0.04))
                elif char in 'fv':
                    phonemes.append(("F" if char == 'f' else "V", 0.06))
                elif char in 'sz':
                    phonemes.append(("S" if char == 's' else "Z", 0.06))
                elif char == 'r':
                    phonemes.append(("R", 0.05))
                elif char == 'l':
                    phonemes.append(("L", 0.05))
                elif char == 'n':
                    phonemes.append(("N", 0.05))
                elif char == 'm':
                    phonemes.append(("M", 0.05))
                else:
                    # Default consonant
                    phonemes.append(("K", 0.04))
            
            i += 1
        
        return phonemes
    
    async def update_animation(self, delta_time: float):
        """Update animation systems"""
        self.facial_animator.update(delta_time)
        self.emotion_renderer.update(delta_time)
    
    async def end_session(self):
        """End the current interaction session"""
        if self.active_session:
            self.active_session.end_time = datetime.now()
            
            # Save session data
            session_data = {
                "session_id": self.active_session.session_id,
                "user_id": self.active_session.user_id,
                "start_time": self.active_session.start_time.isoformat(),
                "end_time": self.active_session.end_time.isoformat(),
                "total_interactions": self.active_session.total_interactions,
                "conversation_history": self.active_session.conversation_context.conversation_history
            }
            
            # Reset state
            self.current_state = SystemState.IDLE
            self.active_session = None
            self.facial_animator.reset()
            
            return session_data
        
        return None
    
    def register_event_handler(self, event: str, handler):
        """Register event handler"""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)
    
    async def _fire_event(self, event: str, data: Any):
        """Fire event to registered handlers"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "state": self.current_state.value,
            "session_active": self.active_session is not None,
            "session_id": self.active_session.session_id if self.active_session else None,
            "device": self.device,
            "model": self.config.get("model_name"),
            "performance_metrics": self.performance_metrics.get_summary() if self.performance_metrics else None
        }


class StateManager:
    """Manages digital human system state"""
    
    def __init__(self):
        self.state_history = []
        self.current_state = SystemState.IDLE
        self.state_transitions = {
            SystemState.IDLE: [SystemState.LISTENING, SystemState.PROCESSING],
            SystemState.LISTENING: [SystemState.PROCESSING, SystemState.IDLE],
            SystemState.PROCESSING: [SystemState.SPEAKING, SystemState.THINKING, SystemState.ERROR],
            SystemState.SPEAKING: [SystemState.IDLE, SystemState.LISTENING],
            SystemState.THINKING: [SystemState.SPEAKING, SystemState.ERROR],
            SystemState.ERROR: [SystemState.IDLE]
        }
    
    def transition_to(self, new_state: SystemState) -> bool:
        """Transition to new state if valid"""
        if new_state in self.state_transitions.get(self.current_state, []):
            self.state_history.append({
                "from": self.current_state,
                "to": new_state,
                "timestamp": datetime.now()
            })
            self.current_state = new_state
            return True
        return False
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state transition history"""
        return self.state_history


class ResponseGenerator:
    """Generates responses combining multiple systems"""
    
    def __init__(
        self,
        conversation_engine: SgLangConversationEngine,
        emotion_mapper: EmotionalResponseMapper,
        facial_animator: FacialAnimationSystem
    ):
        self.conversation_engine = conversation_engine
        self.emotion_mapper = emotion_mapper
        self.facial_animator = facial_animator
    
    async def generate(
        self,
        user_input: str,
        context: ConversationContext,
        emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """Generate complete response with all components"""
        # Get base response
        response_text, metadata = await self.conversation_engine.process_message(
            user_input,
            context
        )
        
        # Apply emotional adaptation
        adapted_text = await self.emotion_mapper.adapt_response_tone(
            response_text,
            emotional_state
        )
        
        # Generate expression
        expression = self.emotion_mapper.map_emotion_to_expression(emotional_state)
        
        # Generate voice parameters
        voice_params = self.emotion_mapper.map_emotion_to_voice_params(emotional_state)
        
        return {
            "text": adapted_text,
            "original_text": response_text,
            "metadata": metadata,
            "expression": expression,
            "voice": voice_params,
            "emotion": emotional_state
        }


class PerformanceMonitor:
    """Monitors digital human system performance"""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "gpu_utilization": [],
            "memory_usage": [],
            "frame_rates": [],
            "error_count": 0
        }
    
    def record_response_time(self, duration: float):
        """Record response generation time"""
        self.metrics["response_times"].append(duration)
    
    def record_gpu_usage(self):
        """Record current GPU utilization"""
        if torch.cuda.is_available():
            utilization = torch.cuda.utilization()
            memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.metrics["gpu_utilization"].append(utilization)
            self.metrics["memory_usage"].append(memory)
    
    def record_frame_rate(self, fps: float):
        """Record rendering frame rate"""
        self.metrics["frame_rates"].append(fps)
    
    def record_error(self):
        """Record error occurrence"""
        self.metrics["error_count"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "avg_response_time": np.mean(self.metrics["response_times"]) if self.metrics["response_times"] else 0,
            "avg_gpu_utilization": np.mean(self.metrics["gpu_utilization"]) if self.metrics["gpu_utilization"] else 0,
            "avg_memory_usage": np.mean(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
            "avg_frame_rate": np.mean(self.metrics["frame_rates"]) if self.metrics["frame_rates"] else 0,
            "error_count": self.metrics["error_count"]
        }


# Utility functions for easy instantiation
async def create_digital_human(config: Dict[str, Any]) -> DigitalHumanOrchestrator:
    """Create and initialize a digital human system"""
    orchestrator = DigitalHumanOrchestrator(config)
    return orchestrator