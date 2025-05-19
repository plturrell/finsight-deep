"""
Avatar controller for digital human system.

Manages avatar state, rendering pipeline, and animation coordination.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time
import numpy as np
import torch

from aiq.digital_human.avatar.facial_animator import FacialAnimationSystem
from aiq.digital_human.avatar.emotion_renderer import EmotionRenderer
from aiq.digital_human.conversation.emotional_mapper import EmotionalState


@dataclass
class AvatarState:
    """Complete avatar state at a point in time"""
    timestamp: float
    facial_expression: Dict[str, float]
    head_pose: Tuple[float, float, float]
    eye_gaze: Tuple[float, float, float]
    body_pose: Dict[str, Any]
    active_animations: List[str]
    render_settings: Dict[str, Any]


class AvatarController:
    """
    Controls the digital human avatar system, coordinating
    facial animation, emotion rendering, and state management.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        target_fps: float = 60.0
    ):
        self.config = config
        self.device = device
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Initialize subsystems
        self.facial_animator = FacialAnimationSystem(
            device=device,
            fps=target_fps,
            enable_gpu_skinning=True
        )
        
        self.emotion_renderer = EmotionRenderer(
            device=device,
            render_resolution=config.get("resolution", (1920, 1080)),
            enable_effects=True,
            use_gpu_rendering=True
        )
        
        # Avatar state
        self.current_state = self._get_default_state()
        self.state_history: List[AvatarState] = []
        self.max_history_size = 1000
        
        # Animation queue
        self.animation_queue: List[Dict[str, Any]] = []
        self.active_animations: Dict[str, Any] = {}
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_times: List[float] = []
        self.gpu_memory_usage: List[float] = []
        
        # Async update task
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    def _get_default_state(self) -> AvatarState:
        """Get default avatar state"""
        return AvatarState(
            timestamp=time.time(),
            facial_expression=self.facial_animator.get_expression_weights(),
            head_pose=(0.0, 0.0, 0.0),
            eye_gaze=(0.0, 0.0, 1.0),
            body_pose={},
            active_animations=[],
            render_settings={
                "lighting": "default",
                "background": "neutral",
                "effects_enabled": True
            }
        )
    
    async def start(self):
        """Start avatar controller update loop"""
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
    
    async def stop(self):
        """Stop avatar controller"""
        self.is_running = False
        if self.update_task:
            await self.update_task
    
    async def _update_loop(self):
        """Main update loop for avatar system"""
        while self.is_running:
            start_time = time.time()
            
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Update subsystems
            await self.update(delta_time)
            
            # Track performance
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
            
            # Track GPU memory
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024**2
                self.gpu_memory_usage.append(memory_mb)
                if len(self.gpu_memory_usage) > 100:
                    self.gpu_memory_usage.pop(0)
            
            # Sleep to maintain target FPS
            sleep_time = self.frame_time - frame_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def update(self, delta_time: float):
        """Update avatar state and animations"""
        # Process animation queue
        self._process_animation_queue()
        
        # Update active animations
        self._update_active_animations(delta_time)
        
        # Update facial animation
        self.facial_animator.update(delta_time)
        
        # Update emotion renderer
        self.emotion_renderer.update(delta_time)
        
        # Update state
        self._update_state()
        
        # Save state to history
        self._save_state_snapshot()
    
    def _process_animation_queue(self):
        """Process queued animations"""
        current_time = time.time()
        
        # Start queued animations
        for animation in self.animation_queue[:]:
            if animation["start_time"] <= current_time:
                self.animation_queue.remove(animation)
                self._start_animation(animation)
    
    def _start_animation(self, animation: Dict[str, Any]):
        """Start an animation"""
        animation_id = animation["id"]
        animation_type = animation["type"]
        
        if animation_type == "expression":
            self.facial_animator.set_expression(
                animation["expression_name"],
                animation.get("intensity", 1.0),
                animation.get("duration", 1.0)
            )
        elif animation_type == "blendshape":
            self.facial_animator.set_blendshape(
                animation["blendshape"],
                animation["weight"]
            )
        elif animation_type == "head_pose":
            self.set_head_pose(
                animation["rotation"],
                animation.get("duration", 1.0)
            )
        elif animation_type == "eye_gaze":
            self.set_eye_gaze(
                animation["target"],
                animation.get("duration", 0.5)
            )
        
        self.active_animations[animation_id] = {
            **animation,
            "start_time": time.time()
        }
    
    def _update_active_animations(self, delta_time: float):
        """Update currently active animations"""
        current_time = time.time()
        completed_animations = []
        
        for animation_id, animation in self.active_animations.items():
            elapsed = current_time - animation["start_time"]
            duration = animation.get("duration", 1.0)
            
            if elapsed >= duration:
                completed_animations.append(animation_id)
            else:
                # Update animation progress
                progress = elapsed / duration
                animation["progress"] = progress
        
        # Remove completed animations
        for animation_id in completed_animations:
            del self.active_animations[animation_id]
    
    def _update_state(self):
        """Update current avatar state"""
        self.current_state = AvatarState(
            timestamp=time.time(),
            facial_expression=self.facial_animator.get_expression_weights(),
            head_pose=self.facial_animator.head_rotation,
            eye_gaze=self.facial_animator.eye_target,
            body_pose=self.emotion_renderer.body_pose.__dict__,
            active_animations=list(self.active_animations.keys()),
            render_settings=self.get_render_settings()
        )
    
    def _save_state_snapshot(self):
        """Save current state to history"""
        self.state_history.append(self.current_state)
        
        # Maintain history size limit
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
    
    def set_emotional_state(self, emotional_state: EmotionalState):
        """Set avatar emotional state"""
        # Update facial expression
        expression_params = self.emotion_renderer.facial_animator.emotion_mapper.map_emotion_to_expression(
            emotional_state
        )
        self.facial_animator.set_emotion_expression(expression_params)
        
        # Update emotion renderer
        self.emotion_renderer.current_emotion = emotional_state.primary_emotion
    
    def queue_animation(
        self,
        animation_type: str,
        parameters: Dict[str, Any],
        delay: float = 0.0
    ) -> str:
        """Queue an animation for execution"""
        animation_id = f"anim_{int(time.time() * 1000)}"
        
        animation = {
            "id": animation_id,
            "type": animation_type,
            "start_time": time.time() + delay,
            **parameters
        }
        
        self.animation_queue.append(animation)
        return animation_id
    
    def set_head_pose(
        self,
        rotation: Tuple[float, float, float],
        duration: float = 1.0
    ):
        """Set head pose (pitch, yaw, roll)"""
        # Animate to new head pose
        self.queue_animation(
            "head_pose",
            {
                "rotation": rotation,
                "duration": duration
            }
        )
    
    def set_eye_gaze(
        self,
        target: Tuple[float, float, float],
        duration: float = 0.5
    ):
        """Set eye gaze target"""
        self.queue_animation(
            "eye_gaze",
            {
                "target": target,
                "duration": duration
            }
        )
    
    def trigger_expression(
        self,
        expression_name: str,
        intensity: float = 1.0,
        duration: float = 2.0
    ):
        """Trigger a facial expression"""
        self.queue_animation(
            "expression",
            {
                "expression_name": expression_name,
                "intensity": intensity,
                "duration": duration
            }
        )
    
    def play_lip_sync(
        self,
        phonemes: List[Tuple[str, float, float]],
        audio_duration: float
    ):
        """Play lip sync animation"""
        self.facial_animator.process_lip_sync(phonemes)
        self.facial_animator.play_animation()
    
    def get_render_settings(self) -> Dict[str, Any]:
        """Get current render settings"""
        return {
            "lighting": "default",
            "background": "neutral",
            "effects_enabled": True,
            "resolution": self.emotion_renderer.resolution,
            "device": self.device
        }
    
    def update_render_settings(self, settings: Dict[str, Any]):
        """Update render settings"""
        # Apply settings to emotion renderer
        if "resolution" in settings:
            self.emotion_renderer.resolution = settings["resolution"]
        if "effects_enabled" in settings:
            self.emotion_renderer.enable_effects = settings["effects_enabled"]
    
    def get_current_frame(self) -> np.ndarray:
        """Get current rendered frame"""
        # Render current state
        render_data = self.emotion_renderer.render_emotion(
            self.emotion_renderer.current_emotion,
            None  # No message content for current frame
        )
        
        return render_data.get("frame", np.zeros((1080, 1920, 3), dtype=np.uint8))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "avg_frame_time": np.mean(self.frame_times) if self.frame_times else 0,
            "current_fps": 1.0 / self.frame_times[-1] if self.frame_times else 0,
            "target_fps": self.target_fps,
            "gpu_memory_mb": np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            "active_animations": len(self.active_animations),
            "queued_animations": len(self.animation_queue)
        }
    
    def get_state_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[AvatarState]:
        """Get state history within time range"""
        if not start_time and not end_time:
            return self.state_history
        
        filtered_history = []
        for state in self.state_history:
            if start_time and state.timestamp < start_time:
                continue
            if end_time and state.timestamp > end_time:
                continue
            filtered_history.append(state)
        
        return filtered_history
    
    def reset(self):
        """Reset avatar to default state"""
        self.facial_animator.reset()
        self.emotion_renderer.current_emotion = EmotionType.NEUTRAL
        self.current_state = self._get_default_state()
        self.animation_queue.clear()
        self.active_animations.clear()
    
    def cleanup(self):
        """Clean up resources"""
        self.emotion_renderer.cleanup()
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()