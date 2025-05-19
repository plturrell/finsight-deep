"""
Emotion rendering system for digital human avatars.

Combines facial animation, body language, and visual effects
to convey emotions effectively.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from enum import Enum

try:
    import vispy
    from vispy import app, scene
    from vispy.visuals.transforms import STTransform
except ImportError:
    vispy = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    plt = None
    cm = None

from aiq.digital_human.conversation.emotional_mapper import EmotionType, EmotionalState
from aiq.digital_human.avatar.facial_animator import FacialAnimationSystem
from aiq.visualization.gpu_visualizer import GPUVisualizationEngine


@dataclass
class BodyPose:
    """Represents body posture and gestures"""
    head_tilt: float
    shoulder_height: float
    torso_lean: float
    arm_position: Dict[str, Tuple[float, float, float]]
    gesture_type: Optional[str] = None
    gesture_intensity: float = 0.0


@dataclass
class VisualEffect:
    """Visual effects for emotion enhancement"""
    effect_type: str
    color: Tuple[float, float, float, float]  # RGBA
    intensity: float
    duration: float
    position: Optional[Tuple[float, float]] = None


class EmotionRenderer:
    """
    Renders emotions through avatar visualization combining
    facial expressions, body language, and visual effects.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        render_resolution: Tuple[int, int] = (1920, 1080),
        enable_effects: bool = True,
        use_gpu_rendering: bool = True
    ):
        self.device = device
        self.resolution = render_resolution
        self.enable_effects = enable_effects
        
        # Initialize subsystems
        self.facial_animator = FacialAnimationSystem(device=device)
        
        if use_gpu_rendering:
            self.gpu_visualizer = GPUVisualizationEngine(
                backend="vispy" if vispy else "matplotlib"
            )
        else:
            self.gpu_visualizer = None
        
        # Avatar state
        self.current_emotion = EmotionType.NEUTRAL
        self.body_pose = self._get_neutral_pose()
        self.active_effects: List[VisualEffect] = []
        
        # Color palettes for emotions
        self.emotion_colors = self._initialize_emotion_colors()
        
        # Gesture library
        self.gesture_library = self._initialize_gestures()
        
        # Initialize rendering scene if VisPy available
        if vispy is not None:
            self._initialize_vispy_scene()
        else:
            self.scene = None
    
    def _initialize_emotion_colors(self) -> Dict[EmotionType, Tuple[float, float, float, float]]:
        """Initialize color schemes for each emotion"""
        return {
            EmotionType.NEUTRAL: (0.8, 0.8, 0.8, 1.0),      # Light gray
            EmotionType.HAPPY: (1.0, 0.9, 0.3, 1.0),        # Warm yellow
            EmotionType.EMPATHETIC: (0.6, 0.8, 1.0, 1.0),   # Soft blue
            EmotionType.CONCERNED: (0.9, 0.6, 0.5, 1.0),    # Muted red
            EmotionType.EXCITED: (1.0, 0.5, 0.0, 1.0),      # Bright orange
            EmotionType.THOUGHTFUL: (0.7, 0.6, 0.9, 1.0),   # Lavender
            EmotionType.ENCOURAGING: (0.3, 0.9, 0.6, 1.0),  # Fresh green
            EmotionType.PROFESSIONAL: (0.4, 0.4, 0.6, 1.0)  # Business blue
        }
    
    def _initialize_gestures(self) -> Dict[str, Dict[str, Any]]:
        """Initialize gesture animations for different emotions"""
        return {
            "thinking": {
                "arm_position": {"right": (0.3, 0.7, 0.1)},  # Hand to chin
                "head_tilt": 15.0,
                "duration": 2.0
            },
            "explaining": {
                "arm_position": {"right": (0.5, 0.5, 0.3), "left": (0.4, 0.4, 0.2)},
                "head_tilt": 0.0,
                "duration": 3.0
            },
            "welcoming": {
                "arm_position": {"right": (0.8, 0.4, 0.2), "left": (0.8, 0.4, 0.2)},
                "shoulder_height": 0.1,
                "duration": 1.5
            },
            "emphasizing": {
                "arm_position": {"right": (0.6, 0.6, 0.4)},
                "torso_lean": 10.0,
                "duration": 1.0
            }
        }
    
    def _initialize_vispy_scene(self):
        """Initialize VisPy 3D scene for avatar rendering"""
        if vispy is None:
            return
        
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor='black',
            size=self.resolution
        )
        self.canvas.show()
        
        # Add a ViewBox
        self.view = self.canvas.central_widget.add_view()
        
        # Create avatar mesh placeholder
        # In production, this would load actual 3D avatar model
        vertices = np.random.normal(size=(100, 3))
        self.avatar_mesh = scene.visuals.Markers(
            pos=vertices,
            edge_color=None,
            face_color=(0.8, 0.6, 0.5, 1),
            size=10,
            scaling=True,
            parent=self.view.scene
        )
        
        # Create lighting
        self.light = scene.visuals.Light(
            pos=(0, 0, 10),
            color=(1.0, 1.0, 1.0, 1.0),
            parent=self.view.scene
        )
        
        # Set up camera
        self.camera = scene.TurntableCamera(
            elevation=10,
            azimuth=0,
            distance=5,
            fov=60,
            parent=self.view.scene
        )
        self.view.camera = self.camera
    
    def _get_neutral_pose(self) -> BodyPose:
        """Get neutral body pose"""
        return BodyPose(
            head_tilt=0.0,
            shoulder_height=0.0,
            torso_lean=0.0,
            arm_position={
                "left": (0.0, -0.5, 0.0),
                "right": (0.0, -0.5, 0.0)
            }
        )
    
    def render_emotion(
        self,
        emotional_state: EmotionalState,
        message_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Render complete emotional state including face, body, and effects.
        
        Args:
            emotional_state: Current emotional state from mapper
            message_content: Optional message for gesture selection
            
        Returns:
            Rendering metadata and frame buffer
        """
        # Update facial expression
        expression_params = self.facial_animator.emotion_mapper.map_emotion_to_expression(
            emotional_state
        )
        self.facial_animator.set_emotion_expression(expression_params)
        
        # Update body pose
        self.body_pose = self._map_emotion_to_body_pose(
            emotional_state,
            message_content
        )
        
        # Update visual effects
        if self.enable_effects:
            self.active_effects = self._generate_visual_effects(emotional_state)
        
        # Render frame
        frame_data = self._render_frame()
        
        # Return metadata
        return {
            "frame": frame_data,
            "emotion": emotional_state.primary_emotion.value,
            "intensity": emotional_state.intensity,
            "expression_weights": self.facial_animator.get_expression_weights(),
            "body_pose": self._serialize_body_pose(self.body_pose),
            "active_effects": [self._serialize_effect(e) for e in self.active_effects]
        }
    
    def _map_emotion_to_body_pose(
        self,
        emotional_state: EmotionalState,
        message_content: Optional[str] = None
    ) -> BodyPose:
        """Map emotional state to appropriate body posture"""
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Base poses for each emotion
        base_poses = {
            EmotionType.NEUTRAL: self._get_neutral_pose(),
            EmotionType.HAPPY: BodyPose(
                head_tilt=5.0,
                shoulder_height=0.05,
                torso_lean=0.0,
                arm_position={"left": (0.1, -0.4, 0.1), "right": (0.1, -0.4, 0.1)}
            ),
            EmotionType.EMPATHETIC: BodyPose(
                head_tilt=10.0,
                shoulder_height=-0.05,
                torso_lean=5.0,
                arm_position={"left": (0.2, -0.3, 0.2), "right": (0.2, -0.3, 0.2)}
            ),
            EmotionType.CONCERNED: BodyPose(
                head_tilt=7.0,
                shoulder_height=-0.03,
                torso_lean=3.0,
                arm_position={"left": (0.0, -0.4, 0.1), "right": (0.0, -0.4, 0.1)}
            ),
            EmotionType.EXCITED: BodyPose(
                head_tilt=-5.0,
                shoulder_height=0.08,
                torso_lean=-2.0,
                arm_position={"left": (0.3, -0.2, 0.3), "right": (0.3, -0.2, 0.3)}
            ),
            EmotionType.THOUGHTFUL: BodyPose(
                head_tilt=12.0,
                shoulder_height=0.0,
                torso_lean=0.0,
                arm_position={"left": (0.0, -0.5, 0.0), "right": (0.3, 0.7, 0.1)}
            ),
            EmotionType.ENCOURAGING: BodyPose(
                head_tilt=0.0,
                shoulder_height=0.05,
                torso_lean=2.0,
                arm_position={"left": (0.2, -0.3, 0.2), "right": (0.4, -0.2, 0.3)}
            ),
            EmotionType.PROFESSIONAL: BodyPose(
                head_tilt=0.0,
                shoulder_height=0.0,
                torso_lean=0.0,
                arm_position={"left": (0.0, -0.5, 0.0), "right": (0.0, -0.5, 0.0)}
            )
        }
        
        base_pose = base_poses.get(emotion, self._get_neutral_pose())
        
        # Apply intensity scaling
        base_pose.head_tilt *= intensity
        base_pose.shoulder_height *= intensity
        base_pose.torso_lean *= intensity
        
        # Add contextual gestures if message provided
        if message_content:
            gesture = self._select_gesture(message_content, emotion)
            if gesture:
                base_pose.gesture_type = gesture["type"]
                base_pose.gesture_intensity = intensity
                # Blend gesture positions
                for limb, position in gesture.get("arm_position", {}).items():
                    current = base_pose.arm_position[limb]
                    base_pose.arm_position[limb] = tuple(
                        c + (p - c) * intensity for c, p in zip(current, position)
                    )
        
        return base_pose
    
    def _generate_visual_effects(
        self,
        emotional_state: EmotionalState
    ) -> List[VisualEffect]:
        """Generate visual effects based on emotional state"""
        effects = []
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Ambient glow effect
        if intensity > 0.6:
            color = self.emotion_colors[emotion]
            effects.append(VisualEffect(
                effect_type="glow",
                color=(*color[:3], color[3] * intensity * 0.5),
                intensity=intensity,
                duration=2.0
            ))
        
        # Particle effects for excitement
        if emotion == EmotionType.EXCITED and intensity > 0.7:
            effects.append(VisualEffect(
                effect_type="particles",
                color=(1.0, 0.8, 0.3, 0.8),
                intensity=intensity,
                duration=3.0,
                position=(0, 0)  # Center
            ))
        
        # Aura effect for empathy
        if emotion == EmotionType.EMPATHETIC:
            effects.append(VisualEffect(
                effect_type="aura",
                color=(0.6, 0.8, 1.0, 0.3),
                intensity=intensity * 0.7,
                duration=4.0
            ))
        
        # Thinking sparkles
        if emotion == EmotionType.THOUGHTFUL and intensity > 0.5:
            effects.append(VisualEffect(
                effect_type="sparkles",
                color=(0.9, 0.9, 1.0, 0.9),
                intensity=intensity,
                duration=2.5,
                position=(0.3, 0.7)  # Near head
            ))
        
        return effects
    
    def _select_gesture(
        self,
        message_content: str,
        emotion: EmotionType
    ) -> Optional[Dict[str, Any]]:
        """Select appropriate gesture based on message and emotion"""
        message_lower = message_content.lower()
        
        # Keyword-based gesture selection
        if any(word in message_lower for word in ["think", "consider", "wonder"]):
            return {
                "type": "thinking",
                **self.gesture_library["thinking"]
            }
        elif any(word in message_lower for word in ["explain", "describe", "tell"]):
            return {
                "type": "explaining",
                **self.gesture_library["explaining"]
            }
        elif any(word in message_lower for word in ["welcome", "hello", "hi"]):
            return {
                "type": "welcoming",
                **self.gesture_library["welcoming"]
            }
        elif any(word in message_lower for word in ["important", "crucial", "key"]):
            return {
                "type": "emphasizing",
                **self.gesture_library["emphasizing"]
            }
        
        # Emotion-based fallback
        emotion_gestures = {
            EmotionType.THOUGHTFUL: "thinking",
            EmotionType.EXCITED: "emphasizing",
            EmotionType.HAPPY: "welcoming"
        }
        
        if emotion in emotion_gestures:
            gesture_type = emotion_gestures[emotion]
            return {
                "type": gesture_type,
                **self.gesture_library[gesture_type]
            }
        
        return None
    
    def _render_frame(self) -> np.ndarray:
        """Render current frame with avatar and effects"""
        if self.scene is not None and vispy is not None:
            # Update avatar mesh with current deformation
            deformed_mesh = self.facial_animator.get_current_mesh()
            if deformed_mesh is not None:
                self.avatar_mesh.set_data(pos=deformed_mesh[:, :3])
            
            # Apply body pose transformations
            self._apply_body_pose_to_mesh()
            
            # Render effects
            self._render_visual_effects()
            
            # Capture frame
            frame = self.canvas.render()
            return np.array(frame)
        else:
            # Fallback to matplotlib rendering
            return self._render_matplotlib_frame()
    
    def _apply_body_pose_to_mesh(self):
        """Apply body pose to avatar mesh in scene"""
        if self.avatar_mesh is None:
            return
        
        # Create transformation matrix
        transform = STTransform()
        
        # Apply rotations
        transform.rotate(self.body_pose.head_tilt, (1, 0, 0))  # Pitch
        transform.rotate(self.body_pose.torso_lean, (0, 0, 1))  # Roll
        
        # Apply translations
        transform.translate((0, self.body_pose.shoulder_height, 0))
        
        self.avatar_mesh.transform = transform
    
    def _render_visual_effects(self):
        """Render active visual effects in scene"""
        for effect in self.active_effects:
            if effect.effect_type == "glow":
                self._render_glow_effect(effect)
            elif effect.effect_type == "particles":
                self._render_particle_effect(effect)
            elif effect.effect_type == "aura":
                self._render_aura_effect(effect)
            elif effect.effect_type == "sparkles":
                self._render_sparkle_effect(effect)
    
    def _render_glow_effect(self, effect: VisualEffect):
        """Render glow effect around avatar"""
        # Would implement actual glow shader or post-processing
        pass
    
    def _render_particle_effect(self, effect: VisualEffect):
        """Render particle system"""
        # Would implement particle system
        pass
    
    def _render_aura_effect(self, effect: VisualEffect):
        """Render aura effect"""
        # Would implement aura visualization
        pass
    
    def _render_sparkle_effect(self, effect: VisualEffect):
        """Render sparkle effects"""
        # Would implement sparkle particles
        pass
    
    def _render_matplotlib_frame(self) -> np.ndarray:
        """Fallback rendering using matplotlib"""
        if plt is None:
            # Return placeholder frame
            return np.zeros((*self.resolution, 3), dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=(self.resolution[0]/100, self.resolution[1]/100))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw avatar placeholder
        emotion_color = self.emotion_colors[self.current_emotion]
        circle = plt.Circle((0, 0), 0.3, color=emotion_color[:3], alpha=emotion_color[3])
        ax.add_patch(circle)
        
        # Draw expression indicators
        expression_weights = self.facial_animator.get_expression_weights()
        y_pos = 0.6
        for expression, weight in list(expression_weights.items())[:5]:
            if weight > 0.1:
                ax.text(0, y_pos, f"{expression}: {weight:.2f}",
                       ha='center', va='center', fontsize=8)
                y_pos -= 0.1
        
        # Convert to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return frame
    
    def _serialize_body_pose(self, pose: BodyPose) -> Dict[str, Any]:
        """Convert body pose to serializable format"""
        return {
            "head_tilt": pose.head_tilt,
            "shoulder_height": pose.shoulder_height,
            "torso_lean": pose.torso_lean,
            "arm_position": pose.arm_position,
            "gesture_type": pose.gesture_type,
            "gesture_intensity": pose.gesture_intensity
        }
    
    def _serialize_effect(self, effect: VisualEffect) -> Dict[str, Any]:
        """Convert visual effect to serializable format"""
        return {
            "type": effect.effect_type,
            "color": effect.color,
            "intensity": effect.intensity,
            "duration": effect.duration,
            "position": effect.position
        }
    
    def update(self, delta_time: float):
        """Update animation states"""
        # Update facial animation
        self.facial_animator.update(delta_time)
        
        # Update effects
        self.active_effects = [
            effect for effect in self.active_effects
            if effect.duration > 0
        ]
        
        for effect in self.active_effects:
            effect.duration -= delta_time
            # Could update effect parameters here
    
    def cleanup(self):
        """Clean up rendering resources"""
        if self.canvas is not None:
            self.canvas.close()