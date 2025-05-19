"""
GPU-accelerated facial animation system for digital humans.

Provides real-time facial animation with blendshape morphing,
lip-sync, and emotional expressions.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
import time
from enum import Enum

try:
    import vispy
    from vispy import app, scene
except ImportError:
    vispy = None

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer
from aiq.cuda_kernels.similarity_kernels import cosine_similarity_cuda


class BlendshapeType(Enum):
    """Standard facial blendshape types"""
    BROW_DOWN_L = "browDown_L"
    BROW_DOWN_R = "browDown_R"
    BROW_UP_L = "browUp_L"
    BROW_UP_R = "browUp_R"
    EYE_BLINK_L = "eyeBlink_L"
    EYE_BLINK_R = "eyeBlink_R"
    EYE_WIDE_L = "eyeWide_L"
    EYE_WIDE_R = "eyeWide_R"
    JAW_OPEN = "jawOpen"
    JAW_FORWARD = "jawForward"
    JAW_LEFT = "jawLeft"
    JAW_RIGHT = "jawRight"
    MOUTH_SMILE_L = "mouthSmile_L"
    MOUTH_SMILE_R = "mouthSmile_R"
    MOUTH_FROWN_L = "mouthFrown_L"
    MOUTH_FROWN_R = "mouthFrown_R"
    MOUTH_PUCKER = "mouthPucker"
    CHEEK_PUFF = "cheekPuff"
    NOSE_SNEER_L = "noseSneer_L"
    NOSE_SNEER_R = "noseSneer_R"


@dataclass
class AnimationKeyframe:
    """Represents a single animation keyframe"""
    time: float
    blendshape_weights: Dict[str, float]
    head_rotation: Tuple[float, float, float]  # pitch, yaw, roll
    eye_target: Tuple[float, float, float]  # 3D gaze target


@dataclass
class FacialExpression:
    """Defines a complete facial expression"""
    name: str
    blendshape_weights: Dict[BlendshapeType, float]
    duration: float = 1.0
    intensity: float = 1.0


class FacialAnimationSystem:
    """
    Real-time facial animation system with GPU acceleration.
    Handles blendshape morphing, lip-sync, and expression blending.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        fps: float = 60.0,
        blend_rate: float = 0.15,
        enable_gpu_skinning: bool = True
    ):
        self.device = device
        self.fps = fps
        self.blend_rate = blend_rate
        self.frame_time = 1.0 / fps
        
        # GPU optimization
        if enable_gpu_skinning:
            self.tensor_optimizer = TensorCoreOptimizer()
        else:
            self.tensor_optimizer = None
        
        # Animation state
        self.current_weights = {bs: 0.0 for bs in BlendshapeType}
        self.target_weights = {bs: 0.0 for bs in BlendshapeType}
        self.head_rotation = (0.0, 0.0, 0.0)
        self.eye_target = (0.0, 0.0, 1.0)  # Looking forward
        
        # Expression library
        self.expressions = self._initialize_expressions()
        
        # Lip-sync phoneme mapping
        self.phoneme_mapping = self._initialize_phoneme_mapping()
        
        # Animation timeline
        self.keyframes: List[AnimationKeyframe] = []
        self.current_time = 0.0
        self.is_playing = False
        
        # Mesh data (placeholder - would be loaded from 3D model)
        self.base_mesh = None
        self.blendshape_deltas = None
        self.vertex_count = 0
        
        # Initialize GPU tensors if available
        if torch.cuda.is_available() and device == "cuda":
            self._initialize_gpu_tensors()
    
    def _initialize_expressions(self) -> Dict[str, FacialExpression]:
        """Initialize library of facial expressions"""
        return {
            "neutral": FacialExpression(
                name="neutral",
                blendshape_weights={}
            ),
            "happy": FacialExpression(
                name="happy",
                blendshape_weights={
                    BlendshapeType.MOUTH_SMILE_L: 0.8,
                    BlendshapeType.MOUTH_SMILE_R: 0.8,
                    BlendshapeType.CHEEK_PUFF: 0.3,
                    BlendshapeType.EYE_WIDE_L: 0.2,
                    BlendshapeType.EYE_WIDE_R: 0.2
                }
            ),
            "sad": FacialExpression(
                name="sad",
                blendshape_weights={
                    BlendshapeType.MOUTH_FROWN_L: 0.6,
                    BlendshapeType.MOUTH_FROWN_R: 0.6,
                    BlendshapeType.BROW_DOWN_L: 0.4,
                    BlendshapeType.BROW_DOWN_R: 0.4,
                    BlendshapeType.EYE_BLINK_L: 0.2,
                    BlendshapeType.EYE_BLINK_R: 0.2
                }
            ),
            "surprised": FacialExpression(
                name="surprised",
                blendshape_weights={
                    BlendshapeType.EYE_WIDE_L: 0.9,
                    BlendshapeType.EYE_WIDE_R: 0.9,
                    BlendshapeType.BROW_UP_L: 0.8,
                    BlendshapeType.BROW_UP_R: 0.8,
                    BlendshapeType.JAW_OPEN: 0.4
                }
            ),
            "angry": FacialExpression(
                name="angry",
                blendshape_weights={
                    BlendshapeType.BROW_DOWN_L: 0.8,
                    BlendshapeType.BROW_DOWN_R: 0.8,
                    BlendshapeType.NOSE_SNEER_L: 0.5,
                    BlendshapeType.NOSE_SNEER_R: 0.5,
                    BlendshapeType.MOUTH_FROWN_L: 0.4,
                    BlendshapeType.MOUTH_FROWN_R: 0.4
                }
            ),
            "thinking": FacialExpression(
                name="thinking",
                blendshape_weights={
                    BlendshapeType.BROW_UP_L: 0.4,
                    BlendshapeType.BROW_DOWN_R: 0.3,
                    BlendshapeType.MOUTH_PUCKER: 0.2,
                    BlendshapeType.EYE_BLINK_L: 0.1
                }
            )
        }
    
    def _initialize_phoneme_mapping(self) -> Dict[str, Dict[BlendshapeType, float]]:
        """Initialize phoneme to blendshape mapping for lip-sync"""
        return {
            # Vowels
            "AA": {BlendshapeType.JAW_OPEN: 0.8, BlendshapeType.MOUTH_SMILE_L: 0.2, BlendshapeType.MOUTH_SMILE_R: 0.2},
            "AE": {BlendshapeType.JAW_OPEN: 0.6, BlendshapeType.MOUTH_SMILE_L: 0.4, BlendshapeType.MOUTH_SMILE_R: 0.4},
            "EH": {BlendshapeType.JAW_OPEN: 0.4, BlendshapeType.MOUTH_SMILE_L: 0.3, BlendshapeType.MOUTH_SMILE_R: 0.3},
            "IY": {BlendshapeType.MOUTH_SMILE_L: 0.7, BlendshapeType.MOUTH_SMILE_R: 0.7},
            "OW": {BlendshapeType.MOUTH_PUCKER: 0.8, BlendshapeType.JAW_OPEN: 0.3},
            "UW": {BlendshapeType.MOUTH_PUCKER: 0.9, BlendshapeType.JAW_FORWARD: 0.3},
            
            # Consonants
            "B": {BlendshapeType.MOUTH_PUCKER: 0.6},
            "P": {BlendshapeType.MOUTH_PUCKER: 0.7},
            "M": {BlendshapeType.MOUTH_PUCKER: 0.5},
            "F": {BlendshapeType.JAW_OPEN: 0.1, BlendshapeType.MOUTH_FROWN_L: 0.2, BlendshapeType.MOUTH_FROWN_R: 0.2},
            "V": {BlendshapeType.JAW_OPEN: 0.15, BlendshapeType.MOUTH_FROWN_L: 0.15, BlendshapeType.MOUTH_FROWN_R: 0.15},
            "TH": {BlendshapeType.JAW_OPEN: 0.2},
            "S": {BlendshapeType.MOUTH_SMILE_L: 0.3, BlendshapeType.MOUTH_SMILE_R: 0.3},
            "SH": {BlendshapeType.MOUTH_PUCKER: 0.4, BlendshapeType.JAW_FORWARD: 0.2},
            "K": {BlendshapeType.JAW_OPEN: 0.3},
            "G": {BlendshapeType.JAW_OPEN: 0.35},
            "L": {BlendshapeType.JAW_OPEN: 0.2},
            "R": {BlendshapeType.MOUTH_PUCKER: 0.3, BlendshapeType.JAW_FORWARD: 0.1},
            "W": {BlendshapeType.MOUTH_PUCKER: 0.7},
            "Y": {BlendshapeType.MOUTH_SMILE_L: 0.4, BlendshapeType.MOUTH_SMILE_R: 0.4}
        }
    
    def _initialize_gpu_tensors(self):
        """Initialize GPU tensors for mesh deformation"""
        # These would be populated with actual mesh data
        if torch.cuda.is_available():
            # Placeholder dimensions
            self.vertex_count = 5000  # Typical face mesh
            self.blendshape_count = len(BlendshapeType)
            
            # Base mesh vertices (3D positions)
            self.base_mesh_gpu = torch.zeros(
                self.vertex_count, 3,
                device=self.device,
                dtype=torch.float32
            )
            
            # Blendshape deltas (per blendshape vertex displacement)
            self.blendshape_deltas_gpu = torch.zeros(
                self.blendshape_count, self.vertex_count, 3,
                device=self.device,
                dtype=torch.float32
            )
            
            # Current blendshape weights
            self.weights_gpu = torch.zeros(
                self.blendshape_count,
                device=self.device,
                dtype=torch.float32
            )
    
    def set_expression(
        self,
        expression_name: str,
        intensity: float = 1.0,
        duration: float = 0.5
    ):
        """Set facial expression with optional intensity and transition duration"""
        if expression_name not in self.expressions:
            print(f"Unknown expression: {expression_name}")
            return
        
        expression = self.expressions[expression_name]
        
        # Clear current targets
        self.target_weights = {bs: 0.0 for bs in BlendshapeType}
        
        # Apply expression weights with intensity
        for blendshape, weight in expression.blendshape_weights.items():
            self.target_weights[blendshape] = weight * intensity
        
        # Set transition duration
        self.transition_duration = duration
        self.transition_start_time = time.time()
    
    def set_blendshape(
        self,
        blendshape: BlendshapeType,
        weight: float
    ):
        """Set individual blendshape weight"""
        self.target_weights[blendshape] = np.clip(weight, 0.0, 1.0)
    
    def set_emotion_expression(
        self,
        expression_params: Dict[str, Any]
    ):
        """Set expression from emotion mapper parameters"""
        # Map emotion parameters to blendshapes
        eyebrow_raise = expression_params.get("eyebrow_raise", 0.0)
        smile = expression_params.get("smile", 0.0)
        eye_openness = expression_params.get("eye_openness", 1.0)
        head_tilt = expression_params.get("head_tilt", 0.0)
        
        # Clear weights
        self.target_weights = {bs: 0.0 for bs in BlendshapeType}
        
        # Eyebrows
        if eyebrow_raise > 0:
            self.target_weights[BlendshapeType.BROW_UP_L] = eyebrow_raise
            self.target_weights[BlendshapeType.BROW_UP_R] = eyebrow_raise
        else:
            self.target_weights[BlendshapeType.BROW_DOWN_L] = -eyebrow_raise
            self.target_weights[BlendshapeType.BROW_DOWN_R] = -eyebrow_raise
        
        # Mouth
        if smile > 0:
            self.target_weights[BlendshapeType.MOUTH_SMILE_L] = smile
            self.target_weights[BlendshapeType.MOUTH_SMILE_R] = smile
        else:
            self.target_weights[BlendshapeType.MOUTH_FROWN_L] = -smile
            self.target_weights[BlendshapeType.MOUTH_FROWN_R] = -smile
        
        # Eyes
        if eye_openness > 1.0:
            self.target_weights[BlendshapeType.EYE_WIDE_L] = (eye_openness - 1.0) * 2
            self.target_weights[BlendshapeType.EYE_WIDE_R] = (eye_openness - 1.0) * 2
        else:
            blink_amount = 1.0 - eye_openness
            self.target_weights[BlendshapeType.EYE_BLINK_L] = blink_amount
            self.target_weights[BlendshapeType.EYE_BLINK_R] = blink_amount
        
        # Head rotation
        self.head_rotation = (head_tilt * 10, 0.0, head_tilt * 5)  # degrees
    
    def process_lip_sync(
        self,
        phonemes: List[Tuple[str, float, float]]
    ):
        """
        Process phoneme sequence for lip-sync animation.
        
        Args:
            phonemes: List of (phoneme, start_time, duration) tuples
        """
        self.keyframes = []
        
        for phoneme, start_time, duration in phonemes:
            if phoneme in self.phoneme_mapping:
                # Create keyframe for phoneme
                weights = self.current_weights.copy()
                
                # Apply phoneme blendshapes
                for blendshape, weight in self.phoneme_mapping[phoneme].items():
                    weights[blendshape] = weight
                
                # Add keyframe
                self.keyframes.append(AnimationKeyframe(
                    time=start_time,
                    blendshape_weights=weights,
                    head_rotation=self.head_rotation,
                    eye_target=self.eye_target
                ))
                
                # Add transition keyframe
                self.keyframes.append(AnimationKeyframe(
                    time=start_time + duration * 0.7,
                    blendshape_weights=self.current_weights.copy(),
                    head_rotation=self.head_rotation,
                    eye_target=self.eye_target
                ))
    
    def update(self, delta_time: float):
        """Update animation state"""
        # Smooth blending towards target weights
        for blendshape in BlendshapeType:
            current = self.current_weights[blendshape]
            target = self.target_weights[blendshape]
            
            # Exponential smoothing
            self.current_weights[blendshape] = current + (target - current) * self.blend_rate
        
        # Update timeline animation if playing
        if self.is_playing and self.keyframes:
            self.current_time += delta_time
            self._update_timeline_animation()
        
        # Update GPU tensors if available
        if torch.cuda.is_available() and self.device == "cuda":
            self._update_gpu_mesh()
    
    def _update_timeline_animation(self):
        """Update animation based on timeline keyframes"""
        if not self.keyframes:
            return
        
        # Find current and next keyframes
        current_kf = None
        next_kf = None
        
        for i, kf in enumerate(self.keyframes):
            if kf.time <= self.current_time:
                current_kf = kf
                if i + 1 < len(self.keyframes):
                    next_kf = self.keyframes[i + 1]
            else:
                break
        
        if current_kf and next_kf:
            # Interpolate between keyframes
            t = (self.current_time - current_kf.time) / (next_kf.time - current_kf.time)
            t = np.clip(t, 0.0, 1.0)
            
            for blendshape in BlendshapeType:
                current_weight = current_kf.blendshape_weights.get(blendshape, 0.0)
                next_weight = next_kf.blendshape_weights.get(blendshape, 0.0)
                self.target_weights[blendshape] = current_weight + (next_weight - current_weight) * t
        elif current_kf:
            # Use last keyframe
            for blendshape, weight in current_kf.blendshape_weights.items():
                self.target_weights[blendshape] = weight
    
    def _update_gpu_mesh(self):
        """Update GPU mesh with current blendshape weights"""
        if not hasattr(self, 'base_mesh_gpu'):
            return
        
        # Convert current weights to GPU tensor
        weight_values = [self.current_weights[bs] for bs in BlendshapeType]
        self.weights_gpu = torch.tensor(
            weight_values,
            device=self.device,
            dtype=torch.float32
        )
        
        # Compute deformed mesh: base + sum(weight * delta)
        # Using Einstein notation for efficient computation
        deformed_mesh = self.base_mesh_gpu + torch.einsum(
            'b,bvc->vc',
            self.weights_gpu,
            self.blendshape_deltas_gpu
        )
        
        # Apply any additional transformations (head rotation, etc.)
        if any(r != 0 for r in self.head_rotation):
            deformed_mesh = self._apply_rotation(deformed_mesh, self.head_rotation)
        
        return deformed_mesh
    
    def _apply_rotation(
        self,
        vertices: torch.Tensor,
        rotation: Tuple[float, float, float]
    ) -> torch.Tensor:
        """Apply rotation to vertices"""
        pitch, yaw, roll = [np.radians(r) for r in rotation]
        
        # Rotation matrices
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ], device=self.device, dtype=torch.float32)
        
        Ry = torch.tensor([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ], device=self.device, dtype=torch.float32)
        
        Rz = torch.tensor([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Combined rotation
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)
        
        # Apply rotation
        return torch.matmul(vertices, R.T)
    
    def get_current_mesh(self) -> Optional[np.ndarray]:
        """Get current deformed mesh vertices"""
        if torch.cuda.is_available() and self.device == "cuda":
            mesh = self._update_gpu_mesh()
            return mesh.cpu().numpy()
        return None
    
    def get_expression_weights(self) -> Dict[str, float]:
        """Get current blendshape weights"""
        return {bs.value: weight for bs, weight in self.current_weights.items()}
    
    def play_animation(self):
        """Start timeline animation playback"""
        self.is_playing = True
        self.current_time = 0.0
    
    def stop_animation(self):
        """Stop timeline animation playback"""
        self.is_playing = False
    
    def reset(self):
        """Reset to neutral expression"""
        self.set_expression("neutral")
        self.keyframes = []
        self.current_time = 0.0
        self.is_playing = False