"""
Expression library for digital human avatar system.

Provides a comprehensive library of facial expressions, gestures,
and animation presets.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

from aiq.digital_human.avatar.facial_animator import BlendshapeType


class ExpressionCategory(Enum):
    """Categories of expressions"""
    BASIC = "basic"
    EMOTIONAL = "emotional"
    CONVERSATIONAL = "conversational"
    THINKING = "thinking"
    SOCIAL = "social"
    COMPOUND = "compound"


@dataclass
class Expression:
    """Defines a complete expression preset"""
    name: str
    category: ExpressionCategory
    blendshapes: Dict[BlendshapeType, float]
    duration: float = 1.0
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    tags: List[str] = None


@dataclass
class GestureSequence:
    """Defines a gesture animation sequence"""
    name: str
    keyframes: List[Dict[str, Any]]
    total_duration: float
    loop: bool = False
    tags: List[str] = None


class ExpressionLibrary:
    """
    Comprehensive library of facial expressions and gestures
    for digital human avatars.
    """
    
    def __init__(self):
        self.expressions: Dict[str, Expression] = {}
        self.gestures: Dict[str, GestureSequence] = {}
        self.expression_transitions: Dict[Tuple[str, str], float] = {}
        
        # Initialize default library
        self._initialize_basic_expressions()
        self._initialize_emotional_expressions()
        self._initialize_conversational_expressions()
        self._initialize_thinking_expressions()
        self._initialize_social_expressions()
        self._initialize_compound_expressions()
        self._initialize_gestures()
        self._initialize_transitions()
    
    def _initialize_basic_expressions(self):
        """Initialize basic facial expressions"""
        basic_expressions = [
            Expression(
                name="neutral",
                category=ExpressionCategory.BASIC,
                blendshapes={},
                tags=["default", "rest"]
            ),
            Expression(
                name="smile",
                category=ExpressionCategory.BASIC,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.7,
                    BlendshapeType.MOUTH_SMILE_R: 0.7
                },
                tags=["positive", "friendly"]
            ),
            Expression(
                name="frown",
                category=ExpressionCategory.BASIC,
                blendshapes={
                    BlendshapeType.MOUTH_FROWN_L: 0.6,
                    BlendshapeType.MOUTH_FROWN_R: 0.6
                },
                tags=["negative", "displeasure"]
            ),
            Expression(
                name="blink",
                category=ExpressionCategory.BASIC,
                blendshapes={
                    BlendshapeType.EYE_BLINK_L: 1.0,
                    BlendshapeType.EYE_BLINK_R: 1.0
                },
                duration=0.15,
                tags=["reflex", "natural"]
            ),
            Expression(
                name="eyebrow_raise",
                category=ExpressionCategory.BASIC,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.8,
                    BlendshapeType.BROW_UP_R: 0.8
                },
                tags=["surprise", "interest"]
            )
        ]
        
        for expr in basic_expressions:
            self.expressions[expr.name] = expr
    
    def _initialize_emotional_expressions(self):
        """Initialize emotional expressions"""
        emotional_expressions = [
            Expression(
                name="happy",
                category=ExpressionCategory.EMOTIONAL,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.9,
                    BlendshapeType.MOUTH_SMILE_R: 0.9,
                    BlendshapeType.CHEEK_PUFF: 0.4,
                    BlendshapeType.EYE_WIDE_L: 0.2,
                    BlendshapeType.EYE_WIDE_R: 0.2
                },
                tags=["positive", "joy", "emotion"]
            ),
            Expression(
                name="sad",
                category=ExpressionCategory.EMOTIONAL,
                blendshapes={
                    BlendshapeType.MOUTH_FROWN_L: 0.7,
                    BlendshapeType.MOUTH_FROWN_R: 0.7,
                    BlendshapeType.BROW_DOWN_L: 0.4,
                    BlendshapeType.BROW_DOWN_R: 0.4,
                    BlendshapeType.EYE_BLINK_L: 0.3,
                    BlendshapeType.EYE_BLINK_R: 0.3
                },
                tags=["negative", "sorrow", "emotion"]
            ),
            Expression(
                name="angry",
                category=ExpressionCategory.EMOTIONAL,
                blendshapes={
                    BlendshapeType.BROW_DOWN_L: 0.9,
                    BlendshapeType.BROW_DOWN_R: 0.9,
                    BlendshapeType.NOSE_SNEER_L: 0.5,
                    BlendshapeType.NOSE_SNEER_R: 0.5,
                    BlendshapeType.MOUTH_FROWN_L: 0.4,
                    BlendshapeType.MOUTH_FROWN_R: 0.4,
                    BlendshapeType.JAW_FORWARD: 0.2
                },
                tags=["negative", "anger", "emotion"]
            ),
            Expression(
                name="surprised",
                category=ExpressionCategory.EMOTIONAL,
                blendshapes={
                    BlendshapeType.EYE_WIDE_L: 1.0,
                    BlendshapeType.EYE_WIDE_R: 1.0,
                    BlendshapeType.BROW_UP_L: 0.9,
                    BlendshapeType.BROW_UP_R: 0.9,
                    BlendshapeType.JAW_OPEN: 0.5
                },
                duration=0.3,
                tags=["shock", "amazement", "emotion"]
            ),
            Expression(
                name="disgusted",
                category=ExpressionCategory.EMOTIONAL,
                blendshapes={
                    BlendshapeType.NOSE_SNEER_L: 0.8,
                    BlendshapeType.NOSE_SNEER_R: 0.8,
                    BlendshapeType.MOUTH_FROWN_L: 0.5,
                    BlendshapeType.MOUTH_FROWN_R: 0.5,
                    BlendshapeType.EYE_BLINK_L: 0.2,
                    BlendshapeType.EYE_BLINK_R: 0.2
                },
                tags=["negative", "disgust", "emotion"]
            ),
            Expression(
                name="fearful",
                category=ExpressionCategory.EMOTIONAL,
                blendshapes={
                    BlendshapeType.EYE_WIDE_L: 0.8,
                    BlendshapeType.EYE_WIDE_R: 0.8,
                    BlendshapeType.BROW_UP_L: 0.7,
                    BlendshapeType.BROW_UP_R: 0.7,
                    BlendshapeType.MOUTH_FROWN_L: 0.3,
                    BlendshapeType.MOUTH_FROWN_R: 0.3,
                    BlendshapeType.JAW_OPEN: 0.2
                },
                tags=["negative", "fear", "emotion"]
            )
        ]
        
        for expr in emotional_expressions:
            self.expressions[expr.name] = expr
    
    def _initialize_conversational_expressions(self):
        """Initialize conversational expressions"""
        conversational_expressions = [
            Expression(
                name="attentive",
                category=ExpressionCategory.CONVERSATIONAL,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.2,
                    BlendshapeType.BROW_UP_R: 0.2,
                    BlendshapeType.EYE_WIDE_L: 0.1,
                    BlendshapeType.EYE_WIDE_R: 0.1
                },
                tags=["listening", "engaged"]
            ),
            Expression(
                name="questioning",
                category=ExpressionCategory.CONVERSATIONAL,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.6,
                    BlendshapeType.BROW_DOWN_R: 0.2,
                    BlendshapeType.MOUTH_SMILE_L: 0.1,
                    BlendshapeType.MOUTH_SMILE_R: 0.2
                },
                tags=["inquiry", "curiosity"]
            ),
            Expression(
                name="explaining",
                category=ExpressionCategory.CONVERSATIONAL,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.3,
                    BlendshapeType.BROW_UP_R: 0.3,
                    BlendshapeType.MOUTH_SMILE_L: 0.2,
                    BlendshapeType.MOUTH_SMILE_R: 0.2
                },
                tags=["teaching", "clarifying"]
            ),
            Expression(
                name="agreeing",
                category=ExpressionCategory.CONVERSATIONAL,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.4,
                    BlendshapeType.MOUTH_SMILE_R: 0.4,
                    BlendshapeType.EYE_BLINK_L: 0.1,
                    BlendshapeType.EYE_BLINK_R: 0.1
                },
                duration=0.5,
                tags=["affirmative", "nodding"]
            ),
            Expression(
                name="disagreeing",
                category=ExpressionCategory.CONVERSATIONAL,
                blendshapes={
                    BlendshapeType.MOUTH_FROWN_L: 0.3,
                    BlendshapeType.MOUTH_FROWN_R: 0.3,
                    BlendshapeType.BROW_DOWN_L: 0.2,
                    BlendshapeType.BROW_DOWN_R: 0.2
                },
                tags=["negative", "rejection"]
            )
        ]
        
        for expr in conversational_expressions:
            self.expressions[expr.name] = expr
    
    def _initialize_thinking_expressions(self):
        """Initialize thinking expressions"""
        thinking_expressions = [
            Expression(
                name="pondering",
                category=ExpressionCategory.THINKING,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.4,
                    BlendshapeType.BROW_DOWN_R: 0.3,
                    BlendshapeType.MOUTH_PUCKER: 0.2,
                    BlendshapeType.EYE_BLINK_L: 0.15
                },
                tags=["contemplation", "thought"]
            ),
            Expression(
                name="concentrating",
                category=ExpressionCategory.THINKING,
                blendshapes={
                    BlendshapeType.BROW_DOWN_L: 0.5,
                    BlendshapeType.BROW_DOWN_R: 0.5,
                    BlendshapeType.EYE_BLINK_L: 0.2,
                    BlendshapeType.EYE_BLINK_R: 0.2,
                    BlendshapeType.MOUTH_PUCKER: 0.1
                },
                tags=["focus", "intense"]
            ),
            Expression(
                name="remembering",
                category=ExpressionCategory.THINKING,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.3,
                    BlendshapeType.BROW_UP_R: 0.3,
                    BlendshapeType.EYE_BLINK_L: 0.1,
                    BlendshapeType.EYE_BLINK_R: 0.1
                },
                tags=["recall", "memory"]
            ),
            Expression(
                name="analyzing",
                category=ExpressionCategory.THINKING,
                blendshapes={
                    BlendshapeType.BROW_DOWN_L: 0.3,
                    BlendshapeType.BROW_DOWN_R: 0.3,
                    BlendshapeType.EYE_WIDE_L: 0.1,
                    BlendshapeType.EYE_WIDE_R: 0.1
                },
                tags=["processing", "computation"]
            )
        ]
        
        for expr in thinking_expressions:
            self.expressions[expr.name] = expr
    
    def _initialize_social_expressions(self):
        """Initialize social expressions"""
        social_expressions = [
            Expression(
                name="greeting",
                category=ExpressionCategory.SOCIAL,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.6,
                    BlendshapeType.MOUTH_SMILE_R: 0.6,
                    BlendshapeType.BROW_UP_L: 0.2,
                    BlendshapeType.BROW_UP_R: 0.2,
                    BlendshapeType.EYE_WIDE_L: 0.1,
                    BlendshapeType.EYE_WIDE_R: 0.1
                },
                duration=1.5,
                tags=["hello", "welcome"]
            ),
            Expression(
                name="farewell",
                category=ExpressionCategory.SOCIAL,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.4,
                    BlendshapeType.MOUTH_SMILE_R: 0.4,
                    BlendshapeType.EYE_BLINK_L: 0.3,
                    BlendshapeType.EYE_BLINK_R: 0.3
                },
                tags=["goodbye", "parting"]
            ),
            Expression(
                name="apologetic",
                category=ExpressionCategory.SOCIAL,
                blendshapes={
                    BlendshapeType.MOUTH_FROWN_L: 0.3,
                    BlendshapeType.MOUTH_FROWN_R: 0.3,
                    BlendshapeType.BROW_UP_L: 0.4,
                    BlendshapeType.BROW_UP_R: 0.4,
                    BlendshapeType.EYE_BLINK_L: 0.2,
                    BlendshapeType.EYE_BLINK_R: 0.2
                },
                tags=["sorry", "regret"]
            ),
            Expression(
                name="grateful",
                category=ExpressionCategory.SOCIAL,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.7,
                    BlendshapeType.MOUTH_SMILE_R: 0.7,
                    BlendshapeType.EYE_BLINK_L: 0.2,
                    BlendshapeType.EYE_BLINK_R: 0.2,
                    BlendshapeType.CHEEK_PUFF: 0.3
                },
                tags=["thanks", "appreciation"]
            )
        ]
        
        for expr in social_expressions:
            self.expressions[expr.name] = expr
    
    def _initialize_compound_expressions(self):
        """Initialize compound expressions combining multiple emotions"""
        compound_expressions = [
            Expression(
                name="amused",
                category=ExpressionCategory.COMPOUND,
                blendshapes={
                    BlendshapeType.MOUTH_SMILE_L: 0.6,
                    BlendshapeType.MOUTH_SMILE_R: 0.7,
                    BlendshapeType.BROW_UP_L: 0.3,
                    BlendshapeType.BROW_UP_R: 0.2,
                    BlendshapeType.EYE_BLINK_L: 0.1,
                    BlendshapeType.EYE_BLINK_R: 0.1
                },
                tags=["humor", "entertainment"]
            ),
            Expression(
                name="skeptical",
                category=ExpressionCategory.COMPOUND,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.5,
                    BlendshapeType.BROW_DOWN_R: 0.4,
                    BlendshapeType.MOUTH_SMILE_L: 0.1,
                    BlendshapeType.MOUTH_FROWN_R: 0.2,
                    BlendshapeType.EYE_BLINK_L: 0.2
                },
                tags=["doubt", "questioning"]
            ),
            Expression(
                name="concerned",
                category=ExpressionCategory.COMPOUND,
                blendshapes={
                    BlendshapeType.BROW_DOWN_L: 0.3,
                    BlendshapeType.BROW_DOWN_R: 0.3,
                    BlendshapeType.MOUTH_FROWN_L: 0.2,
                    BlendshapeType.MOUTH_FROWN_R: 0.2,
                    BlendshapeType.EYE_WIDE_L: 0.2,
                    BlendshapeType.EYE_WIDE_R: 0.2
                },
                tags=["worry", "empathy"]
            ),
            Expression(
                name="impressed",
                category=ExpressionCategory.COMPOUND,
                blendshapes={
                    BlendshapeType.BROW_UP_L: 0.5,
                    BlendshapeType.BROW_UP_R: 0.5,
                    BlendshapeType.MOUTH_SMILE_L: 0.3,
                    BlendshapeType.MOUTH_SMILE_R: 0.3,
                    BlendshapeType.EYE_WIDE_L: 0.3,
                    BlendshapeType.EYE_WIDE_R: 0.3
                },
                tags=["admiration", "respect"]
            )
        ]
        
        for expr in compound_expressions:
            self.expressions[expr.name] = expr
    
    def _initialize_gestures(self):
        """Initialize gesture sequences"""
        gestures = [
            GestureSequence(
                name="head_nod",
                keyframes=[
                    {"time": 0.0, "head_pitch": 0.0},
                    {"time": 0.2, "head_pitch": 10.0},
                    {"time": 0.4, "head_pitch": 0.0},
                    {"time": 0.6, "head_pitch": 8.0},
                    {"time": 0.8, "head_pitch": 0.0}
                ],
                total_duration=0.8,
                tags=["agreement", "yes"]
            ),
            GestureSequence(
                name="head_shake",
                keyframes=[
                    {"time": 0.0, "head_yaw": 0.0},
                    {"time": 0.2, "head_yaw": -15.0},
                    {"time": 0.4, "head_yaw": 15.0},
                    {"time": 0.6, "head_yaw": -10.0},
                    {"time": 0.8, "head_yaw": 0.0}
                ],
                total_duration=0.8,
                tags=["disagreement", "no"]
            ),
            GestureSequence(
                name="thinking_gesture",
                keyframes=[
                    {"time": 0.0, "head_tilt": 0.0, "expression": "neutral"},
                    {"time": 0.5, "head_tilt": 10.0, "expression": "pondering"},
                    {"time": 2.0, "head_tilt": 12.0, "eye_gaze": (0.2, 0.8, 0.5)},
                    {"time": 3.0, "head_tilt": 0.0, "expression": "neutral"}
                ],
                total_duration=3.0,
                tags=["contemplation", "processing"]
            )
        ]
        
        for gesture in gestures:
            self.gestures[gesture.name] = gesture
    
    def _initialize_transitions(self):
        """Initialize expression transition timing"""
        # Define smooth transition times between expressions
        transitions = [
            (("neutral", "happy"), 0.5),
            (("neutral", "sad"), 0.7),
            (("happy", "sad"), 1.0),
            (("happy", "surprised"), 0.3),
            (("sad", "happy"), 1.2),
            (("angry", "neutral"), 0.8),
            (("surprised", "neutral"), 0.4),
            (("thinking", "explaining"), 0.6),
            (("questioning", "understanding"), 0.5)
        ]
        
        for (from_expr, to_expr), duration in transitions:
            self.expression_transitions[(from_expr, to_expr)] = duration
            # Add reverse transition with same duration
            self.expression_transitions[(to_expr, from_expr)] = duration
    
    def get_expression(self, name: str) -> Optional[Expression]:
        """Get expression by name"""
        return self.expressions.get(name)
    
    def get_expressions_by_category(
        self,
        category: ExpressionCategory
    ) -> List[Expression]:
        """Get all expressions in a category"""
        return [
            expr for expr in self.expressions.values()
            if expr.category == category
        ]
    
    def get_expressions_by_tag(self, tag: str) -> List[Expression]:
        """Get expressions with specific tag"""
        return [
            expr for expr in self.expressions.values()
            if expr.tags and tag in expr.tags
        ]
    
    def get_gesture(self, name: str) -> Optional[GestureSequence]:
        """Get gesture sequence by name"""
        return self.gestures.get(name)
    
    def get_transition_duration(
        self,
        from_expression: str,
        to_expression: str
    ) -> float:
        """Get recommended transition duration between expressions"""
        return self.expression_transitions.get(
            (from_expression, to_expression),
            0.5  # Default transition time
        )
    
    def create_custom_expression(
        self,
        name: str,
        blendshapes: Dict[BlendshapeType, float],
        category: ExpressionCategory = ExpressionCategory.COMPOUND,
        tags: Optional[List[str]] = None
    ) -> Expression:
        """Create and register custom expression"""
        expression = Expression(
            name=name,
            category=category,
            blendshapes=blendshapes,
            tags=tags or []
        )
        
        self.expressions[name] = expression
        return expression
    
    def blend_expressions(
        self,
        expr1_name: str,
        expr2_name: str,
        blend_factor: float = 0.5
    ) -> Dict[BlendshapeType, float]:
        """Blend two expressions together"""
        expr1 = self.get_expression(expr1_name)
        expr2 = self.get_expression(expr2_name)
        
        if not expr1 or not expr2:
            return {}
        
        blended = {}
        
        # Blend expr1 weights
        for blendshape, weight in expr1.blendshapes.items():
            blended[blendshape] = weight * (1.0 - blend_factor)
        
        # Blend expr2 weights
        for blendshape, weight in expr2.blendshapes.items():
            if blendshape in blended:
                blended[blendshape] += weight * blend_factor
            else:
                blended[blendshape] = weight * blend_factor
        
        return blended
    
    def export_library(self, filepath: str):
        """Export expression library to JSON"""
        library_data = {
            "expressions": {
                name: {
                    "category": expr.category.value,
                    "blendshapes": {bs.value: weight for bs, weight in expr.blendshapes.items()},
                    "duration": expr.duration,
                    "intensity_range": expr.intensity_range,
                    "tags": expr.tags
                }
                for name, expr in self.expressions.items()
            },
            "gestures": {
                name: {
                    "keyframes": gesture.keyframes,
                    "total_duration": gesture.total_duration,
                    "loop": gesture.loop,
                    "tags": gesture.tags
                }
                for name, gesture in self.gestures.items()
            },
            "transitions": {
                f"{from_expr}->{to_expr}": duration
                for (from_expr, to_expr), duration in self.expression_transitions.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(library_data, f, indent=2)
    
    def import_library(self, filepath: str):
        """Import expression library from JSON"""
        with open(filepath, 'r') as f:
            library_data = json.load(f)
        
        # Import expressions
        for name, expr_data in library_data.get("expressions", {}).items():
            blendshapes = {
                BlendshapeType[bs]: weight
                for bs, weight in expr_data["blendshapes"].items()
            }
            
            expression = Expression(
                name=name,
                category=ExpressionCategory(expr_data["category"]),
                blendshapes=blendshapes,
                duration=expr_data.get("duration", 1.0),
                intensity_range=tuple(expr_data.get("intensity_range", [0.0, 1.0])),
                tags=expr_data.get("tags", [])
            )
            
            self.expressions[name] = expression
        
        # Import gestures
        for name, gesture_data in library_data.get("gestures", {}).items():
            gesture = GestureSequence(
                name=name,
                keyframes=gesture_data["keyframes"],
                total_duration=gesture_data["total_duration"],
                loop=gesture_data.get("loop", False),
                tags=gesture_data.get("tags", [])
            )
            
            self.gestures[name] = gesture
        
        # Import transitions
        for transition_str, duration in library_data.get("transitions", {}).items():
            from_expr, to_expr = transition_str.split("->")
            self.expression_transitions[(from_expr, to_expr)] = duration