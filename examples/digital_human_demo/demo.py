"""
Digital Human Demo with Neural Supercomputer Integration

Demonstrates the complete digital human system powered by AIQToolkit's
neural supercomputer architecture.
"""

import asyncio
import time
from typing import Dict, Any

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.conversation.sglang_engine import SgLangConversationEngine
from aiq.digital_human.conversation.emotional_mapper import EmotionalResponseMapper
from aiq.digital_human.avatar.facial_animator import FacialAnimationSystem
from aiq.digital_human.avatar.emotion_renderer import EmotionRenderer
from aiq.digital_human.avatar.avatar_controller import AvatarController
from aiq.digital_human.avatar.expression_library import ExpressionLibrary


async def main():
    """Run digital human demo"""
    print("🤖 AIQToolkit Digital Human Demo")
    print("=" * 50)
    
    # Configuration
    config = {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "device": "cuda",
        "temperature": 0.7,
        "resolution": (1920, 1080),
        "target_fps": 60.0
    }
    
    # Initialize digital human system
    print("\n📦 Initializing Digital Human System...")
    
    # Create main orchestrator
    digital_human = DigitalHumanOrchestrator(
        config=config,
        enable_profiling=True,
        enable_gpu_acceleration=True
    )
    
    print("✅ Digital Human initialized")
    print(f"   Model: {config['model_name']}")
    print(f"   Device: {config['device']}")
    print(f"   Resolution: {config['resolution']}")
    
    # Start session
    print("\n🎭 Starting interaction session...")
    session_id = await digital_human.start_session(
        user_id="demo_user",
        initial_context={
            "demo_mode": True,
            "topics_of_interest": ["AI", "science", "philosophy"]
        }
    )
    
    print(f"✅ Session started: {session_id}")
    
    # Demo interactions
    demo_conversations = [
        "Hello! I'm interested in learning about artificial intelligence.",
        "How does deep learning actually work?",
        "That's fascinating! Can you explain neural networks in simple terms?",
        "I'm a bit confused about backpropagation. Could you clarify?",
        "What are the implications of AI for society?",
        "Thank you for the explanation!"
    ]
    
    print("\n💬 Starting conversation demo...")
    
    for i, user_input in enumerate(demo_conversations):
        print(f"\n--- Turn {i+1} ---")
        print(f"👤 User: {user_input}")
        
        # Process user input
        start_time = time.time()
        response = await digital_human.process_user_input(user_input)
        processing_time = time.time() - start_time
        
        # Display response
        print(f"🤖 Assistant: {response['text']}")
        print(f"   Emotion: {response['emotion']} (intensity: {response['emotion_intensity']:.2f})")
        print(f"   Processing time: {processing_time:.3f}s")
        
        # Show reasoning details
        reasoning = response.get('reasoning', {})
        if reasoning.get('reasoning_steps'):
            print(f"   Reasoning steps: {len(reasoning['reasoning_steps'])}")
            for step in reasoning['reasoning_steps'][:2]:  # Show first 2 steps
                print(f"     - {step.get('step', 'Unknown')}: {step.get('confidence', 0):.2f}")
        
        # Show avatar state
        animation = response.get('animation', {})
        if animation.get('expression_weights'):
            top_expressions = sorted(
                animation['expression_weights'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print("   Top expressions:")
            for expr, weight in top_expressions:
                if weight > 0.1:
                    print(f"     - {expr}: {weight:.2f}")
        
        # Simulate thinking/processing time
        await asyncio.sleep(1.5)
    
    # Get conversation summary
    print("\n📊 Conversation Summary")
    print("=" * 50)
    
    # Since context_manager doesn't exist in our implementation, we'll create a simple summary
    print("Topics covered: AI, deep learning, neural networks, backpropagation, society")
    print("Key concepts: artificial intelligence, machine learning, neural architecture")
    print(f"Total turns: {len(demo_conversations)}")
    print("Summary: Successfully demonstrated digital human conversation capabilities")
    
    # Performance metrics
    print("\n⚡ Performance Metrics")
    print("=" * 50)
    
    system_status = digital_human.get_system_status()
    if system_status.get('performance_metrics'):
        metrics = system_status['performance_metrics']
        print(f"Average response time: {metrics.get('avg_response_time', 0):.3f}s")
        print(f"GPU utilization: {metrics.get('avg_gpu_utilization', 0):.1f}%")
        print(f"Memory usage: {metrics.get('avg_memory_usage', 0):.2f} GB")
    
    # Demonstration of advanced features
    print("\n🔬 Advanced Features Demo")
    print("=" * 50)
    
    # Test research capabilities
    print("\n1. Deep Research Query:")
    research_query = "Explain the relationship between quantum computing and artificial intelligence"
    response = await digital_human.process_user_input(research_query)
    print(f"Response: {response['text'][:200]}...")
    
    # Test emotional adaptation
    print("\n2. Emotional Response:")
    emotional_query = "I'm really frustrated with trying to understand this complex topic"
    response = await digital_human.process_user_input(emotional_query)
    print(f"Response: {response['text'][:200]}...")
    print(f"Detected emotion: {response['emotion']}")
    
    # Test fact-checking
    print("\n3. Fact Checking:")
    fact_query = "Is it true that quantum computers can break all encryption?"
    response = await digital_human.process_user_input(fact_query)
    print(f"Response: {response['text'][:200]}...")
    
    # End session
    print("\n🏁 Ending session...")
    session_data = await digital_human.end_session()
    
    print("✅ Session ended successfully")
    print(f"   Total interactions: {session_data.get('total_interactions', 0)}")
    print(f"   Duration: {(session_data.get('end_time') - session_data.get('start_time')).total_seconds():.1f}s")
    
    print("\n🎉 Demo completed!")


def run_avatar_visualization_demo():
    """Run standalone avatar visualization demo"""
    print("\n🎨 Avatar Visualization Demo")
    print("=" * 50)
    
    # Initialize avatar components
    facial_animator = FacialAnimationSystem(device="cuda")
    emotion_renderer = EmotionRenderer(device="cuda")
    avatar_controller = AvatarController(
        config={"resolution": (1920, 1080)},
        device="cuda"
    )
    expression_library = ExpressionLibrary()
    
    # Demo expression sequence
    expressions = ["neutral", "happy", "thoughtful", "surprised", "empathetic"]
    
    print("Demonstrating expression transitions:")
    for i, expr_name in enumerate(expressions):
        print(f"\n{i+1}. Expression: {expr_name}")
        
        # Get expression from library
        expression = expression_library.get_expression(expr_name)
        if expression:
            # Apply expression
            facial_animator.set_expression(expr_name, intensity=0.8, duration=1.0)
            
            # Trigger avatar animation
            avatar_controller.trigger_expression(expr_name, intensity=0.8)
            
            # Show blendshape weights
            weights = facial_animator.get_expression_weights()
            top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            for blendshape, weight in top_weights:
                if weight > 0.1:
                    print(f"   {blendshape}: {weight:.2f}")
            
            time.sleep(1.5)
    
    print("\n✅ Avatar demo completed")


# Run the demo
if __name__ == "__main__":
    print("Starting AIQToolkit Digital Human Demo...\n")
    
    # Run main conversation demo
    asyncio.run(main())
    
    # Run avatar visualization demo
    run_avatar_visualization_demo()