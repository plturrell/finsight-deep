"""
Real gRPC client implementation for NVIDIA Audio2Face-3D
Based on NVIDIA's official gRPC infrastructure documentation
"""

import asyncio
import grpc
import logging
import os
import struct
from typing import AsyncIterator, Dict, List, Optional, Tuple
import numpy as np

# These will be generated from the proto file
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "generated"))

try:
    import nvidia_audio2face_pb2 as audio2face_pb2
    import nvidia_audio2face_pb2_grpc as audio2face_pb2_grpc
except ImportError:
    try:
        import audio2face_pb2
        import audio2face_pb2_grpc
    except ImportError:
        # If not compiled yet, we'll define stub classes
        print("Warning: Proto files not compiled. Run compile_protos.sh first.")
        audio2face_pb2 = None
        audio2face_pb2_grpc = None

logger = logging.getLogger(__name__)


class NvidiaAudio2FaceGrpcClient:
    """Real gRPC client for NVIDIA Audio2Face-3D service"""
    
    # Official NVIDIA models with their Function IDs
    MODELS = {
        "james": "9327c39f-a361-4e02-bd72-e11b4c9b7b5e",
        "claire": "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d",
        "mark": "8efc55f5-6f00-424e-afe9-26212cd2c630"
    }
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "james",
        endpoint: str = "grpc.nvcf.nvidia.com:443",
        secure: bool = True
    ):
        """
        Initialize the NVIDIA Audio2Face gRPC client
        
        Args:
            api_key: NVIDIA API key for authentication
            model_name: Avatar model to use (james, claire, or mark)
            endpoint: gRPC endpoint URL
            secure: Whether to use secure channel (TLS)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model_id = self.MODELS.get(model_name, self.MODELS["james"])
        self.endpoint = endpoint
        self.secure = secure
        
        # Channel and stub
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[audio2face_pb2_grpc.Audio2FaceServiceStub] = None
        
        # Metadata for authentication
        self.metadata = [
            ("authorization", f"Bearer {self.api_key}"),
            ("x-nv-function-id", self.model_id),
        ]
        
    async def connect(self):
        """Establish gRPC connection"""
        if self.secure:
            # Create secure channel with proper credentials
            credentials = grpc.ssl_channel_credentials()
            self.channel = grpc.aio.secure_channel(
                self.endpoint,
                credentials,
                options=[
                    ("grpc.max_send_message_length", 10 * 1024 * 1024),  # 10MB
                    ("grpc.max_receive_message_length", 10 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                ]
            )
        else:
            self.channel = grpc.aio.insecure_channel(self.endpoint)
        
        # Check which stub is available
        if hasattr(audio2face_pb2_grpc, 'Audio2FaceStub'):
            self.stub = audio2face_pb2_grpc.Audio2FaceStub(self.channel)
        elif hasattr(audio2face_pb2_grpc, 'Audio2FaceServiceStub'):
            self.stub = audio2face_pb2_grpc.Audio2FaceServiceStub(self.channel)
        else:
            raise AttributeError("Could not find appropriate stub class in generated code")
        logger.info(f"Connected to NVIDIA Audio2Face at {self.endpoint}")
        
    async def disconnect(self):
        """Close gRPC connection"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from NVIDIA Audio2Face")
    
    async def get_available_models(self) -> List[Dict[str, any]]:
        """Get list of available avatar models"""
        # NVIDIA's Audio2Face service might not have GetAvailableModels
        # Return predefined models
        return [
            {"id": self.MODELS["james"], "name": "james", "version": "1.0", "features": ["photorealistic", "male", "tongue"]},
            {"id": self.MODELS["claire"], "name": "claire", "version": "1.0", "features": ["photorealistic", "female", "tongue"]},
            {"id": self.MODELS["mark"], "name": "mark", "version": "1.0", "features": ["photorealistic", "male", "tongue"]}
        ]
    
    async def initialize_model(
        self,
        enable_tongue: bool = True,
        animation_fps: float = 30.0,
        quality: str = "high"
    ) -> bool:
        """Initialize the avatar model with specific configuration"""
        if not self.stub:
            await self.connect()
            
        try:
            config = audio2face_pb2.ModelConfig(
                avatar_id=self.model_id,
                high_quality=(quality == "high"),
                fps=animation_fps,
                enable_tongue=enable_tongue,
                expression_scale=1.0
            )
            
            # For NVIDIA's version, we might need to use GetFacialAnimation
            # or the service might not have InitializeModel
            # Let's use the actual audio request for initialization
            
            # Skip initialization for now - will initialize on first use
            logger.info(f"Configured model {self.model_name} with settings")
            return True
            
            if response.success:
                logger.info(f"Initialized model {self.model_name}: {response.message}")
                return True
            else:
                logger.error(f"Failed to initialize model: {response.message}")
                return False
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error initializing model: {e}")
            raise
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        encoding: str = "PCM_16",
        language: str = "en-US",
        emotion_strength: float = 1.0
    ) -> AsyncIterator[Dict[str, any]]:
        """
        Process audio stream and get animation data
        
        Args:
            audio_stream: Async iterator of audio bytes
            sample_rate: Audio sample rate (default 16kHz)
            encoding: Audio encoding format
            language: Language code for lip sync
            emotion_strength: Strength of emotional expression (0-1)
            
        Yields:
            Animation frame data with blendshapes and facial pose
        """
        if not self.stub:
            await self.connect()
        
        async def request_generator():
            """Generate request stream from audio"""
            sequence_number = 0
            
            # Model config
            model_config = audio2face_pb2.ModelConfig(
                avatar_id=self.model_id,
                high_quality=True,
                fps=30.0,
                enable_tongue=True,
                expression_scale=emotion_strength
            )
            
            # Audio config
            audio_config = audio2face_pb2.AudioConfig(
                sample_rate=sample_rate,
                channels=1,
                format=encoding,
                language_code=language
            )
            
            # For NVIDIA's protocol, use AudioData messages for streaming
            async for audio_chunk in audio_stream:
                audio_data = audio2face_pb2.AudioData(
                    data=audio_chunk,
                    sequence_number=sequence_number,
                    is_last_chunk=False
                )
                sequence_number += 1
                yield audio_data
        
        try:
            # Use the NVIDIA Audio2Face streaming method
            stream = self.stub.GetFacialAnimationStream(
                request_generator(),
                metadata=self.metadata
            )
            
            async for response in stream:
                # NVIDIA AnimationData message structure
                animation_data = {
                    "timestamp": response.timestamp,
                    "blendshapes": dict(response.blendshapes),  # It's already a map
                    "bones": [],
                    "sequence_number": response.sequence_number
                }
                
                # Add bone transforms if available
                for bone in response.bones:
                    animation_data["bones"].append({
                        "name": bone.name,
                        "translation": {"x": bone.translation.x, "y": bone.translation.y, "z": bone.translation.z},
                        "rotation": {"x": bone.rotation.x, "y": bone.rotation.y, "z": bone.rotation.z, "w": bone.rotation.w},
                        "scale": {"x": bone.scale.x, "y": bone.scale.y, "z": bone.scale.z}
                    })
                
                # Add audio-visual sync data if available
                if hasattr(response, 'av_sync') and response.av_sync:
                    animation_data["av_sync"] = {
                        "phoneme": response.av_sync.phoneme,
                        "viseme_weight": response.av_sync.viseme_weight,
                        "audio_level": response.av_sync.audio_level
                    }
                
                yield animation_data
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error in audio processing: {e}")
            raise
    
    async def process_audio_data(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        encoding: str = "PCM_16",
        language: str = "en-US",
        emotion_strength: float = 1.0
    ) -> List[Dict[str, any]]:
        """
        Process a complete audio buffer and return all animation frames
        
        Args:
            audio_data: Complete audio data as bytes
            sample_rate: Audio sample rate
            encoding: Audio encoding format
            language: Language code
            emotion_strength: Emotion expression strength
            
        Returns:
            List of animation frames
        """
        if not self.stub:
            await self.connect()
        
        try:
            # Model config
            model_config = audio2face_pb2.ModelConfig(
                avatar_id=self.model_id,
                high_quality=True,
                fps=30.0,
                enable_tongue=True,
                expression_scale=emotion_strength
            )
            
            # Audio config
            audio_config = audio2face_pb2.AudioConfig(
                sample_rate=sample_rate,
                channels=1,
                format=encoding,
                language_code=language
            )
            
            # Create request
            request = audio2face_pb2.AudioRequest(
                audio_data=audio_data,
                config=model_config,
                audio_config=audio_config
            )
            
            # Call the non-streaming RPC
            response = await self.stub.GetFacialAnimation(
                request,
                metadata=self.metadata
            )
            
            animation_frames = []
            
            # Check response status
            if response.status.code == audio2face_pb2.StatusCode.OK:
                animation_data = response.animation
                
                # Parse the animation data into frames
                # For now, return a single frame since it's not streaming
                frame = {
                    "timestamp": animation_data.timestamp,
                    "blendshapes": dict(animation_data.blendshapes),
                    "bones": [],
                    "sequence_number": animation_data.sequence_number
                }
                
                # Add bone transforms
                for bone in animation_data.bones:
                    frame["bones"].append({
                        "name": bone.name,
                        "translation": {"x": bone.translation.x, "y": bone.translation.y, "z": bone.translation.z},
                        "rotation": {"x": bone.rotation.x, "y": bone.rotation.y, "z": bone.rotation.z, "w": bone.rotation.w},
                        "scale": {"x": bone.scale.x, "y": bone.scale.y, "z": bone.scale.z}
                    })
                
                animation_frames.append(frame)
            else:
                logger.error(f"Animation error: {response.status.message}")
                if response.error_message:
                    logger.error(f"Details: {response.error_message}")
            
            return animation_frames
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error processing audio: {e}")
            raise


# Example usage function
async def example_usage():
    """Example of how to use the NVIDIA Audio2Face gRPC client"""
    
    # Initialize client
    client = NvidiaAudio2FaceGrpcClient(
        api_key=os.getenv("NVIDIA_API_KEY"),
        model_name="james"
    )
    
    try:
        # Connect to service
        await client.connect()
        
        # Get available models
        models = await client.get_available_models()
        print(f"Available models: {models}")
        
        # Initialize model
        success = await client.initialize_model(
            enable_tongue=True,
            animation_fps=30.0,
            quality="high"
        )
        
        if success:
            # Process some audio (example with dummy data)
            # In production, this would be real audio from TTS
            dummy_audio = b"\x00" * 16000  # 1 second of silence
            
            frames = await client.process_audio_data(
                dummy_audio,
                sample_rate=16000,
                encoding="PCM_16",
                language="en-US"
            )
            
            print(f"Generated {len(frames)} animation frames")
            
            # Print first frame data
            if frames:
                first_frame = frames[0]
                print(f"First frame timestamp: {first_frame['timestamp']}")
                print(f"Blendshapes: {list(first_frame['blendshapes'].keys())[:5]}...")
                print(f"Facial pose: {first_frame['facial_pose']}")
                print(f"Emotion: {first_frame['emotion']}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())