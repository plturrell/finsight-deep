"""
NVIDIA Audio2Face-3D Client Implementation
Uses NVIDIA's actual James/Claire/Mark models
"""

import grpc
import asyncio
import wave
import struct
import os
from typing import Optional, Dict, Any, List
import numpy as np
import json

# Import NVIDIA ACE protos (you need to install these from NVIDIA)
# pip install nvidia_ace-1.2.0-py3-none-any.whl
try:
    from nvidia_ace.controller.v1 import aceagent_pb2
    from nvidia_ace.controller.v1 import aceagent_pb2_grpc
    from nvidia_ace.nemo.v1 import nemo_pb2
    from nvidia_ace.a2f.v1 import a2e_pb2
    from nvidia_ace.a2f.v1 import a2e_pb2_grpc
    from nvidia_ace.audio.v1 import audio_pb2
except ImportError:
    print("Warning: NVIDIA ACE protos not installed. Please install nvidia_ace package.")
    # Fallback to basic types
    class aceagent_pb2:
        pass
    class a2e_pb2:
        pass

class Audio2Face3DClient:
    """Client for NVIDIA Audio2Face-3D models (James, Claire, Mark)"""
    
    # Function IDs for different models
    MODELS = {
        "james": "9327c39f-a361-4e02-bd72-e11b4c9b7b5e",  # With tongue animation
        "claire": "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d", 
        "mark": "8efc55f5-6f00-424e-afe9-26212cd2c630"
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "james",
        grpc_endpoint: str = "grpc.nvcf.nvidia.com:443"
    ):
        """Initialize Audio2Face-3D client
        
        Args:
            api_key: NVIDIA API key
            model: One of "james", "claire", "mark"
            grpc_endpoint: NVIDIA gRPC endpoint
        """
        self.api_key = api_key
        self.model = model
        self.function_id = self.MODELS.get(model, self.MODELS["james"])
        self.grpc_endpoint = grpc_endpoint
        
        # Setup gRPC channel with SSL
        self.metadata = [
            ('authorization', f'Bearer {self.api_key}'),
            ('function-id', self.function_id)
        ]
        
        # Create secure channel
        self.channel = grpc.secure_channel(
            self.grpc_endpoint,
            grpc.ssl_channel_credentials()
        )
        
        # Create stubs
        self.controller_stub = aceagent_pb2_grpc.AceAgentStub(self.channel)
        self.a2e_stub = a2e_pb2_grpc.A2EStub(self.channel)
        
    def read_audio_file(self, file_path: str) -> bytes:
        """Read WAV file and return PCM data"""
        with wave.open(file_path, 'rb') as wav_file:
            # Verify it's 16-bit PCM
            if wav_file.getsampwidth() != 2:
                raise ValueError("Audio must be 16-bit PCM")
            
            # Read all frames
            frames = wav_file.readframes(wav_file.getnframes())
            return frames
    
    def create_audio_header(self, sample_rate: int = 16000) -> audio_pb2.AudioHeader:
        """Create audio header for streaming"""
        return audio_pb2.AudioHeader(
            encoding=audio_pb2.AudioEncoding.ENCODING_LINEAR_PCM,
            sample_rate_hz=sample_rate,
            bits_per_sample=16,
            channel_count=1
        )
    
    async def process_audio_stream(
        self,
        audio_data: bytes,
        emotions: Dict[str, float] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process audio through Audio2Face-3D and get animation data
        
        Args:
            audio_data: PCM audio data (16-bit)
            emotions: Emotion parameters (e.g., {"happy": 0.8})
            parameters: Processing parameters
            
        Returns:
            Animation data including blendshapes and emotion results
        """
        emotions = emotions or {}
        parameters = parameters or {}
        
        try:
            # Create bidirectional stream
            stream = self.controller_stub.ProcessAudioStream(metadata=self.metadata)
            
            # Send audio header
            audio_header = self.create_audio_header()
            audio_stream = aceagent_pb2.AudioStream(audio_header=audio_header)
            await stream.write(audio_stream)
            
            # Send parameters
            params = aceagent_pb2.AudioStream(
                input_params=aceagent_pb2.InputParameters(
                    emotion_strength=aceagent_pb2.EmotionStrength(
                        **emotions
                    ),
                    face_params=a2e_pb2.FaceParameters(
                        # Add any face-specific parameters
                    )
                )
            )
            await stream.write(params)
            
            # Send audio data in chunks
            chunk_size = 4096  # 4KB chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                audio_stream = aceagent_pb2.AudioStream(
                    audio_buffer=chunk
                )
                await stream.write(audio_stream)
            
            # Send end of audio
            await stream.write(aceagent_pb2.AudioStream(end_of_audio=True))
            
            # Collect results
            results = {
                "blendshapes": [],
                "emotions": [],
                "audio": None
            }
            
            async for response in stream:
                if response.HasField("audio_header"):
                    # Audio header received
                    pass
                elif response.HasField("audio_buffer"):
                    # Audio data (echo back)
                    if results["audio"] is None:
                        results["audio"] = b""
                    results["audio"] += response.audio_buffer
                elif response.HasField("face_data"):
                    # Animation data
                    face_data = response.face_data
                    
                    # Extract blendshapes
                    for blendshape in face_data.blendshapes:
                        results["blendshapes"].append({
                            "name": blendshape.name,
                            "weight": blendshape.weight,
                            "time_code": blendshape.time_code
                        })
                    
                    # Extract emotions
                    for emotion in face_data.emotions:
                        results["emotions"].append({
                            "name": emotion.name,
                            "weight": emotion.weight,
                            "time_code": emotion.time_code
                        })
            
            await stream.done_writing()
            return results
            
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
            return None
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()


# Example usage
async def main():
    # Initialize client
    client = Audio2Face3DClient(
        api_key="YOUR_NVIDIA_API_KEY",
        model="james"  # or "claire" or "mark"
    )
    
    # Read audio file
    audio_data = client.read_audio_file("example_audio.wav")
    
    # Process with emotions
    results = await client.process_audio_stream(
        audio_data,
        emotions={
            "happy": 0.8,
            "amazement": 0.3
        }
    )
    
    if results:
        # Save blendshapes to file
        with open("blendshapes.json", "w") as f:
            json.dump(results["blendshapes"], f, indent=2)
        
        # Save emotions
        with open("emotions.json", "w") as f:
            json.dump(results["emotions"], f, indent=2)
        
        # Save audio if returned
        if results["audio"]:
            with wave.open("output_audio.wav", "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(results["audio"])
    
    client.close()

if __name__ == "__main__":
    asyncio.run(main())