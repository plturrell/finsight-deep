"""NVIDIA Audio2Face-3D Integration for Digital Human"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import requests
from pathlib import Path

import torch
import torchaudio
from PIL import Image
import cv2

from aiq.digital_human.avatar.base import AvatarRenderer
from aiq.llm.nim_llm import NVIDIAInferenceMicroservice


@dataclass
class Audio2FaceConfig:
    """Configuration for Audio2Face integration"""
    api_key: str
    endpoint: str = "https://api.nvidia.com/audio2face/v1"
    model_version: str = "3d"
    fps: int = 30
    audio_sample_rate: int = 16000
    enable_emotions: bool = True
    blend_shape_mapping: Dict[str, str] = None
    cache_animations: bool = True
    cache_dir: str = ".cache/audio2face"
    timeout: int = 30


@dataclass
class FacialAnimation:
    """Facial animation data"""
    timestamp: float
    blend_shapes: Dict[str, float]
    head_rotation: Tuple[float, float, float]
    eye_gaze: Tuple[float, float]
    emotion: str
    intensity: float


@dataclass
class LipSyncData:
    """Lip sync data for audio"""
    phonemes: List[Tuple[str, float, float]]  # phoneme, start_time, end_time
    visemes: List[Tuple[str, float, float]]   # viseme, start_time, duration
    blend_shapes: List[FacialAnimation]
    audio_duration: float


class Audio2FaceIntegration:
    """NVIDIA Audio2Face-3D integration for realistic facial animation"""
    
    def __init__(self, config: Audio2FaceConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        })
        
        # Initialize cache
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Blend shape mapping
        self.blend_shapes = config.blend_shape_mapping or self._get_default_blend_shapes()
        
    def _get_default_blend_shapes(self) -> Dict[str, str]:
        """Get default blend shape mapping for common 3D avatar systems"""
        return {
            # Mouth shapes
            "jaw_open": "mouthOpen",
            "mouth_wide": "mouthWide",
            "mouth_smile_left": "mouthSmileLeft",
            "mouth_smile_right": "mouthSmileRight",
            "mouth_frown_left": "mouthFrownLeft",
            "mouth_frown_right": "mouthFrownRight",
            "mouth_pucker": "mouthPucker",
            "mouth_funnel": "mouthFunnel",
            "mouth_left": "mouthLeft",
            "mouth_right": "mouthRight",
            
            # Lip shapes
            "lip_upper_up": "lipUpperUp",
            "lip_lower_down": "lipLowerDown",
            "lip_corner_pull_left": "lipCornerPullLeft",
            "lip_corner_pull_right": "lipCornerPullRight",
            
            # Eye shapes
            "eye_blink_left": "eyeBlinkLeft",
            "eye_blink_right": "eyeBlinkRight",
            "eye_wide_left": "eyeWideLeft",
            "eye_wide_right": "eyeWideRight",
            "eye_squint_left": "eyeSquintLeft",
            "eye_squint_right": "eyeSquintRight",
            
            # Eyebrow shapes
            "brow_up_left": "browUpLeft",
            "brow_up_right": "browUpRight",
            "brow_down_left": "browDownLeft",
            "brow_down_right": "browDownRight",
            "brow_inner_up": "browInnerUp",
            
            # Nose shapes
            "nose_flare_left": "noseFlareLeft",
            "nose_flare_right": "noseFlareRight",
            
            # Cheek shapes
            "cheek_puff": "cheekPuff",
            "cheek_squint_left": "cheekSquintLeft",
            "cheek_squint_right": "cheekSquintRight",
            
            # Tongue (if supported)
            "tongue_out": "tongueOut"
        }
    
    async def process_audio(
        self,
        audio_path: Union[str, Path],
        text: Optional[str] = "",
        emotion: Optional[str] = "neutral",
        intensity: float = 1.0
    ) -> LipSyncData:
        """Process audio file with Audio2Face for lip sync animation"""
        
        # Check cache first
        cache_key = self._get_cache_key(audio_path, text, emotion, intensity)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Load and preprocess audio
        audio_data = await self._load_audio(audio_path)
        
        # Prepare request
        request_data = {
            "audio": self._encode_audio(audio_data),
            "text": text,
            "emotion": emotion,
            "intensity": intensity,
            "model_version": self.config.model_version,
            "fps": self.config.fps,
            "enable_emotions": self.config.enable_emotions
        }
        
        # Send request to Audio2Face API
        response = await self._send_request("process", request_data)
        
        # Parse response
        lip_sync_data = self._parse_response(response)
        
        # Cache results
        if self.config.cache_animations:
            self._save_to_cache(cache_key, lip_sync_data)
        
        return lip_sync_data
    
    async def process_realtime_audio(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        emotion: str = "neutral",
        context: Optional[Dict] = None
    ) -> List[FacialAnimation]:
        """Process real-time audio chunk for live animation"""
        
        # Prepare audio chunk
        audio_data = self._prepare_audio_chunk(audio_chunk, sample_rate)
        
        # Prepare request
        request_data = {
            "audio_chunk": self._encode_audio(audio_data),
            "sample_rate": sample_rate,
            "emotion": emotion,
            "context": context or {},
            "model_version": self.config.model_version,
            "streaming": True
        }
        
        # Send request (using streaming endpoint)
        response = await self._send_request("stream", request_data)
        
        # Parse streaming response
        animations = self._parse_streaming_response(response)
        
        return animations
    
    async def generate_emotion_animation(
        self,
        emotion: str,
        intensity: float = 1.0,
        duration: float = 1.0,
        blend_with_current: bool = True
    ) -> List[FacialAnimation]:
        """Generate facial animation for specific emotion"""
        
        request_data = {
            "emotion": emotion,
            "intensity": intensity,
            "duration": duration,
            "blend_with_current": blend_with_current,
            "fps": self.config.fps
        }
        
        response = await self._send_request("emotion", request_data)
        
        return self._parse_animation_response(response)
    
    async def text_to_animation(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: str = "neutral",
        speaking_style: str = "conversational"
    ) -> Tuple[bytes, LipSyncData]:
        """Generate audio and facial animation from text"""
        
        request_data = {
            "text": text,
            "voice_id": voice_id or "default",
            "emotion": emotion,
            "speaking_style": speaking_style,
            "model_version": self.config.model_version,
            "fps": self.config.fps,
            "audio_format": "wav"
        }
        
        response = await self._send_request("text_to_animation", request_data)
        
        # Extract audio and animation data
        audio_data = self._decode_audio(response["audio"])
        lip_sync_data = self._parse_response(response["animation"])
        
        return audio_data, lip_sync_data
    
    async def _load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess audio file"""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Resample if necessary
        if sample_rate != self.config.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, 
                self.config.audio_sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.numpy().flatten()
    
    def _prepare_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare audio chunk for real-time processing"""
        # Resample if necessary
        if sample_rate != self.config.audio_sample_rate:
            import librosa
            audio_chunk = librosa.resample(
                audio_chunk,
                orig_sr=sample_rate,
                target_sr=self.config.audio_sample_rate
            )
        
        # Normalize audio
        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        return audio_chunk
    
    def _encode_audio(self, audio_data: np.ndarray) -> str:
        """Encode audio data to base64"""
        import base64
        import io
        from scipy.io import wavfile
        
        # Convert to WAV format
        buffer = io.BytesIO()
        wavfile.write(buffer, self.config.audio_sample_rate, audio_data)
        
        # Encode to base64
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def _decode_audio(self, audio_base64: str) -> bytes:
        """Decode base64 audio data"""
        import base64
        return base64.b64decode(audio_base64)
    
    async def _send_request(self, endpoint: str, data: Dict) -> Dict:
        """Send request to Audio2Face API"""
        url = f"{self.config.endpoint}/{endpoint}"
        
        try:
            response = self.session.post(
                url,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Audio2Face API error: {e}")
    
    def _parse_response(self, response: Dict) -> LipSyncData:
        """Parse Audio2Face API response"""
        # Extract phoneme data
        phonemes = [
            (p["phoneme"], p["start_time"], p["end_time"])
            for p in response.get("phonemes", [])
        ]
        
        # Extract viseme data
        visemes = [
            (v["viseme"], v["start_time"], v["duration"])
            for v in response.get("visemes", [])
        ]
        
        # Extract blend shape animations
        blend_shapes = []
        for frame in response.get("frames", []):
            animation = FacialAnimation(
                timestamp=frame["timestamp"],
                blend_shapes=frame["blend_shapes"],
                head_rotation=tuple(frame.get("head_rotation", [0, 0, 0])),
                eye_gaze=tuple(frame.get("eye_gaze", [0, 0])),
                emotion=frame.get("emotion", "neutral"),
                intensity=frame.get("intensity", 1.0)
            )
            blend_shapes.append(animation)
        
        return LipSyncData(
            phonemes=phonemes,
            visemes=visemes,
            blend_shapes=blend_shapes,
            audio_duration=response.get("duration", 0.0)
        )
    
    def _parse_streaming_response(self, response: Dict) -> List[FacialAnimation]:
        """Parse streaming response for real-time animation"""
        animations = []
        
        for frame in response.get("frames", []):
            animation = FacialAnimation(
                timestamp=frame["timestamp"],
                blend_shapes=frame["blend_shapes"],
                head_rotation=tuple(frame.get("head_rotation", [0, 0, 0])),
                eye_gaze=tuple(frame.get("eye_gaze", [0, 0])),
                emotion=frame.get("emotion", "neutral"),
                intensity=frame.get("intensity", 1.0)
            )
            animations.append(animation)
        
        return animations
    
    def _parse_animation_response(self, response: Dict) -> List[FacialAnimation]:
        """Parse animation response"""
        return self._parse_streaming_response(response)
    
    def _get_cache_key(
        self,
        audio_path: Union[str, Path],
        text: str,
        emotion: str,
        intensity: float
    ) -> str:
        """Generate cache key for animation data"""
        import hashlib
        
        # Create unique key based on input parameters
        key_data = f"{audio_path}_{text}_{emotion}_{intensity}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[LipSyncData]:
        """Load animation data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with cache_file.open('r') as f:
                    data = json.load(f)
                
                # Reconstruct LipSyncData
                blend_shapes = []
                for frame in data["blend_shapes"]:
                    animation = FacialAnimation(
                        timestamp=frame["timestamp"],
                        blend_shapes=frame["blend_shapes"],
                        head_rotation=tuple(frame["head_rotation"]),
                        eye_gaze=tuple(frame["eye_gaze"]),
                        emotion=frame["emotion"],
                        intensity=frame["intensity"]
                    )
                    blend_shapes.append(animation)
                
                return LipSyncData(
                    phonemes=[tuple(p) for p in data["phonemes"]],
                    visemes=[tuple(v) for v in data["visemes"]],
                    blend_shapes=blend_shapes,
                    audio_duration=data["audio_duration"]
                )
            except Exception:
                return None
        
        return None
    
    def _save_to_cache(self, cache_key: str, lip_sync_data: LipSyncData):
        """Save animation data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Convert to serializable format
        data = {
            "phonemes": lip_sync_data.phonemes,
            "visemes": lip_sync_data.visemes,
            "blend_shapes": [
                {
                    "timestamp": anim.timestamp,
                    "blend_shapes": anim.blend_shapes,
                    "head_rotation": anim.head_rotation,
                    "eye_gaze": anim.eye_gaze,
                    "emotion": anim.emotion,
                    "intensity": anim.intensity
                }
                for anim in lip_sync_data.blend_shapes
            ],
            "audio_duration": lip_sync_data.audio_duration
        }
        
        with cache_file.open('w') as f:
            json.dump(data, f)


class ParakeetASRIntegration:
    """NVIDIA Parakeet CTC ASR integration for speech recognition"""
    
    def __init__(self, api_key: str, endpoint: str = "https://api.nvidia.com/parakeet/v1"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    async def transcribe(
        self,
        audio_path: Union[str, Path, np.ndarray],
        language: str = "en",
        model: str = "ctc-1.1b",
        enable_punctuation: bool = True,
        enable_diarization: bool = False,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Transcribe audio using Parakeet ASR"""
        
        # Prepare audio data
        if isinstance(audio_path, (str, Path)):
            audio_data = await self._load_audio_file(audio_path)
        else:
            audio_data = audio_path
        
        # Prepare request
        request_data = {
            "audio": self._encode_audio(audio_data),
            "language": language,
            "model": model,
            "enable_punctuation": enable_punctuation,
            "enable_diarization": enable_diarization,
            "num_speakers": num_speakers
        }
        
        # Send request
        response = await self._send_request("transcribe", request_data)
        
        return self._parse_transcription(response)
    
    async def stream_transcribe(
        self,
        audio_stream: asyncio.Queue,
        language: str = "en",
        model: str = "ctc-1.1b"
    ) -> asyncio.Queue:
        """Stream transcription for real-time audio"""
        results_queue = asyncio.Queue()
        
        async def process_stream():
            buffer = b""
            
            while True:
                audio_chunk = await audio_stream.get()
                
                if audio_chunk is None:  # End of stream
                    break
                
                buffer += audio_chunk
                
                # Process when buffer reaches threshold
                if len(buffer) >= 16000:  # ~1 second at 16kHz
                    request_data = {
                        "audio_chunk": self._encode_audio(buffer),
                        "language": language,
                        "model": model,
                        "streaming": True
                    }
                    
                    response = await self._send_request("stream_transcribe", request_data)
                    result = self._parse_streaming_transcription(response)
                    
                    await results_queue.put(result)
                    buffer = b""
        
        # Start processing in background
        asyncio.create_task(process_stream())
        
        return results_queue
    
    async def _load_audio_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load audio file"""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to 16kHz mono
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.numpy().flatten()
    
    def _encode_audio(self, audio_data: np.ndarray) -> str:
        """Encode audio to base64"""
        import base64
        return base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    async def _send_request(self, endpoint: str, data: Dict) -> Dict:
        """Send request to Parakeet API"""
        url = f"{self.endpoint}/{endpoint}"
        
        try:
            response = self.session.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Parakeet API error: {e}")
    
    def _parse_transcription(self, response: Dict) -> Dict[str, Any]:
        """Parse transcription response"""
        return {
            "text": response.get("text", ""),
            "confidence": response.get("confidence", 0.0),
            "words": response.get("words", []),
            "speakers": response.get("speakers", []),
            "metadata": response.get("metadata", {})
        }
    
    def _parse_streaming_transcription(self, response: Dict) -> Dict[str, Any]:
        """Parse streaming transcription response"""
        return {
            "partial_text": response.get("partial_text", ""),
            "is_final": response.get("is_final", False),
            "confidence": response.get("confidence", 0.0),
            "timestamp": response.get("timestamp", 0.0)
        }


class DigitalHumanNVIDIAIntegration:
    """Integration layer for NVIDIA Digital Human technologies"""
    
    def __init__(
        self,
        audio2face_config: Audio2FaceConfig,
        llm_config: Dict[str, Any],
        asr_config: Dict[str, Any]
    ):
        # Initialize Audio2Face
        self.audio2face = Audio2FaceIntegration(audio2face_config)
        
        # Initialize Llama3-8B through NIM
        self.llm = NVIDIAInferenceMicroservice(
            api_key=llm_config["api_key"],
            model_name="meta/llama3-8b-instruct",
            endpoint=llm_config.get("endpoint", "https://integrate.api.nvidia.com/v1")
        )
        
        # Initialize Parakeet ASR
        self.asr = ParakeetASRIntegration(
            api_key=asr_config["api_key"],
            endpoint=asr_config.get("endpoint", "https://api.nvidia.com/parakeet/v1")
        )
        
        self.active_sessions = {}
    
    async def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new digital human session"""
        session = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "conversation_history": [],
            "emotion_state": "neutral",
            "animation_queue": asyncio.Queue(),
            "audio_queue": asyncio.Queue()
        }
        
        self.active_sessions[session_id] = session
        
        # Start background processors
        asyncio.create_task(self._process_animation_queue(session_id))
        asyncio.create_task(self._process_audio_queue(session_id))
        
        return session
    
    async def process_user_input(
        self,
        session_id: str,
        audio_input: Optional[Union[str, Path, np.ndarray]] = None,
        text_input: Optional[str] = None,
        emotion_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process user input and generate digital human response"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Transcribe audio if provided
        if audio_input is not None:
            transcription = await self.asr.transcribe(audio_input)
            user_text = transcription["text"]
        else:
            user_text = text_input
        
        # Add to conversation history
        session["conversation_history"].append({
            "role": "user",
            "content": user_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate LLM response
        llm_response = await self.llm.chat_completion_async(
            messages=session["conversation_history"][-10:],  # Last 10 messages
            max_tokens=500,
            temperature=0.7
        )
        
        response_text = llm_response["choices"][0]["message"]["content"]
        
        # Add to conversation history
        session["conversation_history"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Detect emotion from context
        emotion = self._detect_emotion(response_text, emotion_context)
        session["emotion_state"] = emotion
        
        # Generate audio and facial animation
        audio_data, lip_sync_data = await self.audio2face.text_to_animation(
            text=response_text,
            emotion=emotion,
            speaking_style="conversational"
        )
        
        # Queue animation and audio
        await session["animation_queue"].put(lip_sync_data)
        await session["audio_queue"].put(audio_data)
        
        return {
            "session_id": session_id,
            "text": response_text,
            "emotion": emotion,
            "audio": audio_data,
            "animation": lip_sync_data,
            "conversation_history": session["conversation_history"]
        }
    
    async def stream_response(
        self,
        session_id: str,
        user_text: str,
        stream_audio: bool = True,
        stream_animation: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream digital human response in real-time"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Stream LLM response
        async for chunk in self.llm.stream_chat_completion(
            messages=session["conversation_history"][-10:] + [
                {"role": "user", "content": user_text}
            ],
            max_tokens=500,
            temperature=0.7
        ):
            text_chunk = chunk["choices"][0]["delta"].get("content", "")
            
            if text_chunk:
                # Generate audio and animation for chunk
                if stream_audio or stream_animation:
                    audio_chunk, animation_chunk = await self.audio2face.text_to_animation(
                        text=text_chunk,
                        emotion=session["emotion_state"],
                        speaking_style="conversational"
                    )
                    
                    yield {
                        "type": "chunk",
                        "text": text_chunk,
                        "audio": audio_chunk if stream_audio else None,
                        "animation": animation_chunk if stream_animation else None
                    }
                else:
                    yield {
                        "type": "chunk",
                        "text": text_chunk
                    }
        
        # Final response
        yield {
            "type": "complete",
            "session_id": session_id
        }
    
    def _detect_emotion(
        self,
        text: str,
        emotion_context: Optional[Dict] = None
    ) -> str:
        """Detect emotion from text and context"""
        # Simple emotion detection (in production, use a proper emotion model)
        emotions = {
            "happy": ["great", "wonderful", "excellent", "happy", "glad"],
            "sad": ["sorry", "sad", "unfortunate", "regret"],
            "excited": ["exciting", "amazing", "fantastic", "wow"],
            "concerned": ["worried", "concern", "problem", "issue"],
            "confident": ["certainly", "definitely", "absolutely", "sure"]
        }
        
        text_lower = text.lower()
        
        for emotion, keywords in emotions.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        # Use context if available
        if emotion_context and "detected_emotion" in emotion_context:
            return emotion_context["detected_emotion"]
        
        return "neutral"
    
    async def _process_animation_queue(self, session_id: str):
        """Process animation queue in background"""
        session = self.active_sessions[session_id]
        
        while session_id in self.active_sessions:
            try:
                animation_data = await asyncio.wait_for(
                    session["animation_queue"].get(),
                    timeout=1.0
                )
                
                # Send animation data to renderer
                # In production, this would communicate with the 3D renderer
                print(f"Processing animation for session {session_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Animation processing error: {e}")
    
    async def _process_audio_queue(self, session_id: str):
        """Process audio queue in background"""
        session = self.active_sessions[session_id]
        
        while session_id in self.active_sessions:
            try:
                audio_data = await asyncio.wait_for(
                    session["audio_queue"].get(),
                    timeout=1.0
                )
                
                # Play audio
                # In production, this would send to audio output system
                print(f"Playing audio for session {session_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
    
    async def close_session(self, session_id: str):
        """Close a digital human session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            print(f"Closed session {session_id}")
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            "id": session["id"],
            "created_at": session["created_at"],
            "emotion_state": session["emotion_state"],
            "conversation_length": len(session["conversation_history"]),
            "animation_queue_size": session["animation_queue"].qsize(),
            "audio_queue_size": session["audio_queue"].qsize()
        }