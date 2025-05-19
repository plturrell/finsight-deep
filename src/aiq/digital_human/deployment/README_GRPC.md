# FinSight Deep - Production gRPC Implementation

This is the production-ready implementation of FinSight Deep using NVIDIA's official gRPC Audio2Face-3D service.

## Architecture

- **nvidia_grpc_client.py**: Real gRPC client that connects to NVIDIA's Audio2Face-3D service
- **finsight_grpc_production.py**: FastAPI application with WebSocket integration
- **protos/audio2face.proto**: Protocol buffer definitions for NVIDIA Audio2Face
- **audio2face_proto_compiler.py**: Script to compile the proto files

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export NVIDIA_API_KEY="your_nvidia_api_key"
export TOGETHER_API_KEY="your_together_api_key"
```

3. Compile proto files:
```bash
python audio2face_proto_compiler.py
```

4. Run the application:
```bash
python finsight_grpc_production.py
```

## Features

- Real gRPC connection to NVIDIA Audio2Face-3D service
- Support for James, Claire, and Mark photorealistic models
- Bidirectional streaming for real-time audio processing
- Blendshape animation data for facial expressions
- Emotion detection and facial pose tracking
- WebSocket interface for real-time chat

## NVIDIA Models

The following photorealistic models are available:

- **James**: Male avatar (ID: 9327c39f-a361-4e02-bd72-e11b4c9b7b5e)
- **Claire**: Female avatar (ID: 0961a6da-fb9e-4f2e-8491-247e5fd7bf8d)
- **Mark**: Male avatar (ID: 8efc55f5-6f00-424e-afe9-26212cd2c630)

All models support:
- High-quality facial animation
- Tongue animation
- Emotion expression
- Real-time lip sync
- Eye gaze tracking

## gRPC Protocol

The client implements the official NVIDIA Audio2Face gRPC protocol:

```protobuf
service Audio2FaceService {
  rpc ProcessAudioStream(stream AudioRequest) returns (stream AnimationResponse);
  rpc GetAvailableModels(Empty) returns (ModelList);
  rpc InitializeModel(ModelRequest) returns (ModelResponse);
}
```

## Production Deployment

For production deployment:

1. Use proper TLS certificates
2. Configure firewall rules for gRPC port
3. Set up monitoring and logging
4. Implement proper error handling
5. Use connection pooling for multiple clients

## Security

- All connections use TLS encryption
- API keys are transmitted via secure metadata
- No sensitive data is logged
- Rate limiting should be implemented

## Performance

- Supports 30 FPS animation output
- Low-latency bidirectional streaming
- Efficient protocol buffer serialization
- Connection pooling for scalability