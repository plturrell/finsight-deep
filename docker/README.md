# AIQToolkit Docker Setup

This directory contains Docker configurations for running AIQToolkit in containerized environments.

## Available Configurations

### Dockerfiles

This directory now contains all Docker configurations:

- `Dockerfile` - Base Dockerfile
- `Dockerfile.cpu` - CPU-only container
- `Dockerfile.gpu` - GPU-enabled container (recommended)
- `Dockerfile.digital_human` - Digital Human UI configuration
- `Dockerfile.digital_human_production` - Production Digital Human setup
- `Dockerfile.distributed_manager` - Distributed processing manager node
- `Dockerfile.distributed_worker` - Distributed processing worker node
- `Dockerfile.manager-nvidia` - NVIDIA-specific manager configuration
- `Dockerfile.worker-nvidia` - NVIDIA-specific worker configuration
- `Dockerfile.nvidia-test` - NVIDIA testing configuration
- `Dockerfile.simple` - Minimal configuration for simple deployments

### GPU-enabled Container (Recommended)

The GPU-enabled container includes all CUDA libraries and GPU acceleration support:

```bash
# Build the GPU container
docker build -f docker/Dockerfile.gpu -t aiqtoolkit:gpu .

# Run with GPU support
docker run --gpus all -it --rm \
    -v $(pwd):/app \
    -p 8000:8000 \
    -p 8080:8080 \
    aiqtoolkit:gpu
```

### CPU-only Container

For environments without GPU access:

```bash
# Build the CPU container
docker build -f docker/Dockerfile.cpu -t aiqtoolkit:cpu .

# Run CPU version
docker run -it --rm \
    -v $(pwd):/app \
    -p 8000:8000 \
    -p 8080:8080 \
    aiqtoolkit:cpu
```

## Docker Compose

Available docker-compose configurations:

- `docker-compose.yml` - Default development setup
- `docker-compose.production.yml` - Production deployment
- `docker-compose.digital_human.yml` - Digital Human UI setup
- `docker-compose.hackathon.yml` - Hackathon configuration
- `docker-compose.nvidia-real.yml` - NVIDIA production setup
- `docker-compose-nvidia-test.yml` - NVIDIA testing setup

For a complete development environment with all services:

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# Start specific service
docker-compose -f docker/docker-compose.yml up aiqtoolkit

# Run in background
docker-compose -f docker/docker-compose.yml up -d
```

### Available Services

- **aiqtoolkit**: Main AIQToolkit container with GPU support
- **milvus**: Vector database for retrieval
- **phoenix**: Observability platform
- **jupyter**: Jupyter Lab for development

## Environment Variables

- `NVIDIA_VISIBLE_DEVICES`: GPU devices to use (default: all)
- `CUDA_VISIBLE_DEVICES`: Specific GPU index (default: 0)
- `DISPLAY`: For GUI applications (requires X11 forwarding)

## Volume Mounts

- `/app/src`: Source code
- `/app/examples`: Example workflows
- `/app/tests`: Test suite
- `/data`: Persistent data storage

## Ports

- `8000`: FastAPI server
- `8080`: Web UI
- `6006`: TensorBoard
- `6007`: Phoenix observability
- `8888`: Jupyter Lab
- `19530`: Milvus vector database

## GPU Support

### Requirements

1. NVIDIA Docker runtime:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   ```

2. Docker configuration:
   ```json
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### Testing GPU Access

```bash
# Inside container
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

## Development Workflow

1. Start the development environment:
   ```bash
   docker-compose up -d
   ```

2. Access the main container:
   ```bash
   docker exec -it aiqtoolkit bash
   ```

3. Run AIQToolkit commands:
   ```bash
   # Inside container
   aiq --help
   aiq run --config_file examples/simple/configs/config.yml
   aiq start  # Launch UI
   ```

4. Access services:
   - Web UI: http://localhost:8080
   - API: http://localhost:8000
   - Jupyter: http://localhost:8888

## Building Custom Images

To add custom dependencies:

1. Create a custom Dockerfile:
   ```dockerfile
   FROM aiqtoolkit:gpu
   
   # Add your dependencies
   RUN pip install custom-package
   ```

2. Build your image:
   ```bash
   docker build -f Dockerfile.custom -t aiqtoolkit:custom .
   ```

## Troubleshooting

### GPU Not Detected

1. Check NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

2. Verify Docker runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### Permission Issues

Add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Memory Issues

Increase Docker memory limits in Docker Desktop settings or daemon.json.

## Security Considerations

- Don't run containers as root in production
- Use secrets management for API keys
- Limit container capabilities
- Network isolation for production deployments

## Support

For issues with Docker setup, please file an issue at:
https://github.com/NVIDIA/AIQToolkit/issues