# Docker Integration Scripts

This directory contains scripts for Docker integration with the AIQToolkit, particularly focusing on NVIDIA API integration.

## Security Notice
All scripts have been updated to use environment variables for sensitive information. Before running these scripts:

1. Set your NVIDIA API key in your environment:
```bash
export NIM_API_KEY='your-nvidia-api-key'
```

2. Or add it to your `.env.production` file:
```
NIM_API_KEY=your-nvidia-api-key
```

## Available Scripts

### `docker_nvidia_real.sh`
Tests real Docker integration with NVIDIA API. Creates a test container that makes API calls to verify connectivity.

### `docker-nvidia-real-test.sh`
Similar to above but with a more compact test implementation.

### `simple_nvidia_docker.sh`
A minimal Docker + NVIDIA API test script.

## Usage

From the repository root:
```bash
# Set your API key
export NIM_API_KEY='your-nvidia-api-key'

# Run the test script
./scripts/docker/docker_nvidia_real.sh
```
