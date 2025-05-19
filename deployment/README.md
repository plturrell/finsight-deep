# Deployment Scripts

This directory contains deployment scripts for AIQToolkit across various platforms.

## Structure

- `gpu_cluster/` - GPU cluster deployment scripts
  - `deploy_production.py` - Production deployment for GPU clusters

## Purpose

These scripts facilitate deployment of AIQToolkit in:
- GPU-enabled environments
- Production clusters
- Cloud platforms
- On-premise installations

## Related Documentation

For detailed deployment guides, see the documentation in `/docs/deployment/`:
- [Deployment Guide](../docs/deployment/DEPLOYMENT_GUIDE.md)
- [NVIDIA Deployment Guide](../docs/deployment/NVIDIA_DEPLOYMENT_GUIDE.md)
- [Deployment Summary](../docs/deployment/DEPLOYMENT_SUMMARY.md)

## Usage

Deploy to production GPU cluster:
```python
python deployment/gpu_cluster/deploy_production.py
```