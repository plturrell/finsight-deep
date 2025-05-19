# Kubernetes Manifests

This directory contains Kubernetes deployment manifests for AIQToolkit components.

## Structure

- `digital-human-deployment.yaml` - Digital Human component deployment
- `distributed/` - Distributed processing components
  - `manager-deployment.yaml` - Manager node deployment
  - `worker-statefulset.yaml` - Worker nodes StatefulSet
- `istio/` - Service mesh configurations
  - `service-mesh.yaml` - Istio service mesh setup

## Components

### Digital Human
Deploys the Digital Human interface with GPU support for facial rendering.

### Distributed Processing
- **Manager**: Coordinates distributed tasks
- **Workers**: Process tasks in parallel using StatefulSet for persistence

### Service Mesh
Istio configuration for:
- Traffic management
- Security policies
- Observability

## Deployment

Deploy all components:
```bash
kubectl apply -f kubernetes/
kubectl apply -f kubernetes/distributed/
kubectl apply -f kubernetes/istio/
```

## Requirements

- Kubernetes 1.20+
- GPU nodes for Digital Human
- Istio (optional) for service mesh