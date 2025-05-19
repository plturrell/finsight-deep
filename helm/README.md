# Helm Charts

This directory contains Helm charts for deploying AIQToolkit on Kubernetes.

## Structure

- `aiqtoolkit/` - Main AIQToolkit Helm chart
  - `Chart.yaml` - Chart metadata
  - `values.yaml` - Default configuration values

## Usage

Install AIQToolkit using Helm:

```bash
helm install aiqtoolkit ./helm/aiqtoolkit
```

Customize values:

```bash
helm install aiqtoolkit ./helm/aiqtoolkit -f custom-values.yaml
```

## Configuration

The `values.yaml` file contains configurable parameters for:
- Image versions
- Resource limits
- Service configurations
- Persistence settings
- Security options

## Prerequisites

- Kubernetes cluster
- Helm 3.x installed
- Proper RBAC permissions