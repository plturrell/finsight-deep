# Digital Human Deployment Guide

## Overview

This guide covers the deployment of the AIQToolkit Digital Human system in production environments, including cloud platforms, on-premise installations, and hybrid architectures.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA A100 (recommended) or RTX 4090 minimum
- **CPU**: 16+ cores recommended
- **RAM**: 64GB minimum, 128GB recommended
- **Storage**: 500GB SSD minimum for models and cache
- **Network**: 10Gbps for optimal streaming performance

### Software Requirements

- Docker 24.0+
- Kubernetes 1.28+ (for cluster deployment)
- NVIDIA Driver 535+
- CUDA 12.1+
- Python 3.10+

## Quick Start Deployment

### 1. Single Instance Deployment

```bash
# Clone repository
git clone https://github.com/aiq/aiqtoolkit.git
cd aiqtoolkit

# Run deployment script
./scripts/deploy_digital_human.sh

# Verify deployment
curl http://localhost:8000/health
```

### 2. Docker Compose Deployment

```yaml
# docker/docker-compose.digital_human.yml

version: '3.9'

services:
  digital-human:
    image: aiqtoolkit/digital-human:latest
    runtime: nvidia
    ports:
      - "8000:8000"
      - "8080:8080"
      - "50051:50051"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NAME=llama3:70b
      - AUDIO2FACE_ENABLED=true
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  audio2face:
    image: nvidia/audio2face:3.0
    runtime: nvidia
    ports:
      - "50051:50051"
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  redis-data:
```

```bash
# Start services
docker-compose -f docker/docker-compose.digital_human.yml up -d

# Check logs
docker-compose logs -f digital-human
```

## Cloud Deployment

### AWS EC2 Deployment

```bash
# scripts/deploy_aws_ec2.sh

#!/bin/bash

# Create EC2 instance with GPU
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p4d.24xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --user-data file://user_data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=digital-human-prod}]'

# User data script (user_data.sh)
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
usermod -a -G docker ec2-user

# Install NVIDIA drivers
yum install -y nvidia-driver-latest-dkms
yum install -y cuda-toolkit-12-1

# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-container-toolkit
systemctl restart docker

# Clone and start application
git clone https://github.com/aiq/aiqtoolkit.git
cd aiqtoolkit
docker-compose -f docker/docker-compose.digital_human.yml up -d
```

### Google Cloud Platform Deployment

```yaml
# kubernetes/gcp-deployment.yaml

apiVersion: v1
kind: Namespace
metadata:
  name: digital-human
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human
  namespace: digital-human
spec:
  replicas: 3
  selector:
    matchLabels:
      app: digital-human
  template:
    metadata:
      labels:
        app: digital-human
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
      containers:
      - name: main
        image: gcr.io/project-id/digital-human:latest
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
        - containerPort: 8080
        env:
        - name: PROJECT_ID
          value: "my-project-id"
        - name: AUDIO2FACE_SERVER
          value: "audio2face-service:50051"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: digital-human-service
  namespace: digital-human
spec:
  type: LoadBalancer
  selector:
    app: digital-human
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: websocket
    port: 8080
    targetPort: 8080
```

```bash
# Deploy to GKE
gcloud container clusters get-credentials digital-human-cluster --zone us-central1-a
kubectl apply -f kubernetes/gcp-deployment.yaml

# Check deployment
kubectl get pods -n digital-human
kubectl get services -n digital-human
```

### Azure Deployment

```json
// azure/arm-template.json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmName": {
      "type": "string",
      "defaultValue": "digital-human-vm"
    },
    "adminUsername": {
      "type": "string"
    },
    "adminPassword": {
      "type": "securestring"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Compute/virtualMachines",
      "apiVersion": "2021-07-01",
      "name": "[parameters('vmName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "hardwareProfile": {
          "vmSize": "Standard_NC24ads_A100_v4"
        },
        "osProfile": {
          "computerName": "[parameters('vmName')]",
          "adminUsername": "[parameters('adminUsername')]",
          "adminPassword": "[parameters('adminPassword')]"
        },
        "storageProfile": {
          "imageReference": {
            "publisher": "Canonical",
            "offer": "0001-com-ubuntu-server-focal",
            "sku": "20_04-lts-gen2",
            "version": "latest"
          }
        },
        "networkProfile": {
          "networkInterfaces": [
            {
              "id": "[resourceId('Microsoft.Network/networkInterfaces', 'digital-human-nic')]"
            }
          ]
        }
      }
    }
  ]
}
```

## Production Configuration

### High Availability Setup

```yaml
# kubernetes/ha-deployment.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: digital-human-config
data:
  config.yaml: |
    digital_human:
      ha:
        enabled: true
        min_replicas: 3
        max_replicas: 10
        target_cpu_utilization: 70
        target_gpu_utilization: 80
      
      load_balancing:
        algorithm: "least_connections"
        health_check_interval: 10s
        session_affinity: true
        
      failover:
        enabled: true
        timeout: 30s
        max_retries: 3
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: digital-human-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: digital-human
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: digital-human-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: digital-human
```

### Security Configuration

```yaml
# kubernetes/security-config.yaml

apiVersion: v1
kind: Secret
metadata:
  name: digital-human-secrets
type: Opaque
data:
  api-key: <base64-encoded-api-key>
  jwt-secret: <base64-encoded-jwt-secret>
  ssl-cert: <base64-encoded-ssl-cert>
  ssl-key: <base64-encoded-ssl-key>
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: digital-human-network-policy
spec:
  podSelector:
    matchLabels:
      app: digital-human
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: backend-services
    ports:
    - protocol: TCP
      port: 50051
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: digital-human-sa
  namespace: digital-human
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: digital-human-role
  namespace: digital-human
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: digital-human-rolebinding
  namespace: digital-human
subjects:
- kind: ServiceAccount
  name: digital-human-sa
  namespace: digital-human
roleRef:
  kind: Role
  name: digital-human-role
  apiGroup: rbac.authorization.k8s.io
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus-config.yaml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'digital-human'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - digital-human
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: digital-human
    metric_relabel_configs:
    - source_labels: [__name__]
      regex: 'digital_human_.*'
      action: keep
```

### Grafana Dashboard

```json
// monitoring/grafana-dashboard.json
{
  "dashboard": {
    "title": "Digital Human Monitoring",
    "panels": [
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(digital_human_response_time_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "avg(digital_human_gpu_utilization)"
          }
        ]
      },
      {
        "title": "Active Sessions",
        "targets": [
          {
            "expr": "sum(digital_human_active_sessions)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(digital_human_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Backup Strategy

```bash
#!/bin/bash
# scripts/backup_digital_human.sh

# Backup configuration
BACKUP_DIR="/backup/digital-human"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup models
kubectl cp digital-human/digital-human-0:/app/models "$BACKUP_DIR/$DATE/models"

# Backup configurations
kubectl cp digital-human/digital-human-0:/app/configs "$BACKUP_DIR/$DATE/configs"

# Backup Redis data
kubectl exec -n digital-human redis-0 -- redis-cli BGSAVE
kubectl cp digital-human/redis-0:/data/dump.rdb "$BACKUP_DIR/$DATE/redis-dump.rdb"

# Backup Kubernetes configurations
kubectl get all -n digital-human -o yaml > "$BACKUP_DIR/$DATE/k8s-resources.yaml"

# Create tarball
tar -czf "$BACKUP_DIR/digital-human-backup-$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/digital-human-backup-$DATE.tar.gz" s3://backups/digital-human/

# Clean up old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

```yaml
# kubernetes/disaster-recovery.yaml

apiVersion: batch/v1
kind: Job
metadata:
  name: restore-digital-human
spec:
  template:
    spec:
      containers:
      - name: restore
        image: aiqtoolkit/restore-tools:latest
        command: ["/bin/bash", "-c"]
        args:
          - |
            # Download latest backup
            aws s3 cp s3://backups/digital-human/latest.tar.gz /tmp/backup.tar.gz
            
            # Extract backup
            tar -xzf /tmp/backup.tar.gz -C /tmp/
            
            # Restore models
            kubectl cp /tmp/backup/models digital-human/digital-human-0:/app/
            
            # Restore configs
            kubectl cp /tmp/backup/configs digital-human/digital-human-0:/app/
            
            # Restore Redis data
            kubectl cp /tmp/backup/redis-dump.rdb digital-human/redis-0:/data/
            kubectl exec -n digital-human redis-0 -- redis-cli SHUTDOWN NOSAVE
            kubectl exec -n digital-human redis-0 -- redis-server --appendonly no
            
            # Restart pods
            kubectl rollout restart deployment/digital-human -n digital-human
      restartPolicy: OnFailure
```

## Performance Tuning

### GPU Optimization

```yaml
# configs/gpu-optimization.yaml

gpu_optimization:
  cuda_settings:
    memory_fraction: 0.85
    allow_growth: false
    force_gpu_compatible: true
    
  tensorrt:
    enabled: true
    precision: "fp16"
    workspace_size: 8192  # MB
    
  model_optimization:
    quantization:
      enabled: true
      method: "int8"
      calibration_samples: 1000
      
    pruning:
      enabled: false
      sparsity: 0.5
      
    compilation:
      backend: "torch.compile"
      mode: "max-performance"
      dynamic: false
```

### Network Optimization

```nginx
# nginx/digital-human.conf

upstream digital_human_backend {
    least_conn;
    server digital-human-1:8000 max_fails=3 fail_timeout=30s;
    server digital-human-2:8000 max_fails=3 fail_timeout=30s;
    server digital-human-3:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name digital-human.example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # Enable OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # WebSocket configuration
    location /ws {
        proxy_pass http://digital_human_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
    
    # API endpoints
    location /api {
        proxy_pass http://digital_human_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Enable compression
        gzip on;
        gzip_types application/json;
        
        # Caching
        proxy_cache_bypass $http_pragma;
        proxy_cache_valid 200 1m;
    }
}
```

## Health Checks

### Kubernetes Probes

```yaml
# kubernetes/health-checks.yaml

apiVersion: v1
kind: Service
metadata:
  name: digital-human-health
spec:
  selector:
    app: digital-human
  ports:
  - name: health
    port: 9090
    targetPort: 9090
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human
spec:
  template:
    spec:
      containers:
      - name: main
        livenessProbe:
          httpGet:
            path: /health/live
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 9090
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 2
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health/startup
            port: 9090
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 30
```

### Health Check Implementation

```python
# src/aiq/digital_human/health/health_checks.py

from fastapi import FastAPI, Response
from typing import Dict, Any
import asyncio

app = FastAPI()

@app.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """Basic liveness check"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Comprehensive readiness check"""
    
    checks = {
        "database": await check_database(),
        "gpu": await check_gpu(),
        "model": await check_model_loaded(),
        "audio2face": await check_audio2face_connection()
    }
    
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }

@app.get("/health/startup")
async def startup_check() -> Dict[str, Any]:
    """Startup check for model loading"""
    
    startup_status = {
        "models_loaded": await check_models_loaded(),
        "gpu_initialized": await check_gpu_initialized(),
        "services_connected": await check_service_connections()
    }
    
    all_started = all(startup_status.values())
    
    return {
        "status": "started" if all_started else "starting",
        "startup": startup_status
    }

async def check_gpu() -> bool:
    """Check GPU availability"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

async def check_audio2face_connection() -> bool:
    """Check Audio2Face service connection"""
    try:
        # Test gRPC connection
        channel = grpc.insecure_channel('audio2face:50051')
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
        response = await stub.HealthCheck(
            audio2face_pb2.HealthRequest()
        )
        return response.status == "healthy"
    except:
        return False
```

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   ```bash
   # Clear GPU memory
   nvidia-smi --gpu-reset
   
   # Adjust batch size
   kubectl set env deployment/digital-human BATCH_SIZE=1
   ```

2. **Audio2Face Connection Issues**
   ```bash
   # Check service status
   kubectl logs -n digital-human audio2face-0
   
   # Test connection
   grpcurl -plaintext audio2face:50051 list
   ```

3. **High Latency**
   ```bash
   # Check network latency
   kubectl exec -it digital-human-0 -- ping audio2face
   
   # Profile application
   kubectl exec -it digital-human-0 -- python -m cProfile app.py
   ```

### Debug Commands

```bash
# Check pod status
kubectl describe pod digital-human-0 -n digital-human

# View logs
kubectl logs -f digital-human-0 -n digital-human

# Check GPU usage
kubectl exec -it digital-human-0 -- nvidia-smi

# Test endpoints
curl -X POST http://digital-human:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Monitor metrics
kubectl port-forward -n digital-human svc/prometheus 9090:9090
```

## Maintenance

### Update Procedure

```bash
#!/bin/bash
# scripts/update_digital_human.sh

# Blue-green deployment
kubectl apply -f kubernetes/digital-human-blue.yaml

# Wait for new version to be ready
kubectl wait --for=condition=ready pod -l version=blue -n digital-human

# Switch traffic
kubectl patch service digital-human-service \
  -p '{"spec":{"selector":{"version":"blue"}}}'

# Verify new version
curl http://digital-human:8000/version

# Remove old version
kubectl delete deployment digital-human-green -n digital-human
```

### Performance Testing

```python
# tests/load_test.py

import asyncio
import aiohttp
import time

async def load_test(url: str, num_requests: int = 1000):
    """Load test digital human endpoint"""
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.time()
        
        for i in range(num_requests):
            task = session.post(
                f"{url}/api/chat",
                json={"message": f"Test message {i}"}
            )
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful = sum(1 for r in responses if r.status == 200)
        
        print(f"Completed {num_requests} requests in {duration:.2f}s")
        print(f"Success rate: {successful/num_requests*100:.1f}%")
        print(f"Requests per second: {num_requests/duration:.2f}")

if __name__ == "__main__":
    asyncio.run(load_test("http://localhost:8000"))
```

## Cost Optimization

### GPU Instance Selection

```python
# scripts/cost_optimizer.py

def calculate_optimal_instance(workload_profile):
    """Calculate most cost-effective GPU instance"""
    
    instances = {
        "p4d.24xlarge": {"gpus": 8, "cost": 32.77, "memory": 320},
        "p3.8xlarge": {"gpus": 4, "cost": 12.24, "memory": 244},
        "g4dn.12xlarge": {"gpus": 4, "cost": 3.912, "memory": 192}
    }
    
    required_gpus = workload_profile["gpu_count"]
    required_memory = workload_profile["memory_gb"]
    
    valid_instances = [
        (name, specs) for name, specs in instances.items()
        if specs["gpus"] >= required_gpus and 
           specs["memory"] >= required_memory
    ]
    
    # Sort by cost efficiency
    valid_instances.sort(key=lambda x: x[1]["cost"] / x[1]["gpus"])
    
    return valid_instances[0] if valid_instances else None
```

### Spot Instance Configuration

```yaml
# kubernetes/spot-instances.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: spot-instance-config
data:
  ec2-spot-config: |
    {
      "SpotPrice": "1.50",
      "TargetCapacity": 3,
      "IamFleetRole": "arn:aws:iam::123456789012:role/fleet-role",
      "LaunchSpecifications": [
        {
          "ImageId": "ami-12345678",
          "InstanceType": "p3.8xlarge",
          "KeyName": "my-key",
          "SecurityGroups": [{"GroupId": "sg-12345678"}],
          "UserData": "base64-encoded-startup-script"
        }
      ],
      "AllocationStrategy": "lowestPrice",
      "InstanceInterruptionBehavior": "terminate"
    }
```

## Compliance and Security

### Data Privacy

```python
# src/aiq/digital_human/security/data_privacy.py

class DataPrivacyManager:
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.encryptor = DataEncryptor()
        
    def anonymize_user_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive user data"""
        
        anonymized = data.copy()
        
        # Remove PII
        pii_fields = ["name", "email", "phone", "ssn"]
        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = self.hash_field(anonymized[field])
                
        # Encrypt sensitive data
        if self.config.encrypt_conversations:
            anonymized["conversation"] = self.encryptor.encrypt(
                data.get("conversation", "")
            )
            
        return anonymized
```

### Audit Logging

```python
# src/aiq/digital_human/security/audit_logger.py

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("audit")
        
    async def log_interaction(
        self,
        user_id: str,
        action: str,
        details: Dict[str, Any]
    ):
        """Log user interaction for audit trail"""
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "ip_address": details.get("ip_address"),
            "session_id": details.get("session_id"),
            "result": details.get("result"),
            "duration_ms": details.get("duration_ms")
        }
        
        # Log to secure audit trail
        self.logger.info(json.dumps(audit_entry))
        
        # Store in compliance database
        await self.store_audit_entry(audit_entry)
```

## Next Steps

- Review [Examples](examples.md) for implementation patterns
- Check [Technical Guide](technical-guide.md) for detailed architecture
- See [Performance Guide](../performance/index.md) for optimization