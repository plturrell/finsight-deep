#!/bin/bash

# Production Deployment Script for Digital Human System
# This script deploys the complete production system with all real services

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
        error "NVIDIA Docker runtime not available or no GPU detected"
    fi
    
    # Check Kubernetes
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
    fi
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        error "Helm is not installed"
    fi
    
    log "✓ All requirements satisfied"
}

# Validate environment variables
validate_env() {
    log "Validating environment variables..."
    
    required_vars=(
        "NVIDIA_API_KEY"
        "NEURAL_SUPERCOMPUTER_ENDPOINT"
        "NEURAL_SUPERCOMPUTER_API_KEY"
        "GOOGLE_API_KEY"
        "GOOGLE_CSE_ID"
        "YAHOO_API_KEY"
        "ALPHA_VANTAGE_API_KEY"
        "POLYGON_API_KEY"
        "QUANDL_API_KEY"
        "JWT_SECRET_KEY"
        "API_KEY_ENCRYPTION_KEY"
        "SSL_CERT_FILE"
        "SSL_KEY_FILE"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            error "Required environment variable $var is not set"
        fi
    done
    
    log "✓ All environment variables validated"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure components..."
    
    # Create namespace
    kubectl create namespace digital-human-prod --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy PostgreSQL
    log "Deploying PostgreSQL..."
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace digital-human-prod \
        --set auth.postgresPassword=$POSTGRES_PASSWORD \
        --set auth.database=digital_human \
        --set persistence.size=100Gi \
        --set metrics.enabled=true \
        --wait
    
    # Deploy Redis
    log "Deploying Redis..."
    helm upgrade --install redis bitnami/redis \
        --namespace digital-human-prod \
        --set auth.password=$REDIS_PASSWORD \
        --set replica.replicaCount=3 \
        --set sentinel.enabled=true \
        --set metrics.enabled=true \
        --wait
    
    # Deploy Milvus
    log "Deploying Milvus..."
    helm upgrade --install milvus milvus/milvus \
        --namespace digital-human-prod \
        --set cluster.enabled=true \
        --set minio.persistence.size=100Gi \
        --set etcd.replicaCount=3 \
        --set pulsar.enabled=true \
        --set metrics.enabled=true \
        --wait
    
    log "✓ Infrastructure deployed"
}

# Build production images
build_images() {
    log "Building production Docker images..."
    
    # Build main application image
    docker build \
        -f docker/Dockerfile.digital_human \
        -t aiqtoolkit/digital-human-prod:latest \
        --build-arg PRODUCTION=true \
        .
    
    # Push to registry
    docker push aiqtoolkit/digital-human-prod:latest
    
    log "✓ Images built and pushed"
}

# Deploy application
deploy_application() {
    log "Deploying Digital Human application..."
    
    # Create ConfigMap from production config
    kubectl create configmap digital-human-config \
        --from-file=production_config.yaml=src/aiq/digital_human/deployment/production_config.yaml \
        --namespace digital-human-prod \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic digital-human-secrets \
        --from-literal=nvidia-api-key=$NVIDIA_API_KEY \
        --from-literal=neural-api-key=$NEURAL_SUPERCOMPUTER_API_KEY \
        --from-literal=google-api-key=$GOOGLE_API_KEY \
        --from-literal=yahoo-api-key=$YAHOO_API_KEY \
        --from-literal=jwt-secret=$JWT_SECRET_KEY \
        --namespace digital-human-prod \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy application
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human
  namespace: digital-human-prod
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
      containers:
      - name: digital-human
        image: aiqtoolkit/digital-human-prod:latest
        ports:
        - containerPort: 8000
        env:
        - name: PRODUCTION_CONFIG_PATH
          value: /config/production_config.yaml
        envFrom:
        - secretRef:
            name: digital-human-secrets
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /config
        - name: ssl-certs
          mountPath: /ssl
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: digital-human-config
      - name: ssl-certs
        secret:
          secretName: ssl-certificates
---
apiVersion: v1
kind: Service
metadata:
  name: digital-human
  namespace: digital-human-prod
spec:
  selector:
    app: digital-human
  ports:
  - port: 80
    targetPort: 8000
    name: http
  - port: 443
    targetPort: 8443
    name: https
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: digital-human
  namespace: digital-human-prod
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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
    
    log "✓ Application deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace digital-human-prod \
        --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.app=digital-human \
        --wait
    
    # Deploy Grafana dashboards
    kubectl apply -f docker/monitoring/grafana/dashboards/
    
    # Deploy Elasticsearch & Kibana
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace digital-human-prod \
        --set replicas=3 \
        --set volumeClaimTemplate.resources.requests.storage=100Gi \
        --wait
    
    helm upgrade --install kibana elastic/kibana \
        --namespace digital-human-prod \
        --set elasticsearchHosts="http://elasticsearch-master:9200" \
        --wait
    
    log "✓ Monitoring stack deployed"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=digital-human \
        --namespace digital-human-prod \
        --timeout=300s
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get svc digital-human -n digital-human-prod -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        warning "Service IP not yet assigned. Waiting..."
        sleep 30
        SERVICE_IP=$(kubectl get svc digital-human -n digital-human-prod -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    fi
    
    # Test health endpoint
    if curl -f -k https://$SERVICE_IP/health; then
        log "✓ Health check passed"
    else
        error "Health check failed"
    fi
    
    # Run integration tests
    log "Running integration tests..."
    kubectl run integration-test \
        --image=aiqtoolkit/digital-human-prod:latest \
        --namespace=digital-human-prod \
        --rm -i --tty \
        --command -- python -m pytest tests/integration/
    
    log "✓ Deployment verified"
    
    # Print access information
    echo ""
    echo "=========================================="
    echo "Digital Human Production Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Access Points:"
    echo "- API Endpoint: https://$SERVICE_IP"
    echo "- Grafana: http://$SERVICE_IP:3000"
    echo "- Kibana: http://$SERVICE_IP:5601"
    echo ""
    echo "To view logs:"
    echo "kubectl logs -f deployment/digital-human -n digital-human-prod"
    echo ""
    echo "To scale deployment:"
    echo "kubectl scale deployment digital-human --replicas=5 -n digital-human-prod"
    echo ""
}

# Main deployment flow
main() {
    log "Starting Digital Human production deployment..."
    
    check_requirements
    validate_env
    deploy_infrastructure
    build_images
    deploy_application
    deploy_monitoring
    verify_deployment
    
    log "Deployment completed successfully!"
}

# Run main function
main "$@"