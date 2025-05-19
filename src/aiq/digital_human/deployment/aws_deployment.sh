#!/bin/bash

# AWS Deployment Script for Digital Human System
# This script deploys the system to AWS EKS with proper security

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# AWS Configuration
export AWS_REGION="us-east-1"
export CLUSTER_NAME="digital-human-prod"
export NODE_GROUP_NAME="gpu-nodes"
export NAMESPACE="digital-human"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not installed"
    fi
    
    # Check eksctl
    if ! command -v eksctl &> /dev/null; then
        error "eksctl not installed"
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not installed"
    fi
    
    log "✓ Prerequisites satisfied"
}

# Setup AWS credentials securely
setup_aws_credentials() {
    log "Setting up AWS credentials..."
    
    # IMPORTANT: Never hardcode credentials in scripts
    # Use environment variables or AWS SSM
    
    # Create AWS credentials file
    mkdir -p ~/.aws
    
    # Configure AWS CLI
    aws configure set region $AWS_REGION
    
    # Verify credentials
    if aws sts get-caller-identity &> /dev/null; then
        log "✓ AWS credentials configured"
    else
        error "AWS credentials not valid"
    fi
}

# Store secrets in AWS Secrets Manager
store_secrets() {
    log "Storing secrets in AWS Secrets Manager..."
    
    # Create secrets
    aws secretsmanager create-secret \
        --name digital-human/nvidia-api-key \
        --secret-string "$NVIDIA_API_KEY" \
        --region $AWS_REGION || true
    
    # Store other API keys
    aws secretsmanager create-secret \
        --name digital-human/api-keys \
        --secret-string '{
            "neural_api_key": "'$NEURAL_SUPERCOMPUTER_API_KEY'",
            "google_api_key": "'$GOOGLE_API_KEY'",
            "alpha_vantage_key": "'$ALPHA_VANTAGE_API_KEY'"
        }' \
        --region $AWS_REGION || true
    
    log "✓ Secrets stored securely"
}

# Create EKS cluster
create_eks_cluster() {
    log "Creating EKS cluster..."
    
    # Check if cluster exists
    if aws eks describe-cluster --name $CLUSTER_NAME --region $AWS_REGION &> /dev/null; then
        log "Cluster already exists"
    else
        # Create cluster with GPU support
        eksctl create cluster \
            --name $CLUSTER_NAME \
            --region $AWS_REGION \
            --version 1.28 \
            --nodegroup-name $NODE_GROUP_NAME \
            --node-type g4dn.xlarge \
            --nodes 3 \
            --nodes-min 3 \
            --nodes-max 10 \
            --node-ami auto \
            --managed
        
        # Install NVIDIA device plugin
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
    fi
    
    # Update kubeconfig
    aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME
    
    log "✓ EKS cluster ready"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy RDS PostgreSQL
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: rds-postgres
  namespace: $NAMESPACE
type: Opaque
data:
  password: $(echo -n "$POSTGRES_PASSWORD" | base64)
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  type: ExternalName
  externalName: $RDS_ENDPOINT
  ports:
  - port: 5432
    targetPort: 5432
EOF
    
    # Deploy ElastiCache Redis
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  type: ExternalName
  externalName: $ELASTICACHE_ENDPOINT
  ports:
  - port: 6379
    targetPort: 6379
EOF
    
    log "✓ Infrastructure deployed"
}

# Deploy application
deploy_application() {
    log "Deploying Digital Human application..."
    
    # Build and push to ECR
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPO="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/digital-human"
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
    
    # Create repository if not exists
    aws ecr create-repository --repository-name digital-human --region $AWS_REGION || true
    
    # Build and push image
    docker build -f docker/Dockerfile.digital_human_production -t digital-human:latest .
    docker tag digital-human:latest $ECR_REPO:latest
    docker push $ECR_REPO:latest
    
    # Deploy to Kubernetes
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human
  namespace: $NAMESPACE
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
      serviceAccountName: digital-human-sa
      containers:
      - name: digital-human
        image: $ECR_REPO:latest
        ports:
        - containerPort: 8000
        env:
        - name: AWS_REGION
          value: $AWS_REGION
        - name: NVIDIA_API_KEY
          valueFrom:
            secretKeyRef:
              name: nvidia-secret
              key: api-key
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: digital-human
  namespace: $NAMESPACE
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "$SSL_CERT_ARN"
spec:
  type: LoadBalancer
  selector:
    app: digital-human
  ports:
  - name: https
    port: 443
    targetPort: 8000
    protocol: TCP
EOF
    
    log "✓ Application deployed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Deploy CloudWatch Container Insights
    curl https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluentd-quickstart.yaml | \
    sed "s/{{cluster_name}}/$CLUSTER_NAME/;s/{{region_name}}/$AWS_REGION/" | \
    kubectl apply -f -
    
    # Deploy Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace $NAMESPACE \
        --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.app=digital-human
    
    log "✓ Monitoring configured"
}

# Create IAM roles and policies
setup_iam() {
    log "Setting up IAM roles..."
    
    # Create service account
    eksctl create iamserviceaccount \
        --cluster $CLUSTER_NAME \
        --namespace $NAMESPACE \
        --name digital-human-sa \
        --attach-policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy \
        --attach-policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy \
        --attach-policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite \
        --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
        --override-existing-serviceaccounts \
        --approve
    
    log "✓ IAM roles configured"
}

# Configure autoscaling
setup_autoscaling() {
    log "Setting up autoscaling..."
    
    # Deploy cluster autoscaler
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
    
    # Configure HPA
    cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: digital-human-hpa
  namespace: $NAMESPACE
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
    
    log "✓ Autoscaling configured"
}

# Configure security
setup_security() {
    log "Setting up security..."
    
    # Create security group for EKS
    SG_ID=$(aws ec2 create-security-group \
        --group-name digital-human-sg \
        --description "Security group for Digital Human" \
        --vpc-id $(aws eks describe-cluster --name $CLUSTER_NAME --query 'cluster.resourcesVpcConfig.vpcId' --output text) \
        --region $AWS_REGION \
        --output text)
    
    # Allow HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    # Configure network policies
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: digital-human-netpol
  namespace: $NAMESPACE
spec:
  podSelector:
    matchLabels:
      app: digital-human
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: digital-human
    ports:
    - port: 8000
  egress:
  - to:
    - podSelector: {}
  - to:
    - namespaceSelector: {}
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 169.254.169.254/32
EOF
    
    log "✓ Security configured"
}

# Create backup strategy
setup_backup() {
    log "Setting up backup strategy..."
    
    # Create S3 bucket for backups
    BACKUP_BUCKET="digital-human-backups-$(date +%s)"
    aws s3api create-bucket \
        --bucket $BACKUP_BUCKET \
        --region $AWS_REGION \
        --create-bucket-configuration LocationConstraint=$AWS_REGION
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket $BACKUP_BUCKET \
        --versioning-configuration Status=Enabled
    
    # Configure lifecycle
    cat <<EOF > lifecycle.json
{
    "Rules": [{
        "ID": "ArchiveOldBackups",
        "Status": "Enabled",
        "Transitions": [{
            "Days": 30,
            "StorageClass": "GLACIER"
        }],
        "Expiration": {
            "Days": 365
        }
    }]
}
EOF
    
    aws s3api put-bucket-lifecycle-configuration \
        --bucket $BACKUP_BUCKET \
        --lifecycle-configuration file://lifecycle.json
    
    log "✓ Backup configured"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Wait for pods
    kubectl wait --for=condition=ready pod -l app=digital-human \
        --namespace $NAMESPACE \
        --timeout=300s
    
    # Get load balancer URL
    LB_URL=$(kubectl get svc digital-human -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    # Wait for LB to be ready
    sleep 30
    
    # Test health endpoint
    if curl -f -k https://$LB_URL/health; then
        log "✓ Health check passed"
    else
        error "Health check failed"
    fi
    
    # Display access information
    echo ""
    echo "=========================================="
    echo "Digital Human Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Access Points:"
    echo "- API Endpoint: https://$LB_URL"
    echo "- CloudWatch: https://console.aws.amazon.com/cloudwatch"
    echo "- EKS Console: https://console.aws.amazon.com/eks"
    echo ""
    echo "Useful Commands:"
    echo "- View logs: kubectl logs -f deployment/digital-human -n $NAMESPACE"
    echo "- Scale: kubectl scale deployment digital-human --replicas=5 -n $NAMESPACE"
    echo "- SSH to pod: kubectl exec -it deployment/digital-human -n $NAMESPACE -- bash"
    echo ""
}

# Main deployment flow
main() {
    log "Starting AWS deployment..."
    
    check_prerequisites
    setup_aws_credentials
    store_secrets
    create_eks_cluster
    setup_iam
    deploy_infrastructure
    deploy_application
    setup_monitoring
    setup_autoscaling
    setup_security
    setup_backup
    verify_deployment
    
    log "AWS deployment completed successfully!"
}

# Run main
main "$@"