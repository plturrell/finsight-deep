#!/bin/bash
# Deployment script for Digital Human Financial Advisor

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-development}
NAMESPACE=${2:-aiqtoolkit}

echo -e "${GREEN}Deploying Digital Human Financial Advisor - Environment: $ENVIRONMENT${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check if running on Kubernetes
    if [ "$ENVIRONMENT" == "production" ] || [ "$ENVIRONMENT" == "staging" ]; then
        kubectl cluster-info &> /dev/null || {
            echo -e "${RED}Not connected to Kubernetes cluster${NC}"
            exit 1
        }
    fi
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Build Docker images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    # Build backend image
    docker build -t aiqtoolkit/digital-human:latest -f docker/Dockerfile.digital_human .
    
    if [ "$ENVIRONMENT" == "production" ] || [ "$ENVIRONMENT" == "staging" ]; then
        # Tag with version
        VERSION=$(git describe --tags --always)
        docker tag aiqtoolkit/digital-human:latest aiqtoolkit/digital-human:$VERSION
        
        # Push to registry
        echo -e "${YELLOW}Pushing images to registry...${NC}"
        docker push aiqtoolkit/digital-human:latest
        docker push aiqtoolkit/digital-human:$VERSION
    fi
    
    echo -e "${GREEN}Docker images built successfully${NC}"
}

# Deploy to development
deploy_development() {
    echo -e "${YELLOW}Deploying to development environment...${NC}"
    
    # Create .env file if not exists
    if [ ! -f .env ]; then
        cp .env.example .env
        echo -e "${YELLOW}Created .env file from example${NC}"
    fi
    
    # Start services with docker-compose
    docker-compose -f docker/docker-compose.digital_human.yml up -d
    
    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10
    
    # Initialize database
    docker-compose -f docker/docker-compose.digital_human.yml exec -T backend python -m aiq.digital_human.security.initialize_db
    
    echo -e "${GREEN}Development deployment complete${NC}"
    echo -e "${GREEN}Access the application at:${NC}"
    echo -e "  - Digital Human UI: http://localhost:8080"
    echo -e "  - Research Dashboard: http://localhost:8081"
    echo -e "  - API Documentation: http://localhost:8000/docs"
    echo -e "  - Grafana: http://localhost:3000 (admin/admin)"
    echo -e "  - Kibana: http://localhost:5601"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo -e "${YELLOW}Deploying to Kubernetes environment: $ENVIRONMENT${NC}"
    
    # Create namespace if doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    echo -e "${YELLOW}Creating secrets...${NC}"
    kubectl create secret generic digital-human-secrets \
        --from-literal=database-url=$DATABASE_URL \
        --from-literal=redis-url=$REDIS_URL \
        --from-literal=nvidia-api-key=$NVIDIA_API_KEY \
        --from-literal=jwt-secret-key=$JWT_SECRET_KEY \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy PostgreSQL
    echo -e "${YELLOW}Deploying PostgreSQL...${NC}"
    kubectl apply -f kubernetes/postgres-deployment.yaml -n $NAMESPACE
    
    # Deploy Redis
    echo -e "${YELLOW}Deploying Redis...${NC}"
    kubectl apply -f kubernetes/redis-deployment.yaml -n $NAMESPACE
    
    # Deploy backend
    echo -e "${YELLOW}Deploying backend...${NC}"
    kubectl apply -f kubernetes/digital-human-deployment.yaml -n $NAMESPACE
    
    # Deploy monitoring stack
    echo -e "${YELLOW}Deploying monitoring stack...${NC}"
    kubectl apply -f kubernetes/monitoring-deployment.yaml -n $NAMESPACE
    
    # Wait for deployments
    echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/digital-human-backend -n $NAMESPACE
    
    # Initialize database
    echo -e "${YELLOW}Initializing database...${NC}"
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=digital-human,component=backend -o jsonpath="{.items[0].metadata.name}")
    kubectl exec -n $NAMESPACE $POD_NAME -- python -m aiq.digital_human.security.initialize_db
    
    # Get service endpoints
    BACKEND_URL=$(kubectl get service digital-human-backend -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    FRONTEND_URL=$(kubectl get service digital-human-frontend -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    echo -e "${GREEN}Kubernetes deployment complete${NC}"
    echo -e "${GREEN}Access the application at:${NC}"
    echo -e "  - Digital Human UI: http://$FRONTEND_URL"
    echo -e "  - API: http://$BACKEND_URL"
}

# Run health checks
health_check() {
    echo -e "${YELLOW}Running health checks...${NC}"
    
    if [ "$ENVIRONMENT" == "development" ]; then
        # Check Docker containers
        docker-compose -f docker/docker-compose.digital_human.yml ps
        
        # Check backend health
        curl -f http://localhost:8000/health || {
            echo -e "${RED}Backend health check failed${NC}"
            exit 1
        }
    else
        # Check Kubernetes pods
        kubectl get pods -n $NAMESPACE
        
        # Check backend health
        BACKEND_URL=$(kubectl get service digital-human-backend -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        curl -f http://$BACKEND_URL:8000/health || {
            echo -e "${RED}Backend health check failed${NC}"
            exit 1
        }
    fi
    
    echo -e "${GREEN}Health checks passed${NC}"
}

# Rollback deployment
rollback() {
    echo -e "${YELLOW}Rolling back deployment...${NC}"
    
    if [ "$ENVIRONMENT" == "development" ]; then
        docker-compose -f docker/docker-compose.digital_human.yml down
    else
        kubectl rollout undo deployment/digital-human-backend -n $NAMESPACE
        kubectl rollout undo deployment/digital-human-frontend -n $NAMESPACE
    fi
    
    echo -e "${GREEN}Rollback complete${NC}"
}

# Main deployment flow
main() {
    check_prerequisites
    
    case "$1" in
        "development")
            build_images
            deploy_development
            health_check
            ;;
        "staging"|"production")
            build_images
            deploy_kubernetes
            health_check
            ;;
        "rollback")
            rollback
            ;;
        *)
            echo "Usage: $0 {development|staging|production|rollback} [namespace]"
            exit 1
            ;;
    esac
}

# Run main function
main $@