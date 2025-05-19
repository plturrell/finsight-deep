#!/bin/bash
# Pre-deployment verification script for Digital Human Financial Advisor

echo "=== Digital Human Financial Advisor - Pre-Deployment Check ==="
echo "============================================================"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check functions
check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

check_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 1. Check environment variables
echo ""
echo "1. Checking Environment Variables..."
if [ -z "$NVIDIA_API_KEY" ]; then
    check_fail "NVIDIA_API_KEY not set. Run: source production_env.sh"
else
    check_pass "NVIDIA_API_KEY is set"
fi

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    check_fail "AWS credentials not set"
else
    check_pass "AWS credentials configured"
fi

# 2. Check required tools
echo ""
echo "2. Checking Required Tools..."
for tool in docker kubectl aws; do
    if command -v $tool &> /dev/null; then
        check_pass "$tool is installed"
    else
        check_fail "$tool is not installed"
    fi
done

# 3. Check AWS connectivity
echo ""
echo "3. Checking AWS Connectivity..."
if aws sts get-caller-identity &> /dev/null; then
    check_pass "AWS authentication successful"
else
    check_fail "AWS authentication failed"
fi

# 4. Check configuration files
echo ""
echo "4. Checking Configuration Files..."
CONFIG_FILES=(
    "llama3_config.yaml"
    "deploy_llama3_aws.sh"
    "production_env.sh"
    "docker-compose.production.yml"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file exists"
    else
        check_fail "$file not found"
    fi
done

# 5. Check Docker daemon
echo ""
echo "5. Checking Docker..."
if docker info &> /dev/null; then
    check_pass "Docker daemon is running"
else
    check_fail "Docker daemon is not running"
fi

# 6. Check deployment readiness
echo ""
echo "6. Checking Deployment Readiness..."
if [ -f "deploy_llama3_aws.sh" ] && [ -x "deploy_llama3_aws.sh" ]; then
    check_pass "Deployment script is executable"
else
    check_warn "Making deployment script executable..."
    chmod +x deploy_llama3_aws.sh
    check_pass "Deployment script is now executable"
fi

# 7. Verify model configuration
echo ""
echo "7. Verifying Model Configuration..."
if grep -q "llama-3-8b-instruct" llama3_config.yaml; then
    check_pass "Llama3-8B-Instruct configured correctly"
else
    check_fail "Llama3 model not properly configured"
fi

# 8. Check GPU support
echo ""
echo "8. Checking GPU Support..."
if grep -q "g4dn.xlarge" deploy_llama3_aws.sh; then
    check_pass "GPU nodes configured (g4dn.xlarge)"
else
    check_fail "GPU nodes not properly configured"
fi

# Summary
echo ""
echo "============================================================"
echo "Pre-Deployment Check Complete!"
echo ""
echo "To deploy the Digital Human Financial Advisor:"
echo "1. Ensure all checks above are green"
echo "2. Run: ./deploy_llama3_aws.sh deploy"
echo ""
echo "Deployment will create:"
echo "- EKS cluster with GPU nodes"
echo "- Digital Human interface with NVIDIA ACE"
echo "- Llama3-8B-Instruct language model"
echo "- Complete financial analysis system"
echo ""
echo "Expected deployment time: 5-7 minutes"
echo "============================================================"