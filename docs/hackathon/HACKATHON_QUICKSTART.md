# Digital Human Hackathon Quick Start

## 🚀 Deploy in 15 Minutes

This guide will get your Digital Human system running on AWS with NVIDIA Audio2Face-3D.

### Prerequisites
- Docker installed locally
- AWS CLI installed
- kubectl installed
- eksctl installed

### Step 1: Clone and Setup
```bash
git clone https://github.com/your-repo/AIQToolkit.git
cd AIQToolkit
```

### Step 2: Export Credentials
```bash
# AWS Credentials
export AWS_ACCESS_KEY_ID="<provided>"
export AWS_SECRET_ACCESS_KEY="<provided>"

# NVIDIA Audio2Face-3D API Key
export NVIDIA_API_KEY="<provided>"
```

### Step 3: Deploy to AWS
```bash
# Quick deploy with all services
./src/aiq/digital_human/deployment/deploy_aws_optimized.sh deploy
```

This script will:
1. Create an EKS cluster with GPU nodes
2. Deploy the Digital Human system
3. Configure NVIDIA Audio2Face-3D
4. Set up load balancer
5. Provide access URL

### Step 4: Access Your System

After deployment (about 10-15 minutes), you'll get:
```
Access URL: http://[your-lb-url].elb.amazonaws.com
```

### Step 5: Test the System

1. **Health Check**:
```bash
curl http://[your-lb-url]/health
```

2. **Create Session**:
```bash
curl -X POST http://[your-lb-url]/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user"}'
```

3. **Send Message**:
```bash
curl -X POST http://[your-lb-url]/messages \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "[session-id]",
    "content": "Tell me about the market today"
  }'
```

### Step 6: View Logs
```bash
kubectl logs -f deployment/digital-human -n digital-human
```

## 🎮 Demo Features

### 1. Real-time Avatar
- NVIDIA Audio2Face-3D integration
- Photorealistic 2D rendering
- Emotional expressions
- Lip-sync with speech

### 2. Financial Analysis
- Real-time market data
- Portfolio optimization with MCTS
- Risk assessment
- Neural supercomputer reasoning

### 3. Natural Conversation
- Context-aware responses
- Multi-turn dialogue
- Emotional intelligence
- Voice interaction

## 📊 Architecture

```
┌─────────────────────────────────────────┐
│      Digital Human Interface (ACE)      │
├─────────────────────────────────────────┤
│   Model Context Server (RAG + Search)   │
├─────────────────────────────────────────┤
│     Neural Supercomputer Connector      │
├─────────────────────────────────────────┤
│    Financial Analysis Engine (MCTS)     │
└─────────────────────────────────────────┘
```

## 🛠️ Troubleshooting

### GPU Not Available
```bash
kubectl describe nodes | grep nvidia
```

### Pods Not Starting
```bash
kubectl describe pod [pod-name] -n digital-human
```

### Load Balancer Not Ready
```bash
kubectl get svc -n digital-human -w
```

## 💰 Cost Optimization

The deployment uses:
- Spot instances for GPU nodes
- Auto-scaling (1-5 nodes)
- g4dn.xlarge instances (~$0.50/hour)

Estimated cost: ~$50-100 for hackathon duration

## 🧹 Cleanup

When done, clean up all resources:
```bash
./src/aiq/digital_human/deployment/deploy_aws_optimized.sh cleanup
```

## 📝 Important Notes

1. **API Keys**: Keep your API keys secure
2. **Region**: Deployment is in us-east-1
3. **GPU**: Each pod uses 1 NVIDIA T4 GPU
4. **Storage**: Uses AWS managed services

## 🏆 Hackathon Tips

1. **Focus on Demo**: System is ready, focus on your use case
2. **Monitor Costs**: Use AWS Cost Explorer
3. **Scale as Needed**: `kubectl scale deployment`
4. **Save Outputs**: Record demo videos early

## 📞 Support

- Slack: #digital-human-support
- Issues: GitHub Issues
- Docs: See `/docs` folder

Good luck with your hackathon! 🚀