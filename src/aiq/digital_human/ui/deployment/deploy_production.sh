#!/bin/bash
# Production deployment script for Digital Human Unified Backend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DEPLOYMENT_ENV=${1:-production}
DEPLOYMENT_USER="aiqtoolkit"
DEPLOYMENT_DIR="/opt/aiqtoolkit/digital_human/ui"
VENV_DIR="/opt/aiqtoolkit/venv"
LOG_DIR="/opt/aiqtoolkit/logs"
DATA_DIR="/opt/aiqtoolkit/data"

echo -e "${GREEN}Digital Human Production Deployment${NC}"
echo -e "${YELLOW}Environment: ${DEPLOYMENT_ENV}${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Create deployment user if not exists
if ! id "$DEPLOYMENT_USER" &>/dev/null; then
    echo -e "${YELLOW}Creating deployment user...${NC}"
    useradd -r -s /bin/bash -m -d /home/$DEPLOYMENT_USER $DEPLOYMENT_USER
fi

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p $DEPLOYMENT_DIR
mkdir -p $VENV_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
mkdir -p /etc/aiqtoolkit

# Set permissions
chown -R $DEPLOYMENT_USER:$DEPLOYMENT_USER /opt/aiqtoolkit
chmod 755 /opt/aiqtoolkit

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get update
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    redis-server \
    postgresql \
    nginx \
    certbot \
    python3-certbot-nginx \
    supervisor \
    curl \
    git

# Setup Python virtual environment
echo -e "${YELLOW}Setting up Python environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip wheel setuptools
pip install -r $DEPLOYMENT_DIR/requirements.txt
pip install gunicorn uvloop httptools

# Generate secrets if not exist
echo -e "${YELLOW}Generating secrets...${NC}"
if [ ! -f /etc/aiqtoolkit/jwt.secret ]; then
    openssl rand -hex 64 > /etc/aiqtoolkit/jwt.secret
    chmod 600 /etc/aiqtoolkit/jwt.secret
    chown $DEPLOYMENT_USER:$DEPLOYMENT_USER /etc/aiqtoolkit/jwt.secret
fi

# Setup SSL certificates
echo -e "${YELLOW}Setting up SSL certificates...${NC}"
if [ ! -f /etc/ssl/certs/digitalhuman.crt ]; then
    # For production, use Let's Encrypt
    if [ "$DEPLOYMENT_ENV" == "production" ]; then
        certbot certonly --nginx -d api.digitalhuman.ai -d digitalhuman.ai
        ln -sf /etc/letsencrypt/live/digitalhuman.ai/fullchain.pem /etc/ssl/certs/digitalhuman.crt
        ln -sf /etc/letsencrypt/live/digitalhuman.ai/privkey.pem /etc/ssl/private/digitalhuman.key
    else
        # Self-signed for development/staging
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout /etc/ssl/private/digitalhuman.key \
            -out /etc/ssl/certs/digitalhuman.crt \
            -subj "/C=US/ST=State/L=City/O=AIQToolkit/CN=digitalhuman.local"
    fi
fi

# Configure PostgreSQL
echo -e "${YELLOW}Configuring PostgreSQL...${NC}"
sudo -u postgres psql <<EOF
CREATE USER aiqtoolkit WITH PASSWORD 'secure_password_here';
CREATE DATABASE digital_human OWNER aiqtoolkit;
GRANT ALL PRIVILEGES ON DATABASE digital_human TO aiqtoolkit;
EOF

# Configure Redis
echo -e "${YELLOW}Configuring Redis...${NC}"
cat > /etc/redis/redis.conf <<EOF
bind 127.0.0.1
protected-mode yes
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
maxmemory 1gb
maxmemory-policy allkeys-lru
EOF

systemctl restart redis

# Configure NGINX
echo -e "${YELLOW}Configuring NGINX...${NC}"
cat > /etc/nginx/sites-available/digital-human <<'EOF'
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.digitalhuman.ai digitalhuman.ai;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.digitalhuman.ai;
    
    ssl_certificate /etc/ssl/certs/digitalhuman.crt;
    ssl_certificate_key /etc/ssl/private/digitalhuman.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    client_max_body_size 10M;
    
    location /ws {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    }
    
    location /static {
        alias /opt/aiqtoolkit/digital_human/ui/frontend;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

ln -sf /etc/nginx/sites-available/digital-human /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# Install systemd service
echo -e "${YELLOW}Installing systemd service...${NC}"
cp $DEPLOYMENT_DIR/deployment/digital-human-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable digital-human-backend.service

# Setup log rotation
echo -e "${YELLOW}Setting up log rotation...${NC}"
cat > /etc/logrotate.d/digital-human <<EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 $DEPLOYMENT_USER $DEPLOYMENT_USER
    sharedscripts
    postrotate
        systemctl reload digital-human-backend
    endscript
}
EOF

# Setup monitoring with Prometheus
echo -e "${YELLOW}Setting up monitoring...${NC}"
cat > /etc/prometheus/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'digital-human-backend'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
EOF

# Create environment file
echo -e "${YELLOW}Creating environment configuration...${NC}"
cat > /etc/aiqtoolkit/environment <<EOF
# Digital Human Production Environment
ENVIRONMENT=$DEPLOYMENT_ENV
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://aiqtoolkit:secure_password_here@localhost/digital_human
JWT_SECRET_FILE=/etc/aiqtoolkit/jwt.secret
SSL_CERT_FILE=/etc/ssl/certs/digitalhuman.crt
SSL_KEY_FILE=/etc/ssl/private/digitalhuman.key
LOG_LEVEL=INFO
ENABLE_CONSENSUS=true
ENABLE_NVIDIA=false
ENABLE_MCP=true
# Add your API keys here
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
EOF

chmod 600 /etc/aiqtoolkit/environment
chown $DEPLOYMENT_USER:$DEPLOYMENT_USER /etc/aiqtoolkit/environment

# Start services
echo -e "${YELLOW}Starting services...${NC}"
systemctl start digital-human-backend
systemctl status digital-human-backend

# Setup automatic backups
echo -e "${YELLOW}Setting up automatic backups...${NC}"
cat > /etc/cron.daily/digital-human-backup <<EOF
#!/bin/bash
BACKUP_DIR="/opt/aiqtoolkit/backups"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)

mkdir -p \$BACKUP_DIR

# Backup database
pg_dump -U aiqtoolkit digital_human | gzip > \$BACKUP_DIR/db_\$TIMESTAMP.sql.gz

# Backup Redis
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb \$BACKUP_DIR/redis_\$TIMESTAMP.rdb

# Backup configuration
tar czf \$BACKUP_DIR/config_\$TIMESTAMP.tar.gz /etc/aiqtoolkit

# Keep only last 7 days of backups
find \$BACKUP_DIR -type f -mtime +7 -delete
EOF

chmod +x /etc/cron.daily/digital-human-backup

# Health check
echo -e "${YELLOW}Running health check...${NC}"
sleep 5
curl -f https://localhost:8000/health || {
    echo -e "${RED}Health check failed!${NC}"
    journalctl -u digital-human-backend -n 50
    exit 1
}

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${YELLOW}Service Status:${NC}"
systemctl status digital-human-backend --no-pager

echo -e "\n${GREEN}Access URLs:${NC}"
echo -e "API: https://api.digitalhuman.ai"
echo -e "Health: https://api.digitalhuman.ai/health"
echo -e "Metrics: https://api.digitalhuman.ai/metrics"
echo -e "WebSocket: wss://api.digitalhuman.ai/ws"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Update DNS records to point to this server"
echo "2. Configure monitoring alerts"
echo "3. Set up backup retention policy"
echo "4. Review security settings"
echo "5. Load test the deployment"

echo -e "\n${GREEN}Useful Commands:${NC}"
echo "View logs: journalctl -u digital-human-backend -f"
echo "Restart service: systemctl restart digital-human-backend"
echo "Check status: systemctl status digital-human-backend"
echo "View metrics: curl http://localhost:8000/metrics"