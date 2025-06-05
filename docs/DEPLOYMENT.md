# Supply Chain Risk Intelligence System - Deployment Guide

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Database Setup](#database-setup)
8. [Redis Configuration](#redis-configuration)
9. [Monitoring Setup](#monitoring-setup)
10. [Security Configuration](#security-configuration)
11. [Performance Tuning](#performance-tuning)
12. [Maintenance](#maintenance)

## Deployment Overview

The Supply Chain Risk Intelligence System can be deployed in various configurations:

- **Local Development**: Single-machine setup for development and testing
- **Docker Compose**: Containerized deployment for easy scaling
- **Production Server**: Traditional server deployment with external services
- **Cloud Platform**: AWS, Azure, or GCP deployment with managed services
- **Kubernetes**: Container orchestration for high availability

### Architecture Components

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Web UI    │    │  API Server │    │ ML Models   │
│ (Dashboard) │◄──►│ (FastAPI)   │◄──►│ (PyTorch)   │
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Database   │    │ Cache Layer │
                   │ (SQLite/PG) │    │   (Redis)   │
                   └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │ Monitoring  │
                   │(Prometheus) │
                   └─────────────┘
```

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores (2.0 GHz)
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: Broadband internet connection
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+

#### Recommended Requirements
- **CPU**: 8 cores (3.0 GHz)
- **RAM**: 16 GB
- **Storage**: 100 GB NVMe SSD
- **Network**: High-speed internet with low latency
- **OS**: Linux (Ubuntu 22.04 LTS)

### Software Dependencies

#### Core Dependencies
- **Python**: 3.9+ (3.11 recommended)
- **pip**: Latest version
- **Git**: For source code management
- **Docker**: 20.10+ (optional but recommended)
- **Docker Compose**: 2.0+ (for containerized deployment)

#### Optional Dependencies
- **Redis**: 6.0+ (for caching)
- **PostgreSQL**: 13+ (for production database)
- **Nginx**: For reverse proxy and load balancing
- **Prometheus**: For monitoring
- **Grafana**: For visualization

### External Services

#### Required API Keys
- **OpenWeatherMap**: Weather data (`OPENWEATHER_API_KEY`)
- **NewsAPI**: News data (`NEWS_API_KEY`)
- **Twitter**: Social media data (`TWITTER_BEARER_TOKEN`)

#### Optional Services
- **AWS S3**: For model and data storage
- **Google Cloud Storage**: Alternative cloud storage
- **SendGrid**: For email notifications
- **Slack**: For alert notifications

## Local Development Setup

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd supply-chain-risk-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Initialize database
python -c "from database.manager import get_db_manager; get_db_manager().init_database()"

# 6. Download and set up models
python scripts/setup_models.py

# 7. Run the system
python main.py
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Application Settings
DEBUG=true
LOG_LEVEL=INFO
HOST=localhost
PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///supply_chain_risk.db
ASYNC_DATABASE_URL=sqlite+aiosqlite:///supply_chain_risk.db

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_DEFAULT_TTL=3600
CACHE_MAX_MEMORY_ITEMS=1000

# Authentication
JWT_SECRET_KEY=your-secret-key-here
API_DEMO_TOKEN=demo_user_token_12345
API_ADMIN_TOKEN=admin_user_token_67890

# External APIs
OPENWEATHER_API_KEY=your_openweather_api_key
NEWS_API_KEY=your_news_api_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Model Configuration
MODEL_PATH=./models
CHECKPOINT_PATH=./checkpoints
MODEL_CACHE_SIZE=5

# Monitoring
PROMETHEUS_PORT=9090
METRICS_ENABLED=true
```

### Development Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Code formatting
black .
isort .

# Linting
flake8 .
mypy .

# Security scanning
bandit -r .
```

## Production Deployment

### Server Setup

#### Ubuntu 22.04 LTS Setup

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx redis-server postgresql-14

# 3. Create application user
sudo useradd -m -s /bin/bash scr
sudo usermod -aG sudo scr

# 4. Switch to application user
sudo su - scr

# 5. Clone and setup application
git clone <repository-url> /home/scr/app
cd /home/scr/app
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### PostgreSQL Database Setup

```bash
# 1. Create database and user
sudo -u postgres psql
CREATE DATABASE supply_chain_risk;
CREATE USER scr_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE supply_chain_risk TO scr_user;
\q

# 2. Update environment variables
DATABASE_URL=postgresql://scr_user:secure_password@localhost/supply_chain_risk
ASYNC_DATABASE_URL=postgresql+asyncpg://scr_user:secure_password@localhost/supply_chain_risk
```

#### Nginx Configuration

Create `/etc/nginx/sites-available/scr`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files
    location /static/ {
        alias /home/scr/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Metrics endpoint (restricted)
    location /metrics {
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
```

#### Systemd Service

Create `/etc/systemd/system/scr.service`:

```ini
[Unit]
Description=Supply Chain Risk Intelligence System
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=exec
User=scr
Group=scr
WorkingDirectory=/home/scr/app
Environment=PATH=/home/scr/app/venv/bin
ExecStart=/home/scr/app/venv/bin/python main.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/scr/app/logs /home/scr/app/data

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable scr.service
sudo systemctl start scr.service
sudo systemctl status scr.service
```

## Docker Deployment

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://scr_user:scr_password@db:5432/supply_chain_risk
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=supply_chain_risk
      - POSTGRES_USER=scr_user
      - POSTGRES_PASSWORD=scr_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 scr && chown -R scr:scr /app
USER scr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "main.py"]
```

### Docker Commands

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3

# Update application
docker-compose pull
docker-compose up -d

# Backup database
docker-compose exec db pg_dump -U scr_user supply_chain_risk > backup.sql

# Restore database
docker-compose exec -T db psql -U scr_user supply_chain_risk < backup.sql
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# 1. Launch EC2 instance (t3.large or larger)
# 2. Security groups: 80, 443, 22
# 3. Elastic IP for static IP
# 4. Install application following production setup

# Install AWS CLI
pip install awscli
aws configure

# Use RDS for database
# Use ElastiCache for Redis
# Use S3 for model storage
```

#### RDS Database Setup

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier scr-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username scruser \
    --master-user-password securepassword \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-xxxxxxxxx

# Update environment variables
DATABASE_URL=postgresql://scruser:securepassword@scr-db.xxxxxxxxx.us-east-1.rds.amazonaws.com:5432/scrdb
```

#### ElastiCache Redis Setup

```bash
# Create ElastiCache Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id scr-redis \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1

# Update environment variables
REDIS_URL=redis://scr-redis.xxxxxx.cache.amazonaws.com:6379/0
```

### Kubernetes Deployment

#### Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scr-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scr-app
  template:
    metadata:
      labels:
        app: scr-app
    spec:
      containers:
      - name: app
        image: scr:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: scr-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: scr-secrets
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: scr-service
spec:
  selector:
    app: scr-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Database Setup

### SQLite (Development)

```python
# Automatic setup with default configuration
from database.manager import get_db_manager
db = get_db_manager()  # Creates SQLite database automatically
```

### PostgreSQL (Production)

```bash
# 1. Install PostgreSQL
sudo apt install postgresql-14 postgresql-client-14

# 2. Create database and user
sudo -u postgres createuser --createdb scr_user
sudo -u postgres createdb supply_chain_risk -O scr_user
sudo -u postgres psql -c "ALTER USER scr_user PASSWORD 'secure_password';"

# 3. Configure PostgreSQL
sudo nano /etc/postgresql/14/main/postgresql.conf
# Add: listen_addresses = 'localhost'

sudo nano /etc/postgresql/14/main/pg_hba.conf
# Add: local all scr_user md5

# 4. Restart PostgreSQL
sudo systemctl restart postgresql

# 5. Test connection
psql -h localhost -U scr_user -d supply_chain_risk
```

### Database Migrations

```python
# Create migration script
from database.manager import DatabaseManager
from database.models import Base

def migrate_database():
    db = DatabaseManager()
    # Create all tables
    Base.metadata.create_all(bind=db.engine)
    print("Database migration completed")

if __name__ == "__main__":
    migrate_database()
```

### Database Backup and Restore

```bash
# Backup PostgreSQL
pg_dump -h localhost -U scr_user supply_chain_risk > backup_$(date +%Y%m%d).sql

# Restore PostgreSQL
psql -h localhost -U scr_user supply_chain_risk < backup_20240101.sql

# Backup SQLite
cp supply_chain_risk.db backup_$(date +%Y%m%d).db

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U scr_user supply_chain_risk | gzip > $BACKUP_DIR/backup_$DATE.sql.gz
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

## Redis Configuration

### Redis Installation

```bash
# Ubuntu/Debian
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Uncomment: requirepass yourpassword
# Set: maxmemory 256mb
# Set: maxmemory-policy allkeys-lru

# Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test Redis
redis-cli ping
```

### Redis Configuration File

```bash
# /etc/redis/redis.conf
bind 127.0.0.1
port 6379
requirepass yourpassword
maxmemory 512mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
save 900 1
save 300 10
save 60 10000
```

### Redis Monitoring

```bash
# Monitor Redis performance
redis-cli info memory
redis-cli info stats
redis-cli monitor

# Check slow queries
redis-cli slowlog get 10

# Memory usage analysis
redis-cli --bigkeys
```

## Monitoring Setup

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'scr-app'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
```

### Grafana Dashboard

Import the provided dashboard configuration:

```json
{
  "dashboard": {
    "id": null,
    "title": "Supply Chain Risk Intelligence",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Log Management

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/scr

/home/scr/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload scr
    endscript
}
```

## Security Configuration

### SSL/TLS Setup

```bash
# Install certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration

```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Check status
sudo ufw status
```

### Security Headers

Add to Nginx configuration:

```nginx
# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload";
```

### API Security

```python
# Update environment variables
JWT_SECRET_KEY=$(openssl rand -hex 32)
API_DEMO_TOKEN=$(openssl rand -hex 16)
API_ADMIN_TOKEN=$(openssl rand -hex 16)

# Rate limiting configuration
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
RATE_LIMIT_BURST=20
```

## Performance Tuning

### Application Optimization

```python
# Update .env for production
DEBUG=false
LOG_LEVEL=WARNING
CACHE_DEFAULT_TTL=7200
MODEL_CACHE_SIZE=10
WORKER_PROCESSES=4
```

### Database Optimization

```sql
-- PostgreSQL performance tuning
-- Add to postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

-- Create indexes
CREATE INDEX CONCURRENTLY idx_predictions_location_timestamp 
ON risk_predictions(location, timestamp);

CREATE INDEX CONCURRENTLY idx_alerts_severity_status 
ON alerts(severity, status);
```

### Redis Optimization

```bash
# Redis performance tuning
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# Add to /etc/rc.local
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

### System Optimization

```bash
# Increase file limits
echo '* soft nofile 65536' >> /etc/security/limits.conf
echo '* hard nofile 65536' >> /etc/security/limits.conf

# TCP optimization
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
sysctl -p
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
```bash
#!/bin/bash
# Daily maintenance script

# Check system health
systemctl status scr
systemctl status postgresql
systemctl status redis-server
systemctl status nginx

# Check disk space
df -h

# Check application logs
tail -n 100 /home/scr/app/logs/app.log | grep ERROR

# Database cleanup (if needed)
python -c "
from database.manager import get_db_manager
db = get_db_manager()
result = db.cleanup_old_data(days=30)
print(f'Cleaned up: {result}')
"
```

#### Weekly Tasks
```bash
#!/bin/bash
# Weekly maintenance script

# Update system packages
sudo apt update && sudo apt upgrade -y

# Backup database
pg_dump -h localhost -U scr_user supply_chain_risk | gzip > /backup/weekly_$(date +%Y%m%d).sql.gz

# Analyze database
sudo -u postgres psql -d supply_chain_risk -c "ANALYZE;"

# Check certificate expiry
certbot certificates

# Review security logs
grep "authentication failure" /var/log/auth.log
```

#### Monthly Tasks
```bash
#!/bin/bash
# Monthly maintenance script

# Review and update dependencies
pip list --outdated

# Performance analysis
# Review Grafana dashboards
# Check error rates and response times
# Analyze resource utilization

# Capacity planning
# Check storage usage trends
# Review memory and CPU utilization
# Plan for scaling if needed

# Security review
# Check for security updates
# Review access logs
# Update SSL certificates if needed
```

### Backup Strategy

#### Automated Backup Script
```bash
#!/bin/bash
# Backup script - run daily via cron

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U scr_user supply_chain_risk | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Application data backup
tar -czf $BACKUP_DIR/app_$DATE.tar.gz /home/scr/app/models /home/scr/app/checkpoints

# Configuration backup
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /home/scr/app/.env /etc/nginx/sites-available/scr

# Cleanup old backups
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete

# Upload to cloud storage (optional)
# aws s3 sync $BACKUP_DIR s3://your-backup-bucket/scr/
```

### Health Checks

#### Automated Health Check Script
```bash
#!/bin/bash
# Health check script

# Check application health
APP_HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$APP_HEALTH" != "healthy" ]; then
    echo "Application health check failed"
    # Send alert
fi

# Check database connectivity
DB_CHECK=$(python -c "
from database.manager import get_db_manager
try:
    db = get_db_manager()
    with db.get_session() as session:
        session.execute('SELECT 1')
    print('healthy')
except:
    print('unhealthy')
")

if [ "$DB_CHECK" != "healthy" ]; then
    echo "Database health check failed"
    # Send alert
fi

# Check Redis connectivity
REDIS_CHECK=$(redis-cli ping)
if [ "$REDIS_CHECK" != "PONG" ]; then
    echo "Redis health check failed"
    # Send alert
fi
```

### Troubleshooting Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -20

# Check for memory leaks
cat /proc/meminfo
sudo dmesg | grep -i "killed process"

# Solutions:
# - Restart application: sudo systemctl restart scr
# - Adjust cache settings: reduce CACHE_MAX_MEMORY_ITEMS
# - Scale horizontally: add more instances
```

#### High CPU Usage
```bash
# Check CPU usage
top -p $(pgrep -f "python main.py")
iostat 1 5

# Profile application
python -m cProfile -o profile.stats main.py

# Solutions:
# - Optimize code bottlenecks
# - Add more worker processes
# - Scale horizontally
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# Check connection limits
sudo -u postgres psql -c "SHOW max_connections;"
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Solutions:
# - Increase max_connections in postgresql.conf
# - Implement connection pooling
# - Check for long-running queries
```

This deployment guide provides comprehensive instructions for setting up the Supply Chain Risk Intelligence System in various environments. Choose the deployment method that best fits your requirements and infrastructure.
