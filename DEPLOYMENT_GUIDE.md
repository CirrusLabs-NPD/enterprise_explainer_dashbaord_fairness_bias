# ML Explainer Dashboard - Deployment Guide

This guide covers deployment options for the ML Explainer Dashboard in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Environment Configuration](#environment-configuration)
- [Database Setup](#database-setup)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **CPU**: 2+ cores (4+ recommended for production)
- **Memory**: 4GB+ (8GB+ recommended for production)
- **Storage**: 10GB+ free space
- **Network**: Internet connection for package installation

### Required Software

- **Node.js**: 18.0+ (for frontend)
- **Python**: 3.11+ (for backend)
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for multi-container setup)
- **PostgreSQL**: 15+ (for production database)
- **Redis**: 7+ (for caching)

## Local Development

### 1. Clone Repository

```bash
git clone <repository-url>
cd explainer_dashboard
```

### 2. Setup Environment

```bash
# Copy environment file
cp .env.development .env

# Create data directories
mkdir -p dev_data/uploads dev_data/models
```

### 3. Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### 4. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start development server
python app/main.py
```

The backend will be available at `http://localhost:8000`.

### 5. Run Tests

```bash
# Frontend tests
npm run test

# Backend tests
cd backend
pytest
```

## Docker Deployment

### 1. Quick Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services will be available at:
- Application: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001` (admin/admin123)

### 2. Build Custom Images

```bash
# Build application image
docker build -t ml-explainer-dashboard .

# Run with custom configuration
docker run -d \
  --name ml-explainer \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_URL=redis://host:6379 \
  ml-explainer-dashboard
```

### 3. Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'
services:
  app:
    volumes:
      - ./local-data:/app/data
    environment:
      - DEBUG=true
    ports:
      - "8001:8000"  # Use different port
```

## Production Deployment

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin

# Create deployment user
sudo useradd -m -s /bin/bash ml-deploy
sudo usermod -aG docker ml-deploy
```

### 2. Environment Setup

```bash
# Create production directories
sudo mkdir -p /opt/ml-explainer/{data,logs,config}
sudo chown -R ml-deploy:ml-deploy /opt/ml-explainer

# Copy environment file
cp .env.production /opt/ml-explainer/.env

# Update environment variables
nano /opt/ml-explainer/.env
```

### 3. SSL/TLS Setup

```bash
# Install Certbot
sudo apt install certbot

# Generate SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Create nginx configuration
sudo nano /etc/nginx/sites-available/ml-explainer
```

Nginx configuration:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 4. Database Setup

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
```

```sql
CREATE DATABASE ml_explainer;
CREATE USER ml_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ml_explainer TO ml_user;
\q
```

### 5. Deploy Application

```bash
# Navigate to deployment directory
cd /opt/ml-explainer

# Pull latest code
git pull origin main

# Start services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
docker-compose logs app
```

### 6. Setup Monitoring

```bash
# Configure Prometheus
sudo mkdir -p /opt/prometheus/data
sudo chown 65534:65534 /opt/prometheus/data

# Configure Grafana
sudo mkdir -p /opt/grafana/data
sudo chown 472:472 /opt/grafana/data

# Import dashboards
curl -X POST \
  http://admin:admin123@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/ml-explainer.json
```

## Environment Configuration

### Required Environment Variables

```bash
# Application
DEBUG=false
SECRET_KEY=your-256-bit-secret-key
DATABASE_URL=postgresql://user:pass@host:5432/db_name
REDIS_URL=redis://host:6379

# Security
CORS_ORIGINS=["https://your-domain.com"]
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
PROMETHEUS_GATEWAY=http://prometheus:9090
LOG_LEVEL=INFO

# Email notifications
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your-email@domain.com
SMTP_PASSWORD=your-app-password
```

### Optional Configuration

```bash
# Cloud storage
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=ml-explainer-models

# Slack integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#ml-alerts

# Feature flags
ENABLE_ADVANCED_EXPLANATIONS=true
ENABLE_DRIFT_DETECTION=true
ENABLE_REAL_TIME_MONITORING=true
```

## Database Setup

### 1. PostgreSQL Configuration

```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/15/main/postgresql.conf
```

Recommended settings:
```
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 4MB

# Connection settings
max_connections = 100
```

### 2. Database Initialization

```bash
# Run migrations
docker-compose exec app python -c "
from app.core.database import init_db
import asyncio
asyncio.run(init_db())
"
```

### 3. Backup Setup

```bash
# Create backup script
sudo nano /opt/ml-explainer/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/opt/ml-explainer/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ml_explainer_$DATE.sql"

mkdir -p $BACKUP_DIR
pg_dump -h localhost -U ml_user ml_explainer > $BACKUP_FILE
gzip $BACKUP_FILE

# Keep only last 7 days
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
```

```bash
# Make executable and add to cron
chmod +x /opt/ml-explainer/backup.sh
sudo crontab -e
# Add: 0 2 * * * /opt/ml-explainer/backup.sh
```

## Monitoring and Logging

### 1. Log Aggregation

```bash
# Install Filebeat for log shipping
curl -L -O https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.8.0-amd64.deb
sudo dpkg -i filebeat-8.8.0-amd64.deb

# Configure Filebeat
sudo nano /etc/filebeat/filebeat.yml
```

### 2. Metrics Collection

Prometheus targets:
- Application metrics: `http://app:8000/metrics`
- System metrics: `http://node-exporter:9100/metrics`
- Database metrics: `http://postgres-exporter:9187/metrics`

### 3. Alerting Rules

Create `/opt/ml-explainer/monitoring/rules/alerts.yml`:

```yaml
groups:
  - name: ml-explainer-alerts
    rules:
      - alert: HighCPUUsage
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High CPU usage detected
          
      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.8
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: Model accuracy below threshold
```

## Security Considerations

### 1. Network Security

```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8000/tcp   # Block direct app access
```

### 2. Application Security

- Use strong, unique secret keys
- Enable HTTPS/TLS encryption
- Implement rate limiting
- Regular security updates
- Audit logs monitoring

### 3. Database Security

```bash
# Secure PostgreSQL
sudo nano /etc/postgresql/15/main/pg_hba.conf
```

```
# Restrict connections
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             ml_user         10.0.0.0/8              md5
```

### 4. Container Security

```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ml-explainer-dashboard:latest

# Use non-root user in containers
# Already configured in Dockerfile
```

## Troubleshooting

### Common Issues

1. **Application won't start**
   ```bash
   # Check logs
   docker-compose logs app
   
   # Verify environment variables
   docker-compose exec app env | grep -E "(DATABASE|REDIS)"
   
   # Test database connection
   docker-compose exec app python -c "
   from app.core.database import check_database_health
   import asyncio
   print(asyncio.run(check_database_health()))
   "
   ```

2. **Database connection issues**
   ```bash
   # Test PostgreSQL connectivity
   docker-compose exec postgres pg_isready
   
   # Check PostgreSQL logs
   docker-compose logs postgres
   
   # Verify credentials
   docker-compose exec postgres psql -U ml_user -d ml_explainer -c "\l"
   ```

3. **Performance issues**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check application metrics
   curl http://localhost:8000/metrics
   
   # Analyze slow queries
   docker-compose exec postgres psql -U ml_user -d ml_explainer -c "
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC LIMIT 10;
   "
   ```

4. **Memory issues**
   ```bash
   # Increase memory limits in docker-compose.yml
   services:
     app:
       deploy:
         resources:
           limits:
             memory: 2G
           reservations:
             memory: 1G
   ```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
docker-compose exec app python -c "
from app.core.database import check_database_health
import asyncio
print(asyncio.run(check_database_health()))
"

# System health
curl http://localhost:9100/metrics | grep cpu
```

### Support

For additional support:
1. Check application logs: `docker-compose logs app`
2. Review monitoring dashboards in Grafana
3. Consult the application documentation
4. Contact the development team

## Maintenance

### Regular Tasks

1. **Weekly**
   - Review monitoring dashboards
   - Check application logs for errors
   - Verify backup integrity

2. **Monthly**
   - Update system packages
   - Review and rotate logs
   - Performance optimization

3. **Quarterly**
   - Security audit
   - Dependency updates
   - Disaster recovery testing

### Scaling

For horizontal scaling:
1. Use a load balancer (nginx, HAProxy)
2. Deploy multiple application instances
3. Use external PostgreSQL and Redis clusters
4. Implement shared storage for models