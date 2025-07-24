# CPUC RAG System - Server Migration Guide

## Overview
This guide will help you migrate your entire CPUC RAG system from your laptop to an external server for hosting. This is designed as a simple MVP deployment, not a scalable production solution.

## Project Analysis
Your project contains:
- **Streamlit web application** (`app.py`)
- **Document processing system** (Docling, ChromaDB)
- **PDF scraping tools** (Selenium-based)
- **1,100+ processed documents** in vector database
- **Authentication system** with OAuth
- **Multiple Python dependencies** including ML models

## Recommended Hosting Options (Simple → Advanced)

### Option 1: DigitalOcean Droplet (Recommended for MVP)
**Cost**: ~$24/month for adequate specs
**Pros**: Simple, affordable, good performance
**Cons**: Manual setup required

### Option 2: AWS EC2 (Alternative)
**Cost**: ~$30-50/month for comparable specs
**Pros**: More features, better ecosystem
**Cons**: Steeper learning curve

### Option 3: Google Cloud Compute Engine
**Cost**: ~$25-40/month
**Pros**: Good for ML workloads
**Cons**: Complex pricing

## Step-by-Step Migration Process

### Step 1: Choose Server Specifications

**Minimum Requirements for CPUC RAG:**
- **CPU**: 4 cores (for Docling processing)
- **RAM**: 8GB (for ChromaDB + ML models)
- **Storage**: 50GB SSD (for documents + vector DB)
- **OS**: Ubuntu 22.04 LTS

**Recommended DigitalOcean Droplet**: "Basic Premium AMD - 4 vCPUs, 8GB RAM, 160GB SSD" (~$48/month)

### Step 2: Server Setup

#### 2.1 Create Server
1. Go to [DigitalOcean](https://digitalocean.com)
2. Create account
3. Create new Droplet:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic Premium AMD (4 vCPUs, 8GB RAM)
   - **Datacenter**: Choose closest to your users
   - **Authentication**: SSH keys (recommended) or password
   - **Hostname**: `cpuc-rag-server`

#### 2.2 Initial Server Configuration
```bash
# Connect to your server
ssh root@your_server_ip

# Update system
apt update && apt upgrade -y

# Install essential packages
apt install -y python3.11 python3.11-venv python3-pip git curl wget htop

# Install Chrome for Selenium (needed for scraping)
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
apt update
apt install -y google-chrome-stable

# Install ChromeDriver
CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | cut -d. -f1)
wget -O /tmp/chromedriver.zip "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION}/chromedriver_linux64.zip"
unzip /tmp/chromedriver.zip -d /usr/local/bin/
chmod +x /usr/local/bin/chromedriver

# Create app user (security best practice)
useradd -m -s /bin/bash cpucrag
usermod -aG sudo cpucrag
```

### Step 3: Transfer Files to Server

#### 3.1 Prepare Your Project for Transfer
```bash
# On your laptop, create a clean version without unnecessary files
cd /Users/anthony.liu/Downloads/CPUC_REG_RAG

# Create transfer package (exclude logs, temp files)
tar -czf cpuc_rag_system.tar.gz \
  --exclude='*.log' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='docling_performance_comparison_*.json' \
  .
```

#### 3.2 Transfer Files
```bash
# Transfer the package to server
scp cpuc_rag_system.tar.gz root@your_server_ip:/home/cpucrag/

# On server, extract files
ssh root@your_server_ip
su - cpucrag
tar -xzf cpuc_rag_system.tar.gz
rm cpuc_rag_system.tar.gz
```

### Step 4: Environment Setup on Server

#### 4.1 Create Python Environment
```bash
# As cpucrag user
cd /home/cpucrag
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### 4.2 Install Dependencies
```bash
# Install requirements
pip install -r requirements.txt
pip install -r requirements_auth.txt

# Install additional system dependencies for Docling
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Configure Application for Server

#### 5.1 Update Configuration
```bash
# Edit config.py for server environment
nano config.py
```

**Add these server-specific settings:**
```python
# Server-specific configuration
import os

# Set server mode
SERVER_MODE = True
DEBUG = False

# Streamlit configuration for server
STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
STREAMLIT_SERVER_PORT = 8501

# File paths (use absolute paths)
PROJECT_ROOT = Path("/home/cpucrag")

# Chrome options for headless server operation
CHROME_OPTIONS = [
    "--headless",
    "--no-sandbox", 
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--remote-debugging-port=9222"
]
```

#### 5.2 Create Environment Variables
```bash
# Create .env file for sensitive data
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_OAUTH_CLIENT_ID=your_google_oauth_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_google_oauth_client_secret
SERVER_MODE=true
EOF

# Secure the environment file
chmod 600 .env
```

### Step 6: Configure Firewall and Security

```bash
# As root user
# Configure UFW firewall
ufw enable
ufw allow ssh
ufw allow 8501/tcp  # Streamlit port
ufw status
```

### Step 7: Create System Services

#### 7.1 Create Systemd Service for Streamlit
```bash
# As root, create service file
cat > /etc/systemd/system/cpuc-rag.service << EOF
[Unit]
Description=CPUC RAG Streamlit Application
After=network.target

[Service]
Type=simple
User=cpucrag
WorkingDirectory=/home/cpucrag
Environment=PATH=/home/cpucrag/venv/bin
ExecStart=/home/cpucrag/venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable cpuc-rag.service
systemctl start cpuc-rag.service
```

### Step 8: Setup Reverse Proxy (Optional but Recommended)

#### 8.1 Install and Configure Nginx
```bash
# Install Nginx
apt install -y nginx

# Create Nginx configuration
cat > /etc/nginx/sites-available/cpuc-rag << EOF
server {
    listen 80;
    server_name your_domain_or_ip;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Enable the site
ln -s /etc/nginx/sites-available/cpuc-rag /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
```

### Step 9: Test Deployment

#### 9.1 Verify Services
```bash
# Check if services are running
systemctl status cpuc-rag.service
systemctl status nginx

# Check if application is accessible
curl http://localhost:8501
```

#### 9.2 Test Application Features
1. **Web Interface**: Visit `http://your_server_ip:8501`
2. **Document Search**: Test a simple query
3. **Authentication**: Test login functionality
4. **Document Processing**: Run a small test batch

### Step 10: Create Backup and Monitoring Scripts

#### 10.1 Create Backup Script
```bash
cat > /home/cpucrag/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/cpucrag/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup vector database
tar -czf "$BACKUP_DIR/vector_db_$DATE.tar.gz" local_chroma_db/

# Backup documents
tar -czf "$BACKUP_DIR/documents_$DATE.tar.gz" cpuc_proceedings/

# Backup configuration
cp config.py .env "$BACKUP_DIR/"

# Keep only last 7 days of backups
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /home/cpucrag/backup.sh

# Add to crontab for daily backups
crontab -e
# Add: 0 2 * * * /home/cpucrag/backup.sh >> /home/cpucrag/backup.log 2>&1
```

## Cost Estimation

### Monthly Costs:
- **DigitalOcean Droplet**: $48/month (4 vCPUs, 8GB RAM)
- **Storage**: Included in droplet
- **Bandwidth**: 8TB included (sufficient for MVP)
- **Domain** (optional): $12/year

**Total Monthly Cost**: ~$48-52

### Comparison with Laptop:
- **Electricity**: $0 (no laptop running 24/7)
- **Internet**: No additional load on home internet
- **Reliability**: 99.9% uptime vs laptop availability
- **Performance**: Dedicated resources, faster processing

## Maintenance Tasks

### Daily:
- Monitor application logs: `tail -f /var/log/syslog | grep cpuc-rag`

### Weekly:
- Check disk space: `df -h`
- Review system updates: `apt list --upgradable`

### Monthly:
- Update system packages: `apt update && apt upgrade`
- Review and rotate logs
- Check backup integrity

## Troubleshooting Common Issues

### Issue 1: Application Won't Start
```bash
# Check service status
systemctl status cpuc-rag.service

# Check logs
journalctl -u cpuc-rag.service -f

# Restart service
systemctl restart cpuc-rag.service
```

### Issue 2: High Memory Usage
```bash
# Monitor memory
htop

# Restart application if needed
systemctl restart cpuc-rag.service
```

### Issue 3: ChromeDriver Issues
```bash
# Update ChromeDriver
which chromedriver
google-chrome --version
# Download matching version from ChromeDriver releases
```

## Security Considerations

1. **Change default passwords** for all services
2. **Set up SSH key authentication** (disable password auth)
3. **Configure UFW firewall** (only open necessary ports)
4. **Regular security updates** (unattended-upgrades)
5. **Monitor access logs** regularly
6. **Use environment variables** for sensitive data

## Next Steps After Migration

1. **Domain Setup**: Point a domain name to your server IP
2. **SSL Certificate**: Use Let's Encrypt for HTTPS
3. **Monitoring**: Set up monitoring and alerting
4. **Automated Deployment**: Create deployment scripts for updates
5. **Database Backups**: Implement automated backups to cloud storage

## Quick Commands Reference

```bash
# Start/stop application
systemctl start cpuc-rag.service
systemctl stop cpuc-rag.service
systemctl restart cpuc-rag.service

# View logs
journalctl -u cpuc-rag.service -f

# Update application
cd /home/cpucrag
git pull  # if using git
systemctl restart cpuc-rag.service

# Monitor resources
htop
df -h
free -h
```

This migration approach prioritizes simplicity and cost-effectiveness for an MVP deployment while maintaining the ability to scale later if needed.