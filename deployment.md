# VoiceGuardAI Deployment Guide

This document outlines the steps to deploy VoiceGuardAI to a production server.

## Server Requirements

- AWS t2.micro instance or equivalent (1GB RAM minimum)
- Ubuntu 22.04 LTS
- Python 3.9+
- Nginx
- Gunicorn

## Step-by-Step Deployment

### 1. Server Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system dependencies
sudo apt install -y python3-pip python3-dev nginx

# Create application directory
sudo mkdir -p /home/ubuntu/voiceguardai
sudo chown ubuntu:ubuntu /home/ubuntu/voiceguardai
```

### 2. Upload Application Files

From your local machine:

```bash
# Option 1: Upload directories individually
scp -r static/ ubuntu@your-server-ip:/home/ubuntu/voiceguardai/
scp -r api/ ubuntu@your-server-ip:/home/ubuntu/voiceguardai/
scp voiceguard_model.pth ubuntu@your-server-ip:/home/ubuntu/voiceguardai/

# Option 2: Upload as zip archive
zip -r voiceguardai_deploy.zip static/ api/ voiceguard_model.pth requirements.txt
scp voiceguardai_deploy.zip ubuntu@your-server-ip:/home/ubuntu/voiceguardai/
```

On the server:

```bash
# If using zip method
cd /home/ubuntu/voiceguardai
unzip voiceguardai_deploy.zip

# Create temp directory
mkdir -p /home/ubuntu/voiceguardai/temp
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
cd /home/ubuntu/voiceguardai
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn
```

### 4. Configure Gunicorn Service

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/voiceguard.service
```

Add the following content:

```ini
[Unit]
Description=VoiceGuard Gunicorn Service
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/voiceguardai
Environment="PATH=/home/ubuntu/voiceguardai/venv/bin"
ExecStart=/home/ubuntu/voiceguardai/venv/bin/gunicorn -w 2 -k uvicorn.workers.UvicornWorker api.web_test:app -b 0.0.0.0:8009

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable voiceguard
sudo systemctl start voiceguard
```

### 5. Configure Nginx

Create an Nginx site configuration:

```bash
sudo nano /etc/nginx/sites-available/voiceguard
```

Add the following content:

```nginx
server {
    listen 80;
    server_name voiceguard.ai www.voiceguard.ai;

    location / {
        proxy_pass http://localhost:8009;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    client_max_body_size 10M;
}
```

Enable the site and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/voiceguard /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### 6. Set Up SSL (Optional but Recommended)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d voiceguard.ai -d www.voiceguard.ai
```

### 7. Validation

Test that everything is working:

```bash
# Check Gunicorn service status
sudo systemctl status voiceguard

# Check Nginx status
sudo systemctl status nginx

# Test API endpoints
curl http://localhost:8009/api/recent
```

Visit your domain to verify the site is working properly.

## Backup and Rollback Procedure

### Creating Backups

```bash
cd /home/ubuntu
mkdir -p backups
tar -czvf backups/voiceguardai_$(date +%F).tar.gz /home/ubuntu/voiceguardai
```

### Rolling Back

```bash
# Stop services
sudo systemctl stop voiceguard

# Restore from backup
cd /home/ubuntu
rm -rf voiceguardai
mkdir voiceguardai
tar -xzvf backups/voiceguardai_YYYY-MM-DD.tar.gz -C /

# Restart services
sudo systemctl start voiceguard
sudo systemctl restart nginx
```

## Monitoring

Monitor server resources:

```bash
# Check memory usage
free -h

# Check disk usage
df -h

# Monitor application logs
sudo journalctl -u voiceguard -f
```
