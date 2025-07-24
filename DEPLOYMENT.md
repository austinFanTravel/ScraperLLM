# ScraperLLM Deployment Guide

This guide will help you deploy ScraperLLM to a production environment.

## Prerequisites

- Docker and Docker Compose
- Domain name (recommended)
- SSL certificates (can be obtained with Let's Encrypt)
- At least 2GB of RAM and 2 CPU cores recommended

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/scraperllm.git
   cd scraperllm
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   nano .env  # Update with your configuration
   ```

3. **Set up SSL certificates**
   ```bash
   mkdir -p nginx/ssl/your-domain.com
   # Place your SSL certificates in nginx/ssl/your-domain.com/
   # or use Let's Encrypt (see SSL Setup section)
   ```

4. **Build and start the services**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

5. **Verify the installation**
   ```bash
   ./healthcheck.sh
   ```

## Detailed Setup

### 1. Server Requirements

- Linux server (Ubuntu 20.04/22.04 recommended)
- Docker 20.10+ and Docker Compose 1.29+
- At least 2GB RAM (4GB+ recommended for production)
- At least 10GB free disk space

### 2. Environment Configuration

Edit the `.env` file with your configuration:

```bash
# Application Settings
APP_NAME=ScraperLLM
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO

# Security
SECRET_KEY=generate-a-secure-key-here
API_KEYS=your-api-key-1,your-api-key-2

# Model Configuration
MODEL_NAME=all-MiniLM-L6-v2
DATA_DIR=/app/data/search_assistant

# Nginx
NGINX_HOST=your-domain.com
NGINX_PORT=80
NGINX_SSL_PORT=443

# Docker
COMPOSE_PROJECT_NAME=scraperllm

# Development (set to False in production)
DEVELOPMENT=False
```

### 3. SSL Setup (Let's Encrypt)

1. Stop Nginx:
   ```bash
   docker-compose stop nginx
   ```

2. Install Certbot:
   ```bash
   sudo apt update
   sudo apt install -y certbot python3-certbot-nginx
   ```

3. Get certificates:
   ```bash
   sudo certbot certonly --standalone -d your-domain.com -d www.your-domain.com
   ```

4. Copy certificates:
   ```bash
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/your-domain.com/
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/your-domain.com/
   ```

5. Set proper permissions:
   ```bash
   sudo chown -R $USER:$USER nginx/ssl
   chmod 600 nginx/ssl/your-domain.com/privkey.pem
   ```

6. Restart services:
   ```bash
   ./deploy.sh
   ```

### 4. Configure Nginx

Update `nginx/nginx.conf` with your domain name:

```nginx
server_name your-domain.com www.your-domain.com;
```

### 5. Start the Services

```bash
./deploy.sh
```

## Maintenance

### Updating

To update to the latest version:

```bash
git pull origin main
./deploy.sh
```

### Backups

Create a backup:

```bash
./backup.sh
```

This will create timestamped backups in the `backups/` directory.

### Monitoring

Check service health:

```bash
./healthcheck.sh
```

View logs:

```bash
docker-compose logs -f
```

## Security Considerations

1. **Firewall**
   ```bash
   sudo ufw allow ssh
   sudo ufw allow http
   sudo ufw allow https
   sudo ufw enable
   ```

2. **SSH Security**
   - Use SSH keys instead of passwords
   - Disable root login
   - Change default SSH port

3. **Regular Updates**
   ```bash
   sudo apt update && sudo apt upgrade -y
   docker-compose pull
   ```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Ensure ports 80 and 443 are not in use by other services
   - Check with: `sudo lsof -i :80` and `sudo lsof -i :443`

2. **Permission Issues**
   ```bash
   sudo chown -R $USER:$USER .
   chmod -R 755 .
   ```

3. **Docker Issues**
   ```bash
   # Check container status
   docker ps -a
   
   # View logs
   docker-compose logs -f
   
   # Rebuild containers
   docker-compose up -d --build
   ```

## Support

For support, please open an issue on GitHub or contact your system administrator.

## License

[Your License Here]
