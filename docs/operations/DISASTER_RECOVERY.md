# Disaster Recovery Plan

Comprehensive disaster recovery procedures for claude-code-manager.

## Overview

This document outlines the disaster recovery procedures for the claude-code-manager system, including data backup, system restoration, and business continuity measures.

## Recovery Time Objectives (RTO)

| **Severity Level** | **RTO Target** | **Description** |
|-------------------|----------------|-----------------|
| **Critical** | < 1 hour | Complete system outage affecting all users |
| **High** | < 4 hours | Major functionality impacted, partial service |
| **Medium** | < 24 hours | Minor functionality issues, degraded performance |
| **Low** | < 72 hours | Non-critical features affected |

## Recovery Point Objectives (RPO)

| **Data Type** | **RPO Target** | **Backup Frequency** |
|---------------|----------------|---------------------|
| **Task Data** | < 1 hour | Continuous (every 15 minutes) |
| **Configuration** | < 4 hours | Every 4 hours |
| **Logs** | < 24 hours | Daily |
| **Performance Metrics** | < 1 hour | Real-time with 1-hour snapshots |

## Backup Strategy

### Automated Backups

```bash
# Daily automated backup (configured in crontab)
0 2 * * * /path/to/claude-code-manager/scripts/backup-system.sh

# Hourly task data backup
0 * * * * /path/to/claude-code-manager/scripts/backup-data.sh
```

### Manual Backup Procedures

#### 1. Database Backup
```bash
# SQLite backup
cp data/tasks.db backups/tasks-$(date +%Y%m%d_%H%M%S).db

# PostgreSQL backup (if using PostgreSQL)
pg_dump -h localhost -U claude_user claude_manager > backups/postgres-$(date +%Y%m%d_%H%M%S).sql
```

#### 2. Configuration Backup
```bash
# Backup all configuration files
tar -czf backups/config-$(date +%Y%m%d_%H%M%S).tar.gz \
    config.json .env pyproject.toml docker-compose*.yml
```

#### 3. Complete System Backup
```bash
# Full system backup excluding cache and temp files
tar -czf backups/full-system-$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    --exclude='logs/*.log' \
    --exclude='temp/*' \
    .
```

## Recovery Procedures

### Scenario 1: Database Corruption

#### Symptoms
- Application fails to start with database errors
- Data inconsistency or corruption messages
- Unable to read/write task data

#### Recovery Steps
```bash
# 1. Stop the application
docker-compose down

# 2. Restore from latest backup
cp backups/tasks-latest.db data/tasks.db

# 3. Verify database integrity
sqlite3 data/tasks.db "PRAGMA integrity_check;"

# 4. Restart application
docker-compose up -d

# 5. Verify functionality
curl http://localhost:5000/health
```

### Scenario 2: Container Registry Failure

#### Symptoms
- Unable to pull Docker images
- Container startup failures
- Image not found errors

#### Recovery Steps
```bash
# 1. Rebuild images locally
docker build -t claude-code-manager:latest .

# 2. Tag for alternative registry
docker tag claude-code-manager:latest backup-registry/claude-code-manager:latest

# 3. Push to backup registry
docker push backup-registry/claude-code-manager:latest

# 4. Update docker-compose.yml to use backup registry
sed -i 's|ghcr.io/terragon-labs/claude-code-manager|backup-registry/claude-code-manager|g' docker-compose.yml

# 5. Restart services
docker-compose up -d
```

### Scenario 3: Complete System Failure

#### Symptoms
- Server/infrastructure completely unavailable
- All services down
- Network connectivity issues

#### Recovery Steps

##### Phase 1: Infrastructure Recovery (0-30 minutes)
```bash
# 1. Provision new infrastructure
# - Create new VM/container instances
# - Configure networking and security groups
# - Install required software (Docker, Docker Compose, etc.)

# 2. Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

##### Phase 2: System Restoration (30-60 minutes)
```bash
# 1. Clone repository or restore from backup
git clone https://github.com/terragon-labs/claude-code-manager.git
cd claude-code-manager

# 2. Restore configuration
tar -xzf /backups/config-latest.tar.gz

# 3. Restore data
mkdir -p data
cp /backups/tasks-latest.db data/tasks.db

# 4. Start services
docker-compose up -d

# 5. Verify all services are running
docker-compose ps
curl http://localhost:5000/health
```

##### Phase 3: Verification and Monitoring (60+ minutes)
```bash
# 1. Run comprehensive health checks
make health

# 2. Verify data integrity
python scripts/verify-data-integrity.py

# 3. Check all integrations
python scripts/test-integrations.py

# 4. Update DNS/load balancer to point to new instance
# (Infrastructure-specific steps)
```

### Scenario 4: GitHub API Rate Limit Exceeded

#### Symptoms
- 403 Forbidden errors from GitHub API
- Unable to create issues or pull requests
- Rate limit exceeded messages

#### Recovery Steps
```bash
# 1. Check current rate limit status
curl -H "Authorization: token $GITHUB_TOKEN" \
    https://api.github.com/rate_limit

# 2. Implement exponential backoff
# (Already implemented in github_api.py)

# 3. Use alternative GitHub tokens (if available)
export GITHUB_TOKEN_BACKUP=your_backup_token
export GITHUB_TOKEN=$GITHUB_TOKEN_BACKUP

# 4. Enable rate limiting bypass mode
export GITHUB_API_RATE_LIMIT_BYPASS=true

# 5. Restart services
docker-compose restart
```

## Business Continuity

### Communication Plan

#### Internal Team
1. **Primary**: Slack #incident-response channel
2. **Secondary**: Email incident-response@terragon.ai  
3. **Escalation**: Phone/SMS for critical incidents

#### External Users
1. **Status Page**: Update system status at status.terragon.ai
2. **Documentation**: Update README with known issues
3. **GitHub Issues**: Create incident issue for tracking

### Roles and Responsibilities

| **Role** | **Primary** | **Backup** | **Responsibilities** |
|----------|-------------|------------|---------------------|
| **Incident Commander** | DevOps Lead | Engineering Manager | Overall incident response coordination |
| **Technical Lead** | Senior Engineer | Platform Engineer | Technical troubleshooting and resolution |
| **Communications** | Product Manager | Engineering Manager | User communication and status updates |
| **Documentation** | Technical Writer | Senior Engineer | Incident documentation and post-mortem |

## Testing and Validation

### Monthly DR Tests
```bash
# Automated DR test script
#!/bin/bash
# File: scripts/dr-test.sh

echo "Starting Disaster Recovery Test..."

# 1. Create test backup
./scripts/backup-system.sh

# 2. Simulate failure
docker-compose down
mv data/tasks.db data/tasks.db.backup

# 3. Test recovery
cp backups/tasks-latest.db data/tasks.db
docker-compose up -d

# 4. Verify recovery
if curl -f http://localhost:5000/health; then
    echo " DR Test PASSED"
else
    echo "L DR Test FAILED"
    exit 1
fi

# 5. Restore original state
docker-compose down
mv data/tasks.db.backup data/tasks.db
docker-compose up -d
```

### Quarterly Full DR Exercises
- Complete infrastructure failure simulation
- Cross-team coordination testing
- Communication plan validation
- Documentation updates

## Monitoring and Alerting

### Critical System Alerts
```yaml
# Prometheus alerting rules
groups:
  - name: disaster_recovery
    rules:
      - alert: SystemDown
        expr: up{job="claude-manager"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Claude Code Manager is down"
          description: "System has been down for more than 5 minutes"
          
      - alert: DatabaseConnectionFailed
        expr: database_connections_failed > 5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failures detected"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 3m
        labels:
          severity: high
        annotations:
          summary: "High error rate detected"
```

### Backup Monitoring
```bash
# Daily backup verification
#!/bin/bash
# File: scripts/verify-backups.sh

BACKUP_DIR="backups"
TODAY=$(date +%Y%m%d)

# Check if today's backup exists
if [ ! -f "$BACKUP_DIR/tasks-${TODAY}*.db" ]; then
    echo "L ERROR: Today's database backup missing"
    # Send alert
    curl -X POST $SLACK_WEBHOOK_URL -d '{"text":"  Database backup missing for '$TODAY'"}'
    exit 1
fi

# Verify backup integrity
sqlite3 "$BACKUP_DIR/tasks-${TODAY}"*.db "PRAGMA integrity_check;" | grep -q "ok"
if [ $? -eq 0 ]; then
    echo " Backup integrity verified"
else
    echo "L ERROR: Backup integrity check failed"
    exit 1
fi
```

## Contact Information

### Emergency Contacts
- **On-Call Engineer**: +1-555-0123 (24/7)
- **Infrastructure Team**: infrastructure@terragon.ai
- **Security Team**: security@terragon.ai

### Vendor Contacts
- **Cloud Provider Support**: [Provider-specific contact]
- **DNS Provider**: support@cloudflare.com
- **Monitoring Service**: support@datadog.com

## Post-Incident Procedures

### Immediate Actions (0-24 hours)
1. Conduct incident review meeting
2. Document timeline and actions taken
3. Identify root cause
4. Update monitoring/alerting if needed

### Follow-up Actions (1-7 days)
1. Write detailed post-mortem report
2. Implement preventive measures
3. Update disaster recovery procedures
4. Schedule follow-up testing

### Long-term Actions (1-4 weeks)
1. Review and update RTO/RPO targets
2. Conduct team training on new procedures
3. Update infrastructure automation
4. Schedule next DR exercise

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-31  
**Next Review**: 2025-10-31  
**Owner**: DevOps Team