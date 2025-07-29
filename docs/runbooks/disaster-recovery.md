# Disaster Recovery Runbook 

## Overview

This runbook provides comprehensive procedures for recovering from various disaster scenarios affecting the Claude Code Manager system. It covers data backup, system restoration, and business continuity procedures.

## Emergency Contacts

- **Primary On-Call**: [Contact Information]
- **Secondary On-Call**: [Contact Information]  
- **Escalation Manager**: [Contact Information]
- **External Vendor Support**: [Contact Information]

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Component | RTO Target | RPO Target | Priority |
|-----------|------------|------------|----------|
| API Service | 1 hour | 15 minutes | Critical |
| Task Database | 2 hours | 5 minutes | Critical |
| GitHub Integration | 30 minutes | 5 minutes | High |
| Monitoring/Metrics | 4 hours | 1 hour | Medium |
| Documentation | 24 hours | 24 hours | Low |

## Pre-Disaster Preparation

### Automated Backups

1. **Database Backups** (Daily at 2 AM UTC)
   ```bash
   # Verify backup schedule
   docker exec claude-manager-postgres pg_dump -U claude_user claude_manager > /backups/postgres-$(date +%Y%m%d).sql
   
   # Compress and store
   gzip /backups/postgres-$(date +%Y%m%d).sql
   aws s3 cp /backups/postgres-$(date +%Y%m%d).sql.gz s3://claude-manager-backups/postgres/
   ```

2. **Application Data Backups**
   ```bash
   # Backup application data directory
   tar -czf /backups/data-$(date +%Y%m%d).tar.gz /app/data/
   aws s3 cp /backups/data-$(date +%Y%m%d).tar.gz s3://claude-manager-backups/data/
   ```

3. **Configuration Backups**
   ```bash
   # Backup configurations
   tar -czf /backups/config-$(date +%Y%m%d).tar.gz /app/config/ /app/.env /app/docker-compose.yml
   aws s3 cp /backups/config-$(date +%Y%m%d).tar.gz s3://claude-manager-backups/config/
   ```

### Backup Verification

Weekly backup verification process:

```bash
# Test database backup restoration
./scripts/test-backup-restore.sh --type=postgres --backup-date=$(date -d '1 day ago' +%Y%m%d)

# Verify data integrity
python scripts/verify-backup-integrity.py --backup-path=/backups/
```

## Disaster Scenarios and Recovery Procedures

### Scenario 1: Complete System Failure

**Symptoms:**
- Application completely unresponsive
- All services down
- Infrastructure unavailable

**Impact Assessment:**
- Service downtime: Complete outage
- Data loss risk: High without proper backups
- Business impact: Critical

**Recovery Steps:**

1. **Immediate Response (0-15 minutes)**
   ```bash
   # Assess system status
   curl -f https://claude-manager.example.com/health || echo "System Down"
   
   # Check infrastructure status
   docker ps -a
   docker-compose logs --tail=50
   
   # Activate incident response
   echo "INCIDENT: Complete system failure at $(date)" | mail -s "CRITICAL: System Down" oncall@company.com
   ```

2. **Infrastructure Recovery (15-45 minutes)**
   ```bash
   # Stop all services
   docker-compose down
   
   # Clean up corrupted containers/volumes
   docker system prune -f
   docker volume prune -f
   
   # Restore from backup
   ./scripts/restore-full-system.sh --backup-date=latest
   ```

3. **Service Restoration (45-60 minutes)**
   ```bash
   # Start core services
   docker-compose up -d postgres redis
   
   # Wait for database initialization
   ./scripts/wait-for-postgres.sh
   
   # Start application services
   docker-compose up -d claude-manager
   
   # Verify service health
   ./scripts/health-check.sh
   ```

### Scenario 2: Database Corruption/Loss

**Symptoms:**
- Database connection errors
- Data inconsistency
- Database container failing to start

**Recovery Steps:**

1. **Stop Application Services**
   ```bash
   docker-compose stop claude-manager
   ```

2. **Assess Database Damage**
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Try to connect to database
   docker exec -it claude-manager-postgres psql -U claude_user -d claude_manager
   ```

3. **Restore Database**
   ```bash
   # Stop database service
   docker-compose stop postgres
   
   # Remove corrupted volume
   docker volume rm claude-manager_postgres_data
   
   # Start fresh database
   docker-compose up -d postgres
   
   # Restore from latest backup
   ./scripts/restore-database.sh --backup-file=s3://claude-manager-backups/postgres/postgres-$(date +%Y%m%d).sql.gz
   ```

4. **Verification**
   ```bash
   # Verify data integrity
   python scripts/verify-database-integrity.py
   
   # Test application functionality
   curl -f http://localhost:5000/api/v1/tasks
   ```

### Scenario 3: GitHub API Integration Failure

**Symptoms:**
- Unable to create GitHub issues
- Authentication errors with GitHub
- GitHub API rate limit exceeded

**Recovery Steps:**

1. **Check GitHub Status**
   ```bash
   curl -f https://api.github.com/status
   curl -f https://www.githubstatus.com/api/v2/status.json
   ```

2. **Verify Credentials**
   ```bash
   # Test GitHub token
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   
   # Check token permissions
   curl -I -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/repos/your-org/your-repo
   ```

3. **Implement Fallback Procedures**
   ```bash
   # Enable offline mode
   export GITHUB_OFFLINE_MODE=true
   docker-compose restart claude-manager
   
   # Queue GitHub operations for later
   python scripts/queue-github-operations.py
   ```

### Scenario 4: Data Center/Cloud Provider Outage

**Symptoms:**
- Complete infrastructure unavailable
- Network connectivity lost
- Cloud provider services down

**Recovery Steps:**

1. **Activate Secondary Region**
   ```bash
   # Switch DNS to backup region
   aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://failover-to-backup.json
   
   # Start services in backup region
   cd /backup-region/claude-manager/
   docker-compose up -d
   ```

2. **Restore Data in Backup Region**
   ```bash
   # Download latest backups
   aws s3 cp s3://claude-manager-backups/postgres/postgres-latest.sql.gz ./
   aws s3 cp s3://claude-manager-backups/data/data-latest.tar.gz ./
   
   # Restore data
   ./scripts/restore-full-system.sh --backup-files=./postgres-latest.sql.gz,./data-latest.tar.gz
   ```

## Recovery Scripts

### Full System Restore Script

Create `/scripts/restore-full-system.sh`:

```bash
#!/bin/bash
set -e

BACKUP_DATE=${1:-latest}
BACKUP_PATH="/tmp/restore-$(date +%s)"

echo "Starting full system restore with backup date: $BACKUP_DATE"

# Create temporary restore directory
mkdir -p "$BACKUP_PATH"

# Download backups from S3
aws s3 cp "s3://claude-manager-backups/postgres/postgres-$BACKUP_DATE.sql.gz" "$BACKUP_PATH/"
aws s3 cp "s3://claude-manager-backups/data/data-$BACKUP_DATE.tar.gz" "$BACKUP_PATH/"
aws s3 cp "s3://claude-manager-backups/config/config-$BACKUP_DATE.tar.gz" "$BACKUP_PATH/"

# Stop all services
docker-compose down

# Restore configurations
tar -xzf "$BACKUP_PATH/config-$BACKUP_DATE.tar.gz" -C /

# Restore data directory
tar -xzf "$BACKUP_PATH/data-$BACKUP_DATE.tar.gz" -C /

# Start database
docker-compose up -d postgres
sleep 30

# Restore database
gunzip < "$BACKUP_PATH/postgres-$BACKUP_DATE.sql.gz" | docker exec -i claude-manager-postgres psql -U claude_user -d claude_manager

# Start all services
docker-compose up -d

# Cleanup
rm -rf "$BACKUP_PATH"

echo "Full system restore completed successfully"
```

### Health Check Script

Create `/scripts/health-check.sh`:

```bash
#!/bin/bash

SERVICES=("postgres" "redis" "claude-manager")
ENDPOINTS=("/health" "/api/v1/tasks" "/metrics")

echo "Performing health checks..."

# Check container status
for service in "${SERVICES[@]}"; do
    if docker-compose ps "$service" | grep -q "Up"; then
        echo "✓ $service container is running"
    else
        echo "✗ $service container is not running"
        exit 1
    fi
done

# Check HTTP endpoints
for endpoint in "${ENDPOINTS[@]}"; do
    if curl -f -s "http://localhost:5000$endpoint" > /dev/null; then
        echo "✓ $endpoint is responding"
    else
        echo "✗ $endpoint is not responding"
        exit 1
    fi
done

echo "All health checks passed"
```

## Post-Recovery Procedures

### Data Validation

After any recovery operation:

1. **Verify Data Integrity**
   ```bash
   python scripts/verify-data-integrity.py --full-check
   ```

2. **Check Recent Transactions**
   ```bash
   # Verify no data loss in recent operations
   python scripts/check-recent-transactions.py --hours=24
   ```

3. **Validate GitHub Integration**
   ```bash
   # Test GitHub operations
   python scripts/test-github-integration.py
   ```

### Performance Monitoring

1. **Monitor System Resources**
   ```bash
   # Check system performance after restore
   docker stats --no-stream
   free -h
   df -h
   ```

2. **Application Performance**
   ```bash
   # Run performance tests
   pytest tests/performance/ --benchmark-only
   ```

### Communication

1. **Update Status Page**
   - Update incident status
   - Provide recovery timeline
   - Document lessons learned

2. **Notify Stakeholders**
   ```bash
   # Send recovery notification
   echo "System recovery completed at $(date). Services are operational." | mail -s "RESOLVED: System Recovery Complete" stakeholders@company.com
   ```

## Monitoring and Alerting

### Recovery Metrics to Monitor

- **Recovery Time**: Time from incident start to full restoration
- **Data Loss**: Amount of data lost during incident
- **Service Availability**: Percentage uptime during recovery
- **Performance Impact**: System performance post-recovery

### Automated Monitoring

```bash
# Monitor backup success
*/30 * * * * /scripts/check-backup-status.sh

# Test restore procedures weekly  
0 3 * * 0 /scripts/test-disaster-recovery.sh

# Monitor disk space for backups
*/15 * * * * /scripts/check-backup-disk-space.sh
```

## Continuous Improvement

### Post-Incident Review

After each disaster recovery event:

1. **Document Timeline**
2. **Identify Root Cause**
3. **Review Response Effectiveness**
4. **Update Procedures**
5. **Plan Prevention Measures**

### Regular Testing

- **Monthly**: Backup restore tests
- **Quarterly**: Full disaster recovery simulation
- **Annually**: Complete runbook review and update

### Training

- **New Team Members**: Disaster recovery orientation
- **Quarterly**: Incident response training
- **Annual**: Full disaster recovery drill

## Contact Information and Resources

### Emergency Escalation

1. **Level 1**: On-call engineer
2. **Level 2**: Technical lead
3. **Level 3**: Engineering manager
4. **Level 4**: CTO/VP Engineering

### External Resources

- **Cloud Provider Support**: [Support phone number]
- **DNS Provider**: [Support contact]
- **Monitoring Service**: [Support contact]

### Documentation References

- [System Architecture Documentation]
- [Backup Strategy Documentation]  
- [Incident Response Procedures]
- [Business Continuity Plan]