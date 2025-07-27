# Runbook: Application Down

## Alert: ApplicationDown

**Severity:** Critical  
**Service:** claude-manager  

## Description

The Claude Manager application is not responding to health checks and appears to be down.

## Immediate Response (< 5 minutes)

### 1. Check Application Status
```bash
# Check if containers are running
docker ps | grep claude-manager

# Check application logs
docker logs claude-manager-app --tail=100

# Check health endpoint directly
curl -f http://localhost:5000/health
```

### 2. Quick Diagnostics
```bash
# Check system resources
docker stats

# Check if port is bound
netstat -tlnp | grep 5000

# Check recent system events
journalctl -u docker --since "10 minutes ago"
```

### 3. Immediate Recovery Actions

#### Option A: Restart Application Container
```bash
docker restart claude-manager-app
```

#### Option B: Restart Full Stack
```bash
docker-compose restart
```

#### Option C: Emergency Rebuild
```bash
docker-compose down
docker-compose up -d --force-recreate
```

## Investigation (5-30 minutes)

### 1. Log Analysis
```bash
# Application logs
docker logs claude-manager-app --since="1h" | grep -i error

# System logs
journalctl -xe --since="1h"

# Database logs
docker logs claude-manager-postgres --since="1h"

# Redis logs
docker logs claude-manager-redis --since="1h"
```

### 2. Resource Investigation
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check system load
uptime

# Check for OOM kills
dmesg | grep -i "killed process"
```

### 3. Network Connectivity
```bash
# Check if database is reachable
docker exec claude-manager-app pg_isready -h postgres -p 5432

# Check if Redis is reachable
docker exec claude-manager-app redis-cli -h redis ping

# Check GitHub API connectivity
curl -I https://api.github.com
```

## Root Cause Analysis

### Common Causes

1. **Out of Memory (OOM)**
   - Check: `dmesg | grep -i "out of memory"`
   - Solution: Increase memory limits or optimize application

2. **Database Connection Issues**
   - Check: Database logs and connection pool status
   - Solution: Restart database or fix connection configuration

3. **Configuration Errors**
   - Check: Environment variables and configuration files
   - Solution: Fix configuration and restart

4. **Resource Exhaustion**
   - Check: CPU, memory, disk, and network usage
   - Solution: Scale resources or optimize performance

5. **Dependency Failures**
   - Check: External service connectivity (GitHub API, etc.)
   - Solution: Wait for service recovery or implement fallbacks

### Detailed Investigation

```bash
# Check application health status
curl http://localhost:5000/health | jq '.'

# Check application configuration
docker exec claude-manager-app env | grep -E "(GITHUB|DATABASE|REDIS)"

# Check recent deployments
git log --oneline -10

# Check for recent configuration changes
git diff HEAD~5 -- config.json .env*
```

## Recovery Procedures

### 1. Standard Recovery
```bash
# Stop services gracefully
docker-compose down --timeout 30

# Clear any problematic volumes if needed
docker volume prune

# Start services
docker-compose up -d

# Verify recovery
make health
```

### 2. Database Recovery
```bash
# If database corruption suspected
docker-compose stop claude-manager
docker exec claude-manager-postgres pg_dumpall > emergency_backup.sql

# Restore from latest backup
make restore BACKUP_FILE=backups/latest.sql

# Restart application
docker-compose start claude-manager
```

### 3. Complete Environment Reset
```bash
# Emergency full reset (last resort)
make emergency-stop
make clean-docker
make deploy
```

## Prevention

### 1. Monitoring Improvements
- Increase health check frequency
- Add more granular metrics
- Set up predictive alerting

### 2. Resource Management
- Implement resource limits
- Add auto-scaling policies
- Monitor resource trends

### 3. Dependency Management
- Implement circuit breakers
- Add retry mechanisms
- Create fallback procedures

## Communication

### 1. Incident Response
- Notify stakeholders immediately
- Post status updates every 15 minutes
- Document actions taken

### 2. Status Page Template
```
🔴 INCIDENT: Claude Manager Application Down
Started: [TIMESTAMP]
Impact: All application functionality unavailable
Status: Investigating | Identified | Monitoring | Resolved

Actions Taken:
- [TIMESTAMP] Investigating application logs
- [TIMESTAMP] Restarted application container
- [TIMESTAMP] Verified database connectivity

Next Steps:
- [Expected action and timeline]
```

## Post-Incident

### 1. Service Recovery Verification
```bash
# Comprehensive health check
make health

# Verify all endpoints
curl http://localhost:5000/health
curl http://localhost:5000/metrics
curl http://localhost:5000/ready

# Run smoke tests
make test-e2e
```

### 2. Post-Mortem Actions
- Document root cause
- Update monitoring and alerting
- Implement prevention measures
- Schedule follow-up review

## Contact Information

- **Primary On-Call:** [Team Lead]
- **Secondary On-Call:** [Senior Developer]
- **Escalation:** [Engineering Manager]
- **Status Page:** https://status.claude-manager.com

## Related Runbooks

- [Database Connection Issues](./database-issues.md)
- [High Memory Usage](./high-memory-usage.md)
- [Performance Degradation](./performance-issues.md)
- [GitHub API Issues](./github-api-issues.md)