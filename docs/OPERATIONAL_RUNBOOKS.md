# Operational Runbooks

This document provides step-by-step procedures for common operational scenarios and emergency situations.

## 🚨 Emergency Procedures

### Application Down
See [application-down.md](runbooks/application-down.md) for detailed steps.

**Quick Response:**
1. Check application health: `curl -f http://localhost:5000/health`
2. Check container status: `docker-compose ps`
3. Check logs: `docker-compose logs claude-manager`
4. Restart if needed: `make restart`

### High CPU/Memory Usage
1. **Identify Resource Usage:**
   ```bash
   make top
   docker stats
   ```

2. **Check Application Metrics:**
   ```bash
   curl http://localhost:5000/metrics
   ```

3. **Scale If Needed:**
   ```bash
   docker-compose up --scale claude-manager=3 -d
   ```

### Database Connection Issues
1. **Check Database Status:**
   ```bash
   docker-compose exec postgres pg_isready
   ```

2. **Check Connections:**
   ```bash
   docker-compose exec postgres psql -U claude_user -c "SELECT count(*) FROM pg_stat_activity;"
   ```

3. **Restart Database:**
   ```bash
   docker-compose restart postgres
   ```

## 📊 Monitoring & Alerting

### Key Metrics to Monitor
- **Application Response Time:** < 200ms for 95th percentile
- **Error Rate:** < 1% of total requests
- **Memory Usage:** < 80% of allocated memory
- **CPU Usage:** < 70% sustained
- **Database Connections:** < 80% of max connections

### Setting Up Alerts
1. **Prometheus Alerts:** Configure in `monitoring/rules/alerts.yml`
2. **Grafana Dashboards:** Import dashboards from `monitoring/grafana/`
3. **Email Notifications:** Configure SMTP in environment variables

## 🔧 Maintenance Procedures

### Daily Health Checks
```bash
# Automated daily health check
make health
make metrics
```

### Weekly Maintenance
1. **Update Dependencies:**
   ```bash
   make update-deps
   make test
   ```

2. **Database Maintenance:**
   ```bash
   make db-migrate
   # Analyze query performance
   docker-compose exec postgres psql -U claude_user -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
   ```

3. **Log Rotation:**
   ```bash
   docker-compose exec claude-manager logrotate /etc/logrotate.conf
   ```

### Monthly Maintenance
1. **Security Updates:**
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d --force-recreate
   
   # Run security scans
   make security-scan
   ```

2. **Performance Review:**
   ```bash
   # Generate performance report
   python performance_report.py
   ```

3. **Backup Verification:**
   ```bash
   make backup
   # Test backup restoration in staging
   ```

## 🚀 Deployment Procedures

### Production Deployment
1. **Pre-deployment Checklist:**
   - [ ] All tests passing in CI
   - [ ] Security scans clean
   - [ ] Performance benchmarks within thresholds
   - [ ] Database migrations reviewed
   - [ ] Rollback plan prepared

2. **Deploy:**
   ```bash
   git tag v1.x.x
   git push origin v1.x.x
   # GitHub Actions will handle the rest
   ```

3. **Post-deployment Verification:**
   ```bash
   # Check application health
   curl -f https://api.example.com/health
   
   # Monitor for 15 minutes
   watch -n 30 'curl -s https://api.example.com/metrics | grep error_rate'
   ```

### Rollback Procedure
1. **Immediate Rollback:**
   ```bash
   # Rollback to previous container version
   docker-compose down
   docker tag claude-manager:previous claude-manager:latest
   docker-compose up -d
   ```

2. **Database Rollback (if needed):**
   ```bash
   # Restore from backup
   make restore BACKUP_FILE=backups/postgres-YYYYMMDD_HHMMSS.sql
   ```

## 🔍 Troubleshooting

### Common Issues

#### High Response Times
**Symptoms:** API responses > 1s
**Investigation:**
```bash
# Check database query performance
docker-compose exec postgres psql -U claude_user -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Check application metrics
curl http://localhost:5000/metrics | grep response_time
```

**Resolution:**
- Add database indexes for slow queries
- Increase application replicas
- Review and optimize slow code paths

#### Memory Leaks
**Symptoms:** Gradual memory increase over time
**Investigation:**
```bash
# Memory profiling
python -m memory_profiler src/main.py

# Check for memory leaks in logs
grep -i "memory" logs/application.log
```

**Resolution:**
- Review code for unclosed resources
- Implement proper connection pooling
- Restart application as temporary fix

#### Database Lock Contention
**Symptoms:** Database timeouts, slow queries
**Investigation:**
```bash
# Check for locks
docker-compose exec postgres psql -U claude_user -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Resolution:**
- Optimize transaction scope
- Review query patterns
- Consider read replicas

## 📈 Capacity Planning

### Scaling Guidelines
- **CPU Usage > 70%:** Add horizontal replicas
- **Memory Usage > 80%:** Investigate memory leaks or add memory
- **Database Connections > 80%:** Implement connection pooling or add read replicas
- **Disk Usage > 85%:** Clean logs or add storage

### Performance Baselines
- **Normal Load:** 100 requests/minute
- **Peak Load:** 500 requests/minute
- **Database:** < 100ms average query time
- **Cache Hit Rate:** > 90%

## 🔐 Security Procedures

### Incident Response
1. **Identify Threat Level**
2. **Isolate Affected Systems**
3. **Collect Evidence**
4. **Notify Stakeholders**
5. **Implement Fixes**
6. **Document Lessons Learned**

### Regular Security Tasks
- **Weekly:** Review security logs
- **Monthly:** Update dependencies and scan for vulnerabilities
- **Quarterly:** Security penetration testing
- **Annually:** Full security audit

## 📞 Emergency Contacts

- **On-Call Engineer:** Slack #on-call
- **DevOps Team:** devops@company.com
- **Security Team:** security@company.com
- **Management:** Escalation via Slack #leadership