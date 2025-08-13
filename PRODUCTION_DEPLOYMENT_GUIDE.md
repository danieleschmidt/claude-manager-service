# TERRAGON SDLC v4.0 - Production Deployment Guide

## 🚀 Overview

This guide provides comprehensive instructions for deploying the TERRAGON SDLC v4.0 system to production environments with enterprise-grade reliability, security, and scalability.

## 📋 Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.24+
- **Container Registry**: Docker Hub, AWS ECR, or Google GCR
- **Monitoring Stack**: Prometheus, Grafana, Jaeger
- **Database**: PostgreSQL 14+ or compatible
- **Cache**: Redis 6.0+
- **Load Balancer**: NGINX, AWS ALB, or Google Cloud Load Balancer

### Resource Requirements

| Environment | CPU | Memory | Storage | Replicas |
|-------------|-----|--------|---------|----------|
| Development | 250m | 512Mi | 10Gi | 1 |
| Staging | 500m | 1Gi | 20Gi | 2 |
| Production | 1 CPU | 2Gi | 50Gi | 3-10 |

### Security Requirements

- TLS certificates for HTTPS
- Network policies for pod-to-pod communication
- Pod Security Standards (PSS) configured
- Image vulnerability scanning enabled
- RBAC policies configured

## 🔧 Pre-Deployment Setup

### 1. Environment Preparation

```bash
# Create namespaces
kubectl create namespace terragon-production
kubectl create namespace terragon-staging
kubectl create namespace terragon-monitoring

# Apply security policies
kubectl apply -f k8s/security/network-policies.yaml
kubectl apply -f k8s/security/pod-security-policies.yaml
```

### 2. Secrets Configuration

```bash
# Create database secrets
kubectl create secret generic db-credentials \
  --from-literal=username=terragon_user \
  --from-literal=password=<secure_password> \
  --namespace=terragon-production

# Create registry secrets
kubectl create secret docker-registry registry-secret \
  --docker-server=<registry_url> \
  --docker-username=<username> \
  --docker-password=<password> \
  --namespace=terragon-production

# Create TLS certificates
kubectl create secret tls terragon-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  --namespace=terragon-production
```

### 3. ConfigMap Setup

```bash
# Apply configuration
kubectl apply -f k8s/configmaps/app-config.yaml
kubectl apply -f k8s/configmaps/monitoring-config.yaml
```

## 🚢 Deployment Strategies

### Blue-Green Deployment (Recommended for Production)

```bash
# Deploy green environment
python3 deployment_orchestrator.py \
  --environment production \
  --strategy blue_green \
  --version v1.2.0

# Monitor deployment
kubectl get deployments -n terragon-production -w

# Verify health
curl -f https://terragon-green.production.com/health
```

### Rolling Deployment (Staging)

```bash
# Deploy with rolling update
python3 deployment_orchestrator.py \
  --environment staging \
  --strategy rolling \
  --version v1.2.0
```

### Canary Deployment (Advanced)

```bash
# Deploy canary version
python3 deployment_orchestrator.py \
  --environment production \
  --strategy canary \
  --version v1.2.0 \
  --canary-percentage 10
```

## 📊 Monitoring and Observability

### Metrics Collection

The system exposes metrics on `/metrics` endpoint:

- **Application Metrics**: Request rates, latency, error rates
- **Business Metrics**: Task completion rates, quality scores
- **Infrastructure Metrics**: CPU, memory, disk usage

### Health Checks

| Endpoint | Purpose | Timeout |
|----------|---------|---------|
| `/health` | General health status | 30s |
| `/ready` | Readiness probe | 5s |
| `/metrics` | Prometheus metrics | 10s |

### Alerting Rules

Critical alerts configured:
- Error rate > 5%
- P99 latency > 2 seconds
- Availability < 99%
- Memory usage > 90%
- Disk usage > 85%

### Log Aggregation

Logs are collected and shipped to:
- **Development**: Console output
- **Staging**: ELK Stack
- **Production**: Splunk/ELK with retention

## 🔐 Security Configuration

### Network Security

```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: terragon-network-policy
spec:
  podSelector:
    matchLabels:
      app: terragon-sdlc
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: terragon-production
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: terragon-production
```

### Pod Security

```yaml
# Pod Security Standards
apiVersion: v1
kind: Namespace
metadata:
  name: terragon-production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### RBAC Configuration

```bash
# Apply RBAC policies
kubectl apply -f k8s/rbac/service-account.yaml
kubectl apply -f k8s/rbac/cluster-role.yaml
kubectl apply -f k8s/rbac/role-binding.yaml
```

## 🔄 Backup and Disaster Recovery

### Database Backup

```bash
# Automated backup
kubectl create job backup-$(date +%Y%m%d) \
  --from=cronjob/database-backup \
  --namespace=terragon-production

# Manual backup
kubectl exec -it postgres-0 -- pg_dump -U terragon_user terragon_db > backup.sql
```

### Configuration Backup

```bash
# Export all configurations
kubectl get all,cm,secrets -n terragon-production -o yaml > production-backup.yaml
```

### Disaster Recovery Procedure

1. **Assess Impact**: Determine scope of failure
2. **Isolate**: Remove failed components from load balancer
3. **Restore**: Deploy from last known good state
4. **Verify**: Run health checks and smoke tests
5. **Route Traffic**: Gradually restore traffic
6. **Monitor**: Watch metrics for stability

## 📈 Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: terragon-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: terragon-sdlc
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: terragon-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: terragon-sdlc
  updatePolicy:
    updateMode: "Auto"
```

## 🔍 Troubleshooting

### Common Issues

#### Deployment Fails

```bash
# Check pod status
kubectl get pods -n terragon-production

# View pod logs
kubectl logs -f deployment/terragon-sdlc -n terragon-production

# Describe failing pods
kubectl describe pod <pod-name> -n terragon-production
```

#### Health Check Failures

```bash
# Test health endpoint
kubectl port-forward svc/terragon-sdlc 8080:80 -n terragon-production
curl -v http://localhost:8080/health

# Check resource constraints
kubectl top pods -n terragon-production
```

#### Performance Issues

```bash
# Check HPA status
kubectl get hpa -n terragon-production

# View resource usage
kubectl top nodes
kubectl top pods -n terragon-production

# Check application metrics
curl http://localhost:8080/metrics | grep terragon
```

### Rollback Procedure

#### Automatic Rollback

The system automatically rolls back if:
- Health checks fail for 5 minutes
- Error rate exceeds 5%
- Success rate drops below 95%

#### Manual Rollback

```bash
# List deployment history
kubectl rollout history deployment/terragon-sdlc -n terragon-production

# Rollback to previous version
kubectl rollout undo deployment/terragon-sdlc -n terragon-production

# Rollback to specific revision
kubectl rollout undo deployment/terragon-sdlc --to-revision=2 -n terragon-production
```

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] Infrastructure resources allocated
- [ ] Secrets and ConfigMaps created
- [ ] Database migrations tested
- [ ] Security policies applied
- [ ] Monitoring stack configured
- [ ] Backup procedures verified

### During Deployment

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Security scans completed
- [ ] Performance benchmarks met

### Post-Deployment

- [ ] Smoke tests passed
- [ ] Integration tests passed
- [ ] Performance tests passed
- [ ] Security validation completed
- [ ] Documentation updated
- [ ] Team notifications sent

## 🚨 Emergency Procedures

### Complete System Failure

1. **Immediate Response**
   ```bash
   # Switch to maintenance mode
   kubectl apply -f k8s/emergency/maintenance-mode.yaml
   
   # Scale down all services
   kubectl scale deployment --all --replicas=0 -n terragon-production
   ```

2. **Investigation**
   ```bash
   # Collect logs
   kubectl logs --previous -l app=terragon-sdlc -n terragon-production
   
   # Check cluster events
   kubectl get events --sort-by=.metadata.creationTimestamp -n terragon-production
   ```

3. **Recovery**
   ```bash
   # Restore from backup
   kubectl apply -f production-backup.yaml
   
   # Verify restoration
   python3 validate_sdlc_quality.py
   ```

### Data Corruption

1. **Stop Processing**
   ```bash
   kubectl scale deployment terragon-sdlc --replicas=0 -n terragon-production
   ```

2. **Restore Database**
   ```bash
   kubectl exec -it postgres-0 -- psql -U terragon_user -d terragon_db < backup.sql
   ```

3. **Verify Data Integrity**
   ```bash
   python3 scripts/verify-data-integrity.py
   ```

## 📞 Support and Contacts

- **DevOps Team**: devops@terragon.ai
- **Engineering Team**: engineering@terragon.ai
- **Security Team**: security@terragon.ai
- **On-Call**: +1-XXX-XXX-XXXX

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [TERRAGON SDLC Architecture](./ARCHITECTURE.md)
- [Security Best Practices](./SECURITY.md)
- [Performance Tuning Guide](./PERFORMANCE_MONITORING.md)