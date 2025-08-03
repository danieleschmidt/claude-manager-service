# Build and Deployment Guide

## Build System Overview

The Claude Manager Service uses a comprehensive build system with multi-stage Docker builds and automated deployment scripts.

## Build Process

### Multi-Stage Docker Build

The Dockerfile defines multiple build targets:

- **base**: Common base layer with Python and system dependencies
- **dev-deps**: Development dependencies and tools
- **prod-deps**: Production-only dependencies
- **development**: Full development environment
- **production**: Optimized production image with security hardening
- **testing**: Testing environment with all test dependencies
- **security**: Security scanning tools

### Build Script Usage

```bash
# Build development image
./scripts/build.sh development

# Build and push production image
./scripts/build.sh production --version=1.0.0 --push

# Build all targets without cache
./scripts/build.sh all --no-cache

# Custom registry
./scripts/build.sh production --registry=my-registry.com --push
```

### Build Arguments and Labels

The build process includes:
- Build date and VCS reference
- Version information
- OCI-compliant image labels
- Security metadata

## Container Environments

### Development Environment

```bash
# Start development environment
docker-compose up -d

# Access logs
docker-compose logs -f claude-manager

# Run tests in container
docker-compose exec claude-manager pytest
```

Services included:
- Claude Manager application
- PostgreSQL database
- Redis cache
- Prometheus monitoring
- Grafana dashboards
- Nginx reverse proxy

### Production Environment

```bash
# Deploy production environment
docker-compose -f docker-compose.production.yml up -d

# Monitor deployment
docker-compose -f docker-compose.production.yml ps
```

Production features:
- Resource limits and reservations
- Health checks and restart policies
- Log aggregation with Fluentd
- Automated backup services
- Enhanced security configurations

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes Cluster**: Version 1.20+ with RBAC enabled
2. **kubectl**: Configured to access your cluster
3. **Storage Class**: For persistent volumes
4. **Container Registry**: Access to pull images

### Quick Start

```bash
# Deploy to development namespace
./scripts/deploy.sh development

# Deploy specific version to production
./scripts/deploy.sh production --version=1.0.0

# Perform dry run
./scripts/deploy.sh staging --dry-run
```

### Manual Deployment

```bash
# Create namespace and apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl rollout status deployment/claude-manager -n claude-manager
```

### Resource Configuration

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| Application | 250m | 512Mi | 1000m | 1Gi |
| PostgreSQL | 250m | 512Mi | 1000m | 1Gi |
| Redis | 100m | 128Mi | 500m | 512Mi |

### Scaling

```bash
# Scale application
kubectl scale deployment claude-manager --replicas=5 -n claude-manager

# Setup auto-scaling
kubectl autoscale deployment claude-manager --cpu-percent=70 --min=3 --max=10 -n claude-manager
```

## Security Considerations

### Container Security

1. **Non-root User**: All containers run as user ID 1000
2. **Read-only Filesystem**: Application files are immutable
3. **Minimal Base Images**: Alpine-based for reduced attack surface
4. **Security Scanning**: Automated vulnerability scans

### Secrets Management

```bash
# Create secret for sensitive data
kubectl create secret generic claude-manager-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=github-token="your-token" \
  -n claude-manager
```

### Network Security

```yaml
# Example Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: claude-manager-netpol
  namespace: claude-manager
spec:
  podSelector:
    matchLabels:
      app: claude-manager-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 5000
```

## Monitoring and Observability

### Health Checks

The application provides several health check endpoints:

- `/health` - Basic health status
- `/ready` - Readiness for traffic
- `/metrics` - Prometheus metrics

### Monitoring Stack

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Fluentd**: Log aggregation and forwarding

### Custom Metrics

```python
# Example custom metrics in application
from prometheus_client import Counter, Histogram

task_counter = Counter('tasks_processed_total', 'Total processed tasks')
response_time = Histogram('response_time_seconds', 'Response time')
```

## Deployment Strategies

### Rolling Updates

Default Kubernetes strategy with zero downtime:

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 1
```

### Blue-Green Deployment

For critical updates requiring full validation:

```bash
# Deploy green environment
kubectl apply -f k8s/deployment-green.yaml

# Switch traffic
kubectl patch service claude-manager-service -p '{"spec":{"selector":{"version":"green"}}}'

# Cleanup blue environment
kubectl delete deployment claude-manager-blue
```

### Canary Deployment

Gradual rollout to subset of users:

```yaml
# Canary deployment with 10% traffic
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: claude-manager-rollout
spec:
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 5m}
      - setWeight: 50
      - pause: {duration: 10m}
      - setWeight: 100
```

## Troubleshooting

### Common Issues

1. **Image Pull Errors**
   ```bash
   # Check image availability
   docker pull ghcr.io/terragon-labs/claude-manager:latest-production
   
   # Verify registry credentials
   kubectl get secret regcred -n claude-manager -o yaml
   ```

2. **Pod Startup Failures**
   ```bash
   # Check pod logs
   kubectl logs -f deployment/claude-manager -n claude-manager
   
   # Describe pod for events
   kubectl describe pod <pod-name> -n claude-manager
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connectivity
   kubectl exec -it deployment/claude-manager -n claude-manager -- \
     python -c "import psycopg2; print('DB connection successful')"
   ```

### Rollback Procedures

```bash
# View rollout history
kubectl rollout history deployment/claude-manager -n claude-manager

# Rollback to previous version
kubectl rollout undo deployment/claude-manager -n claude-manager

# Rollback to specific revision
kubectl rollout undo deployment/claude-manager --to-revision=2 -n claude-manager
```

## Best Practices

### Build Optimization

1. **Layer Caching**: Order Dockerfile instructions by change frequency
2. **Multi-stage Builds**: Separate build and runtime dependencies
3. **Image Size**: Use minimal base images and cleanup unnecessary files
4. **Security Scanning**: Integrate vulnerability scanning in CI/CD

### Deployment Safety

1. **Health Checks**: Implement comprehensive health endpoints
2. **Resource Limits**: Set appropriate CPU and memory limits
3. **Graceful Shutdown**: Handle SIGTERM signals properly
4. **Database Migrations**: Use backward-compatible schema changes

### Monitoring

1. **Metrics**: Export business and technical metrics
2. **Logging**: Use structured logging with correlation IDs
3. **Tracing**: Implement distributed tracing for complex workflows
4. **Alerting**: Set up proactive alerting for critical issues

## Environment-Specific Configurations

### Development
- Debug logging enabled
- Hot reloading for code changes
- Accessible databases and services
- Extended timeouts for debugging

### Staging
- Production-like environment
- Realistic data volumes
- Performance testing capabilities
- Integration with external services

### Production
- Optimized performance settings
- Security hardening enabled
- Monitoring and alerting active
- Backup and disaster recovery configured