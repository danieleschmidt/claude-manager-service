# Claude Manager Service - Developer Guide

## Overview

This guide provides detailed technical information for developers working on the Claude Manager Service codebase, including architecture, development workflows, and contribution guidelines.

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Node.js 16+
- Docker and Docker Compose
- Git with pre-commit hooks
- GitHub CLI (optional but recommended)

### Local Development Setup

1. **Clone and Configure**
   ```bash
   git clone https://github.com/danieleschmidt/claude-manager-service.git
   cd claude-manager-service
   cp config.json.example config.json
   ```

2. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. **Pre-commit Hooks**
   ```bash
   pre-commit install
   ./scripts/setup-git-hooks.sh
   ```

4. **Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   source .env
   ```

5. **Database Setup**
   ```bash
   # Using Docker
   docker-compose up -d database
   
   # Or local PostgreSQL
   createdb claude_manager_dev
   python src/database_migration_utility.py migrate
   ```

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard                            │
│                   (Flask + React)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   API Gateway                              │
│                (Flask-RESTful)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Core Orchestrator                           │
│         (Async Task Management)                            │
└─────┬───────────────┬───────────────────────────┬─────────┘
      │               │                           │
┌─────┴──────┐ ┌─────┴──────┐ ┌──────────────────┴─────────┐
│   Task     │ │  GitHub    │ │      AI Agents            │
│ Analyzer   │ │    API     │ │ (Terragon/Claude Flow)    │
└────────────┘ └────────────┘ └────────────────────────────┘
```

### Core Modules

#### 1. Task Analysis (`src/task_analyzer.py`)
- Repository scanning for TODO comments
- Issue analysis and prioritization
- Code quality assessment
- Security vulnerability detection

#### 2. GitHub Integration (`src/github_api.py`)
- GitHub API wrapper with rate limiting
- Repository management
- Issue and PR operations
- Webhook handling

#### 3. Orchestration (`src/orchestrator.py`)
- Task queue management
- AI agent coordination
- Workflow state management
- Error handling and recovery

#### 4. Performance Monitoring (`src/performance_monitor.py`)
- System metrics collection
- Performance alerting
- Resource usage tracking
- Database performance monitoring

## Development Workflow

### Branch Strategy

We use GitFlow with the following branches:
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

### Code Standards

#### Python Style Guide
- Follow PEP 8 with 100-character line limit
- Use type hints for all functions
- Docstring format: Google style
- Import order: standard library, third-party, local

```python
from typing import List, Optional, Dict, Any
import asyncio
import logging

from github import Github
from flask import Flask

from src.config import Config
from src.models import Task

async def process_tasks(
    tasks: List[Task], 
    config: Config
) -> Dict[str, Any]:
    """Process a list of tasks asynchronously.
    
    Args:
        tasks: List of Task objects to process
        config: Configuration object with processing settings
        
    Returns:
        Dictionary containing processing results and metrics
        
    Raises:
        ProcessingError: If task processing fails
    """
    pass
```

#### Testing Standards
- Minimum 90% test coverage
- Unit tests for all functions
- Integration tests for API endpoints
- End-to-end tests for critical workflows

```python
import pytest
from unittest.mock import Mock, patch

from src.task_analyzer import TaskAnalyzer

class TestTaskAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return TaskAnalyzer(config=Mock())
    
    @patch('src.task_analyzer.GitHubAPI')
    async def test_analyze_repository(self, mock_api, analyzer):
        # Test implementation
        pass
```

### Database Schema

#### Core Tables

```sql
-- Tasks table
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    repository VARCHAR(255) NOT NULL,
    priority INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
```

#### Migration System

```python
# Database migrations in src/database_migration_utility.py
from alembic import command
from alembic.config import Config

def run_migrations():
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
```

## API Documentation

### REST Endpoints

#### Tasks API

```http
GET /api/v1/tasks
GET /api/v1/tasks/{id}
POST /api/v1/tasks
PUT /api/v1/tasks/{id}
DELETE /api/v1/tasks/{id}
```

#### Repositories API

```http
GET /api/v1/repositories
GET /api/v1/repositories/{owner}/{repo}/tasks
POST /api/v1/repositories/{owner}/{repo}/scan
```

#### Metrics API

```http
GET /api/v1/metrics/performance
GET /api/v1/metrics/tasks
GET /api/v1/health
```

### WebSocket Events

```javascript
// Real-time task updates
socket.on('task_updated', (data) => {
    console.log('Task updated:', data);
});

// Performance alerts
socket.on('performance_alert', (alert) => {
    console.warn('Performance alert:', alert);
});
```

## Testing Framework

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=html

# Performance tests
pytest tests/performance/ --benchmark-only
```

### Test Configuration

```python
# conftest.py
import pytest
from src.app import create_app
from src.database import db

@pytest.fixture
def app():
    app = create_app(testing=True)
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()
```

### Mock Strategy

```python
# Use factories for consistent test data
class TaskFactory:
    @staticmethod
    def create(title="Test Task", **kwargs):
        defaults = {
            "description": "Test description",
            "repository": "test/repo",
            "priority": 1
        }
        defaults.update(kwargs)
        return Task(**defaults)
```

## Performance Optimization

### Database Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30
)

# Optimize queries with proper indexing
CREATE INDEX idx_tasks_repository ON tasks(repository);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
```

### Async Operations

```python
import asyncio
import aiohttp

async def fetch_repository_data(session, repo_url):
    async with session.get(repo_url) as response:
        return await response.json()

async def process_repositories(repo_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_repository_data(session, url) for url in repo_urls]
        return await asyncio.gather(*tasks)
```

### Caching Strategy

```python
from functools import lru_cache
import redis

# In-memory caching
@lru_cache(maxsize=1000)
def get_repository_config(repo_name):
    return load_config_from_db(repo_name)

# Redis caching
redis_client = redis.Redis()

def cache_task_result(task_id, result, ttl=3600):
    redis_client.setex(f"task:{task_id}", ttl, json.dumps(result))
```

## Security Considerations

### Authentication and Authorization

```python
from functools import wraps
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        verify_jwt_in_request()
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/tasks')
@require_auth
def get_tasks():
    user_id = get_jwt_identity()
    # Implementation
```

### Input Validation

```python
from marshmallow import Schema, fields, validate

class TaskSchema(Schema):
    title = fields.String(required=True, validate=validate.Length(min=1, max=255))
    description = fields.String(validate=validate.Length(max=5000))
    repository = fields.String(required=True, validate=validate.Regexp(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$'))
    priority = fields.Integer(validate=validate.Range(min=0, max=10))

def validate_task_input(data):
    schema = TaskSchema()
    return schema.load(data)
```

### Secure Configuration

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.environ.get('ENCRYPTION_KEY')
        self.cipher_suite = Fernet(self.encryption_key)
    
    def encrypt_token(self, token):
        return self.cipher_suite.encrypt(token.encode())
    
    def decrypt_token(self, encrypted_token):
        return self.cipher_suite.decrypt(encrypted_token).decode()
```

## Monitoring and Observability

### Logging Configuration

```python
import logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
task_counter = Counter('tasks_processed_total', 'Total tasks processed', ['status'])
request_duration = Histogram('request_duration_seconds', 'Request duration')
active_connections = Gauge('active_connections', 'Active database connections')

# Use in code
task_counter.labels(status='completed').inc()
request_duration.observe(0.5)
active_connections.set(10)
```

### Health Checks

```python
@app.route('/health')
def health_check():
    checks = {
        'database': check_database_connection(),
        'github_api': check_github_api(),
        'redis': check_redis_connection()
    }
    
    status = 'healthy' if all(checks.values()) else 'unhealthy'
    return jsonify({'status': status, 'checks': checks})
```

## Deployment

### Docker Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.app:create_app()"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-manager
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-manager
  template:
    metadata:
      labels:
        app: claude-manager
    spec:
      containers:
      - name: claude-manager
        image: claude-manager:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

## Contributing Guidelines

### Pull Request Process

1. **Fork and Branch**: Create feature branch from `develop`
2. **Implement**: Write code following our standards
3. **Test**: Ensure all tests pass and coverage remains high
4. **Document**: Update relevant documentation
5. **Review**: Submit PR with detailed description
6. **Merge**: Squash and merge after approval

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Test coverage maintained or improved
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Breaking changes documented

### Release Process

1. **Feature Freeze**: Merge all features to `develop`
2. **Release Branch**: Create `release/vX.Y.Z` branch
3. **Testing**: Run full test suite and manual testing
4. **Documentation**: Update changelog and version numbers
5. **Merge**: Merge to `main` and tag release
6. **Deploy**: Deploy to production environment

## Troubleshooting

### Common Development Issues

**Import Errors**:
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Database Connection Issues**:
```bash
# Check database status
docker-compose ps database
docker-compose logs database
```

**Test Failures**:
```bash
# Run with verbose output
pytest -v -s tests/
```

### Performance Debugging

```python
import cProfile
import pstats

def profile_function(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper
```

## Additional Resources

- [API Documentation](../API.md)
- [Architecture Decision Records](../adr/)
- [Deployment Guide](../deployment/)
- [Security Guidelines](../../SECURITY.md)
- [Performance Monitoring](../../PERFORMANCE_MONITORING.md)

---

For questions or support, please reach out to the development team or create an issue on GitHub.