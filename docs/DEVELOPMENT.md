# Development Guide

Welcome to the Claude Code Manager development environment! This guide will help you get started with development, testing, and contributing to the project.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Debugging](#debugging)
- [Database](#database)
- [API Development](#api-development)
- [Contributing](#contributing)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd claude-code-manager

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Development Environment Setup

#### Option A: Docker Development (Recommended)
```bash
# Start development environment
make run

# Or manually
docker-compose up -d
```

#### Option B: Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Set up pre-commit hooks
pre-commit install

# Start development server
python start_dashboard.py
```

### 3. Verify Installation

```bash
# Check health
make health

# Run tests
make test

# Check code quality
make quality
```

## Development Environment

### Using DevContainer

The project includes a complete development container configuration:

```bash
# Open in VS Code with DevContainer
code .
# Then: Ctrl/Cmd + Shift + P -> "Reopen in Container"
```

### Environment Variables

Key environment variables for development:

```bash
# GitHub Integration
GITHUB_TOKEN=your_github_token_here
GITHUB_USERNAME=your_username

# Database
DATABASE_URL=sqlite:///data/tasks.db

# Development Settings
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=DEBUG

# Feature Flags
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_ENHANCED_SECURITY=false
```

## Project Structure

```
claude-code-manager/
├── src/                          # Main application code
│   ├── services/                 # Business logic services
│   ├── async_*.py               # Async implementations
│   ├── github_api.py            # GitHub API integration
│   ├── orchestrator.py          # Task orchestration
│   ├── performance_monitor.py   # Performance tracking
│   └── security.py              # Security utilities
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   ├── performance/             # Performance tests
│   └── security/                # Security tests
├── web/                          # Web dashboard
│   ├── app.py                   # Flask application
│   ├── static/                  # CSS, JS assets
│   └── templates/               # HTML templates
├── docs/                         # Documentation
├── prompts/                      # AI prompt templates
├── scripts/                      # Utility scripts
└── monitoring/                   # Monitoring configuration
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# Write tests first (TDD approach recommended)

# Run tests frequently
make test-unit

# Check code quality
make lint
make type-check
```

### 2. Testing Your Changes

```bash
# Unit tests
make test-unit

# Integration tests
make test-integration

# Full test suite
make test

# Performance tests
make test-performance

# Security tests
make test-security
```

### 3. Code Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# Security scan
make security-scan

# All quality checks
make quality
```

### 4. Commit and Push

```bash
# Pre-commit checks run automatically
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

## Testing

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance and scalability
- **Security Tests**: Test security features

### Running Tests

```bash
# All tests
pytest

# Specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_github_api.py

# Specific test function
pytest tests/unit/test_github_api.py::test_create_issue

# Watch mode (requires pytest-watch)
ptw tests/unit/
```

### Writing Tests

```python
# Unit test example
import pytest
from src.github_api import GitHubAPI

def test_github_api_initialization():
    api = GitHubAPI("test_token")
    assert api.token == "test_token"

# Async test example
@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result is not None

# Integration test example
@pytest.mark.integration
def test_database_integration(test_db):
    # Test database operations
    pass
```

### Test Configuration

```python
# conftest.py provides shared fixtures
@pytest.fixture
def mock_github_api():
    return MagicMock()

@pytest.fixture
def sample_config():
    return {"github": {"username": "test"}}
```

## Code Quality

### Code Style

We follow these standards:

- **Python**: PEP 8 with 88-character line length (Black)
- **Import Order**: isort with Black profile
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all modules, classes, and functions

### Pre-commit Hooks

Pre-commit hooks run automatically on commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

### Code Review Checklist

- [ ] Tests are written and passing
- [ ] Code follows style guidelines
- [ ] Type hints are added
- [ ] Documentation is updated
- [ ] Security considerations addressed
- [ ] Performance impact considered

## Debugging

### Local Debugging

```bash
# Enable debug mode
export FLASK_DEBUG=true
export LOG_LEVEL=DEBUG

# Start with debugger
python -m pdb start_dashboard.py

# Or use IDE debugger (VS Code, PyCharm)
```

### Docker Debugging

```bash
# View logs
docker-compose logs -f claude-manager

# Execute shell in container
docker-compose exec claude-manager bash

# Debug specific service
docker-compose logs -f postgres
```

### Performance Debugging

```bash
# Profile with cProfile
python -m cProfile -o profile.stats start_dashboard.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler start_dashboard.py

# Line profiling
pip install line-profiler
kernprof -l -v start_dashboard.py
```

## Database

### Development Database

```bash
# Initialize database
python -c "from src.services.database_service import DatabaseService; DatabaseService().initialize_database()"

# Reset database
make db-reset

# Run migrations
make db-migrate
```

### Database Migrations

```python
# Create migration
from src.database_migration_utility import create_migration
create_migration("add_new_column")

# Apply migrations
from src.database_migration_utility import migrate
migrate()
```

### Database Tools

```bash
# SQLite browser
sqlite3 data/tasks.db

# PostgreSQL (production)
docker-compose exec postgres psql -U claude_user claude_manager
```

## API Development

### Adding New Endpoints

```python
# In web/app.py
@app.route('/api/new-endpoint', methods=['POST'])
def new_endpoint():
    data = request.get_json()
    # Validate input
    # Process request
    # Return response
    return jsonify({"status": "success"})
```

### API Testing

```bash
# Test endpoints with curl
curl -X POST http://localhost:5000/api/new-endpoint \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'

# Or use httpie
http POST localhost:5000/api/new-endpoint key=value
```

### API Documentation

Update OpenAPI specification in `docs/api/openapi.yaml`:

```yaml
paths:
  /api/new-endpoint:
    post:
      summary: Description of endpoint
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                key:
                  type: string
```

## Contributing

### Contribution Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Write** tests for your changes
4. **Implement** your feature
5. **Run** all tests and quality checks
6. **Submit** a pull request

### Pull Request Guidelines

- Clear, descriptive title
- Detailed description of changes
- Link to related issues
- All tests passing
- Code review completed

### Issue Reporting

When reporting issues:

- Use issue templates
- Provide reproduction steps
- Include environment details
- Add relevant logs/screenshots

### Feature Requests

For new features:

- Check existing issues first
- Provide clear use case
- Consider implementation approach
- Discuss in GitHub Discussions first

## Common Tasks

### Adding a New Service

```python
# 1. Create service file
# src/services/new_service.py

class NewService:
    def __init__(self):
        pass
    
    def process(self, data):
        # Implementation
        pass

# 2. Add tests
# tests/unit/test_new_service.py

# 3. Update configuration if needed
# 4. Add to orchestrator if needed
```

### Adding New Configuration

```python
# 1. Add to config.json schema
# 2. Update environment variables
# 3. Add validation
# 4. Update documentation
```

### Performance Optimization

```python
# 1. Identify bottleneck
# 2. Add performance test
# 3. Optimize implementation
# 4. Verify improvement
# 5. Monitor in production
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use relative imports
from src.module import function
```

**Docker Issues**
```bash
# Clean up Docker
make clean-docker

# Rebuild containers
docker-compose build --no-cache
```

**Database Issues**
```bash
# Reset database
make db-reset

# Check database logs
docker-compose logs postgres
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search GitHub issues
- **Discussions**: Use GitHub Discussions
- **Chat**: Join project Discord/Slack

## Resources

- [Python Style Guide](https://pep8.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub API](https://docs.github.com/en/rest)

## Next Steps

- Read the [Architecture Documentation](../ARCHITECTURE.md)
- Check the [API Documentation](./api/)
- Review [Security Guidelines](../SECURITY.md)
- Explore [Performance Monitoring](./performance.md)