# Development Workflow Guide

## Overview
This guide outlines the standard development workflow for contributing to the Claude Manager Service project.

## Prerequisites
- Python 3.11+
- Git
- Docker and Docker Compose
- GitHub CLI (optional but recommended)

## Setup Development Environment

### 1. Clone and Setup
```bash
git clone https://github.com/danieleschmidt/claude-manager-service.git
cd claude-manager-service
make install-dev
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run Development Setup
```bash
# Start development services
docker-compose up -d

# Run the development server
python start_dashboard.py
```

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code changes ...

# Run tests
make test

# Run linting
make lint

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push branch
git push origin feature/your-feature-name
```

### 2. Pull Request Process
1. **Create PR**: Use GitHub CLI or web interface
2. **PR Template**: Fill out the provided template completely
3. **Code Review**: Address all review comments
4. **CI Checks**: Ensure all checks pass
5. **Merge**: Use "Squash and merge" for clean history

### 3. Testing Strategy
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows
- **Performance Tests**: Validate performance benchmarks

### 4. Code Quality Standards
- **Type Hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public methods
- **Test Coverage**: Minimum 90% coverage required
- **Security**: All code must pass security scans

## Branch Protection Rules
- Main branch requires PR reviews
- Status checks must pass
- Up-to-date branches required
- Administrator approval for sensitive changes

## Release Process
1. **Version Bump**: Update version in package.json and pyproject.toml
2. **Changelog**: Update CHANGELOG.md with release notes
3. **Tag Release**: Create semantic version tag
4. **Deploy**: Automated deployment via GitHub Actions

## Troubleshooting
- **Test Failures**: Check test logs and fix issues before pushing
- **Linting Issues**: Run `make lint-fix` to auto-fix formatting
- **Docker Issues**: Reset with `docker-compose down && docker-compose up --build`

## Getting Help
- **Documentation**: Check docs/ directory
- **Issues**: Search existing issues or create new ones
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Follow SECURITY.md for vulnerability reports