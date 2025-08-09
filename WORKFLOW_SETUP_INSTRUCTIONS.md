# GitHub Workflows Setup Instructions

## ⚠️ Manual Setup Required

The Claude Manager Service includes comprehensive GitHub Actions workflows, but they must be set up manually due to GitHub App permissions.

## 🔧 Setup Steps

### 1. Copy Workflow Templates

```bash
# Copy from templates directory to .github/workflows/
mkdir -p .github/workflows
cp docs/github-workflows-templates/*.yml .github/workflows/
```

### 2. Available Workflows

- **`ci-cd.yml`** - Complete CI/CD pipeline with testing, building, and deployment
- **`security.yml`** - Security scanning and vulnerability assessment
- **`performance.yml`** - Performance testing and benchmarking
- **`release.yml`** - Automated releases and changelog generation
- **`dependency-management.yml`** - Dependency updates and security patches

### 3. Required Secrets

Configure these in your repository settings:

```bash
# Core secrets
GITHUB_TOKEN          # GitHub Personal Access Token
TERRAGON_TOKEN        # Terragon CLI authentication (optional)
CLAUDE_FLOW_TOKEN     # Claude Flow authentication (optional)

# Security scanning
SNYK_TOKEN           # Snyk security scanning
SONAR_TOKEN          # SonarCloud analysis

# Deployment (if using)
DOCKER_HUB_USERNAME  # Docker Hub for container builds
DOCKER_HUB_TOKEN     # Docker Hub authentication
```

### 4. Workflow Permissions

Ensure your repository has these permissions enabled:
- **Actions**: Read and write
- **Contents**: Write
- **Pull requests**: Write
- **Issues**: Write
- **Security events**: Write

## 🚀 Quick Setup Script

```bash
#!/bin/bash
# Quick setup script for GitHub workflows

echo "Setting up GitHub workflows..."

# Create workflows directory
mkdir -p .github/workflows

# Copy templates
cp docs/github-workflows-templates/*.yml .github/workflows/

echo "✅ Workflows copied successfully!"
echo "📋 Next steps:"
echo "1. Configure repository secrets"
echo "2. Enable required permissions"
echo "3. Commit and push the workflow files"
```

## 📊 What Each Workflow Provides

### CI/CD Pipeline (`ci-cd.yml`)
- Python testing with pytest
- Code quality checks with flake8, black, mypy
- Security scanning with bandit
- Docker image building
- Automated deployment to staging/production

### Security Workflow (`security.yml`)
- Dependency vulnerability scanning
- Code security analysis
- Secret detection
- Security report generation
- Automated security fixes

### Performance Workflow (`performance.yml`)
- Load testing with pytest-benchmark
- Memory profiling
- API response time testing
- Performance regression detection
- Capacity planning metrics

### Release Workflow (`release.yml`)
- Semantic versioning
- Automated changelog generation
- GitHub releases with assets
- Package publishing to PyPI
- Docker image tagging

## 🔒 Security Considerations

1. **Secret Management**: Use GitHub secrets for all sensitive data
2. **Branch Protection**: Enable branch protection rules
3. **Required Reviews**: Require pull request reviews
4. **Status Checks**: Require passing CI checks before merge

## 🎯 After Setup

Once workflows are configured, you'll have:
- ✅ Automated testing on every PR
- ✅ Security scanning and reporting
- ✅ Performance monitoring
- ✅ Automated releases
- ✅ Dependency management
- ✅ Quality gates enforcement

The system is designed to be production-ready with enterprise-grade CI/CD practices.