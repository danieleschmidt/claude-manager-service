# GitHub Actions Workflows Setup Guide

This guide contains the comprehensive GitHub Actions workflows created by the autonomous SDLC enhancement process. Due to GitHub App permissions limitations, these workflows need to be manually added to your repository.

## 📋 Quick Setup Instructions

1. **Copy workflow files** from `docs/github-workflows-templates/` to `.github/workflows/`
2. **Configure required secrets** in your repository settings
3. **Test workflows** by creating a pull request or pushing to main branch

## 🔧 Required Repository Secrets

Add these secrets in **Settings > Secrets and variables > Actions**:

### Core Secrets
```bash
# Required for all workflows
GITHUB_TOKEN  # Automatically provided by GitHub

# For security scanning
SNYK_TOKEN    # Get from https://snyk.io/
FOSSA_API_KEY # Get from https://fossa.com/

# For Docker publishing
DOCKERHUB_USERNAME  # Your Docker Hub username  
DOCKERHUB_TOKEN     # Docker Hub access token

# For PyPI publishing (release workflow)
PYPI_API_TOKEN      # PyPI API token

# For notifications (optional)
SLACK_WEBHOOK_URL   # Slack webhook for notifications
```

### How to Generate Tokens

#### Snyk Token
1. Sign up at https://snyk.io/
2. Go to Account Settings > API Token
3. Copy the token and add as `SNYK_TOKEN` secret

#### FOSSA API Key  
1. Sign up at https://fossa.com/
2. Go to Settings > API
3. Generate new token and add as `FOSSA_API_KEY` secret

#### Docker Hub Token
1. Log in to Docker Hub
2. Go to Account Settings > Security
3. Create new access token and add as `DOCKERHUB_TOKEN` secret

#### PyPI Token (for releases)
1. Log in to PyPI
2. Go to Account Settings > API tokens
3. Create token with upload permissions
4. Add as `PYPI_API_TOKEN` secret

## 📁 Workflow Files Overview

### Core CI/CD Pipeline

#### `ci.yml` - Continuous Integration
- **Triggers**: Push to main/develop, pull requests
- **Features**: Multi-Python testing (3.10, 3.11, 3.12), linting, security scans
- **Duration**: ~5-8 minutes
- **Dependencies**: requirements.txt, requirements-dev.txt

#### `ci-cd.yml` - Complete CI/CD Pipeline  
- **Triggers**: Push to main, pull requests
- **Features**: Testing, building, security scanning, deployment preparation
- **Duration**: ~10-15 minutes
- **Dependencies**: Docker, all test dependencies

#### `release.yml` - Release Automation
- **Triggers**: Push to main, manual workflow dispatch
- **Features**: Semantic versioning, changelog generation, multi-platform deployment
- **Duration**: ~15-20 minutes
- **Dependencies**: PyPI token, Docker Hub credentials

### Security & Compliance

#### `security.yml` - Comprehensive Security Scanning
- **Triggers**: Push, pull requests, daily schedule
- **Features**: CodeQL, SAST, dependency scanning, container security, SBOM
- **Duration**: ~8-12 minutes  
- **Dependencies**: Snyk token, FOSSA API key

#### `dependency-management.yml` - Automated Dependencies
- **Triggers**: Weekly schedule, manual dispatch
- **Features**: Dependency updates, security patching, automated testing
- **Duration**: ~5-10 minutes
- **Dependencies**: GitHub token (automatic)

### Performance & Monitoring

#### `performance.yml` - Performance Testing
- **Triggers**: Push to main/develop, weekly schedule
- **Features**: Benchmarking, load testing, memory profiling
- **Duration**: ~10-15 minutes
- **Dependencies**: pytest-benchmark, locust, memory-profiler

## 🚀 Setup Process

### Step 1: Copy Workflow Files
```bash
# From your repository root
cp docs/github-workflows-templates/*.yml .github/workflows/

# Verify files are in place
ls -la .github/workflows/
```

### Step 2: Configure Secrets
1. Go to your repository on GitHub
2. Navigate to **Settings > Secrets and variables > Actions**
3. Click **New repository secret** for each required secret
4. Add the secret name and value

### Step 3: Test Workflows
```bash
# Create a test branch and push to trigger CI
git checkout -b test-workflows
git commit --allow-empty -m "test: trigger workflow testing"  
git push origin test-workflows

# Create a pull request to test PR workflows
gh pr create --title "Test Workflows" --body "Testing new GitHub Actions workflows"
```

### Step 4: Monitor First Run
1. Go to **Actions** tab in your repository
2. Watch the workflow runs complete
3. Check for any configuration issues
4. Review security scan results

## 🔍 Workflow Details

### CI Pipeline (`ci.yml`)
```yaml
# Key features:
- Multi-Python version testing (3.10, 3.11, 3.12)
- Comprehensive linting (flake8, pylint, black, isort)
- Security scanning (bandit, safety)
- Type checking (mypy)
- Code coverage reporting
- Docker image building and testing
- Container security scanning with Trivy
```

### Security Scanning (`security.yml`)
```yaml
# Security tools included:
- CodeQL analysis (GitHub's semantic code analysis)
- Snyk vulnerability scanning
- Bandit SAST scanning
- Semgrep security rules
- TruffleHog secret detection  
- Docker Bench Security
- SBOM generation with Syft
- OpenSSF Scorecard
- License compliance checking
```

### Release Automation (`release.yml`)
```yaml
# Release features:
- Semantic version calculation
- Automated changelog generation
- Multi-platform Docker builds (amd64, arm64)
- PyPI package publishing
- Docker Hub + GitHub Container Registry
- Release announcement automation
- Documentation updates
```

## 📊 Expected Performance

### Workflow Execution Times
- **CI Pipeline**: 5-8 minutes
- **Security Scanning**: 8-12 minutes  
- **Performance Testing**: 10-15 minutes
- **Release Process**: 15-20 minutes
- **Dependency Updates**: 5-10 minutes

### Resource Usage
- **Concurrent Jobs**: Up to 20 (GitHub free tier limit)
- **Storage**: ~500MB for artifacts per run
- **Network**: Moderate (Docker image pulls, package downloads)

## 🚨 Troubleshooting

### Common Issues

#### "Secrets not found" Error
- **Cause**: Required secrets not configured
- **Solution**: Add all required secrets in repository settings
- **Check**: Verify secret names match exactly (case-sensitive)

#### Docker Build Failures
- **Cause**: Missing Dockerfile or build context issues
- **Solution**: Ensure Dockerfile exists in repository root
- **Check**: Test local Docker build: `docker build -t test .`

#### Security Scan Failures
- **Cause**: High-severity vulnerabilities found
- **Solution**: Review and fix vulnerabilities, or adjust thresholds
- **Check**: Run local security scans: `make security-scan`

#### Performance Test Timeouts
- **Cause**: Tests taking longer than expected
- **Solution**: Increase timeout values or optimize tests
- **Check**: Run performance tests locally: `make test-performance`

### Debug Workflow Issues
```bash
# Check workflow syntax locally
act --list  # Requires act CLI tool

# Validate workflow files
yamllint .github/workflows/*.yml

# Test specific workflow step
act -j test  # Runs 'test' job locally
```

## 📈 Monitoring Workflow Health

### Key Metrics to Track
- **Success Rate**: > 95% for CI workflows
- **Average Duration**: Within expected ranges
- **Failure Patterns**: Identify recurring issues
- **Resource Usage**: Monitor Action minutes usage

### Workflow Analytics
1. Go to **Actions** tab
2. Review **Workflow runs** for patterns
3. Check **Usage** for resource consumption
4. Set up **Status badges** in README.md

## 🔧 Customization Options

### Adjusting Test Coverage Thresholds
```yaml
# In ci.yml, modify coverage settings
--cov-fail-under=80  # Change to desired percentage
```

### Modifying Security Scan Sensitivity
```yaml
# In security.yml, adjust severity levels
severity: 'CRITICAL,HIGH'  # Add MEDIUM, LOW as needed
```

### Customizing Release Process
```yaml
# In release.yml, modify version bump rules
release_type: 'patch'  # Options: patch, minor, major, prerelease
```

## 📞 Support

If you encounter issues with the workflows:

1. **Check the workflow logs** in the Actions tab for detailed error messages
2. **Review this setup guide** for missing configuration steps  
3. **Test locally** using the provided make commands
4. **Validate secrets** are correctly configured and have proper permissions

The workflows are designed to be robust and provide clear error messages to help with troubleshooting.

## 🎯 Next Steps After Setup

1. **Configure branch protection rules** to require workflow success
2. **Set up notification preferences** for workflow failures
3. **Review security scan results** and address any findings
4. **Monitor performance benchmarks** and set up alerts
5. **Schedule regular dependency updates** and security reviews

These workflows will transform your repository into an enterprise-grade development environment with comprehensive automation, security, and monitoring capabilities.