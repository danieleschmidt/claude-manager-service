# GitHub Actions Workflow Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying GitHub Actions workflows in the Claude Code Manager repository. These workflows provide comprehensive CI/CD, security scanning, and automation capabilities.

## Prerequisites

### Required Tools

- Git (version 2.0+)
- GitHub CLI (optional but recommended)
- Text editor or IDE
- Repository admin access

### Repository Requirements

- GitHub repository with Actions enabled
- Main branch protection (recommended)
- Appropriate permissions for workflow creation

## Quick Deployment

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
./scripts/setup-workflows.sh

# Follow the prompts to configure secrets
# Push changes to trigger workflows
git push origin main
```

### Option 2: Manual Setup

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/github-workflows-templates/*.yml .github/workflows/

# Commit and push
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows"
git push origin main
```

## Workflow Configuration

### 1. Core CI Pipeline (`ci.yml`)

**Purpose**: Continuous Integration with testing and quality checks

**Triggers**:
- Push to main/develop branches
- Pull requests to main/develop

**Key Features**:
- Multi-Python version testing (3.10, 3.11, 3.12)
- Code quality checks (linting, formatting, type checking)
- Security scanning with Bandit
- Test coverage reporting
- Docker image building

**Configuration Options**:

```yaml
# Customize Python versions
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']  # Modify as needed

# Adjust test commands
- name: Run tests
  run: |
    pytest tests/ --cov=src --cov-report=xml
    # Add your custom test commands here
```

### 2. Security Scanning (`security.yml`)

**Purpose**: Comprehensive security analysis

**Triggers**:
- Push to main branch
- Pull requests
- Weekly scheduled scan

**Security Tools**:
- CodeQL (semantic analysis)
- Bandit (Python security linting)
- Safety (dependency vulnerability scanning)
- Semgrep (SAST)
- Container security scanning
- Secret detection

**Configuration**:

```yaml
# Customize security thresholds
- name: Run Bandit security check
  run: |
    bandit -r src/ -ll  # Adjust severity level (ll = low, l = low)
    
# Configure CodeQL languages
strategy:
  matrix:
    language: ['python']  # Add other languages if needed
```

### 3. Release Automation (`release.yml`)

**Purpose**: Automated releases with semantic versioning

**Triggers**:
- Git tags (v*.*.*)
- Manual workflow dispatch

**Features**:
- Semantic version calculation
- Automated changelog generation
- Multi-platform Docker builds
- Package publishing (PyPI, Docker Hub)
- GitHub release creation

**Configuration**:

```yaml
# Customize release settings
env:
  REGISTRY: ghcr.io  # Change to your preferred registry
  IMAGE_NAME: ${{ github.repository }}
  
# Configure publishing targets
- name: Publish to PyPI
  if: github.event_name == 'release'  # Only on releases
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

### 4. Performance Testing (`performance.yml`)

**Purpose**: Automated performance monitoring

**Triggers**:
- Push to main branch
- Weekly scheduled runs
- Manual dispatch

**Features**:
- Benchmark testing with pytest-benchmark
- Load testing with Locust
- Memory profiling
- Performance regression detection

**Configuration**:

```yaml
# Customize performance thresholds
- name: Run performance tests
  run: |
    pytest tests/performance/ --benchmark-min-rounds=5
    # Adjust benchmark parameters as needed
    
# Configure load testing
- name: Load testing
  run: |
    locust -f tests/load/locustfile.py --headless -u 10 -r 2 -t 30s
```

### 5. Dependency Management (`dependency-management.yml`)

**Purpose**: Automated dependency updates

**Triggers**:
- Weekly schedule (Mondays at 9 AM)
- Manual dispatch

**Features**:
- Security update detection
- Automated PR creation
- Compatibility testing
- Breaking change detection

## Secret Configuration

### Required Secrets

Configure these secrets in your repository settings (`Settings > Secrets and variables > Actions`):

#### Essential Secrets

```bash
# Docker Hub credentials (for image publishing)
DOCKERHUB_USERNAME=your_dockerhub_username
DOCKERHUB_TOKEN=your_dockerhub_access_token

# Security scanning
SNYK_TOKEN=your_snyk_token
```

#### Optional Secrets

```bash
# PyPI publishing
PYPI_API_TOKEN=pypi-your_token_here

# License compliance
FOSSA_API_KEY=your_fossa_api_key

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# External monitoring
DATADOG_API_KEY=your_datadog_key
DATADOG_APP_KEY=your_datadog_app_key
```

### Secret Setup Commands

```bash
# Using GitHub CLI
gh secret set DOCKERHUB_USERNAME --body "your_username"
gh secret set DOCKERHUB_TOKEN --body "your_token"
gh secret set SNYK_TOKEN --body "your_snyk_token"

# Verify secrets
gh secret list
```

## Environment Variables

### Repository Variables

Set these in `Settings > Secrets and variables > Actions > Variables`:

```bash
# Container registry
REGISTRY=ghcr.io
IMAGE_NAME=claude-code-manager

# Performance thresholds
PERF_CPU_THRESHOLD=80
PERF_MEMORY_THRESHOLD=85
PERF_RESPONSE_TIME_THRESHOLD=200

# Security settings
SECURITY_SCAN_LEVEL=medium
VULN_SEVERITY_THRESHOLD=high
```

## Branch Protection Rules

### Recommended Settings

```yaml
# .github/branch-protection.yml (if using settings app)
branchProtectionRules:
  - pattern: "main"
    requiredStatusChecks:
      strict: true
      contexts:
        - "CI Pipeline / test (3.10)"
        - "CI Pipeline / test (3.11)"
        - "CI Pipeline / test (3.12)"
        - "Security Scanning / security-scan"
    enforceAdmins: true
    requiredPullRequestReviews:
      requiredApprovingReviewCount: 1
      dismissStaleReviews: true
    restrictions: null
```

### Manual Configuration

1. Go to repository `Settings > Branches`
2. Add rule for `main` branch
3. Enable "Require status checks to pass before merging"
4. Select the following status checks:
   - CI Pipeline tests
   - Security scanning
   - Code quality checks
5. Enable "Require pull request reviews before merging"

## Monitoring and Troubleshooting

### Workflow Status Monitoring

```bash
# Check workflow runs
gh run list

# View specific run details
gh run view <run-id>

# Download logs
gh run download <run-id>
```

### Common Issues and Solutions

#### 1. Permission Errors

**Error**: `Permission denied` or `403 Forbidden`

**Solutions**:
- Verify repository permissions
- Check if Actions are enabled
- Ensure secrets are properly configured
- Verify token scopes

#### 2. Test Failures

**Error**: Tests fail in CI but pass locally

**Solutions**:
- Check environment differences
- Verify dependency versions
- Review test isolation
- Check for timing issues

#### 3. Security Scan Failures

**Error**: Security tools report vulnerabilities

**Solutions**:
- Review vulnerability reports
- Update dependencies
- Apply security patches
- Add exemptions for false positives

#### 4. Docker Build Issues

**Error**: Docker build or push failures

**Solutions**:
- Verify Dockerfile syntax
- Check Docker Hub credentials
- Review build context
- Ensure sufficient disk space

### Debugging Workflows

#### Enable Debug Logging

```yaml
# Add to workflow file
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

#### Common Debug Commands

```yaml
# Debug environment
- name: Debug environment
  run: |
    echo "GitHub context:"
    echo "$GITHUB_CONTEXT"
    echo "Environment variables:"
    env | sort
    echo "File system:"
    ls -la
```

## Performance Optimization

### Caching Strategies

```yaml
# Cache Python dependencies
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
# Cache Docker layers
- name: Cache Docker layers
  uses: actions/cache@v4
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
    restore-keys: |
      ${{ runner.os }}-buildx-
```

### Matrix Optimization

```yaml
# Optimize test matrix
strategy:
  fail-fast: false  # Continue other jobs if one fails
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest]  # Add other OS if needed
    exclude:
      # Exclude specific combinations if needed
      - python-version: '3.12'
        os: windows-latest
```

## Maintenance and Updates

### Regular Maintenance Tasks

1. **Weekly Review**:
   - Check workflow success rates
   - Review security scan results
   - Update dependencies
   - Monitor performance trends

2. **Monthly Updates**:
   - Update workflow actions to latest versions
   - Review and update security thresholds
   - Optimize performance baselines
   - Update documentation

3. **Quarterly Review**:
   - Evaluate workflow effectiveness
   - Add new security tools if available
   - Review and update processes
   - Conduct workflow training

### Automated Updates

```yaml
# Dependabot configuration for workflow updates
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

## Best Practices

### Security

1. **Secret Management**:
   - Use repository secrets for sensitive data
   - Rotate secrets regularly
   - Use least privilege principle
   - Audit secret access

2. **Permissions**:
   - Use `permissions` key to limit workflow permissions
   - Avoid `contents: write` unless necessary
   - Use specific permissions per job

3. **Supply Chain Security**:
   - Pin action versions to specific SHAs
   - Use verified actions from trusted publishers
   - Regular dependency scanning

### Performance

1. **Resource Management**:
   - Use appropriate runner sizes
   - Implement efficient caching
   - Parallelize independent jobs
   - Use fail-fast strategies

2. **Cost Optimization**:
   - Monitor Actions usage
   - Optimize workflow triggers
   - Use conditional job execution
   - Cache expensive operations

### Reliability

1. **Error Handling**:
   - Use `continue-on-error` judiciously
   - Implement retry mechanisms
   - Add meaningful error messages
   - Use proper exit codes

2. **Testing**:
   - Test workflows in feature branches
   - Use matrix builds for comprehensive testing
   - Implement smoke tests
   - Monitor workflow stability

## Integration with External Tools

### Slack Notifications

```yaml
- name: Slack notification
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#ci-cd'
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Datadog Integration

```yaml
- name: Send metrics to Datadog
  uses: DataDog/datadog-ci-action@v1
  with:
    api-key: ${{ secrets.DATADOG_API_KEY }}
    metrics: |
      ci.pipeline.duration:${{ steps.timing.outputs.duration }}
      ci.tests.count:${{ steps.tests.outputs.count }}
```

### SonarQube Integration

```yaml
- name: SonarQube Scan
  uses: sonarqube-quality-gate-action@master
  env:
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## Support and Resources

### Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

### Community Resources

- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [Awesome Actions](https://github.com/sdras/awesome-actions)
- [GitHub Community Forum](https://github.community/)

### Internal Support

- Repository Issues: Create an issue for workflow problems
- Team Slack: #devops or #ci-cd channels
- Documentation: Check workflow-specific README files

---

**Last Updated**: 2025-08-02  
**Next Review**: 2025-09-02  
**Maintainer**: DevOps Team