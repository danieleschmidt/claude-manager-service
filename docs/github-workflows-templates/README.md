# GitHub Actions Workflow Templates

This directory contains comprehensive GitHub Actions workflows created by the autonomous SDLC enhancement process. These templates provide enterprise-grade CI/CD, security scanning, and operational automation.

## 🚀 Quick Start

```bash
# Copy all workflow files to your .github/workflows/ directory
cp docs/github-workflows-templates/*.yml .github/workflows/

# Commit and push the workflows
git add .github/workflows/
git commit -m "feat: add comprehensive GitHub Actions workflows"
git push
```

## 📁 Workflow Files

| File | Purpose | Triggers | Duration |
|------|---------|----------|----------|
| `ci.yml` | Core CI pipeline | Push, PR | 5-8 min |
| `ci-cd.yml` | Complete CI/CD | Push to main, PR | 10-15 min |
| `release.yml` | Release automation | Tags, manual | 15-20 min |
| `security.yml` | Security scanning | Push, PR, schedule | 8-12 min |
| `dependency-management.yml` | Dependency updates | Weekly schedule | 5-10 min |
| `performance.yml` | Performance testing | Push, schedule | 10-15 min |

## 🔑 Required Secrets

Before using these workflows, configure these secrets in your repository:

### Essential Secrets
- `GITHUB_TOKEN` (automatically provided)
- `SNYK_TOKEN` (security scanning)
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (Docker publishing)

### Optional Secrets  
- `FOSSA_API_KEY` (license compliance)
- `PYPI_API_TOKEN` (PyPI publishing)
- `SLACK_WEBHOOK_URL` (notifications)

## 📖 Detailed Setup

See [GITHUB_WORKFLOWS_SETUP.md](../GITHUB_WORKFLOWS_SETUP.md) for complete setup instructions, troubleshooting, and customization options.

## 🎯 Workflow Features

### CI Pipeline (`ci.yml`)
- Multi-Python version testing (3.10, 3.11, 3.12)
- Code quality checks (linting, type checking)
- Security scanning (bandit, safety)
- Test coverage reporting
- Docker image building and security scanning

### Security Scanning (`security.yml`) 
- CodeQL semantic analysis
- SAST with multiple tools (Bandit, Semgrep)
- Dependency vulnerability scanning
- Container security scanning
- Secret detection
- SBOM generation
- License compliance checking

### Release Automation (`release.yml`)
- Semantic version calculation
- Automated changelog generation  
- Multi-platform Docker builds (amd64, arm64)
- Package publishing (PyPI, Docker registries)
- Release announcement automation

### Performance Testing (`performance.yml`)
- Automated benchmarking with pytest-benchmark
- Load testing with Locust
- Memory profiling
- Performance regression detection

## 🔧 Customization

Each workflow can be customized by modifying:

- **Test commands**: Update to match your testing setup
- **Security thresholds**: Adjust severity levels as needed
- **Performance baselines**: Set appropriate benchmark thresholds
- **Notification settings**: Configure Slack/email alerts

## 📊 Expected Benefits

After implementing these workflows, you'll have:

- ✅ **Automated Quality Gates**: No broken code reaches main branch
- ✅ **Security Assurance**: Comprehensive vulnerability detection
- ✅ **Release Automation**: One-click releases with full traceability
- ✅ **Performance Monitoring**: Automated performance regression detection
- ✅ **Dependency Management**: Automated security updates

## 🚨 Important Notes

1. **GitHub App Limitations**: These files are provided as templates because GitHub Apps cannot directly create workflow files
2. **Resource Usage**: Workflows will consume GitHub Actions minutes from your plan
3. **Security Scanning**: Some tools require external service tokens for full functionality
4. **Permissions**: Ensure repository has necessary permissions for publishing packages

## 📞 Support

For issues with workflow setup or execution, refer to:
1. The comprehensive setup guide: [GITHUB_WORKFLOWS_SETUP.md](../GITHUB_WORKFLOWS_SETUP.md)
2. GitHub Actions documentation: https://docs.github.com/en/actions
3. Individual workflow logs in the Actions tab of your repository

These workflows represent best practices for modern software development and will significantly enhance your repository's maturity and operational excellence.