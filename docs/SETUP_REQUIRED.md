# Manual Setup Requirements

This document lists items that require manual setup due to permission limitations.

## GitHub Actions Workflows

### Required Workflow Files
Create these files in `.github/workflows/` directory:

1. **ci.yml** - Continuous Integration
   * Python testing with pytest
   * Code quality checks (flake8, mypy, black)
   * Security scanning with bandit

2. **security.yml** - Security Scanning
   * Daily dependency vulnerability scans
   * CodeQL analysis for code security
   * Container image scanning

3. **release.yml** - Release Automation
   * Automated releases on version tags
   * Generate release notes from commits
   * Build and publish artifacts

### Workflow Templates
* Reference implementations available in `templates/github-workflows/`
* Customize for your specific environment and requirements

## Repository Settings

### Branch Protection Rules
Navigate to Settings → Branches → Add rule for `main`:
* ✅ Require pull request reviews before merging (1 reviewer minimum)
* ✅ Require status checks to pass before merging
* ✅ Require branches to be up to date before merging
* ✅ Include administrators in restrictions

### Repository Topics
Add these topics in Settings → General:
* `python` `automation` `github-api` `task-management` `ai-assistant`

### Security Settings
Enable in Settings → Security:
* ✅ Dependency graph
* ✅ Dependabot alerts
* ✅ Dependabot security updates
* ✅ Code scanning (CodeQL)

## External Integrations

### Monitoring Setup
* Configure monitoring service integration
* Set up alerting for critical failures
* Enable performance tracking

### Documentation Hosting
* Set up GitHub Pages or external docs hosting
* Configure automatic documentation updates

## Environment Variables

Set these in Settings → Secrets and variables → Actions:
```
GITHUB_TOKEN (automatically provided)
SECURITY_EMAIL (for security notifications)
```

## Review Checklist

- [ ] GitHub Actions workflows created and enabled
- [ ] Branch protection rules configured
- [ ] Security features enabled
- [ ] Repository topics and description set
- [ ] External integrations configured
- [ ] Documentation hosting set up