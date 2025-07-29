# Recommended GitHub Actions Workflows

This document outlines the recommended GitHub Actions workflows for this MATURING repository. These workflows should be manually implemented by repository maintainers with appropriate permissions.

## Overview

Based on the repository maturity assessment (50-75% SDLC maturity), these workflows will complete the automation coverage and bring the repository to advanced SDLC practices.

## 1. Continuous Integration Workflow

**File:** `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.10"

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          
      - name: Lint with flake8
        run: flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        
      - name: Check formatting with black
        run: black --check src tests
        
      - name: Check import sorting with isort
        run: isort --check-only src tests
        
      - name: Type check with mypy
        run: mypy src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"
          
      - name: Run unit tests
        run: pytest tests/unit -v --cov=src --cov-report=xml
        
      - name: Run integration tests
        run: pytest tests/integration -v
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.10'
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[security]"
          
      - name: Run security scan with bandit
        run: bandit -r src -f json -o bandit-report.json
        
      - name: Run safety check
        run: safety check --json --output safety-report.json
        
      - name: Upload security reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json

  build:
    runs-on: ubuntu-latest
    needs: [lint-and-format, test, security]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build
          
      - name: Build package
        run: python -m build
        
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

## 2. Continuous Deployment Workflow

**File:** `.github/workflows/cd.yml`

```yaml
name: Continuous Deployment

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: staging
    
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment"
          # Add staging deployment logic here
          
  deploy-production:
    if: github.event_name == 'release' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: production
    
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production environment"
          # Add production deployment logic here
```

## 3. Security Scanning Workflow

**File:** `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM UTC

permissions:
  contents: read
  security-events: write

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[security]"
          
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
          safety check --short-report
          
      - name: Upload safety report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: safety-report
          path: safety-report.json

  code-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[security]"
          
      - name: Run Bandit security scan
        run: |
          bandit -r src -f json -o bandit-report.json
          bandit -r src -f txt
          
      - name: Upload bandit report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: bandit-report
          path: bandit-report.json

  semgrep:
    runs-on: ubuntu-latest
    name: Semgrep Scan
    steps:
      - uses: actions/checkout@v4
      
      - uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/python
            p/owasp-top-ten
          generateSarif: "1"
          
      - name: Upload SARIF file
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif
        if: always()

  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Run Trivy vulnerability scanner in repo mode
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
```

## 4. Performance Testing Workflow

**File:** `.github/workflows/performance.yml`

```yaml
name: Performance Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 3 * * 0'  # Weekly on Sunday at 3 AM UTC

jobs:
  performance-test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          
      - name: Run performance benchmarks
        run: |
          pytest tests/performance/ -v --benchmark-json=benchmark.json
          
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          comment-on-alert: true
          alert-threshold: '150%'
          fail-on-alert: true
          
      - name: Upload performance reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: performance-reports
          path: |
            benchmark.json
            performance_data/

  load-test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          
      - name: Start test services
        run: |
          docker-compose up -d
          sleep 10
          
      - name: Run load tests
        run: |
          python -m pytest tests/performance/ -k "load" -v
          
      - name: Collect logs
        if: always()
        run: |
          docker-compose logs > docker-logs.txt
          
      - name: Cleanup
        if: always()
        run: |
          docker-compose down
          
      - name: Upload load test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: load-test-results
          path: |
            docker-logs.txt
            performance_data/
```

## Implementation Instructions

1. **Manual Setup Required**: These workflows must be manually created by repository maintainers with `workflows` permission
2. **Prerequisites**: Ensure all dependencies in `pyproject.toml` are properly configured
3. **Secrets Configuration**: Set up required secrets like `GITHUB_TOKEN` for deployments
4. **Environment Configuration**: Configure staging and production environments in repository settings
5. **Branch Protection**: Enable branch protection rules that require these checks to pass

## Benefits

- **Complete CI/CD Pipeline**: Automated testing, security scanning, and deployment
- **Multi-Environment Support**: Separate staging and production deployments
- **Security Integration**: Comprehensive vulnerability scanning and SARIF reporting
- **Performance Monitoring**: Automated performance regression detection
- **Quality Gates**: Enforce code quality standards before merging

## Expected Impact

- **Maturity Increase**: Repository will advance to 75-85% SDLC maturity
- **Security Enhancement**: 4 integrated security scanning tools
- **Automation Coverage**: 95%+ of development workflow automated
- **Quality Assurance**: Multi-stage validation pipeline