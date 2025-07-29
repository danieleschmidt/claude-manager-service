# GitHub Workflows Setup Guide

⚠️ **Important**: Due to GitHub App permission restrictions, the workflow files need to be created manually. The workflow configurations have been generated and are documented below.

## Required Workflows

Please create the following files in your `.github/workflows/` directory:

### 1. Continuous Integration - `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.10"
  NODE_VERSION: "18"

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          flake8 src tests
          black --check .
          isort --check-only .

      - name: Run type checking
        run: mypy src

      - name: Run security analysis
        run: |
          bandit -r src/ -f json -o bandit-report.json
          safety check --json --output safety-report.json

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
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

  docker:
    name: Docker Build & Test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: claude-code-manager:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Test Docker image
        run: |
          docker run --rm claude-code-manager:test python -c "import src; print('Import successful')"

  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run performance tests
        run: pytest tests/performance/ --benchmark-json=benchmark.json

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

### 2. Security Audit - `.github/workflows/security-audit.yml`

```yaml
name: Security Audit

on:
  schedule:
    - cron: '0 2 * * 1' # Weekly on Mondays at 2 AM UTC
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
      - '.pre-commit-config.yaml'

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install safety bandit semgrep

      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true
          safety check --short-report

      - name: Run Bandit security analysis
        run: |
          bandit -r src/ -f json -o bandit-report.json
          bandit -r src/ -f txt

      - name: Run Semgrep security scan
        run: |
          semgrep --config=auto --json --output=semgrep-report.json src/
          semgrep --config=auto src/

      - name: Upload security artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json
            semgrep-report.json
          retention-days: 30

      - name: Run CodeQL Analysis
        uses: github/codeql-action/init@v3
        with:
          languages: python
          queries: security-and-quality

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  dependency-vulnerability-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=medium --file=requirements.txt

      - name: Upload Snyk result to GitHub Code Scanning
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: snyk.sarif
```

### 3. Release Automation - `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      version_bump:
        description: 'Version bump type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  test:
    name: Test Before Release
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements-dev.txt

      - name: Run comprehensive tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
          mypy src
          flake8 src tests
          bandit -r src/

  semantic-release:
    name: Semantic Release
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    outputs:
      released: ${{ steps.semantic.outputs.released }}
      version: ${{ steps.semantic.outputs.version }}
      tag: ${{ steps.semantic.outputs.tag }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Python Semantic Release
        run: pip install python-semantic-release

      - name: Python Semantic Release
        id: semantic
        run: |
          semantic-release version --print > version_output.txt
          if [ -s version_output.txt ]; then
            echo "released=true" >> $GITHUB_OUTPUT
            echo "version=$(cat version_output.txt)" >> $GITHUB_OUTPUT
            echo "tag=v$(cat version_output.txt)" >> $GITHUB_OUTPUT
            semantic-release version
            semantic-release publish
          else
            echo "released=false" >> $GITHUB_OUTPUT
          fi
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  docker-release:
    name: Docker Release
    runs-on: ubuntu-latest
    needs: semantic-release
    if: needs.semantic-release.outputs.released == 'true'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository_owner }}/claude-code-manager
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}},value=${{ needs.semantic-release.outputs.version }}
            type=semver,pattern={{major}}.{{minor}},value=${{ needs.semantic-release.outputs.version }}
            type=semver,pattern={{major}},value=${{ needs.semantic-release.outputs.version }}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  github-release:
    name: GitHub Release
    runs-on: ubuntu-latest
    needs: [semantic-release, docker-release]
    if: needs.semantic-release.outputs.released == 'true'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Generate changelog
        id: changelog
        run: |
          git log --pretty=format:"* %s (%h)" $(git describe --tags --abbrev=0 HEAD~1)..HEAD > CHANGELOG_TEMP.md
          echo "CHANGELOG<<EOF" >> $GITHUB_OUTPUT
          cat CHANGELOG_TEMP.md >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ needs.semantic-release.outputs.tag }}
          release_name: Release ${{ needs.semantic-release.outputs.version }}
          body: |
            ## Changes
            ${{ steps.changelog.outputs.CHANGELOG }}
            
            ## Docker Images
            - `ghcr.io/${{ github.repository_owner }}/claude-code-manager:${{ needs.semantic-release.outputs.version }}`
            - `ghcr.io/${{ github.repository_owner }}/claude-code-manager:latest`
          draft: false
          prerelease: false
```

### 4. Dependency Review - `.github/workflows/dependency-review.yml`

```yaml
name: Dependency Review

on:
  pull_request:
    branches: [main, develop]

permissions:
  contents: read
  pull-requests: write

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: moderate
          allow-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD
          deny-licenses: GPL-2.0, GPL-3.0, LGPL-2.0, LGPL-3.0
          comment-summary-in-pr: always
```

### 5. Performance Monitoring - `.github/workflows/performance-monitoring.yml`

```yaml
name: Performance Monitoring

on:
  schedule:
    - cron: '0 */6 * * *' # Every 6 hours
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'tests/performance/**'

jobs:
  performance-test:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements-dev.txt
          pip install pytest-benchmark memory-profiler py-spy

      - name: Run performance benchmarks
        run: |
          pytest tests/performance/ --benchmark-json=benchmark.json --benchmark-histogram=benchmark-histogram

      - name: Memory profiling
        run: |
          python -m memory_profiler scripts/memory-profile-test.py

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          comment-on-alert: true
          alert-threshold: '150%'
          fail-on-alert: true

      - name: Upload performance artifacts
        uses: actions/upload-artifact@v4
        with:
          name: performance-reports
          path: |
            benchmark.json
            benchmark-histogram.svg
            *.prof
          retention-days: 30
```

### 6. Dependabot Configuration - `.github/dependabot.yml`

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "04:00"
    open-pull-requests-limit: 10
    reviewers:
      - "terragon-labs"
    assignees:
      - "terragon-labs"
    commit-message:
      prefix: "deps"
      prefix-development: "deps-dev"
      include: "scope"
    labels:
      - "dependencies"
      - "python"
    groups:
      development-dependencies:
        patterns:
          - "pytest*"
          - "black"
          - "isort"
          - "mypy"
          - "flake8*"
          - "pylint"
          - "bandit"
          - "safety"
          - "pre-commit"
        update-types:
          - "minor"
          - "patch"
      security-dependencies:
        patterns:
          - "cryptography"
          - "requests"
          - "urllib3"
        update-types:
          - "patch"
      production-dependencies:
        patterns:
          - "*"
        exclude-patterns:
          - "pytest*"
          - "black"
          - "isort" 
          - "mypy"
          - "flake8*"
          - "pylint"
          - "bandit"
          - "safety"
          - "pre-commit"
        update-types:
          - "minor"
          - "patch"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "sunday"
    reviewers:
      - "terragon-labs"
    labels:
      - "dependencies"
      - "docker"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "tuesday"
    reviewers:
      - "terragon-labs"
    labels:
      - "dependencies"
      - "github-actions"
```

## Setup Instructions

1. **Create the `.github/workflows/` directory** in your repository if it doesn't exist
2. **Copy each workflow file** from the code blocks above into the appropriate location
3. **Configure GitHub secrets** for the workflows to function properly:
   - `CODECOV_TOKEN` - For coverage reporting (optional)
   - `SNYK_TOKEN` - For Snyk vulnerability scanning (optional)
   - Ensure `GITHUB_TOKEN` has appropriate permissions

4. **Enable GitHub Actions** in your repository settings
5. **Configure branch protection rules** to require CI checks before merging

## Features Provided

- **Comprehensive CI/CD**: Multi-Python testing, linting, security scanning
- **Automated Security**: Weekly security audits with multiple tools
- **Release Automation**: Semantic versioning with Docker builds
- **Dependency Management**: Automated dependency updates with security prioritization
- **Performance Monitoring**: Automated benchmarking with regression detection

These workflows provide enterprise-grade CI/CD capabilities with comprehensive security, quality, and performance monitoring for your repository.