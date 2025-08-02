# Multi-stage build for Claude Code Manager
# Stage 1: Base image with Python and system dependencies
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for CLI tools
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Stage 2: Development dependencies
FROM base AS dev-deps

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Install Node.js tools
RUN npm install -g \
    @anthropic-ai/claude-code \
    claude-flow@alpha

# Stage 3: Production dependencies
FROM base AS prod-deps

# Copy requirements
COPY requirements.txt ./

# Install only production dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-dev -r requirements.txt

# Install essential Node.js tools
RUN npm install -g \
    @anthropic-ai/claude-code \
    claude-flow@alpha

# Stage 4: Development image
FROM dev-deps AS development

# Set working directory
WORKDIR /app

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Install pre-commit hooks
RUN pre-commit install || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.health_check; src.health_check.check()" || exit 1

# Default command
CMD ["python", "start_dashboard.py"]

# Stage 5: Production image
FROM prod-deps AS production

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code (only necessary files)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser web/ ./web/
COPY --chown=appuser:appuser prompts/ ./prompts/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser *.py ./
COPY --chown=appuser:appuser config.json ./
COPY --chown=appuser:appuser mypy.ini ./
COPY --chown=appuser:appuser pytest.ini ./

# Security: Remove any sensitive files
RUN find . -name "*.md" -type f -delete && \
    find . -name "test_*" -type f -delete && \
    find . -name "*_test.py" -type f -delete

# Set secure permissions
RUN chmod -R 750 /app && \
    chmod -R 640 /app/src && \
    chmod 750 /app/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.health_check; src.health_check.check()" || exit 1

# Expose port
EXPOSE 5000

# Default command
CMD ["python", "start_dashboard.py"]

# Stage 6: Testing image
FROM dev-deps AS testing

# Set working directory
WORKDIR /app

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy everything for testing
COPY --chown=appuser:appuser . .

# Install development dependencies
RUN pip install pytest pytest-cov pytest-asyncio pytest-mock

# Run tests by default
CMD ["pytest", "--cov=src", "--cov-report=xml", "--cov-report=html"]

# Stage 7: Security scanning image
FROM base AS security

# Install security tools
RUN pip install bandit safety semgrep

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/

# Run security scans
RUN bandit -r src/ -f json -o bandit-report.json || true && \
    safety check --json --output safety-report.json || true

# Default command
CMD ["bandit", "-r", "src/"]

# Labels for metadata
LABEL maintainer="Terragon Labs <dev@terragon.ai>" \
      description="Claude Code Manager - Autonomous SDLC Management" \
      version="0.1.0" \
      org.opencontainers.image.title="Claude Code Manager" \
      org.opencontainers.image.description="Autonomous software development lifecycle management system" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/terragon-labs/claude-code-manager" \
      org.opencontainers.image.documentation="https://claude-code-manager.readthedocs.io/"