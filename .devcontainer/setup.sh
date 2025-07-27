#!/bin/bash

set -e

echo "🚀 Setting up Claude Code Manager development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    sqlite3 \
    postgresql-client \
    redis-tools

# Set up Python environment
echo "🐍 Setting up Python environment..."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt 2>/dev/null || echo "requirements-dev.txt not found, skipping..."

# Install development tools
echo "🛠️ Installing development tools..."
pip install \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    pre-commit \
    bandit \
    safety

# Install Node.js tools
echo "📦 Installing Node.js tools..."
npm install -g \
    @anthropic-ai/claude-code \
    claude-flow@alpha \
    prettier \
    eslint

# Set up pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    logs \
    data \
    temp \
    .pytest_cache \
    htmlcov \
    docs/guides \
    docs/runbooks \
    .github/workflows

# Set up environment file template
echo "🌍 Creating environment template..."
cat > .env.example << EOF
# GitHub Configuration
GITHUB_TOKEN=your_github_token_here
GITHUB_USERNAME=your_username
GITHUB_REPO=your_repo_name

# AI Service Configuration
TERRAGON_TOKEN=your_terragon_token
CLAUDE_FLOW_TOKEN=your_claude_flow_token

# Database Configuration
DATABASE_URL=sqlite:///data/tasks.db

# Performance Configuration
PERF_ALERT_DURATION=15.0
PERF_MAX_OPERATIONS=20000
PERF_RETENTION_HOURS=168

# Security Configuration
SECURITY_MAX_CONTENT_LENGTH=75000
SECURITY_ENABLE_RATE_LIMITING=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Feature Flags
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_ENHANCED_SECURITY=true
ENABLE_AUTONOMOUS_MODE=true

# Web Dashboard Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_DEBUG=false
FLASK_PORT=5000
EOF

# Set up Git configuration
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Install Claude CLI if not present
echo "🤖 Setting up Claude CLI..."
if ! command -v claude &> /dev/null; then
    echo "Installing Claude CLI..."
    npm install -g @anthropic-ai/claude-code
fi

# Set up database
echo "🗄️ Initializing database..."
python -c "
from src.services.database_service import DatabaseService
db = DatabaseService()
db.initialize_database()
print('Database initialized successfully')
" 2>/dev/null || echo "Database initialization skipped (missing dependencies)"

# Create initial project structure documentation
echo "📚 Creating project documentation..."
cat > docs/DEVELOPMENT.md << 'EOF'
# Development Guide

## Getting Started

1. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Run Tests**
   ```bash
   pytest
   pytest --cov=src tests/
   ```

4. **Start Development Server**
   ```bash
   python start_dashboard.py
   ```

## Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write Tests First**
   ```bash
   # Create tests in tests/unit/ or tests/integration/
   pytest tests/unit/test_your_feature.py
   ```

3. **Implement Feature**
   ```bash
   # Write code in src/
   # Run tests frequently
   pytest
   ```

4. **Check Code Quality**
   ```bash
   black .
   isort .
   flake8 src tests
   mypy src
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

## Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Coverage Report
```bash
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

## Code Quality

### Formatting
```bash
black .
isort .
```

### Linting
```bash
flake8 src tests
pylint src
```

### Type Checking
```bash
mypy src
```

### Security Scanning
```bash
bandit -r src
safety check
```

## Database Management

### Migrations
```bash
python -c "from src.database_migration_utility import migrate; migrate()"
```

### Reset Database
```bash
rm data/tasks.db
python -c "from src.services.database_service import DatabaseService; DatabaseService().initialize_database()"
```

## Debugging

### Verbose Logging
```bash
export LOG_LEVEL=DEBUG
python your_script.py
```

### Performance Profiling
```bash
python -m cProfile -o profile.stats your_script.py
```

### Memory Profiling
```bash
pip install memory-profiler
python -m memory_profiler your_script.py
```
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy .env.example to .env and configure your tokens"
echo "2. Run 'pytest' to ensure everything is working"
echo "3. Start coding with 'python start_dashboard.py'"
echo ""
echo "📖 Documentation:"
echo "- Architecture: ARCHITECTURE.md"
echo "- Development: docs/DEVELOPMENT.md"
echo "- Roadmap: docs/ROADMAP.md"
echo ""
echo "🔗 Useful commands:"
echo "- Run tests: pytest"
echo "- Format code: black . && isort ."
echo "- Type check: mypy src"
echo "- Start dashboard: python start_dashboard.py"