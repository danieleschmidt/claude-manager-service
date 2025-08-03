#!/bin/bash
set -e

echo "🚀 Setting up Claude Manager Service development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional development tools
echo "🔧 Installing development tools..."
sudo apt-get install -y \
    curl \
    wget \
    unzip \
    jq \
    httpie \
    tree \
    fd-find \
    ripgrep \
    bat \
    git-lfs \
    make \
    build-essential

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development requirements
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
fi

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup PYTHONPATH
echo "📍 Setting up PYTHONPATH..."
echo 'export PYTHONPATH="${PYTHONPATH}:/workspaces/repo/src"' >> ~/.bashrc

# Setup Git configuration for development
echo "🔧 Configuring Git for development..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global push.autoSetupRemote true

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p logs data temp backups

# Setup development database
echo "🗄️ Setting up development database..."
if [ ! -f "data/tasks.db" ]; then
    python -c "
import sqlite3
conn = sqlite3.connect('data/tasks.db')
conn.execute('CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, title TEXT, status TEXT, created_at TIMESTAMP)')
conn.close()
print('✅ Development database initialized')
"
fi

# Install GitHub CLI if not present
if ! command -v gh &> /dev/null; then
    echo "📦 Installing GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list
    sudo apt update
    sudo apt install gh -y
fi

# Setup environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating development environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration"
fi

# Setup git hooks
echo "🪝 Setting up git hooks..."
if [ -f "scripts/setup-git-hooks.sh" ]; then
    bash scripts/setup-git-hooks.sh
fi

# Run initial validation
echo "🔍 Running initial validation..."
if [ -f "scripts/validate-build.py" ]; then
    python scripts/validate-build.py
fi

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Run 'make test' to verify everything works"
echo "3. Run 'python start_dashboard.py' to start the development server"
echo "4. Visit http://localhost:5000 to view the dashboard"