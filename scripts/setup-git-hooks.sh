#!/bin/bash
# Setup Git hooks for autonomous backlog management

set -e

HOOKS_DIR=".git/hooks"

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# prepare-commit-msg hook - Enable rerere
cat > "$HOOKS_DIR/prepare-commit-msg" << 'EOF'
#!/bin/bash
# Enable rerere for this repository
git config rerere.enabled true
git config rerere.autoupdate true
EOF

# pre-push hook - Auto-rebase onto main
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Auto-rebase onto main before push

# Only run on non-main branches
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "main" ]; then
    echo "Auto-rebasing $current_branch onto main..."
    
    # Fetch latest main
    git fetch origin main
    
    # Attempt rebase
    if ! git rebase origin/main; then
        echo "::error::Manual merge required - conflicts detected during rebase"
        echo "Please resolve conflicts manually and retry push"
        exit 1
    fi
    
    echo "Auto-rebase completed successfully"
fi
EOF

# post-merge hook - Update rerere cache
cat > "$HOOKS_DIR/post-merge" << 'EOF'
#!/bin/bash
# Update rerere cache after successful merge

# Share rerere cache if tools branch exists
if git show-ref --verify --quiet refs/heads/tools/rerere-cache; then
    echo "Updating shared rerere cache..."
    git checkout tools/rerere-cache
    cp -r .git/rr-cache/* . 2>/dev/null || true
    git add .
    if ! git diff --cached --quiet; then
        git commit -m "Update rerere cache from merge"
    fi
    git checkout -
fi
EOF

# Make hooks executable
chmod +x "$HOOKS_DIR/prepare-commit-msg"
chmod +x "$HOOKS_DIR/pre-push" 
chmod +x "$HOOKS_DIR/post-merge"

echo "Git hooks installed successfully!"
echo "Hooks installed:"
echo "  - prepare-commit-msg: Enable rerere"
echo "  - pre-push: Auto-rebase onto main"
echo "  - post-merge: Update rerere cache"