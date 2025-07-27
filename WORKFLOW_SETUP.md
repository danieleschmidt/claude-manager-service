# GitHub Workflows Setup Instructions

⚠️ **Manual Setup Required**: Due to GitHub App permissions, the following workflow files need to be added manually:

## 1. Auto-Rebase Workflow

Create `.github/workflows/auto-rebase.yml`:

```yaml
name: auto-rebase
on:
  pull_request_target:
    types: [opened, reopened, synchronize]
jobs:
  rebase:
    runs-on: ubuntu-latest
    permissions: write-all
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
          persist-credentials: false
          token: ${{ secrets.GITHUB_TOKEN }}
      - name: Configure Git
        run: |
          git config --global rerere.enabled true
          git config --global rerere.autoupdate true
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
      - name: Attempt auto-rebase
        run: |
          git fetch origin ${{ github.base_ref }}
          if git rebase origin/${{ github.base_ref }}; then
            echo "✅ Auto-rebase successful"
          else
            echo "::error::Manual merge required - conflicts detected"
            git rebase --abort
            exit 1
          fi
      - name: Push rebased changes
        if: success()
        run: |
          git push origin HEAD:${{ github.head_ref }} --force-with-lease
```

## 2. Rerere Audit Workflow

Create `.github/workflows/rerere-audit.yml`:

```yaml
name: rerere-audit
on:
  pull_request:
    types: [opened, synchronize]
  push:
    branches: [main]
    
jobs:
  audit-rerere:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Configure Git rerere
        run: |
          git config rerere.enabled true
          git config rerere.autoupdate true
      - name: Generate rerere audit report
        run: |
          echo "# Rerere Cache Audit Report" > rerere-audit.md
          echo "Generated: $(date)" >> rerere-audit.md
          echo "" >> rerere-audit.md
          
          if git rerere diff > /dev/null 2>&1; then
            echo "## Automatic Resolutions Available" >> rerere-audit.md
            echo "\`\`\`diff" >> rerere-audit.md
            git rerere diff >> rerere-audit.md || echo "No rerere cache entries found" >> rerere-audit.md
            echo "\`\`\`" >> rerere-audit.md
          else
            echo "## No Automatic Resolutions Found" >> rerere-audit.md
            echo "Clean state - no cached conflict resolutions available." >> rerere-audit.md
          fi
          
          echo "" >> rerere-audit.md
          echo "## Rerere Status" >> rerere-audit.md
          git rerere status >> rerere-audit.md || echo "No rerere status available" >> rerere-audit.md
          
      - name: Upload rerere audit
        uses: actions/upload-artifact@v4
        with:
          name: rerere-audit-report
          path: rerere-audit.md
          retention-days: 30
```

## 3. Setup Instructions

1. Create the `.github/workflows/` directory
2. Add both workflow files with the content above
3. Commit and push the workflows
4. Verify they appear in the Actions tab

## Configuration Summary

✅ **Already Configured:**
- Git rerere enabled globally
- Custom merge drivers (.gitattributes)
- Local Git hooks
- Mergify configuration
- Test validation suite

⏳ **Requires Manual Setup:**
- GitHub Actions workflows (due to permission restrictions)

After adding the workflows, the complete automatic merge conflict resolution system will be active.