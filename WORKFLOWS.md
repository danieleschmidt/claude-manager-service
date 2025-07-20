# GitHub Workflows Setup

Due to permission restrictions, the GitHub workflows need to be created manually. Please create the following files in your repository:

## 1. Create `.github/workflows/1-scan-and-propose.yml`

```yaml
name: 1. Scan Repos and Propose Tasks

on:
  schedule:
    - cron: '0 5 * * *' # Run daily at 5 AM UTC
  workflow_dispatch: # Allows manual triggering

jobs:
  scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write # Required to create issues
    steps:
      - name: Checkout Manager Repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Run Scanner and Create Task Issues
        env:
          GITHUB_TOKEN: ${{ secrets.GH_PAT }}
        run: python src/task_analyzer.py
```

## 2. Create `.github/workflows/2-execute-approved-task.yml`

```yaml
name: 2. Execute Approved Task

on:
  issues:
    types: [labeled]

jobs:
  execute:
    # Run only when the 'approved-for-dev' label is added
    if: github.event.label.name == 'approved-for-dev'
    runs-on: ubuntu-latest
    permissions:
      contents: write # Required to check out code and comment on issues
      issues: write
    steps:
      - name: Checkout Manager Repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Python Dependencies
        run: pip install -r requirements.txt

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install AI Tool CLIs
        run: |
          npm install -g @terragon-labs/cli
          npm install -g claude-flow@alpha
          echo "NOTE: CLI authentication should be handled via secrets in a real environment."
          # Example of auth using tokens from secrets:
          # terry auth --token ${{ secrets.TERRAGON_TOKEN }}

      - name: Run Orchestrator
        env:
          GITHUB_TOKEN: ${{ secrets.GH_PAT }}
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          REPOSITORY_NAME: ${{ github.repository }}
        run: python src/orchestrator.py
```

## Setup Instructions

1. Create the `.github/workflows/` directory in your repository
2. Add both workflow files above to that directory
3. Configure the required GitHub secrets:
   - `GH_PAT` - GitHub Personal Access Token with repo and workflow scopes
   - `TERRAGON_TOKEN` - Terragon CLI authentication token (if available)
   - `CLAUDE_FLOW_TOKEN` - Claude Flow authentication token (if available)

## Testing

1. **Test Workflow 1**: Manually trigger "Scan Repos and Propose Tasks" from the Actions tab
2. **Test Workflow 2**: Add `approved-for-dev` and `terragon-task` labels to a created issue to trigger execution

The workflows are essential for the automation features of the Claude Manager Service.