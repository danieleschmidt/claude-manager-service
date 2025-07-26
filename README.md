# Claude Code Manager Service

This service automates the management of coding tasks across your GitHub repositories by leveraging Terragon and Claude Flow. It scans for potential work, proposes it as a GitHub issue, and executes approved tasks using AI agents.

## 🚀 Development Plan

### Phase 1: Initial Setup & Configuration

- [ ] **1.1: Create GitHub Repository:** Create a new, private GitHub repository named `claude-manager-service`.
- [ ] **1.2: Populate Files:** Populate the repository with all the files from this scaffolding.
- [ ] **1.3: Configure `config.json`:** Edit the `config.json` file to include your GitHub username, the full name of this manager repository, and a list of target repositories you want the service to scan.
- [ ] **1.4: Set up GitHub Secrets:**
  - [ ] Create a GitHub Personal Access Token (PAT) with `repo` and `workflow` scopes.
  - [ ] In the manager repository's settings (`Settings > Secrets and variables > Actions`), create a new repository secret named `GH_PAT` and paste your PAT.
  - [ ] Create a repository secret named `TERRAGON_TOKEN` for Terragon CLI authentication (if available).
  - [ ] Create a repository secret named `CLAUDE_FLOW_TOKEN` for Claude Flow authentication (if available).
- [ ] **1.5: Install Python Dependencies:** Set up a local Python virtual environment and run `pip install -r requirements.txt`.

### Phase 2: Core Logic Implementation

- [ ] **2.1: `github_api.py` - Implement `create_issue`:** Complete the function to check for existing issues with the same title before creating a new one to prevent duplicates.
- [ ] **2.2: `github_api.py` - Implement `add_comment_to_issue`:** Complete the function to post a comment to a specified issue. This will be used to trigger Terragon.
- [ ] **2.3: `task_analyzer.py` - Implement `find_todo_comments`:** Use the GitHub API's code search functionality to find files containing "TODO:" or "FIXME:". For each finding, extract the relevant code block and line number.
- [ ] **2.4: `task_analyzer.py` - Implement `analyze_open_issues`:** Scan through the issues of a target repository. Identify issues with labels like `bug` or `help-wanted` that have not been active recently.
- [ ] **2.5: `task_analyzer.py` - Main Logic:** Integrate the implemented analyzer functions into the main execution block of the script.
- [ ] **2.6: `prompt_builder.py` - Implement `build_prompt`:** Write the logic to load a prompt template and dynamically insert context (like issue title, body, code snippets, etc.) into it.
- [ ] **2.7: `orchestrator.py` - Implement `trigger_terragon_task`:** Use the `github_api.add_comment_to_issue` function to post a fully formatted prompt to the triggering issue. The prompt should mention `@terragon-labs`.
- [ ] **2.8: `orchestrator.py` - Implement `trigger_claude_flow_task`:**
  - [ ] Add logic to `git clone` the specific repository mentioned in the task issue.
  - [ ] Implement the `subprocess.run` call to execute the `claude-flow hive-mind spawn` command within the cloned repository's directory.
  - [ ] Ensure proper error handling and logging of the output from the CLI command.
- [ ] **2.9: `orchestrator.py` - Main Logic:** Refine the decision-making logic to use issue labels (`terragon-task`, `claude-flow-task`) to determine which executor to use, rather than relying on string matching in the title.

### Phase 3: Workflow Activation & Testing

- [ ] **3.1: Test Workflow 1 (`1-scan-and-propose.yml`):** Manually trigger the "Scan Repos and Propose Tasks" workflow from the Actions tab. Verify that it runs successfully and creates new issues in the manager repository.
- [ ] **3.2: Test Workflow 2 (`2-execute-approved-task.yml`):**
  - [ ] Add the `approved-for-dev` and `terragon-task` labels to one of the newly created issues.
  - [ ] Verify that the "Execute Approved Task" workflow is triggered.
  - [ ] Check the workflow logs to confirm that the `trigger_terragon_task` function was called and that a comment was successfully posted to the issue.
- [ ] **3.3: Full End-to-End Test:** Run a full cycle: let the scanner create an issue, approve it, and confirm that the correct AI tool creates a pull request in the target repository.

### Phase 4: Advanced Features & Refinements

- [ ] **4.1: Public-Facing Priority Board:** Create a GitHub Project (Board view) for the manager repository. Add columns like "Proposed," "Prioritized," "In Progress," and "Done." Automate moving issues between columns as labels change.
- [ ] **4.2: Add More Analyzers:** Implement new analysis functions in `task_analyzer.py`, such as:
  - [ ] `check_for_outdated_dependencies`: Scan `package.json` or `requirements.txt` and compare with the latest versions.
  - [ ] `identify_refactor_candidates`: Use code complexity metrics to suggest files for refactoring.
- [ ] **4.3: Enhance Prompt Engineering:** Create more sophisticated prompt templates in the `/prompts` directory for different task types (e.g., `add_tests.txt`, `update_dependencies.txt`).
- [ ] **4.4: Implement Manual Task Trigger:** Complete the `scripts/one_off_task.sh` script to allow for manually creating and executing a task without waiting for the automated scanner.

---

## 📁 File Structure

```
claude-manager-service/
├── README.md
├── config.json
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       ├── 1-scan-and-propose.yml
│       └── 2-execute-approved-task.yml
├── src/
│   ├── github_api.py
│   ├── task_analyzer.py
│   ├── prompt_builder.py
│   └── orchestrator.py
├── prompts/
│   ├── fix_issue.txt
│   └── refactor_code.txt
└── scripts/
    └── one_off_task.sh
```

---

## 🔧 Configuration File

**`config.json`**
```json
{
  "github": {
    "username": "your-github-username",
    "managerRepo": "your-github-username/claude-manager-service",
    "reposToScan": [
      "your-github-username/project-alpha",
      "your-github-username/project-beta"
    ]
  },
  "analyzer": {
    "scanForTodos": true,
    "scanOpenIssues": true
  },
  "executor": {
    "terragonUsername": "@terragon-labs"
  }
}
```

## 🌍 Environment Configuration

The service supports extensive configuration through environment variables for performance tuning, security settings, and feature flags. See [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for complete documentation.

**Key Configuration Categories:**
- **Performance Monitoring**: Adjust thresholds, retention, and alerting
- **Rate Limiting**: Configure API request limits and time windows  
- **Security**: Set content length limits and security features
- **Feature Flags**: Enable/disable monitoring, rate limiting, and enhanced security

**Quick Example:**
```bash
# Performance tuning
export PERF_ALERT_DURATION=15.0
export PERF_MAX_OPERATIONS=20000

# Security configuration  
export SECURITY_MAX_CONTENT_LENGTH=75000
export LOG_LEVEL=INFO
```

---

## 🔄 GitHub Workflows

**`.github/workflows/1-scan-and-propose.yml`**
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

**`.github/workflows/2-execute-approved-task.yml`**
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

---

## 📦 Dependencies

**`requirements.txt`**
```
PyGithub>=1.59
```

---

## 🐍 Python Source Code

**`src/github_api.py`**
```python
import os
from github import Github, GithubException

class GitHubAPI:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable not set.")
        self.client = Github(self.token)

    def get_repo(self, repo_name):
        try:
            return self.client.get_repo(repo_name)
        except GithubException as e:
            print(f"Error getting repo {repo_name}: {e}")
            return None

    def create_issue(self, repo_name, title, body, labels):
        print(f"Creating issue '{title}' in repo {repo_name}")
        repo = self.get_repo(repo_name)
        if repo:
            try:
                # Check for existing issues with similar titles
                existing_issues = repo.get_issues(state='open')
                for issue in existing_issues:
                    if issue.title.lower() == title.lower():
                        print(f"Issue with title '{title}' already exists (#{issue.number})")
                        return
                
                # Create the issue if no duplicate found
                new_issue = repo.create_issue(title=title, body=body, labels=labels)
                print(f"Issue created successfully (#{new_issue.number})")
            except GithubException as e:
                print(f"Error creating issue: {e}")

    def get_issue(self, repo_name, issue_number):
        repo = self.get_repo(repo_name)
        if repo:
            try:
                return repo.get_issue(number=issue_number)
            except GithubException as e:
                print(f"Error getting issue #{issue_number}: {e}")
        return None

    def add_comment_to_issue(self, repo_name, issue_number, comment_body):
        print(f"Adding comment to issue #{issue_number} in {repo_name}")
        issue = self.get_issue(repo_name, issue_number)
        if issue:
            try:
                issue.create_comment(comment_body)
                print("Comment posted successfully.")
            except GithubException as e:
                print(f"Error posting comment: {e}")
```

**`src/task_analyzer.py`**
```python
import json
import datetime
from github_api import GitHubAPI

def find_todo_comments(github_api, repo, manager_repo_name):
    print(f"Scanning {repo.full_name} for TODO comments...")
    try:
        # Search for TODO and FIXME comments in the repository
        search_queries = ['TODO:', 'FIXME:', 'TODO', 'FIXME']
        
        for query in search_queries:
            try:
                # Use GitHub's code search API
                search_results = github_api.client.search_code(
                    query=f"{query} repo:{repo.full_name}"
                )
                
                for result in search_results[:5]:  # Limit to first 5 results per query
                    file_path = result.path
                    
                    # Get the file content to extract context
                    try:
                        file_content = repo.get_contents(file_path)
                        content_lines = file_content.decoded_content.decode('utf-8').split('\n')
                        
                        # Find the line with the TODO/FIXME
                        for line_num, line in enumerate(content_lines, 1):
                            if query.lower() in line.lower():
                                # Extract some context around the TODO
                                start_line = max(0, line_num - 3)
                                end_line = min(len(content_lines), line_num + 2)
                                context = '\n'.join(content_lines[start_line:end_line])
                                
                                title = f"Address {query} in {file_path}:{line_num}"
                                body = f"""A `{query}` comment was found that may require action.

**Repository:** {repo.full_name}
**File:** `{file_path}`
**Line:** {line_num}

**Context:**
```
{context}
```

**Direct Link:** {result.html_url}
"""
                                
                                github_api.create_issue(
                                    manager_repo_name, 
                                    title, 
                                    body, 
                                    ["task-proposal", "refactor", "todo"]
                                )
                                break  # Only create one issue per file
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error searching for {query}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in find_todo_comments: {e}")

def analyze_open_issues(github_api, repo, manager_repo_name):
    print(f"Analyzing open issues in {repo.full_name}...")
    try:
        # Get open issues from the repository
        open_issues = repo.get_issues(state='open')
        
        # Current time for comparison
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for issue in open_issues:
            # Skip if it's a pull request
            if issue.pull_request:
                continue
                
            # Check if issue has relevant labels
            issue_labels = [label.name.lower() for label in issue.labels]
            relevant_labels = ['bug', 'help wanted', 'good first issue', 'enhancement']
            
            has_relevant_label = any(label in issue_labels for label in relevant_labels)
            
            if has_relevant_label:
                # Check if issue is stale (no activity in last 30 days)
                days_since_update = (now - issue.updated_at).days
                
                if days_since_update > 30:
                    title = f"Review stale issue: '{issue.title}'"
                    body = f"""The issue #{issue.number} in `{repo.full_name}` has been inactive for {days_since_update} days and may need attention.

**Original Issue:** {issue.title}
**Labels:** {', '.join([label.name for label in issue.labels])}
**Created:** {issue.created_at.strftime('%Y-%m-%d')}
**Last Updated:** {issue.updated_at.strftime('%Y-%m-%d')}
**Assignees:** {', '.join([assignee.login for assignee in issue.assignees]) if issue.assignees else 'None'}

**Description:**
{issue.body[:500]}{'...' if len(issue.body) > 500 else ''}

**Link:** {issue.html_url}

This issue may be a good candidate for AI-assisted resolution.
"""
                    
                    github_api.create_issue(
                        manager_repo_name, 
                        title, 
                        body, 
                        ["task-proposal", "stale-issue", "review"]
                    )
                    
    except Exception as e:
        print(f"Error in analyze_open_issues: {e}")

if __name__ == "__main__":
    try:
        with open('config.json') as f:
            config = json.load(f)

        api = GitHubAPI()
        manager_repo_name = config['github']['managerRepo']
        
        for repo_name in config['github']['reposToScan']:
            print(f"\n--- Analyzing repository: {repo_name} ---")
            repo = api.get_repo(repo_name)
            if not repo:
                print(f"Skipping {repo_name} - could not access repository")
                continue

            if config['analyzer']['scanForTodos']:
                find_todo_comments(api, repo, manager_repo_name)

            if config['analyzer']['scanOpenIssues']:
                analyze_open_issues(api, repo, manager_repo_name)

        print("\n--- Task analysis cycle complete ---")
        
    except Exception as e:
        print(f"Fatal error in task analyzer: {e}")
        raise
```

**`src/prompt_builder.py`**
```python
import os

def build_prompt(template_file, context):
    """
    Builds a detailed prompt from a template file and a context dictionary.
    """
    print(f"Building prompt from template: {template_file}")
    
    if not os.path.exists(template_file):
        print(f"Warning: Template file {template_file} not found. Using default prompt.")
        return f"Please work on the following task:\n\nTitle: {context.get('issue_title', 'No title')}\n\nDescription:\n{context.get('issue_body', 'No description')}"
    
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Replace placeholders in the template with values from context
        prompt = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
        
        return prompt
        
    except Exception as e:
        print(f"Error building prompt: {e}")
        # Fallback to basic prompt
        return f"Please work on the following task:\n\nTitle: {context.get('issue_title', 'No title')}\n\nDescription:\n{context.get('issue_body', 'No description')}"

def get_template_for_labels(labels):
    """
    Returns the appropriate template file based on issue labels.
    """
    label_names = [label.lower() for label in labels]
    
    if any(word in label_names for word in ['refactor', 'todo', 'cleanup']):
        return 'prompts/refactor_code.txt'
    elif any(word in label_names for word in ['bug', 'fix', 'issue']):
        return 'prompts/fix_issue.txt'
    else:
        return 'prompts/fix_issue.txt'  # Default template
```

**`src/orchestrator.py`**
```python
import os
import json
import subprocess
import tempfile
import shutil
from github_api import GitHubAPI
from prompt_builder import build_prompt, get_template_for_labels

def trigger_terragon_task(api, repo_name, issue, config):
    print(f"Orchestrating Terragon task for issue #{issue.number}")
    
    try:
        # Extract relevant information from issue
        labels = [label.name for label in issue.labels]
        
        context = {
            "issue_title": issue.title,
            "issue_body": issue.body or "No description provided",
            "issue_number": issue.number,
            "repository": repo_name,
            "labels": ", ".join(labels),
            "issue_url": issue.html_url
        }
        
        # Choose appropriate template based on labels
        template_file = get_template_for_labels(labels)
        prompt = build_prompt(template_file, context)
        
        # Build the comment to trigger Terragon
        terragon_username = config['executor']['terragonUsername']
        full_comment = f"""{terragon_username} - please begin work on the following task.

{prompt}

---
*This task was automatically generated and approved by the Claude Manager Service.*
"""
        
        # Post the comment to trigger Terragon
        api.add_comment_to_issue(repo_name, issue.number, full_comment)
        print(f"Terragon task trigger comment posted successfully")
        
    except Exception as e:
        print(f"Error triggering Terragon task: {e}")
        raise

def trigger_claude_flow_task(api, repo_name, issue):
    print(f"Orchestrating Claude Flow task for issue #{issue.number}")
    
    try:
        # Extract target repository from issue body or use the manager repo
        # This is a simplified approach - in practice, you'd parse the issue more carefully
        target_repo = repo_name
        if "**Repository:**" in issue.body:
            lines = issue.body.split('\n')
            for line in lines:
                if line.startswith("**Repository:**"):
                    target_repo = line.replace("**Repository:**", "").strip()
                    break
        
        # Create a temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, "cloned_repo")
            
            # Clone the repository
            clone_url = f"https://x-access-token:{api.token}@github.com/{target_repo}.git"
            clone_command = ["git", "clone", clone_url, repo_dir]
            
            print(f"Cloning repository {target_repo}...")
            result = subprocess.run(clone_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error cloning repository: {result.stderr}")
                return
            
            # Prepare the Claude Flow command
            task_description = f'"{issue.title}"'
            claude_flow_command = [
                "npx", "claude-flow@alpha", "hive-mind", "spawn", 
                task_description, "--claude"
            ]
            
            print(f"Executing Claude Flow command: {' '.join(claude_flow_command)}")
            
            # Execute Claude Flow within the cloned repository
            result = subprocess.run(
                claude_flow_command,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("Claude Flow task completed successfully")
                print(f"Output: {result.stdout}")
                
                # Post a comment with the results
                success_comment = f"""✅ Claude Flow task completed successfully for issue #{issue.number}.

**Command executed:** `{' '.join(claude_flow_command)}`

**Output:**
```
{result.stdout}
```

---
*This task was automatically executed by the Claude Manager Service.*
"""
                api.add_comment_to_issue(repo_name, issue.number, success_comment)
                
            else:
                print(f"Claude Flow task failed: {result.stderr}")
                
                # Post a comment about the failure
                error_comment = f"""❌ Claude Flow task failed for issue #{issue.number}.

**Error:**
```
{result.stderr}
```

**Output:**
```
{result.stdout}
```

---
*This task was automatically executed by the Claude Manager Service.*
"""
                api.add_comment_to_issue(repo_name, issue.number, error_comment)
                
    except subprocess.TimeoutExpired:
        print("Claude Flow task timed out after 5 minutes")
        timeout_comment = f"""⏰ Claude Flow task timed out after 5 minutes for issue #{issue.number}.

Please review the task complexity and consider breaking it into smaller pieces.

---
*This task was automatically executed by the Claude Manager Service.*
"""
        api.add_comment_to_issue(repo_name, issue.number, timeout_comment)
        
    except Exception as e:
        print(f"Error triggering Claude Flow task: {e}")
        error_comment = f"""❌ Error executing Claude Flow task for issue #{issue.number}.

**Error:** {str(e)}

---
*This task was automatically executed by the Claude Manager Service.*
"""
        api.add_comment_to_issue(repo_name, issue.number, error_comment)

if __name__ == "__main__":
    try:
        with open('config.json') as f:
            config = json.load(f)

        issue_number = int(os.getenv("ISSUE_NUMBER"))
        repo_name = os.getenv("REPOSITORY_NAME")

        if not issue_number or not repo_name:
            raise ValueError("ISSUE_NUMBER and REPOSITORY_NAME environment variables must be set")

        api = GitHubAPI()
        issue = api.get_issue(repo_name, issue_number)
        
        if not issue:
            raise Exception(f"Could not retrieve issue #{issue_number} from {repo_name}")

        # Get issue labels
        labels = {label.name.lower() for label in issue.labels}
        
        print(f"Processing issue #{issue_number}: {issue.title}")
        print(f"Labels: {', '.join(labels)}")

        # Determine which executor to use based on labels
        if "terragon-task" in labels:
            trigger_terragon_task(api, repo_name, issue, config)
        elif "claude-flow-task" in labels:
            trigger_claude_flow_task(api, repo_name, issue)
        else:
            # Default to Terragon if no specific label is found
            print("No specific executor label found, defaulting to Terragon.")
            trigger_terragon_task(api, repo_name, issue, config)

        print(f"Orchestration for issue #{issue_number} complete.")
        
    except Exception as e:
        print(f"Fatal error in orchestrator: {e}")
        raise
```

---

## 📝 Prompt Templates

**`prompts/fix_issue.txt`**
```
Task: Address GitHub Issue

**Title:** {issue_title}
**Issue Number:** #{issue_number}
**Repository:** {repository}
**Labels:** {labels}

**Description:**
{issue_body}

**Instructions:**
Please analyze the issue description and associated code. Follow these steps:

1. **Investigation:** Thoroughly examine the issue, related code, and any referenced files
2. **Solution Design:** Propose a clear, well-thought-out solution
3. **Implementation:** Implement the fix with clean, maintainable code
4. **Testing:** Ensure all existing tests pass and add new tests if necessary
5. **Documentation:** Update relevant documentation if needed
6. **Pull Request:** Create a pull request with:
   - Clear title and description
   - Summary of changes made
   - Any breaking changes or migration notes
   - Link to this issue

**Quality Standards:**
- Follow the project's coding standards and conventions
- Write clear, self-documenting code
- Include appropriate error handling
- Consider edge cases and potential side effects
- Maintain backward compatibility where possible

**Reference:** {issue_url}
```

**`prompts/refactor_code.txt`**
```
Task: Code Refactoring

**Title:** {issue_title}
**Issue Number:** #{issue_number}
**Repository:** {repository}
**Labels:** {labels}

**Context:**
{issue_body}

**Refactoring Objectives:**
Please perform a comprehensive code refactoring based on the context above. Focus on:

1. **Code Quality Improvements:**
   - Improve readability and maintainability
   - Remove code duplication
   - Simplify complex logic
   - Apply consistent naming conventions

2. **Performance Optimization:**
   - Identify and fix performance bottlenecks
   - Optimize algorithms and data structures
   - Reduce memory usage where applicable

3. **Best Practices:**
   - Apply design patterns where appropriate
   - Improve error handling
   - Enhance code modularity
   - Follow language-specific best practices

4. **Testing:**
   - Ensure all existing functionality remains intact
   - Add unit tests for refactored components
   - Verify performance improvements with benchmarks

5. **Documentation:**
   - Update code comments and documentation
   - Document any architectural changes
   - Create migration guide if needed

**Deliverables:**
- Refactored code following project standards
- Updated tests covering the refactored functionality
- Pull request with detailed explanation of changes
- Before/after comparison highlighting improvements

**Reference:** {issue_url}
```

---

## 🔧 Utility Scripts

**`scripts/one_off_task.sh`**
```bash
#!/bin/bash

# One-off Task Creation Script
# Usage: ./scripts/one_off_task.sh "Task Title" "Task Description" "target-repo" "executor-type"

set -e

TASK_TITLE="$1"
TASK_DESCRIPTION="$2"
TARGET_REPO="$3"
EXECUTOR_TYPE="$4"  # "terragon" or "claude-flow"

if [ -z "$TASK_TITLE" ] || [ -z "$TASK_DESCRIPTION" ] || [ -z "$TARGET_REPO" ]; then
    echo "Usage: $0 \"Task Title\" \"Task Description\" \"target-repo\" [executor-type]"
    echo "Example: $0 \"Fix login bug\" \"The login form is not validating properly\" \"myorg/myproject\" \"terragon\""
    exit 1
fi

# Default to terragon if no executor specified
if [ -z "$EXECUTOR_TYPE" ]; then
    EXECUTOR_TYPE="terragon"
fi

# Validate executor type
if [ "$EXECUTOR_TYPE" != "terragon" ] && [ "$EXECUTOR_TYPE" != "claude-flow" ]; then
    echo "Error: executor-type must be 'terragon' or 'claude-flow'"
    exit 1
fi

echo "Creating manual task..."
echo "Title: $TASK_TITLE"
echo "Target Repository: $TARGET_REPO"
echo "Executor: $EXECUTOR_TYPE"
echo ""

# Create the issue body
ISSUE_BODY="**Manual Task Creation**

**Target Repository:** $TARGET_REPO
**Requested Executor:** $EXECUTOR_TYPE

**Description:**
$TASK_DESCRIPTION

---
*This task was manually created using the one-off task script.*"

# Determine labels based on executor type
if [ "$EXECUTOR_TYPE" = "terragon" ]; then
    LABELS="manual-task,terragon-task"
else
    LABELS="manual-task,claude-flow-task"
fi

# Create the issue using GitHub CLI (requires gh to be installed and authenticated)
if command -v gh &> /dev/null; then
    echo "Creating issue using GitHub CLI..."
    gh issue create \
        --title "$TASK_TITLE" \
        --body "$ISSUE_BODY" \
        --label "$LABELS"
    
    echo "Issue created successfully!"
    echo "Add the 'approved-for-dev' label to trigger execution."
else
    echo "GitHub CLI not found. Please install 'gh' to use this script."
    echo "Alternatively, create the issue manually with:"
    echo "Title: $TASK_TITLE"
    echo "Labels: $LABELS"
    echo "Body:"
    echo "$ISSUE_BODY"
fi
```

---

## 🚫 Git Ignore

**`.gitignore`**
```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
*.egg-info/
dist/
build/
.pytest_cache/

# IDE / OS
.idea/
.vscode/
*.swp
*~
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp

# Environment variables
.env
.env.local

# Coverage reports
htmlcov/
.coverage
.coverage.*

# Virtual environments
venv/
ENV/
env/
.venv/

# Node.js (for CLI tools)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Claude Manager specific
cloned_repos/
temp/
```
