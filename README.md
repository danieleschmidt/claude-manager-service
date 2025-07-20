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
