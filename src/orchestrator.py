import os
import json
import subprocess
import tempfile
import shutil
from typing import Dict, Any
from github import Issue
from github_api import GitHubAPI
from prompt_builder import build_prompt, get_template_for_labels
from logger import get_logger, log_performance
from security import get_secure_subprocess, SecureTempDir, validate_repo_name
from config_validator import get_validated_config

logger = get_logger(__name__)

@log_performance
def trigger_terragon_task(api: GitHubAPI, repo_name: str, issue: Issue.Issue, config: Dict[str, Any]) -> None:
    """Trigger a Terragon task by posting a formatted comment to the issue"""
    logger.info(f"Orchestrating Terragon task for issue #{issue.number} in {repo_name}")
    logger.debug(f"Issue title: {issue.title}")
    
    try:
        # Extract relevant information from issue
        labels = [label.name for label in issue.labels]
        logger.debug(f"Issue labels: {labels}")
        
        context = {
            "issue_title": issue.title,
            "issue_body": issue.body or "No description provided",
            "issue_number": issue.number,
            "repository": repo_name,
            "labels": ", ".join(labels),
            "issue_url": issue.html_url
        }
        
        # Choose appropriate template based on labels
        logger.debug("Selecting template based on issue labels")
        template_file = get_template_for_labels(labels)
        logger.info(f"Using template: {template_file}")
        
        prompt = build_prompt(template_file, context)
        
        # Build the comment to trigger Terragon
        terragon_username = config['executor']['terragonUsername']
        logger.debug(f"Building Terragon comment for user: {terragon_username}")
        
        full_comment = f"""{terragon_username} - please begin work on the following task.

{prompt}

---
*This task was automatically generated and approved by the Claude Manager Service.*
"""
        
        logger.debug(f"Generated comment length: {len(full_comment)} characters")
        
        # Post the comment to trigger Terragon
        logger.info(f"Posting Terragon trigger comment to issue #{issue.number}")
        api.add_comment_to_issue(repo_name, issue.number, full_comment)
        logger.info(f"Terragon task trigger comment posted successfully for issue #{issue.number}")
        
    except KeyError as e:
        logger.error(f"Missing configuration key for Terragon task: {e}")
        raise
    except Exception as e:
        logger.error(f"Error triggering Terragon task for issue #{issue.number}: {e}")
        raise

@log_performance
def trigger_claude_flow_task(api: GitHubAPI, repo_name: str, issue: Issue.Issue) -> None:
    """Trigger a Claude Flow task by cloning repository and executing command"""
    logger.info(f"Orchestrating Claude Flow task for issue #{issue.number} in {repo_name}")
    logger.debug(f"Issue title: {issue.title}")
    
    try:
        # Extract target repository from issue body or use the manager repo
        logger.debug("Extracting target repository from issue body")
        target_repo = repo_name
        
        if issue.body and "**Repository:**" in issue.body:
            lines = issue.body.split('\n')
            for line in lines:
                if line.startswith("**Repository:**"):
                    extracted_repo = line.replace("**Repository:**", "").strip()
                    # Validate extracted repository name for security
                    if validate_repo_name(extracted_repo):
                        target_repo = extracted_repo
                        logger.info(f"Extracted target repository: {target_repo}")
                    else:
                        logger.warning(f"Invalid repository name extracted: {extracted_repo}, using manager repo")
                    break
        else:
            logger.debug(f"No target repository specified, using manager repo: {target_repo}")
        
        # Create a secure temporary directory for cloning
        logger.debug("Creating secure temporary directory for repository cloning")
        
        with SecureTempDir(prefix="claude_flow_") as temp_dir:
            repo_dir = str(temp_dir / "cloned_repo")
            
            # Use secure subprocess for git clone
            repo_url = f"https://github.com/{target_repo}.git"
            
            logger.info(f"Cloning repository {target_repo}")
            result = get_secure_subprocess().run_git_clone(repo_url, repo_dir, api.token, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Failed to clone repository {target_repo}")
                return
            
            logger.info(f"Successfully cloned repository {target_repo}")
            
            # Prepare the Claude Flow command
            task_description = f'"{issue.title}"'
            claude_flow_command = [
                "npx", "claude-flow@alpha", "hive-mind", "spawn", 
                task_description, "--claude"
            ]
            
            logger.info(f"Executing Claude Flow command in {target_repo}")
            
            # Execute Claude Flow within the cloned repository using secure subprocess
            try:
                result = get_secure_subprocess().run_with_sanitized_logging(
                    claude_flow_command,
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
            except subprocess.TimeoutExpired as e:
                logger.error(f"Claude Flow command timed out after 5 minutes for issue #{issue.number}")
                raise
            
            if result.returncode == 0:
                logger.info(f"Claude Flow task completed successfully for issue #{issue.number}")
                logger.debug(f"Claude Flow stdout: {result.stdout[:500]}...")  # Truncate for logging
                
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
                logger.info(f"Posting success comment for issue #{issue.number}")
                api.add_comment_to_issue(repo_name, issue.number, success_comment)
                
            else:
                logger.error(f"Claude Flow task failed for issue #{issue.number}: {result.stderr}")
                logger.debug(f"Claude Flow stdout: {result.stdout}")
                
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
                logger.info(f"Posting error comment for issue #{issue.number}")
                api.add_comment_to_issue(repo_name, issue.number, error_comment)
                
    except subprocess.TimeoutExpired:
        logger.error(f"Claude Flow task timed out after 5 minutes for issue #{issue.number}")
        timeout_comment = f"""⏰ Claude Flow task timed out after 5 minutes for issue #{issue.number}.

Please review the task complexity and consider breaking it into smaller pieces.

---
*This task was automatically executed by the Claude Manager Service.*
"""
        logger.info(f"Posting timeout comment for issue #{issue.number}")
        api.add_comment_to_issue(repo_name, issue.number, timeout_comment)
        
    except Exception as e:
        logger.error(f"Error triggering Claude Flow task for issue #{issue.number}: {e}")
        error_comment = f"""❌ Error executing Claude Flow task for issue #{issue.number}.

**Error:** {str(e)}

---
*This task was automatically executed by the Claude Manager Service.*
"""
        logger.info(f"Posting error comment for issue #{issue.number}")
        api.add_comment_to_issue(repo_name, issue.number, error_comment)

if __name__ == "__main__":
    logger.info("Starting orchestrator")
    
    try:
        config = get_validated_config('config.json')

        issue_number_str = os.getenv("ISSUE_NUMBER")
        repo_name = os.getenv("REPOSITORY_NAME")
        
        logger.debug(f"Environment variables - ISSUE_NUMBER: {issue_number_str}, REPOSITORY_NAME: {repo_name}")

        if not issue_number_str or not repo_name:
            logger.error("Required environment variables not set")
            raise ValueError("ISSUE_NUMBER and REPOSITORY_NAME environment variables must be set")
        
        try:
            issue_number = int(issue_number_str)
        except ValueError:
            logger.error(f"Invalid issue number: {issue_number_str}")
            raise ValueError(f"ISSUE_NUMBER must be a valid integer, got: {issue_number_str}")

        logger.info(f"Initializing GitHub API")
        api = GitHubAPI()
        
        logger.info(f"Retrieving issue #{issue_number} from {repo_name}")
        issue = api.get_issue(repo_name, issue_number)
        
        if not issue:
            logger.error(f"Could not retrieve issue #{issue_number} from {repo_name}")
            raise Exception(f"Could not retrieve issue #{issue_number} from {repo_name}")

        # Get issue labels
        labels = {label.name.lower() for label in issue.labels}
        
        logger.info(f"Processing issue #{issue_number}: {issue.title}")
        logger.info(f"Issue labels: {', '.join(labels)}")

        # Determine which executor to use based on labels
        if "terragon-task" in labels:
            logger.info("Executing Terragon task based on label")
            trigger_terragon_task(api, repo_name, issue, config)
        elif "claude-flow-task" in labels:
            logger.info("Executing Claude Flow task based on label")
            trigger_claude_flow_task(api, repo_name, issue)
        else:
            # Default to Terragon if no specific label is found
            logger.info("No specific executor label found, defaulting to Terragon")
            trigger_terragon_task(api, repo_name, issue, config)

        logger.info(f"Orchestration for issue #{issue_number} completed successfully")
        
    except SystemExit:
        # Configuration validation error already logged
        raise
    except Exception as e:
        logger.error(f"Fatal error in orchestrator: {e}")
        raise