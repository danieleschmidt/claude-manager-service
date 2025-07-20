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