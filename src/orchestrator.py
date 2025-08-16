import os
import json
import subprocess
import tempfile
import shutil
import shlex
from typing import Dict, Any, List
from github import Issue
from src.github_api import GitHubAPI
from src.prompt_builder import build_prompt, get_template_for_labels
from src.logger import get_logger, log_performance
from src.performance_monitor import monitor_performance
from src.security import get_secure_subprocess, SecureTempDir, validate_repo_name
from src.config_validator import get_validated_config
from src.quantum_task_planner import create_quantum_task_planner, QuantumTaskPlanner

logger = get_logger(__name__)


class Orchestrator:
    """Main orchestrator class for managing task execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api = GitHubAPI()
        self.logger = get_logger(__name__)
    
    def trigger_task(self, repo_name: str, issue_number: int) -> None:
        """Trigger task execution for a given issue."""
        issue = self.api.get_issue(repo_name, issue_number)
        if not issue:
            raise Exception(f"Could not retrieve issue #{issue_number} from {repo_name}")
        
        # Get issue labels
        labels = {label.name.lower() for label in issue.labels}
        
        # Determine which executor to use based on labels
        if "terragon-task" in labels:
            trigger_terragon_task(self.api, repo_name, issue, self.config)
        elif "claude-flow-task" in labels:
            trigger_claude_flow_task(self.api, repo_name, issue)
        else:
            # Default to Terragon if no specific label is found
            self.logger.info("No specific executor label found, defaulting to Terragon.")
            trigger_terragon_task(self.api, repo_name, issue, self.config)

@monitor_performance(track_memory=True, custom_name="terragon_task_orchestration")
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

@monitor_performance(track_memory=True, custom_name="claude_flow_task_orchestration")
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
            
            # Prepare the Claude Flow command - properly escape the issue title to prevent command injection
            # Use issue.title directly as a list argument (no shell interpretation needed)
            claude_flow_command = [
                "npx", "claude-flow@alpha", "hive-mind", "spawn", 
                issue.title, "--claude"
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


# Quantum-Enhanced Orchestration Functions

@monitor_performance(track_memory=True, custom_name="quantum_task_orchestration")
@log_performance
def quantum_orchestrate_tasks(api: GitHubAPI, repo_name: str, tasks: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate multiple tasks using quantum-inspired prioritization and optimization
    
    Args:
        api: GitHub API instance
        repo_name: Repository name
        tasks: List of task dictionaries
        config: Configuration dictionary
        
    Returns:
        Orchestration results with quantum insights
    """
    logger.info(f"Starting quantum orchestration for {len(tasks)} tasks in {repo_name}")
    
    # Initialize quantum task planner
    quantum_planner = create_quantum_task_planner(enable_annealing=True)
    
    try:
        # Apply quantum prioritization
        logger.info("Applying quantum-inspired task prioritization")
        prioritized_tasks = quantum_planner.prioritize_tasks(tasks)
        
        # Generate quantum insights report
        quantum_insights = quantum_planner.generate_quantum_insights_report(prioritized_tasks)
        
        logger.info(f"Quantum prioritization complete. Processing {len(prioritized_tasks)} tasks")
        
        # Execute tasks in quantum-optimized order
        execution_results = []
        for i, task in enumerate(prioritized_tasks[:10]):  # Limit to top 10 for initial execution
            logger.info(f"Executing quantum task {i+1}/{min(10, len(prioritized_tasks))}: {task['id']}")
            
            try:
                # Create GitHub issue for task if it doesn't exist
                issue_title = f"Quantum Task: {task['content'][:80]}..."
                issue_body = f"""**Quantum-Enhanced Task Execution**

**Original Content:** {task['content']}
**File Path:** {task['file_path']}
**Line Number:** {task['line_number']}
**Task Type:** {task['task_type']}

**Quantum Insights:**
- Priority Score: {task['priority_score']:.3f}
- Quantum State: {task['quantum_state']}
- Quantum Rank: {task['quantum_rank']}
- Uncertainty: {task['uncertainty']:.3f}
- Entangled Tasks: {len(task['entangled_tasks'])}

**Priority Reasoning:** {task['priority_reason']}

**Quantum Dimensions:**
{_format_quantum_dimensions(task.get('quantum_insights', {}).get('priority_dimensions', {}))}

---
*This issue was created by the Quantum Task Orchestrator.*
"""
                
                labels = ["quantum-task", "automated", task['task_type']]
                if task['priority_score'] >= 8.0:
                    labels.append("critical")
                elif task['priority_score'] >= 6.0:
                    labels.append("high-priority")
                
                # Create issue
                api.create_issue(repo_name, issue_title, issue_body, labels)
                
                execution_results.append({
                    'task_id': task['id'],
                    'status': 'issue_created',
                    'priority_score': task['priority_score'],
                    'quantum_state': task['quantum_state']
                })
                
                logger.info(f"Created GitHub issue for quantum task {task['id']}")
                
            except Exception as e:
                logger.error(f"Failed to create issue for quantum task {task['id']}: {e}")
                execution_results.append({
                    'task_id': task['id'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Create quantum orchestration summary
        orchestration_summary = {
            'total_tasks_processed': len(tasks),
            'prioritized_tasks': len(prioritized_tasks),
            'executed_tasks': len(execution_results),
            'successful_executions': len([r for r in execution_results if r['status'] == 'issue_created']),
            'failed_executions': len([r for r in execution_results if r['status'] == 'failed']),
            'quantum_insights': quantum_insights,
            'execution_results': execution_results,
            'orchestration_timestamp': logger.info.__self__.created if hasattr(logger.info, '__self__') else 'unknown'
        }
        
        logger.info(f"Quantum orchestration complete. Created {orchestration_summary['successful_executions']} issues")
        
        return orchestration_summary
        
    except Exception as e:
        logger.error(f"Quantum orchestration failed: {e}")
        raise


def _format_quantum_dimensions(dimensions: Dict[str, float]) -> str:
    """Format quantum dimensions for readable display"""
    if not dimensions:
        return "No quantum dimension data available"
    
    formatted_lines = []
    for dimension, value in dimensions.items():
        formatted_lines.append(f"- {dimension.replace('_', ' ').title()}: {value:.2f}")
    
    return '\n'.join(formatted_lines)


@monitor_performance(track_memory=True, custom_name="quantum_issue_analysis")
@log_performance
def analyze_issue_with_quantum_insights(api: GitHubAPI, repo_name: str, issue: Issue.Issue) -> Dict[str, Any]:
    """
    Analyze a GitHub issue using quantum-inspired methods
    
    Args:
        api: GitHub API instance
        repo_name: Repository name
        issue: GitHub issue object
        
    Returns:
        Quantum analysis results
    """
    logger.info(f"Analyzing issue #{issue.number} with quantum insights")
    
    # Convert issue to task format
    task_data = {
        'id': f"issue_{issue.number}",
        'content': f"{issue.title}\n\n{issue.body or ''}",
        'file_path': 'github_issue',
        'line_number': 0,
        'type': 'issue'
    }
    
    # Initialize quantum planner
    quantum_planner = create_quantum_task_planner(enable_annealing=False)  # No annealing for single task
    
    try:
        # Get quantum analysis
        quantum_results = quantum_planner.prioritize_tasks([task_data])
        
        if quantum_results:
            quantum_task = quantum_results[0]
            
            analysis = {
                'issue_number': issue.number,
                'issue_title': issue.title,
                'quantum_priority_score': quantum_task['priority_score'],
                'quantum_state': quantum_task['quantum_state'],
                'task_type_classification': quantum_task['task_type'],
                'priority_reasoning': quantum_task['priority_reason'],
                'uncertainty_level': quantum_task['uncertainty'],
                'quantum_insights': quantum_task.get('quantum_insights', {}),
                'recommended_action': _get_quantum_recommendation(quantum_task),
                'entanglement_potential': len(quantum_task['entangled_tasks'])
            }
            
            logger.info(f"Quantum analysis complete for issue #{issue.number}. Priority: {analysis['quantum_priority_score']:.3f}")
            
            return analysis
        else:
            logger.error(f"No quantum results returned for issue #{issue.number}")
            return {'error': 'No quantum analysis results'}
            
    except Exception as e:
        logger.error(f"Quantum analysis failed for issue #{issue.number}: {e}")
        return {'error': str(e)}


def _get_quantum_recommendation(quantum_task: Dict[str, Any]) -> str:
    """Generate quantum-based recommendation for task handling"""
    
    priority = quantum_task['priority_score']
    task_type = quantum_task['task_type']
    uncertainty = quantum_task['uncertainty']
    
    if priority >= 8.0:
        if task_type == 'security':
            return "IMMEDIATE ACTION REQUIRED: Critical security issue detected by quantum analysis"
        elif task_type == 'bug':
            return "HIGH PRIORITY: Critical bug requires immediate quantum-optimized resolution"
        else:
            return "URGENT: High-priority task identified through quantum prioritization"
    
    elif priority >= 6.0:
        if uncertainty > 0.7:
            return "INVESTIGATE: High priority with significant quantum uncertainty - requires analysis"
        else:
            return "SCHEDULE: High priority task suitable for near-term quantum-guided execution"
    
    elif priority >= 4.0:
        return "PLAN: Medium priority task for future quantum-optimized batch processing"
    
    else:
        return "DEFER: Low priority task suitable for background quantum processing"


@monitor_performance(track_memory=True, custom_name="quantum_batch_orchestration")
@log_performance
def quantum_batch_orchestrate(api: GitHubAPI, config: Dict[str, Any], batch_size: int = 20) -> Dict[str, Any]:
    """
    Perform quantum-inspired batch orchestration across multiple repositories
    
    Args:
        api: GitHub API instance
        config: Configuration dictionary
        batch_size: Maximum number of tasks to process in batch
        
    Returns:
        Batch orchestration results
    """
    logger.info(f"Starting quantum batch orchestration with batch size {batch_size}")
    
    repos_to_scan = config.get('github', {}).get('reposToScan', [])
    if not repos_to_scan:
        logger.warning("No repositories configured for scanning")
        return {'error': 'No repositories configured'}
    
    all_tasks = []
    repo_results = {}
    
    # Collect tasks from all repositories
    for repo_name in repos_to_scan:
        try:
            logger.info(f"Scanning repository {repo_name} for quantum orchestration")
            
            # This is a simplified example - in practice, you'd integrate with task_analyzer
            # to discover actual tasks from the repository
            repo_tasks = _discover_repository_tasks(api, repo_name)
            
            all_tasks.extend(repo_tasks)
            repo_results[repo_name] = {
                'tasks_discovered': len(repo_tasks),
                'status': 'scanned'
            }
            
            logger.info(f"Discovered {len(repo_tasks)} tasks in {repo_name}")
            
        except Exception as e:
            logger.error(f"Failed to scan repository {repo_name}: {e}")
            repo_results[repo_name] = {
                'tasks_discovered': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    if not all_tasks:
        logger.warning("No tasks discovered across all repositories")
        return {
            'total_tasks': 0,
            'repo_results': repo_results,
            'message': 'No tasks discovered'
        }
    
    # Apply quantum orchestration to all discovered tasks
    logger.info(f"Applying quantum orchestration to {len(all_tasks)} total tasks")
    
    # Limit to batch size
    tasks_to_process = all_tasks[:batch_size]
    
    # Use quantum orchestration
    quantum_planner = create_quantum_task_planner(enable_annealing=True)
    prioritized_tasks = quantum_planner.prioritize_tasks(tasks_to_process)
    
    # Generate comprehensive insights
    quantum_insights = quantum_planner.generate_quantum_insights_report(prioritized_tasks)
    
    batch_results = {
        'total_tasks_discovered': len(all_tasks),
        'tasks_processed': len(tasks_to_process),
        'quantum_insights': quantum_insights,
        'repo_results': repo_results,
        'top_priority_tasks': prioritized_tasks[:5],  # Top 5 tasks
        'orchestration_timestamp': 'current_time'  # Would use actual timestamp
    }
    
    logger.info(f"Quantum batch orchestration complete. Processed {len(tasks_to_process)} tasks")
    
    return batch_results


def _discover_repository_tasks(api: GitHubAPI, repo_name: str) -> List[Dict[str, Any]]:
    """
    Simplified task discovery for quantum orchestration demo
    In practice, this would integrate with the full task_analyzer module
    """
    # This is a placeholder implementation
    # In the real system, this would use the existing task discovery logic
    
    mock_tasks = [
        {
            'id': f"{repo_name}_quantum_demo_1",
            'content': 'TODO: Optimize database queries for better performance',
            'file_path': 'src/database.py',
            'line_number': 45,
            'type': 'performance'
        },
        {
            'id': f"{repo_name}_quantum_demo_2", 
            'content': 'FIXME: Security vulnerability in authentication',
            'file_path': 'src/auth.py',
            'line_number': 23,
            'type': 'security'
        }
    ]
    
    return mock_tasks

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