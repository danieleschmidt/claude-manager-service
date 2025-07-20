import json
import datetime
from github_api import GitHubAPI
from logger import get_logger, log_performance
from task_tracker import get_task_tracker
from config_validator import get_validated_config

logger = get_logger(__name__)

@log_performance
def find_todo_comments(github_api, repo, manager_repo_name):
    """Scan repository for TODO and FIXME comments and create issues for them"""
    logger.info(f"Scanning {repo.full_name} for TODO comments")
    todo_count = 0
    try:
        # Search for TODO and FIXME comments in the repository
        search_queries = ['TODO:', 'FIXME:', 'TODO', 'FIXME']
        
        for query in search_queries:
            logger.debug(f"Searching for '{query}' patterns")
            try:
                # Use GitHub's code search API
                search_results = github_api.client.search_code(
                    query=f"{query} repo:{repo.full_name}"
                )
                
                results_count = len(list(search_results[:5]))
                logger.debug(f"Found {results_count} results for query '{query}'")
                
                for result in search_results[:5]:  # Limit to first 5 results per query
                    file_path = result.path
                    logger.debug(f"Processing file: {file_path}")
                    
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
                                
                                # Check if this task has already been processed
                                tracker = get_task_tracker()
                                if tracker.is_task_processed(repo.full_name, file_path, line_num, line.strip()):
                                    logger.debug(f"Skipping already processed task: {file_path}:{line_num}")
                                    break
                                
                                logger.info(f"Creating issue for {query} found in {file_path}:{line_num}")
                                github_api.create_issue(
                                    manager_repo_name, 
                                    title, 
                                    body, 
                                    ["task-proposal", "refactor", "todo"]
                                )
                                
                                # Mark task as processed (we don't have issue number from create_issue)
                                tracker.mark_task_processed(repo.full_name, file_path, line_num, line.strip())
                                
                                todo_count += 1
                                break  # Only create one issue per file
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
                continue
                
        logger.info(f"TODO scan completed. Found {todo_count} TODO/FIXME comments in {repo.full_name}")
                
    except Exception as e:
        logger.error(f"Fatal error in find_todo_comments for {repo.full_name}: {e}")


@log_performance
def find_todo_comments_with_tracking(github_api, repo, manager_repo_name):
    """
    Enhanced version of find_todo_comments with duplicate tracking
    
    This function includes the task tracking functionality to prevent
    creating duplicate issues for the same TODO/FIXME comments.
    """
    return find_todo_comments(github_api, repo, manager_repo_name)

@log_performance
def analyze_open_issues(github_api, repo, manager_repo_name):
    """Analyze open issues in repository and identify stale ones for potential action"""
    logger.info(f"Analyzing open issues in {repo.full_name}")
    stale_count = 0
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
                    logger.info(f"Found stale issue #{issue.number}: '{issue.title}' (inactive for {days_since_update} days)")
                    
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
                    stale_count += 1
                    
        logger.info(f"Issue analysis completed. Found {stale_count} stale issues in {repo.full_name}")
                    
    except Exception as e:
        logger.error(f"Fatal error in analyze_open_issues for {repo.full_name}: {e}")

if __name__ == "__main__":
    logger.info("Starting task analysis cycle")
    
    try:
        config = get_validated_config('config.json')
        
        logger.info("Initializing GitHub API")
        api = GitHubAPI()
        manager_repo_name = config['github']['managerRepo']
        repos_to_scan = config['github']['reposToScan']
        
        logger.info(f"Starting analysis of {len(repos_to_scan)} repositories")
        
        # Perform periodic cleanup of old task entries
        tracker = get_task_tracker()
        cleanup_days = config.get('analyzer', {}).get('cleanupTasksOlderThanDays', 90)
        cleaned_count = tracker.cleanup_old_tasks(days=cleanup_days)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old task entries")
        
        for repo_name in repos_to_scan:
            logger.info(f"Analyzing repository: {repo_name}")
            repo = api.get_repo(repo_name)
            if not repo:
                logger.warning(f"Skipping {repo_name} - could not access repository")
                continue

            if config['analyzer']['scanForTodos']:
                logger.debug(f"Running TODO scan for {repo_name}")
                find_todo_comments(api, repo, manager_repo_name)

            if config['analyzer']['scanOpenIssues']:
                logger.debug(f"Running issue analysis for {repo_name}")
                analyze_open_issues(api, repo, manager_repo_name)

        logger.info("Task analysis cycle completed successfully")
        
    except SystemExit:
        # Configuration validation error already logged
        raise
    except Exception as e:
        logger.error(f"Fatal error in task analyzer: {e}")
        raise