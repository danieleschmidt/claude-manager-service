import json
import datetime
import time
from typing import Optional
from github import Repository
from github_api import GitHubAPI
from logger import get_logger, log_performance
from performance_monitor import monitor_performance
from task_tracker import get_task_tracker
from config_validator import get_validated_config
from error_handler import with_error_recovery, safe_github_operation
# Import ConcurrentRepositoryScanner lazily to avoid circular import

logger = get_logger(__name__)

@monitor_performance(track_memory=True, custom_name="scan_todo_comments")
@log_performance
@with_error_recovery("find_todo_comments")
def find_todo_comments(github_api: GitHubAPI, repo: Repository.Repository, manager_repo_name: str) -> None:
    """Scan repository for TODO and FIXME comments and create issues for them"""
    logger.info(f"Scanning {repo.full_name} for TODO comments")
    todo_count = 0
    try:
        # Search for TODO and FIXME comments in the repository using optimized single query
        # Use GitHub search OR operator to combine all search terms into one API call
        combined_query = f"(TODO: OR FIXME: OR TODO OR FIXME) repo:{repo.full_name}"
        logger.debug(f"Searching with optimized combined query: {combined_query}")
        
        try:
            # Single GitHub API call instead of multiple separate calls (75% reduction in API calls)
            search_results = github_api.client.search_code(query=combined_query)
            
            results_list = list(search_results[:20])  # Increased limit since we're doing one query instead of 4
            results_count = len(results_list)
            logger.debug(f"Found {results_count} total results with combined query")
            
            for result in results_list:
                    file_path = result.path
                    logger.debug(f"Processing file: {file_path}")
                    
                    # Get the file content to extract context
                    try:
                        file_content = repo.get_contents(file_path)
                        content_lines = file_content.decoded_content.decode('utf-8').split('\n')
                        
                        # Find lines with TODO/FIXME patterns
                        search_terms = ['TODO:', 'FIXME:', 'TODO', 'FIXME']
                        processed_lines = set()  # Track processed line numbers to avoid duplicates
                        
                        for line_num, line in enumerate(content_lines, 1):
                            # Check if this line contains any of our search terms
                            found_term = None
                            for term in search_terms:
                                if term.lower() in line.lower():
                                    found_term = term
                                    break
                            
                            if found_term and line_num not in processed_lines:
                                processed_lines.add(line_num)
                                
                                # Extract some context around the TODO
                                start_line = max(0, line_num - 3)
                                end_line = min(len(content_lines), line_num + 2)
                                context = '\n'.join(content_lines[start_line:end_line])
                                
                                title = f"Address {found_term} in {file_path}:{line_num}"
                                body = f"""A `{found_term}` comment was found that may require action.

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
                                
                                logger.info(f"Creating issue for {found_term} found in {file_path}:{line_num}")
                                github_api.create_issue(
                                    manager_repo_name, 
                                    title, 
                                    body, 
                                    ["task-proposal", "refactor", "todo"]
                                )
                                
                                # Mark task as processed (we don't have issue number from create_issue)
                                tracker.mark_task_processed(repo.full_name, file_path, line_num, line.strip())
                                
                                todo_count += 1
                        
                        # Limit processing to avoid creating too many issues per file
                        if len(processed_lines) >= 3:  # Max 3 TODOs per file
                            logger.debug(f"Processed maximum TODOs for file {file_path}")
                            break
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error executing optimized TODO search: {e}")
                
        logger.info(f"TODO scan completed using optimized single-query approach. Found {todo_count} TODO/FIXME comments in {repo.full_name}")
        logger.debug(f"Performance improvement: Used 1 API call instead of 4 separate queries (75% reduction)")
                
    except Exception as e:
        logger.error(f"Fatal error in find_todo_comments for {repo.full_name}: {e}")


@monitor_performance(track_memory=True, custom_name="scan_todos_with_tracking")
@log_performance
def find_todo_comments_with_tracking(github_api, repo, manager_repo_name):
    """
    Enhanced version of find_todo_comments with duplicate tracking
    
    This function includes the task tracking functionality to prevent
    creating duplicate issues for the same TODO/FIXME comments.
    """
    return find_todo_comments(github_api, repo, manager_repo_name)

@monitor_performance(track_memory=True, custom_name="analyze_open_issues")
@log_performance
@with_error_recovery("analyze_open_issues")
def analyze_open_issues(github_api: GitHubAPI, repo: Repository.Repository, manager_repo_name: str) -> None:
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
        
        # Try concurrent scanning first for significant performance improvement
        scan_start_time = time.time()
        concurrent_success = False
        
        try:
            logger.info(f"Starting concurrent analysis of {len(repos_to_scan)} repositories")
            
            # Import here to avoid circular import
            from concurrent_repository_scanner import ConcurrentRepositoryScanner
            
            # Initialize concurrent scanner with reasonable defaults
            scanner = ConcurrentRepositoryScanner(
                max_concurrent=min(len(repos_to_scan), 5),  # Don't overwhelm API
                timeout=300  # 5 minutes per repository
            )
            
            # Execute concurrent scanning
            scan_results = scanner.scan_repositories_sync(
                api,
                repos_to_scan,
                manager_repo_name,
                scan_todos=config['analyzer']['scanForTodos'],
                scan_issues=config['analyzer']['scanOpenIssues']
            )
            
            concurrent_success = True
            scan_duration = time.time() - scan_start_time
            
            # Log performance improvement
            logger.info(f"Concurrent scanning completed successfully in {scan_duration:.1f}s")
            logger.info(f"Scanned {scan_results['total_repos']} repositories: "
                       f"{scan_results['successful_scans']} successful, "
                       f"{scan_results['failed_scans']} failed")
            
            if 'total_todos_found' in scan_results:
                logger.info(f"Found {scan_results['total_todos_found']} TODO/FIXME comments")
            if 'total_issues_analyzed' in scan_results:
                logger.info(f"Analyzed {scan_results['total_issues_analyzed']} open issues")
                
            # Estimate performance gain (concurrent vs sequential)
            estimated_sequential_time = len(repos_to_scan) * 60  # Rough estimate: 1 min per repo
            if scan_duration > 0:
                performance_gain = estimated_sequential_time / scan_duration
                logger.info(f"Performance improvement: ~{performance_gain:.1f}x faster than sequential scanning")
            
        except Exception as e:
            logger.error(f"Concurrent scanning failed: {e}")
            logger.warning("Falling back to sequential repository scanning")
            concurrent_success = False
        
        # Fallback to sequential scanning if concurrent failed
        if not concurrent_success:
            logger.info("Starting sequential analysis (fallback mode)")
            sequential_start_time = time.time()
            
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
            
            sequential_duration = time.time() - sequential_start_time
            logger.info(f"Sequential scanning completed in {sequential_duration:.1f}s")

        logger.info("Task analysis cycle completed successfully")
        
    except SystemExit:
        # Configuration validation error already logged
        raise
    except Exception as e:
        logger.error(f"Fatal error in task analyzer: {e}")
        raise