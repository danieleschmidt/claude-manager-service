"""
Async Task Analyzer for Claude Manager Service

This module provides async task analysis functionality with significant
performance improvements over the synchronous approach through concurrent
processing of multiple repositories and operations.

Features:
- Async/await compatibility for all operations
- Concurrent repository scanning and issue creation
- Non-blocking I/O for GitHub API operations
- Full compatibility with existing error handling and monitoring
- Backward compatibility with synchronous task tracking
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from github import Repository

from src.async_github_api import AsyncGitHubAPI
from src.logger import get_logger, log_performance
from src.performance_monitor import monitor_performance
from src.task_tracker import get_task_tracker
from src.config_validator import get_validated_config
from src.error_handler import NetworkError


logger = get_logger(__name__)


class AsyncTaskAnalyzer:
    """
    Async task analyzer with concurrent repository processing capabilities
    
    This class provides async versions of all task analysis operations
    with significant performance improvements through concurrency.
    """
    
    def __init__(self, config_path: str = 'config.json', max_concurrent_repos: int = 5):
        """
        Initialize async task analyzer
        
        Args:
            config_path: Path to configuration file
            max_concurrent_repos: Maximum concurrent repository operations
        """
        self.logger = get_logger(__name__)
        self.config = get_validated_config(config_path)
        
        # Configuration
        self.manager_repo_name = self.config['github']['managerRepo']
        self.repos_to_scan = self.config['github']['reposToScan']
        self.max_concurrent_repos = max_concurrent_repos
        
        # Initialize task tracker (remains synchronous for persistence)
        self.task_tracker = get_task_tracker()
        
        self.logger.info(
            f"Async task analyzer initialized: "
            f"{len(self.repos_to_scan)} repos, max_concurrent={max_concurrent_repos}"
        )
    
    async def find_todo_comments_async(self, github_api: AsyncGitHubAPI, repo: Repository.Repository, 
                                     manager_repo_name: str) -> int:
        """
        Async version of find_todo_comments with concurrent processing
        
        Args:
            github_api: Async GitHub API instance
            repo: Repository object
            manager_repo_name: Manager repository name for issue creation
            
        Returns:
            Number of TODO comments processed
        """
        self.logger.info(f"Scanning {repo.full_name} for TODO comments (async)")
        todo_count = 0
        
        try:
            # Search for TODO and FIXME comments using optimized query
            combined_query = f"(TODO: OR FIXME: OR TODO OR FIXME) repo:{repo.full_name}"
            self.logger.debug(f"Searching with optimized combined query: {combined_query}")
            
            # Execute search operation asynchronously
            search_results = await github_api.search_code(repo, combined_query)
            results_list = list(search_results[:20])
            
            self.logger.debug(f"Found {len(results_list)} total results with combined query")
            
            # Process results concurrently
            tasks = []
            for result in results_list:
                task = self._process_todo_result_async(
                    github_api, repo, result, manager_repo_name
                )
                tasks.append(task)
            
            # Execute all processing tasks concurrently
            if tasks:
                with monitor_performance("concurrent_todo_processing"):
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successful processes
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"TODO processing failed: {result}")
                    elif isinstance(result, int):
                        todo_count += result
            
            self.logger.info(f"Completed async TODO scan for {repo.full_name}: {todo_count} issues created")
            return todo_count
            
        except Exception as e:
            self.logger.error(f"Error during async TODO scan for {repo.full_name}: {e}")
            raise NetworkError(f"TODO scan failed for {repo.full_name}", "find_todo_comments_async", e)
    
    async def _process_todo_result_async(self, github_api: AsyncGitHubAPI, repo: Repository.Repository,
                                       result: Any, manager_repo_name: str) -> int:
        """
        Process a single TODO search result asynchronously
        
        Args:
            github_api: Async GitHub API instance
            repo: Repository object
            result: Search result from GitHub
            manager_repo_name: Manager repository name
            
        Returns:
            Number of issues created for this result
        """
        issues_created = 0
        file_path = result.path
        
        try:
            # Get file content asynchronously
            file_contents = await github_api.get_repository_contents(repo, file_path)
            
            if not file_contents or not hasattr(file_contents[0], 'decoded_content'):
                return 0
            
            content_lines = file_contents[0].decoded_content.decode('utf-8').split('\n')
            search_terms = ['TODO:', 'FIXME:', 'TODO', 'FIXME']
            processed_lines = set()
            
            # Create issue data for batch processing
            issue_data_list = []
            
            for line_num, line in enumerate(content_lines, 1):
                # Limit to max 3 TODOs per file to avoid spam
                if len(issue_data_list) >= 3:
                    break
                
                # Check if line contains TODO/FIXME
                found_term = None
                for term in search_terms:
                    if term.lower() in line.lower():
                        found_term = term
                        break
                
                if found_term and line_num not in processed_lines:
                    processed_lines.add(line_num)
                    
                    # Check if already processed (synchronous check for data consistency)
                    if self.task_tracker.is_task_processed(repo.full_name, file_path, line_num, line.strip()):
                        self.logger.debug(f"Skipping already processed task: {file_path}:{line_num}")
                        continue
                    
                    # Prepare issue data
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
                    
                    issue_data_list.append({
                        'repo_name': manager_repo_name,
                        'title': title,
                        'body': body,
                        'labels': ["task-proposal", "refactor", "todo"],
                        'metadata': {
                            'repo_full_name': repo.full_name,
                            'file_path': file_path,
                            'line_num': line_num,
                            'line_content': line.strip()
                        }
                    })
            
            # Create issues concurrently if any were found
            if issue_data_list:
                self.logger.info(f"Creating {len(issue_data_list)} issues for {file_path}")
                
                # Extract just the issue creation data
                creation_data = [{k: v for k, v in item.items() if k != 'metadata'} 
                               for item in issue_data_list]
                
                success_flags = await github_api.bulk_create_issues(creation_data)
                
                # Mark successful tasks as processed
                for i, success in enumerate(success_flags):
                    if success:
                        metadata = issue_data_list[i]['metadata']
                        self.task_tracker.mark_task_processed(
                            metadata['repo_full_name'],
                            metadata['file_path'],
                            metadata['line_num'],
                            metadata['line_content']
                        )
                        issues_created += 1
            
            return issues_created
            
        except Exception as e:
            self.logger.error(f"Error processing TODO result for {file_path}: {e}")
            return 0
    
    async def analyze_open_issues_async(self, github_api: AsyncGitHubAPI, repo: Repository.Repository,
                                      manager_repo_name: str) -> int:
        """
        Async version of analyze_open_issues
        
        Args:
            github_api: Async GitHub API instance
            repo: Repository object
            manager_repo_name: Manager repository name
            
        Returns:
            Number of analysis results processed
        """
        self.logger.info(f"Analyzing open issues for {repo.full_name} (async)")
        
        try:
            # Get open issues asynchronously
            open_issues = await github_api.get_open_issues(repo)
            
            if not open_issues:
                self.logger.info(f"No open issues found in {repo.full_name}")
                return 0
            
            # Process issues concurrently (if needed)
            analysis_count = len(open_issues)
            
            # For now, just log the analysis - can be extended for more complex processing
            self.logger.info(f"Found {analysis_count} open issues in {repo.full_name}")
            
            return analysis_count
            
        except Exception as e:
            self.logger.error(f"Error analyzing issues for {repo.full_name}: {e}")
            raise NetworkError(f"Issue analysis failed for {repo.full_name}", "analyze_open_issues_async", e)
    
    async def run_analysis_async(self) -> Dict[str, Any]:
        """
        Run complete async task analysis on all configured repositories
        
        Returns:
            Analysis results summary
        """
        start_time = time.time()
        self.logger.info(f"Starting async task analysis for {len(self.repos_to_scan)} repositories")
        
        results = {
            'repositories_scanned': 0,
            'todos_found': 0,
            'issues_analyzed': 0,
            'errors': [],
            'execution_time': 0
        }
        
        async with AsyncGitHubAPI() as github_api:
            # Get all repositories concurrently
            self.logger.info("Fetching repositories concurrently...")
            repos = await github_api.bulk_get_repos(self.repos_to_scan)
            
            # Filter out None results and create analysis tasks
            valid_repos = [(name, repo) for name, repo in zip(self.repos_to_scan, repos) if repo is not None]
            results['repositories_scanned'] = len(valid_repos)
            
            if not valid_repos:
                self.logger.warning("No valid repositories found for analysis")
                return results
            
            # Create analysis tasks with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent_repos)
            
            async def analyze_single_repo(repo_name: str, repo: Repository.Repository):
                async with semaphore:
                    try:
                        # Run TODO analysis and issue analysis concurrently for each repo
                        todo_task = self.find_todo_comments_async(github_api, repo, self.manager_repo_name)
                        issue_task = self.analyze_open_issues_async(github_api, repo, self.manager_repo_name)
                        
                        todo_count, issue_count = await asyncio.gather(todo_task, issue_task)
                        
                        return {
                            'repo_name': repo_name,
                            'todos_found': todo_count,
                            'issues_analyzed': issue_count,
                            'success': True
                        }
                        
                    except Exception as e:
                        error_msg = f"Analysis failed for {repo_name}: {str(e)}"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)
                        
                        return {
                            'repo_name': repo_name,
                            'todos_found': 0,
                            'issues_analyzed': 0,
                            'success': False,
                            'error': str(e)
                        }
            
            # Execute all repository analyses concurrently
            self.logger.info(f"Analyzing {len(valid_repos)} repositories concurrently...")
            
            with monitor_performance("async_full_analysis"):
                analysis_tasks = [analyze_single_repo(name, repo) for name, repo in valid_repos]
                repo_results = await asyncio.gather(*analysis_tasks)
            
            # Aggregate results
            for repo_result in repo_results:
                if repo_result['success']:
                    results['todos_found'] += repo_result['todos_found']
                    results['issues_analyzed'] += repo_result['issues_analyzed']
        
        results['execution_time'] = time.time() - start_time
        
        self.logger.info(
            f"Async task analysis completed in {results['execution_time']:.2f}s: "
            f"{results['repositories_scanned']} repos, "
            f"{results['todos_found']} TODOs, "
            f"{results['issues_analyzed']} issues analyzed"
        )
        
        return results


# Convenience function for running async analysis
async def run_async_task_analysis(config_path: str = 'config.json', max_concurrent: int = 5) -> Dict[str, Any]:
    """
    Convenience function to run async task analysis
    
    Args:
        config_path: Path to configuration file
        max_concurrent: Maximum concurrent repository operations
        
    Returns:
        Analysis results
    """
    analyzer = AsyncTaskAnalyzer(config_path, max_concurrent)
    return await analyzer.run_analysis_async()


# Example usage
async def example_async_analysis():
    """Example of running async task analysis"""
    try:
        results = await run_async_task_analysis()
        logger.info(f"Analysis completed successfully: {results}")
        return results
    except Exception as e:
        logger.error(f"Async analysis failed: {e}")
        raise


if __name__ == "__main__":
    # Test async task analysis
    asyncio.run(example_async_analysis())