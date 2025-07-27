"""
Async GitHub API wrapper for Claude Manager Service

This module provides async operations for GitHub API interactions,
wrapping the synchronous PyGithub library with asyncio thread execution
for improved performance in concurrent scenarios.

Features:
- Async/await compatibility for all GitHub operations
- Non-blocking I/O using thread pool execution
- Full compatibility with existing error handling and monitoring
- Performance improvements for concurrent repository operations
"""

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Awaitable, Any, Callable
from github import Repository, Issue

from .github_api import GitHubAPI
from .logger import get_logger
from .performance_monitor import monitor_performance


logger = get_logger(__name__)


def async_github_operation(operation_name: str):
    """
    Decorator to convert synchronous GitHub operations to async
    
    Args:
        operation_name: Name of the operation for logging and monitoring
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Use thread pool executor for non-blocking execution
            loop = asyncio.get_event_loop()
            
            # Create partial function for thread execution
            sync_func = functools.partial(func, self, *args, **kwargs)
            
            # Execute in thread pool
            try:
                result = await loop.run_in_executor(
                    self._thread_executor, 
                    sync_func
                )
                return result
            except Exception as e:
                logger.error(f"Async operation {operation_name} failed: {e}")
                raise
                
        return wrapper
    return decorator


class AsyncGitHubAPI:
    """
    Async wrapper for GitHub API operations
    
    This class provides async versions of all GitHub API operations
    by wrapping the synchronous GitHubAPI class with thread pool execution.
    """
    
    def __init__(self, max_workers: int = None):
        """
        Initialize async GitHub API wrapper
        
        Args:
            max_workers: Maximum number of threads for the executor
        """
        self.logger = get_logger(__name__)
        
        # Initialize synchronous GitHub API
        self._sync_api = GitHubAPI()
        
        # Configure thread pool executor
        self._max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._thread_executor = ThreadPoolExecutor(max_workers=self._max_workers)
        
        self.logger.info(f"Async GitHub API initialized with {self._max_workers} max workers")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup thread executor"""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            self.logger.debug("Thread executor shutdown completed")
    
    @async_github_operation("get_repository")
    def get_repo(self, repo_name: str) -> Optional[Repository.Repository]:
        """
        Async version of get_repo
        
        Args:
            repo_name: Repository name in format "owner/repo"
            
        Returns:
            GitHub Repository object or None if not found
        """
        return self._sync_api.get_repo(repo_name)
    
    @async_github_operation("create_issue")
    def create_issue(self, repo_name: str, title: str, body: Optional[str], labels: List[str]) -> None:
        """
        Async version of create_issue
        
        Args:
            repo_name: Repository name in format "owner/repo"
            title: Issue title
            body: Issue body content
            labels: List of labels to apply
        """
        return self._sync_api.create_issue(repo_name, title, body, labels)
    
    @async_github_operation("get_repository_contents")
    def get_repository_contents(self, repo: Repository.Repository, path: str = "") -> List[Any]:
        """
        Async version of get_repository_contents
        
        Args:
            repo: GitHub Repository object
            path: Path within repository
            
        Returns:
            List of repository contents
        """
        return self._sync_api.get_repository_contents(repo, path)
    
    @async_github_operation("search_code")
    def search_code(self, repo: Repository.Repository, query: str) -> List[Any]:
        """
        Async version of search_code
        
        Args:
            repo: GitHub Repository object
            query: Search query
            
        Returns:
            List of search results
        """
        return self._sync_api.search_code(repo, query)
    
    @async_github_operation("get_open_issues")
    def get_open_issues(self, repo: Repository.Repository) -> List[Issue.Issue]:
        """
        Async version of get_open_issues
        
        Args:
            repo: GitHub Repository object
            
        Returns:
            List of open issues
        """
        return self._sync_api.get_open_issues(repo)
    
    async def bulk_get_repos(self, repo_names: List[str]) -> List[Optional[Repository.Repository]]:
        """
        Get multiple repositories concurrently
        
        Args:
            repo_names: List of repository names
            
        Returns:
            List of Repository objects (None for not found)
        """
        self.logger.info(f"Fetching {len(repo_names)} repositories concurrently")
        
        # Create concurrent tasks
        tasks = [self.get_repo(repo_name) for repo_name in repo_names]
        
        # Execute concurrently
        with monitor_performance("bulk_get_repos"):
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Failed to fetch repository {repo_names[i]}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        successful = len([r for r in processed_results if r is not None])
        self.logger.info(f"Successfully fetched {successful}/{len(repo_names)} repositories")
        
        return processed_results
    
    async def bulk_create_issues(self, issue_data: List[dict]) -> List[bool]:
        """
        Create multiple issues concurrently
        
        Args:
            issue_data: List of dicts with keys: repo_name, title, body, labels
            
        Returns:
            List of success flags
        """
        self.logger.info(f"Creating {len(issue_data)} issues concurrently")
        
        # Create concurrent tasks
        tasks = []
        for data in issue_data:
            task = self.create_issue(
                data['repo_name'],
                data['title'], 
                data.get('body'),
                data.get('labels', [])
            )
            tasks.append(task)
        
        # Execute concurrently
        with monitor_performance("bulk_create_issues"):
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_flags = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to create issue {i}: {result}")
                success_flags.append(False)
            else:
                success_flags.append(True)
        
        successful = sum(success_flags)
        self.logger.info(f"Successfully created {successful}/{len(issue_data)} issues")
        
        return success_flags
    
    def shutdown(self):
        """Shutdown the thread executor"""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            self.logger.info("Async GitHub API shutdown completed")


# Convenience function for one-off async operations
async def execute_async_github_operation(operation_func: Callable, *args, **kwargs) -> Any:
    """
    Execute a single async GitHub operation
    
    Args:
        operation_func: Async GitHub API method
        *args, **kwargs: Arguments for the operation
        
    Returns:
        Operation result
    """
    async with AsyncGitHubAPI() as api:
        return await operation_func(api, *args, **kwargs)


# Example usage and testing
async def example_async_operations():
    """Example of how to use the async GitHub API"""
    async with AsyncGitHubAPI() as api:
        # Get multiple repositories concurrently
        repo_names = ["owner/repo1", "owner/repo2", "owner/repo3"]
        repos = await api.bulk_get_repos(repo_names)
        
        # Create multiple issues concurrently  
        issue_data = [
            {"repo_name": "owner/repo1", "title": "Issue 1", "body": "Body 1", "labels": ["bug"]},
            {"repo_name": "owner/repo2", "title": "Issue 2", "body": "Body 2", "labels": ["feature"]},
        ]
        results = await api.bulk_create_issues(issue_data)
        
        logger.info(f"Example completed: {len(repos)} repos fetched, {sum(results)} issues created")


if __name__ == "__main__":
    # Test the async GitHub API
    asyncio.run(example_async_operations())