"""
Repository Service for Claude Manager Service

This service provides repository management and scanning operations
with async support and comprehensive error handling.

Features:
- Repository scanning and analysis
- Concurrent repository operations
- Code search and content analysis
- Integration with GitHub API
- Performance monitoring and caching
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from github import Repository

from ..async_github_api import AsyncGitHubAPI
from ..logger import get_logger
from ..performance_monitor import monitor_performance
from ..error_handler import NetworkError, with_enhanced_error_handling
from ..task_tracker import get_task_tracker


logger = get_logger(__name__)


@dataclass
class RepositoryInfo:
    """Information about a repository"""
    name: str
    full_name: str
    repository: Repository.Repository
    last_scanned: Optional[datetime] = None
    scan_results: Optional[Dict[str, Any]] = None
    error_count: int = 0


@dataclass 
class ScanResult:
    """Result of a repository scan"""
    repository_name: str
    todos_found: int
    issues_analyzed: int
    files_scanned: int
    scan_duration: float
    success: bool
    error_message: Optional[str] = None


class RepositoryService:
    """
    Service for repository management and scanning operations
    
    This service handles all repository-related operations including
    scanning for TODOs, analyzing issues, and managing repository state.
    """
    
    def __init__(self, max_concurrent_repos: int = 5, cache_ttl_minutes: int = 30):
        """
        Initialize repository service
        
        Args:
            max_concurrent_repos: Maximum concurrent repository operations
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.logger = get_logger(__name__)
        self.max_concurrent_repos = max_concurrent_repos
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Repository cache
        self._repository_cache: Dict[str, RepositoryInfo] = {}
        self._cache_lock = asyncio.Lock()
        
        # Task tracker for deduplication
        self.task_tracker = get_task_tracker()
        
        self.logger.info(f"Repository service initialized: max_concurrent={max_concurrent_repos}")
    
    @with_enhanced_error_handling("get_repositories", use_rate_limiter=True, use_circuit_breaker=True)
    async def get_repositories(self, repo_names: List[str], use_cache: bool = True) -> List[RepositoryInfo]:
        """
        Get repository objects for the given names
        
        Args:
            repo_names: List of repository names
            use_cache: Whether to use cached results
            
        Returns:
            List of RepositoryInfo objects
        """
        self.logger.info(f"Getting {len(repo_names)} repositories")
        
        # Check cache first
        if use_cache:
            cached_repos, missing_names = await self._get_cached_repositories(repo_names)
        else:
            cached_repos, missing_names = [], repo_names
        
        # Fetch missing repositories
        if missing_names:
            async with AsyncGitHubAPI() as github_api:
                repos = await github_api.bulk_get_repos(missing_names)
                
                # Create RepositoryInfo objects
                for name, repo in zip(missing_names, repos):
                    if repo:
                        repo_info = RepositoryInfo(
                            name=name,
                            full_name=repo.full_name,
                            repository=repo
                        )
                        cached_repos.append(repo_info)
                        
                        # Update cache
                        await self._update_repository_cache(repo_info)
                    else:
                        self.logger.warning(f"Repository not found or inaccessible: {name}")
        
        self.logger.info(f"Retrieved {len(cached_repos)} repositories successfully")
        return cached_repos
    
    @with_enhanced_error_handling("scan_repository_todos", use_circuit_breaker=True)
    async def scan_repository_todos(self, repo_info: RepositoryInfo, manager_repo_name: str) -> ScanResult:
        """
        Scan repository for TODO comments
        
        Args:
            repo_info: Repository information
            manager_repo_name: Manager repository for issue creation
            
        Returns:
            Scan result information
        """
        start_time = asyncio.get_event_loop().time()
        self.logger.info(f"Scanning {repo_info.full_name} for TODOs")
        
        scan_result = ScanResult(
            repository_name=repo_info.full_name,
            todos_found=0,
            issues_analyzed=0,
            files_scanned=0,
            scan_duration=0.0,
            success=False
        )
        
        try:
            async with AsyncGitHubAPI() as github_api:
                # Search for TODO/FIXME comments
                combined_query = f"(TODO: OR FIXME: OR TODO OR FIXME) repo:{repo_info.full_name}"
                search_results = await github_api.search_code(repo_info.repository, combined_query)
                
                results_list = list(search_results[:20])  # Limit results
                scan_result.files_scanned = len(results_list)
                
                # Process results concurrently
                if results_list:
                    todo_tasks = [
                        self._process_todo_result(github_api, repo_info.repository, result, manager_repo_name)
                        for result in results_list
                    ]
                    
                    with monitor_performance("concurrent_todo_processing"):
                        todo_results = await asyncio.gather(*todo_tasks, return_exceptions=True)
                    
                    # Count successful TODOs
                    for result in todo_results:
                        if isinstance(result, Exception):
                            self.logger.warning(f"TODO processing failed: {result}")
                        elif isinstance(result, int):
                            scan_result.todos_found += result
                
                scan_result.success = True
                self.logger.info(f"TODO scan completed for {repo_info.full_name}: {scan_result.todos_found} TODOs found")
                
        except Exception as e:
            scan_result.error_message = str(e)
            self.logger.error(f"TODO scan failed for {repo_info.full_name}: {e}")
            raise NetworkError(f"TODO scan failed for {repo_info.full_name}", "scan_repository_todos", e)
        
        finally:
            scan_result.scan_duration = asyncio.get_event_loop().time() - start_time
            
            # Update repository info
            repo_info.last_scanned = datetime.now()
            repo_info.scan_results = {
                'todos_found': scan_result.todos_found,
                'files_scanned': scan_result.files_scanned,
                'last_scan': repo_info.last_scanned.isoformat()
            }
            
            await self._update_repository_cache(repo_info)
        
        return scan_result
    
    async def _process_todo_result(self, github_api: AsyncGitHubAPI, repo: Repository.Repository, 
                                 result: Any, manager_repo_name: str) -> int:
        """
        Process a single TODO search result
        
        Args:
            github_api: GitHub API instance
            repo: Repository object
            result: Search result
            manager_repo_name: Manager repository name
            
        Returns:
            Number of issues created
        """
        issues_created = 0
        file_path = result.path
        
        try:
            # Get file content
            file_contents = await github_api.get_repository_contents(repo, file_path)
            
            if not file_contents or not hasattr(file_contents[0], 'decoded_content'):
                return 0
            
            content_lines = file_contents[0].decoded_content.decode('utf-8').split('\n')
            search_terms = ['TODO:', 'FIXME:', 'TODO', 'FIXME']
            processed_lines = set()
            
            # Prepare issue data for batch processing
            issue_data_list = []
            
            for line_num, line in enumerate(content_lines, 1):
                # Limit to max 3 TODOs per file
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
                    
                    # Check if already processed
                    if self.task_tracker.is_task_processed(repo.full_name, file_path, line_num, line.strip()):
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
            
            # Create issues concurrently
            if issue_data_list:
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
    
    async def scan_multiple_repositories(self, repo_names: List[str], 
                                       manager_repo_name: str) -> List[ScanResult]:
        """
        Scan multiple repositories concurrently
        
        Args:
            repo_names: List of repository names to scan
            manager_repo_name: Manager repository for issues
            
        Returns:
            List of scan results
        """
        self.logger.info(f"Scanning {len(repo_names)} repositories concurrently")
        
        # Get repositories
        repo_infos = await self.get_repositories(repo_names)
        
        if not repo_infos:
            self.logger.warning("No repositories available for scanning")
            return []
        
        # Create scanning tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_repos)
        
        async def scan_single_repo(repo_info: RepositoryInfo) -> ScanResult:
            async with semaphore:
                return await self.scan_repository_todos(repo_info, manager_repo_name)
        
        # Execute scans concurrently
        with monitor_performance("bulk_repository_scan"):
            scan_tasks = [scan_single_repo(repo_info) for repo_info in repo_infos]
            results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Process results
        scan_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Repository scan failed: {result}")
                # Create failed scan result
                scan_results.append(ScanResult(
                    repository_name="unknown",
                    todos_found=0,
                    issues_analyzed=0,
                    files_scanned=0,
                    scan_duration=0.0,
                    success=False,
                    error_message=str(result)
                ))
            else:
                scan_results.append(result)
        
        successful_scans = len([r for r in scan_results if r.success])
        self.logger.info(f"Repository scanning completed: {successful_scans}/{len(scan_results)} successful")
        
        return scan_results
    
    async def get_repository_statistics(self, repo_names: List[str]) -> Dict[str, Any]:
        """
        Get statistics for repositories
        
        Args:
            repo_names: List of repository names
            
        Returns:
            Repository statistics
        """
        repo_infos = await self.get_repositories(repo_names)
        
        stats = {
            'total_repositories': len(repo_infos),
            'last_scanned_count': 0,
            'total_todos_found': 0,
            'average_scan_duration': 0.0,
            'repositories': []
        }
        
        total_duration = 0.0
        scanned_count = 0
        
        for repo_info in repo_infos:
            repo_stats = {
                'name': repo_info.full_name,
                'last_scanned': repo_info.last_scanned.isoformat() if repo_info.last_scanned else None,
                'scan_results': repo_info.scan_results
            }
            stats['repositories'].append(repo_stats)
            
            if repo_info.last_scanned:
                stats['last_scanned_count'] += 1
                scanned_count += 1
            
            if repo_info.scan_results:
                stats['total_todos_found'] += repo_info.scan_results.get('todos_found', 0)
        
        if scanned_count > 0:
            stats['average_scan_duration'] = total_duration / scanned_count
        
        return stats
    
    async def _get_cached_repositories(self, repo_names: List[str]) -> tuple[List[RepositoryInfo], List[str]]:
        """Get repositories from cache and return missing names"""
        async with self._cache_lock:
            cached_repos = []
            missing_names = []
            
            for name in repo_names:
                if name in self._repository_cache:
                    repo_info = self._repository_cache[name]
                    # Check if cache is still valid
                    if repo_info.last_scanned and datetime.now() - repo_info.last_scanned < self.cache_ttl:
                        cached_repos.append(repo_info)
                    else:
                        missing_names.append(name)
                else:
                    missing_names.append(name)
            
            return cached_repos, missing_names
    
    async def _update_repository_cache(self, repo_info: RepositoryInfo) -> None:
        """Update repository cache"""
        async with self._cache_lock:
            self._repository_cache[repo_info.name] = repo_info
    
    async def clear_cache(self) -> None:
        """Clear repository cache"""
        async with self._cache_lock:
            self._repository_cache.clear()
            self.logger.info("Repository cache cleared")


# Example usage and testing
async def example_repository_service():
    """Example of using repository service"""
    try:
        # Initialize service
        repo_service = RepositoryService(max_concurrent_repos=3)
        
        # Example repository names (replace with actual repositories)
        repo_names = ["octocat/Hello-World"]
        manager_repo = "manager/repo"
        
        # Scan repositories
        scan_results = await repo_service.scan_multiple_repositories(repo_names, manager_repo)
        
        # Get statistics
        stats = await repo_service.get_repository_statistics(repo_names)
        
        logger.info(f"Repository service example completed")
        logger.info(f"Scan results: {len(scan_results)} repositories processed")
        logger.info(f"Statistics: {stats['total_todos_found']} TODOs found")
        
    except Exception as e:
        logger.error(f"Repository service example failed: {e}")
        raise


if __name__ == "__main__":
    # Test repository service
    asyncio.run(example_repository_service())