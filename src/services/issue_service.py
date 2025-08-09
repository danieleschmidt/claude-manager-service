"""
Issue Service for Claude Manager Service

This service provides issue management operations with GitHub integration,
tracking, metadata management, and bulk operations support.

Features:
- GitHub issue creation and management
- Issue status tracking and metadata
- Bulk issue operations with concurrency
- Integration with task tracking
- Performance monitoring and error handling
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..async_github_api import AsyncGitHubAPI
from ..logger import get_logger
from ..performance_monitor import monitor_performance
from ..error_handler import NetworkError, with_enhanced_error_handling
from ..task_tracker import get_task_tracker
from ..security import sanitize_issue_content_enhanced, validate_repo_name


logger = get_logger(__name__)


class IssueStatus(Enum):
    """Issue status enumeration"""
    PENDING = "pending"
    CREATED = "created"
    UPDATED = "updated"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class IssueMetadata:
    """Metadata for issue tracking"""
    issue_id: str
    title: str
    repository: str
    github_issue_number: Optional[int] = None
    status: IssueStatus = IssueStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    labels: List[str] = None
    priority_score: Optional[float] = None
    source_type: str = "manual"  # manual, todo, fixme, pr_feedback
    source_metadata: Dict[str, Any] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.source_metadata is None:
            self.source_metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BulkIssueResult:
    """Result of bulk issue operations"""
    total_issues: int
    successful_issues: int
    failed_issues: int
    created_issue_numbers: List[int]
    failed_issue_metadata: List[IssueMetadata]
    operation_duration: float
    success_rate: float


class IssueService:
    """
    Service for GitHub issue management and tracking
    
    This service handles all issue-related operations including creation,
    tracking, metadata management, and bulk operations with error recovery.
    """
    
    def __init__(self, max_concurrent_issues: int = 10, retry_attempts: int = 3):
        """
        Initialize issue service
        
        Args:
            max_concurrent_issues: Maximum concurrent issue operations
            retry_attempts: Number of retry attempts for failed operations
        """
        self.logger = get_logger(__name__)
        self.max_concurrent_issues = max_concurrent_issues
        self.retry_attempts = retry_attempts
        
        # Issue tracking
        self._issue_cache: Dict[str, IssueMetadata] = {}
        self._cache_lock = asyncio.Lock()
        
        # Task tracker for deduplication
        self.task_tracker = get_task_tracker()
        
        self.logger.info(f"Issue service initialized: max_concurrent={max_concurrent_issues}")
    
    @with_enhanced_error_handling("create_issue", use_rate_limiter=True, use_circuit_breaker=True)
    async def create_issue(self, repo_name: str, title: str, body: str, 
                          labels: List[str] = None, metadata: Dict[str, Any] = None) -> IssueMetadata:
        """
        Create a single GitHub issue
        
        Args:
            repo_name: Repository name (owner/repo format)
            title: Issue title
            body: Issue body content
            labels: Issue labels
            metadata: Additional metadata for tracking
            
        Returns:
            IssueMetadata object with creation results
        """
        if labels is None:
            labels = []
        if metadata is None:
            metadata = {}
        
        # Generate unique issue ID
        issue_id = f"{repo_name}:{hash(title + body)}"
        
        # Create issue metadata
        issue_metadata = IssueMetadata(
            issue_id=issue_id,
            title=title,
            repository=repo_name,
            labels=labels,
            source_metadata=metadata
        )
        
        self.logger.info(f"Creating issue '{title[:50]}...' in {repo_name}")
        
        try:
            # Validate and sanitize inputs
            validated_repo = validate_repo_name(repo_name)
            sanitized_title = sanitize_issue_content_enhanced(title)
            sanitized_body = sanitize_issue_content_enhanced(body)
            
            # Check for duplicates if enabled
            if await self._is_duplicate_issue(validated_repo, sanitized_title):
                self.logger.warning(f"Duplicate issue detected: '{sanitized_title}'")
                issue_metadata.status = IssueStatus.FAILED
                issue_metadata.error_message = "Duplicate issue detected"
                return issue_metadata
            
            # Create issue via GitHub API
            async with AsyncGitHubAPI() as github_api:
                github_issue = await github_api.create_issue(
                    validated_repo, sanitized_title, sanitized_body, labels
                )
                
                if github_issue:
                    issue_metadata.github_issue_number = github_issue.number
                    issue_metadata.status = IssueStatus.CREATED
                    issue_metadata.updated_at = datetime.now()
                    
                    self.logger.info(f"Issue created successfully: #{github_issue.number}")
                else:
                    raise NetworkError("Issue creation returned None", "create_issue", None)
        
        except Exception as e:
            issue_metadata.status = IssueStatus.FAILED
            issue_metadata.error_message = str(e)
            self.logger.error(f"Failed to create issue '{title[:50]}...': {e}")
            
            # Re-raise for enhanced error handling to process
            raise NetworkError(f"Issue creation failed: {str(e)}", "create_issue", e)
        
        finally:
            # Cache the issue metadata
            await self._cache_issue_metadata(issue_metadata)
        
        return issue_metadata
    
    @with_enhanced_error_handling("bulk_create_issues", use_circuit_breaker=True)
    async def bulk_create_issues(self, issue_data_list: List[Dict[str, Any]]) -> BulkIssueResult:
        """
        Create multiple issues concurrently
        
        Args:
            issue_data_list: List of issue data dictionaries
                Each dict should contain: repo_name, title, body, labels (optional), metadata (optional)
                
        Returns:
            BulkIssueResult with operation statistics
        """
        start_time = asyncio.get_event_loop().time()
        self.logger.info(f"Creating {len(issue_data_list)} issues concurrently")
        
        # Initialize result
        result = BulkIssueResult(
            total_issues=len(issue_data_list),
            successful_issues=0,
            failed_issues=0,
            created_issue_numbers=[],
            failed_issue_metadata=[],
            operation_duration=0.0,
            success_rate=0.0
        )
        
        if not issue_data_list:
            result.operation_duration = asyncio.get_event_loop().time() - start_time
            return result
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_issues)
        
        async def create_single_issue(issue_data: Dict[str, Any]) -> IssueMetadata:
            async with semaphore:
                return await self.create_issue(
                    repo_name=issue_data['repo_name'],
                    title=issue_data['title'],
                    body=issue_data['body'],
                    labels=issue_data.get('labels', []),
                    metadata=issue_data.get('metadata', {})
                )
        
        # Execute issue creation concurrently
        with monitor_performance("bulk_issue_creation"):
            creation_tasks = [create_single_issue(issue_data) for issue_data in issue_data_list]
            issue_results = await asyncio.gather(*creation_tasks, return_exceptions=True)
        
        # Process results
        for issue_result in issue_results:
            if isinstance(issue_result, Exception):
                self.logger.error(f"Issue creation failed: {issue_result}")
                result.failed_issues += 1
                # Create failed metadata for tracking
                failed_metadata = IssueMetadata(
                    issue_id=f"failed_{result.failed_issues}",
                    title="Failed Issue",
                    repository="unknown",
                    status=IssueStatus.FAILED,
                    error_message=str(issue_result)
                )
                result.failed_issue_metadata.append(failed_metadata)
            elif isinstance(issue_result, IssueMetadata):
                if issue_result.status == IssueStatus.CREATED:
                    result.successful_issues += 1
                    if issue_result.github_issue_number:
                        result.created_issue_numbers.append(issue_result.github_issue_number)
                else:
                    result.failed_issues += 1
                    result.failed_issue_metadata.append(issue_result)
        
        # Calculate final statistics
        result.operation_duration = asyncio.get_event_loop().time() - start_time
        result.success_rate = (result.successful_issues / result.total_issues) * 100 if result.total_issues > 0 else 0
        
        self.logger.info(f"Bulk issue creation completed: {result.successful_issues}/{result.total_issues} successful ({result.success_rate:.1f}%)")
        
        return result
    
    async def get_issue_status(self, issue_id: str) -> Optional[IssueMetadata]:
        """
        Get status of a tracked issue
        
        Args:
            issue_id: Unique issue identifier
            
        Returns:
            IssueMetadata if found, None otherwise
        """
        async with self._cache_lock:
            return self._issue_cache.get(issue_id)
    
    async def list_issues_by_repository(self, repo_name: str) -> List[IssueMetadata]:
        """
        List all tracked issues for a repository
        
        Args:
            repo_name: Repository name
            
        Returns:
            List of IssueMetadata for the repository
        """
        async with self._cache_lock:
            return [
                metadata for metadata in self._issue_cache.values()
                if metadata.repository == repo_name
            ]
    
    async def get_issue_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive issue statistics
        
        Returns:
            Dictionary with issue statistics
        """
        async with self._cache_lock:
            total_issues = len(self._issue_cache)
            
            status_counts = {}
            repo_counts = {}
            label_counts = {}
            
            for metadata in self._issue_cache.values():
                # Count by status
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by repository
                repo = metadata.repository
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
                
                # Count by labels
                for label in metadata.labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            success_rate = 0.0
            if total_issues > 0:
                successful = status_counts.get('created', 0)
                success_rate = (successful / total_issues) * 100
            
            return {
                'total_issues': total_issues,
                'status_counts': status_counts,
                'repository_counts': repo_counts,
                'label_counts': label_counts,
                'success_rate': success_rate,
                'most_active_repository': max(repo_counts.items(), key=lambda x: x[1])[0] if repo_counts else None
            }
    
    async def retry_failed_issues(self, max_retries: int = None) -> BulkIssueResult:
        """
        Retry failed issue creation attempts
        
        Args:
            max_retries: Maximum retry attempts (uses service default if None)
            
        Returns:
            BulkIssueResult with retry statistics
        """
        if max_retries is None:
            max_retries = self.retry_attempts
        
        # Find failed issues eligible for retry
        async with self._cache_lock:
            failed_issues = [
                metadata for metadata in self._issue_cache.values()
                if metadata.status == IssueStatus.FAILED and metadata.retry_count < max_retries
            ]
        
        if not failed_issues:
            self.logger.info("No failed issues eligible for retry")
            return BulkIssueResult(
                total_issues=0, successful_issues=0, failed_issues=0,
                created_issue_numbers=[], failed_issue_metadata=[],
                operation_duration=0.0, success_rate=100.0
            )
        
        self.logger.info(f"Retrying {len(failed_issues)} failed issues")
        
        # Convert failed issues back to creation data
        retry_data = []
        for metadata in failed_issues:
            metadata.retry_count += 1
            retry_data.append({
                'repo_name': metadata.repository,
                'title': metadata.title,
                'body': metadata.source_metadata.get('body', ''),
                'labels': metadata.labels,
                'metadata': metadata.source_metadata
            })
        
        # Retry creation
        return await self.bulk_create_issues(retry_data)
    
    async def _is_duplicate_issue(self, repo_name: str, title: str) -> bool:
        """
        Check if an issue with the same title already exists
        
        Args:
            repo_name: Repository name
            title: Issue title
            
        Returns:
            True if duplicate found, False otherwise
        """
        try:
            async with AsyncGitHubAPI() as github_api:
                # Search for existing issues with same title
                existing_issues = await github_api.search_issues(
                    repo_name, f'is:issue is:open in:title "{title}"'
                )
                
                return len(existing_issues) > 0
                
        except Exception as e:
            self.logger.warning(f"Duplicate check failed for '{title}': {e}")
            # If duplicate check fails, assume no duplicate to avoid blocking
            return False
    
    async def _cache_issue_metadata(self, metadata: IssueMetadata) -> None:
        """Cache issue metadata"""
        async with self._cache_lock:
            self._issue_cache[metadata.issue_id] = metadata
    
    async def clear_cache(self) -> None:
        """Clear issue cache"""
        async with self._cache_lock:
            self._issue_cache.clear()
            self.logger.info("Issue cache cleared")


# Global issue service instance
_issue_service: Optional[IssueService] = None


async def get_issue_service(max_concurrent_issues: int = 10) -> IssueService:
    """
    Get global issue service instance
    
    Args:
        max_concurrent_issues: Maximum concurrent issue operations
        
    Returns:
        Initialized issue service
    """
    global _issue_service
    
    if _issue_service is None:
        _issue_service = IssueService(max_concurrent_issues=max_concurrent_issues)
    
    return _issue_service


# Example usage and testing
async def example_issue_service():
    """Example of using issue service"""
    try:
        # Get issue service
        issue_service = await get_issue_service()
        
        # Create single issue
        single_issue = await issue_service.create_issue(
            repo_name="example/repo",
            title="Example Issue",
            body="This is an example issue",
            labels=["example", "test"]
        )
        
        # Create multiple issues
        issue_data = [
            {
                'repo_name': "example/repo",
                'title': f"Bulk Issue {i}",
                'body': f"This is bulk issue number {i}",
                'labels': ["bulk", "test"]
            }
            for i in range(3)
        ]
        
        bulk_result = await issue_service.bulk_create_issues(issue_data)
        
        # Get statistics
        stats = await issue_service.get_issue_statistics()
        
        logger.info(f"Issue service example completed")
        logger.info(f"Single issue status: {single_issue.status}")
        logger.info(f"Bulk creation: {bulk_result.successful_issues}/{bulk_result.total_issues} successful")
        logger.info(f"Statistics: {stats['total_issues']} total issues tracked")
        
    except Exception as e:
        logger.error(f"Issue service example failed: {e}")
        raise


if __name__ == "__main__":
    # Test issue service
    asyncio.run(example_issue_service())