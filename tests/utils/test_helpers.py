"""
Test helper utilities for the Claude Manager Service test suite.
"""

import asyncio
import json
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestHelpers:
    """Collection of helper methods for testing."""

    @staticmethod
    def create_temp_config_file(config_data: Dict[str, Any]) -> str:
        """Create a temporary config file for testing."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(config_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name

    @staticmethod
    def create_temp_directory() -> str:
        """Create a temporary directory for testing."""
        return tempfile.mkdtemp()

    @staticmethod
    def assert_called_with_timeout(mock_obj, timeout: float = 1.0) -> bool:
        """Assert that a mock was called within a timeout period."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if mock_obj.called:
                return True
            time.sleep(0.01)
        return False

    @staticmethod
    def wait_for_condition(
        condition_func, timeout: float = 5.0, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

    @staticmethod
    async def async_wait_for_condition(
        condition_func, timeout: float = 5.0, interval: float = 0.1
    ) -> bool:
        """Async version of wait_for_condition."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        return False

    @staticmethod
    def mock_github_response(status_code: int = 200, json_data: Dict = None):
        """Create a mock HTTP response for GitHub API calls."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        mock_response.text = json.dumps(json_data or {})
        mock_response.headers = {"X-RateLimit-Remaining": "4999"}
        return mock_response

    @staticmethod
    def mock_async_github_response(status_code: int = 200, json_data: Dict = None):
        """Create a mock async HTTP response for GitHub API calls."""
        mock_response = AsyncMock()
        mock_response.status = status_code
        mock_response.json = AsyncMock(return_value=json_data or {})
        mock_response.text = AsyncMock(return_value=json.dumps(json_data or {}))
        mock_response.headers = {"X-RateLimit-Remaining": "4999"}
        return mock_response

    @staticmethod
    def create_mock_task(
        task_id: str = "test_task_001",
        status: str = "pending",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a mock task object for testing."""
        default_task = {
            "id": task_id,
            "title": "Test Task",
            "description": "This is a test task",
            "status": status,
            "priority": "medium",
            "repository": "test-user/test-repo",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
        default_task.update(kwargs)
        return default_task

    @staticmethod
    def simulate_github_rate_limit():
        """Simulate GitHub API rate limiting."""
        def side_effect(*args, **kwargs):
            from github import RateLimitExceededException
            raise RateLimitExceededException(403, "Rate limit exceeded")
        return side_effect

    @staticmethod
    def simulate_network_error():
        """Simulate network connectivity issues."""
        def side_effect(*args, **kwargs):
            import requests
            raise requests.ConnectionError("Network error")
        return side_effect


class AsyncTestHelpers:
    """Helper methods specifically for async testing."""

    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run an async coroutine with a timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    async def collect_async_generator(async_gen, max_items: int = 100):
        """Collect items from an async generator."""
        items = []
        count = 0
        async for item in async_gen:
            items.append(item)
            count += 1
            if count >= max_items:
                break
        return items

    @staticmethod
    @asynccontextmanager
    async def temporary_event_loop():
        """Create a temporary event loop for testing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield loop
        finally:
            loop.close()

    @staticmethod
    async def simulate_async_processing_delay(delay: float = 0.1):
        """Simulate processing delay in async operations."""
        await asyncio.sleep(delay)


class MockFactory:
    """Factory for creating various mock objects."""

    @staticmethod
    def create_github_api_mock() -> MagicMock:
        """Create a comprehensive GitHub API mock."""
        mock_api = MagicMock()
        
        # Repository operations
        mock_api.get_repo = MagicMock()
        mock_api.get_repos = MagicMock()
        
        # Issue operations
        mock_api.create_issue = MagicMock()
        mock_api.get_issue = MagicMock()
        mock_api.list_issues = MagicMock()
        mock_api.update_issue = MagicMock()
        mock_api.add_comment_to_issue = MagicMock()
        
        # Pull request operations
        mock_api.create_pull_request = MagicMock()
        mock_api.get_pull_request = MagicMock()
        mock_api.list_pull_requests = MagicMock()
        
        # Search operations
        mock_api.search_code = MagicMock()
        mock_api.search_issues = MagicMock()
        
        # File operations
        mock_api.get_file_content = MagicMock()
        mock_api.update_file = MagicMock()
        mock_api.create_file = MagicMock()
        
        return mock_api

    @staticmethod
    def create_async_github_api_mock() -> AsyncMock:
        """Create a comprehensive async GitHub API mock."""
        mock_api = AsyncMock()
        
        # Repository operations
        mock_api.get_repo = AsyncMock()
        mock_api.get_repos = AsyncMock()
        
        # Issue operations
        mock_api.create_issue = AsyncMock()
        mock_api.get_issue = AsyncMock()
        mock_api.list_issues = AsyncMock()
        mock_api.update_issue = AsyncMock()
        mock_api.add_comment_to_issue = AsyncMock()
        
        # Pull request operations
        mock_api.create_pull_request = AsyncMock()
        mock_api.get_pull_request = AsyncMock()
        mock_api.list_pull_requests = AsyncMock()
        
        # Search operations
        mock_api.search_code = AsyncMock()
        mock_api.search_issues = AsyncMock()
        
        # File operations
        mock_api.get_file_content = AsyncMock()
        mock_api.update_file = AsyncMock()
        mock_api.create_file = AsyncMock()
        
        return mock_api

    @staticmethod
    def create_database_mock() -> MagicMock:
        """Create a database service mock."""
        mock_db = MagicMock()
        
        # Task operations
        mock_db.create_task = MagicMock()
        mock_db.get_task = MagicMock()
        mock_db.update_task = MagicMock()
        mock_db.delete_task = MagicMock()
        mock_db.list_tasks = MagicMock()
        mock_db.search_tasks = MagicMock()
        
        # Performance metrics operations
        mock_db.store_metrics = MagicMock()
        mock_db.get_metrics = MagicMock()
        mock_db.clear_old_metrics = MagicMock()
        
        # Configuration operations
        mock_db.get_config = MagicMock()
        mock_db.update_config = MagicMock()
        
        return mock_db

    @staticmethod
    def create_performance_monitor_mock() -> MagicMock:
        """Create a performance monitor mock."""
        mock_monitor = MagicMock()
        
        mock_monitor.start_timing = MagicMock()
        mock_monitor.end_timing = MagicMock()
        mock_monitor.record_metric = MagicMock()
        mock_monitor.get_current_metrics = MagicMock()
        mock_monitor.get_metrics_history = MagicMock()
        mock_monitor.check_thresholds = MagicMock()
        mock_monitor.generate_alert = MagicMock()
        
        return mock_monitor


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_task_list(count: int = 10) -> List[Dict[str, Any]]:
        """Generate a list of test tasks."""
        tasks = []
        statuses = ["pending", "in_progress", "completed", "failed"]
        priorities = ["low", "medium", "high", "critical"]
        
        for i in range(count):
            task = TestHelpers.create_mock_task(
                task_id=f"task_{i:03d}",
                title=f"Test Task {i + 1}",
                status=statuses[i % len(statuses)],
                priority=priorities[i % len(priorities)],
                repository=f"test-user/repo-{i % 3 + 1}"
            )
            tasks.append(task)
        
        return tasks

    @staticmethod
    def generate_github_issues(count: int = 5) -> List[Dict[str, Any]]:
        """Generate a list of mock GitHub issues."""
        from tests.fixtures import GitHubFixtures
        
        issues = []
        for i in range(count):
            issue = GitHubFixtures.issue_response(
                number=i + 1,
                title=f"Issue {i + 1}: Sample problem",
                state="open" if i % 2 == 0 else "closed"
            )
            issues.append(issue)
        
        return issues

    @staticmethod
    def generate_performance_data(days: int = 7) -> List[Dict[str, Any]]:
        """Generate performance metrics data for testing."""
        import random
        from datetime import datetime, timedelta, timezone
        
        data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        for day in range(days):
            for hour in range(24):
                timestamp = base_time + timedelta(days=day, hours=hour)
                metrics = {
                    "timestamp": timestamp.isoformat(),
                    "cpu_usage": random.uniform(20, 80),
                    "memory_usage": random.uniform(100, 800),
                    "response_time": random.uniform(50, 500),
                    "throughput": random.uniform(100, 1000),
                    "error_rate": random.uniform(0, 5)
                }
                data.append(metrics)
        
        return data


@contextmanager
def temporary_env_vars(env_vars: Dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set environment variables for testing."""
    import os
    old_values = {}
    
    # Store old values and set new ones
    for key, value in env_vars.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield
    finally:
        # Restore old values
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def capture_logs(logger_name: str = None) -> Generator[List[str], None, None]:
    """Capture log messages for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def test_helpers():
    """Provide test helpers as a fixture."""
    return TestHelpers


@pytest.fixture
def async_test_helpers():
    """Provide async test helpers as a fixture."""
    return AsyncTestHelpers


@pytest.fixture
def mock_factory():
    """Provide mock factory as a fixture."""
    return MockFactory


@pytest.fixture
def test_data_generator():
    """Provide test data generator as a fixture."""
    return TestDataGenerator