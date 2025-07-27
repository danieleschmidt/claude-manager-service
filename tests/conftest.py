"""
Global test configuration and fixtures for Claude Code Manager.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Mock environment variables for testing."""
    env_vars = {
        "GITHUB_TOKEN": "test_token",
        "GITHUB_USERNAME": "test_user",
        "GITHUB_REPO": "test_repo",
        "DATABASE_URL": "sqlite:///:memory:",
        "LOG_LEVEL": "DEBUG",
        "TESTING": "true",
        "ENABLE_PERFORMANCE_MONITORING": "false",
        "ENABLE_ENHANCED_SECURITY": "false",
        "SECURITY_MAX_CONTENT_LENGTH": "1000",
        "RATE_LIMIT_REQUESTS": "100",
        "RATE_LIMIT_WINDOW": "3600",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def mock_github_api():
    """Mock GitHub API client."""
    mock_api = MagicMock()
    mock_api.get_repo = MagicMock()
    mock_api.create_issue = MagicMock()
    mock_api.get_issue = MagicMock()
    mock_api.add_comment_to_issue = MagicMock()
    return mock_api


@pytest_asyncio.fixture
async def mock_async_github_api():
    """Mock async GitHub API client."""
    mock_api = AsyncMock()
    mock_api.get_repo = AsyncMock()
    mock_api.create_issue = AsyncMock()
    mock_api.get_issue = AsyncMock()
    mock_api.add_comment_to_issue = AsyncMock()
    return mock_api


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "github": {
            "username": "test_user",
            "managerRepo": "test_user/test_repo",
            "reposToScan": ["test_user/repo1", "test_user/repo2"]
        },
        "analyzer": {
            "scanForTodos": True,
            "scanOpenIssues": True
        },
        "executor": {
            "terragonUsername": "@terragon-labs"
        }
    }


@pytest.fixture
def sample_issue_data():
    """Sample GitHub issue data for testing."""
    return {
        "number": 123,
        "title": "Test Issue",
        "body": "This is a test issue",
        "state": "open",
        "labels": [
            {"name": "bug"},
            {"name": "priority-high"}
        ],
        "html_url": "https://github.com/test_user/test_repo/issues/123",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_repository_data():
    """Sample GitHub repository data for testing."""
    return {
        "full_name": "test_user/test_repo",
        "name": "test_repo",
        "description": "Test repository",
        "html_url": "https://github.com/test_user/test_repo",
        "default_branch": "main",
        "language": "Python"
    }


@pytest.fixture
def mock_database():
    """Mock database service."""
    mock_db = MagicMock()
    mock_db.create_task = MagicMock()
    mock_db.get_task = MagicMock()
    mock_db.update_task = MagicMock()
    mock_db.delete_task = MagicMock()
    mock_db.list_tasks = MagicMock()
    return mock_db


@pytest_asyncio.fixture
async def mock_async_database():
    """Mock async database service."""
    mock_db = AsyncMock()
    mock_db.create_task = AsyncMock()
    mock_db.get_task = AsyncMock()
    mock_db.update_task = AsyncMock()
    mock_db.delete_task = AsyncMock()
    mock_db.list_tasks = AsyncMock()
    return mock_db


@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor."""
    mock_monitor = MagicMock()
    mock_monitor.start_timing = MagicMock()
    mock_monitor.end_timing = MagicMock()
    mock_monitor.record_metric = MagicMock()
    mock_monitor.get_metrics = MagicMock(return_value={})
    return mock_monitor


@pytest.fixture
def mock_security_scanner():
    """Mock security scanner."""
    mock_scanner = MagicMock()
    mock_scanner.scan_content = MagicMock(return_value={"vulnerabilities": []})
    mock_scanner.validate_input = MagicMock(return_value=True)
    return mock_scanner


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup any test files created during the test
    test_files = [
        "test_output.json",
        "test_config.json",
        "test_log.txt",
    ]
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock_log = MagicMock()
    mock_log.info = MagicMock()
    mock_log.debug = MagicMock()
    mock_log.warning = MagicMock()
    mock_log.error = MagicMock()
    mock_log.critical = MagicMock()
    return mock_log


@pytest.fixture
def performance_data():
    """Sample performance data for testing."""
    return {
        "cpu_usage": 45.2,
        "memory_usage": 256.7,
        "disk_usage": 78.3,
        "network_io": 12.5,
        "response_time": 0.125,
        "throughput": 850.0,
        "error_rate": 0.01
    }


@pytest.fixture
def task_data():
    """Sample task data for testing."""
    return {
        "id": "task_123",
        "title": "Fix authentication bug",
        "description": "Users cannot log in with OAuth",
        "status": "pending",
        "priority": "high",
        "repository": "test_user/test_repo",
        "issue_number": 456,
        "labels": ["bug", "authentication"],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z"
    }


# Pytest plugins configuration
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_cov",
]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "github_api: marks tests that require GitHub API access"
    )
    config.addinivalue_line(
        "markers", "requires_network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "slow_test: marks tests that take more than 5 seconds"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests with 'github' in the name as requiring GitHub API
        if "github" in item.nodeid.lower():
            item.add_marker(pytest.mark.github)
        
        # Mark tests with 'integration' in path as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests with 'unit' in path as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set testing environment variable
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Disable external API calls during testing
    os.environ["MOCK_EXTERNAL_APIS"] = "true"
    
    yield
    
    # Cleanup
    test_env_vars = ["TESTING", "MOCK_EXTERNAL_APIS"]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]