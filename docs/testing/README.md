# Testing Guide

## Overview

The Claude Manager Service uses a comprehensive testing strategy that includes unit tests, integration tests, end-to-end tests, performance tests, and security tests. This guide provides information on how to run tests, write new tests, and understand the testing infrastructure.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Global test configuration and fixtures
├── fixtures/                   # Test data and mock responses
│   ├── __init__.py
│   ├── github_responses.py     # Mock GitHub API responses
│   └── sample_data.py          # Sample data for testing
├── utils/                      # Test utilities and helpers
│   ├── __init__.py
│   └── test_helpers.py         # Test helper functions
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_*.py               # Individual unit test files
├── integration/                # Integration tests
│   ├── test_*_integration.py   # Integration test files
├── e2e/                        # End-to-end tests
│   ├── test_full_workflow.py   # Complete workflow tests
├── performance/                # Performance tests
│   ├── test_performance_benchmarks.py
├── security/                   # Security tests
│   └── test_security_validation.py
├── load_testing/               # Load and stress tests
│   └── test_api_load.py        # Concurrent operations and load testing
└── contract/                   # Contract and schema tests
    └── test_api_contracts.py   # API contract validation
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit/
```

### Integration Tests Only
```bash
pytest tests/integration/
```

### End-to-End Tests Only
```bash
pytest tests/e2e/
```

### Performance Tests Only
```bash
pytest tests/performance/ --benchmark-only
```

### Security Tests Only
```bash
pytest tests/security/
```

### Load Tests Only
```bash
pytest tests/load_testing/ -m "slow"
```

### Contract Tests Only
```bash
pytest tests/contract/
```

### With Coverage Report
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Specific Test File
```bash
pytest tests/unit/test_task_analyzer.py
```

### Specific Test Function
```bash
pytest tests/unit/test_task_analyzer.py::test_find_todo_comments
```

### Run Tests in Parallel
```bash
pytest -n auto  # Requires pytest-xdist
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.github` - Tests requiring GitHub API access
- `@pytest.mark.slow` - Slow tests (>5 seconds)
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.load` - Load and stress tests
- `@pytest.mark.contract` - Contract validation tests

### Running Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only fast tests
pytest -m "not slow"

# Run integration and e2e tests
pytest -m "integration or e2e"

# Skip GitHub API tests
pytest -m "not github"
```

## Test Configuration

### Environment Variables for Testing

Create a `.env.test` file for test-specific configuration:

```bash
# Test environment configuration
TESTING=true
LOG_LEVEL=DEBUG
GITHUB_TOKEN=test_token
DATABASE_URL=sqlite:///:memory:
MOCK_EXTERNAL_APIS=true
PERFORMANCE_MONITORING_ENABLED=false
SECURITY_SCANNING_ENABLED=false
```

### Configuration Files

- `pytest.ini` - Pytest configuration
- `pyproject.toml` - Tool configuration (coverage, mypy, etc.)
- `.coveragerc` - Coverage configuration (if needed)

## Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import MagicMock, patch
from src.task_analyzer import TaskAnalyzer
from tests.fixtures import SampleData
from tests.utils import TestHelpers

class TestTaskAnalyzer:
    @pytest.fixture
    def analyzer(self, sample_config):
        return TaskAnalyzer(config=sample_config)

    @pytest.fixture
    def mock_github_api(self):
        return TestHelpers.create_github_api_mock()

    def test_analyze_todo_comments(self, analyzer, mock_github_api):
        # Arrange
        mock_repo = MagicMock()
        mock_github_api.get_repo.return_value = mock_repo
        
        with patch('src.task_analyzer.GitHubAPI', return_value=mock_github_api):
            # Act
            result = analyzer.find_todo_comments(mock_repo, "test-user/test-repo")
            
            # Assert
            assert result is not None
            mock_github_api.search_code.assert_called()

    @pytest.mark.parametrize("query,expected_count", [
        ("TODO", 5),
        ("FIXME", 3),
        ("HACK", 1)
    ])
    def test_search_patterns(self, analyzer, query, expected_count):
        with patch('src.task_analyzer.GitHubAPI') as mock_api:
            mock_api.return_value.search_code.return_value = [
                MagicMock() for _ in range(expected_count)
            ]
            
            result = analyzer.search_for_pattern(query)
            assert len(result) == expected_count
```

### Integration Test Example

```python
import pytest
import asyncio
from src.orchestrator import Orchestrator
from src.github_api import GitHubAPI
from tests.fixtures import GitHubFixtures

@pytest.mark.integration
@pytest.mark.asyncio
class TestOrchestratorIntegration:
    @pytest.fixture
    async def orchestrator(self, sample_config):
        return Orchestrator(config=sample_config)

    async def test_full_task_workflow(self, orchestrator, mock_env_vars):
        # Arrange
        task_data = {
            "title": "Fix authentication bug",
            "repository": "test-user/test-repo",
            "issue_number": 123
        }
        
        with patch('src.orchestrator.GitHubAPI') as mock_github:
            mock_github.return_value.get_issue.return_value = GitHubFixtures.issue_response()
            
            # Act
            result = await orchestrator.execute_task(task_data)
            
            # Assert
            assert result["status"] == "completed"
            assert "github_comment_added" in result
```

### Performance Test Example

```python
import pytest
import time
from src.performance_monitor import PerformanceMonitor

@pytest.mark.performance
class TestPerformanceBenchmarks:
    def test_task_processing_performance(self, benchmark):
        monitor = PerformanceMonitor()
        
        def process_tasks():
            # Simulate task processing
            tasks = [{"id": i} for i in range(100)]
            for task in tasks:
                monitor.record_metric("task_processed", 1)
            return len(tasks)
        
        result = benchmark(process_tasks)
        assert result == 100

    @pytest.mark.benchmark(group="api_calls")
    def test_github_api_performance(self, benchmark):
        from src.github_api import GitHubAPI
        
        with patch('src.github_api.Github') as mock_github:
            api = GitHubAPI()
            
            def make_api_call():
                return api.get_repo("test-user/test-repo")
            
            benchmark(make_api_call)
```

## Test Data and Fixtures

### Using Built-in Fixtures

```python
def test_with_sample_data(sample_config, sample_issue_data, temp_dir):
    # sample_config, sample_issue_data, temp_dir are provided by conftest.py
    assert sample_config["github"]["username"] == "test_user"
    assert sample_issue_data["number"] == 123
    assert temp_dir.exists()
```

### Creating Custom Fixtures

```python
@pytest.fixture
def custom_task_data():
    return {
        "id": "custom_task_001",
        "title": "Custom test task",
        "priority": "high"
    }

def test_with_custom_fixture(custom_task_data):
    assert custom_task_data["priority"] == "high"
```

### Using Mock Data

```python
from tests.fixtures import GitHubFixtures, SampleData

def test_with_mock_github_data():
    repo_data = GitHubFixtures.repository_response()
    issue_data = GitHubFixtures.issue_response(number=42)
    
    assert repo_data["name"] == "test-repo"
    assert issue_data["number"] == 42
```

## Mocking External Dependencies

### GitHub API Mocking

```python
from unittest.mock import patch, MagicMock

@patch('src.github_api.Github')
def test_github_api_interaction(mock_github_class):
    mock_github = MagicMock()
    mock_github_class.return_value = mock_github
    
    # Your test code here
    api = GitHubAPI()
    api.get_repo("test-user/test-repo")
    
    mock_github.get_repo.assert_called_with("test-user/test-repo")
```

### Database Mocking

```python
@patch('src.services.database_service.DatabaseService')
def test_database_interaction(mock_db_class):
    mock_db = MagicMock()
    mock_db_class.return_value = mock_db
    
    # Your test code here
```

### Environment Variable Mocking

```python
from tests.utils import temporary_env_vars

def test_with_env_vars():
    with temporary_env_vars({"GITHUB_TOKEN": "test_token"}):
        # Test code that uses the environment variable
        assert os.environ["GITHUB_TOKEN"] == "test_token"
```

## Async Testing

### Basic Async Test

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Testing Async Context Managers

```python
@pytest.mark.asyncio
async def test_async_context_manager():
    async with AsyncContextManager() as manager:
        result = await manager.do_something()
        assert result is not None
```

### Testing Async Generators

```python
@pytest.mark.asyncio
async def test_async_generator():
    items = []
    async for item in async_generator():
        items.append(item)
    
    assert len(items) > 0
```

## Test Coverage

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=src --cov-report=xml

# Show missing lines in terminal
pytest --cov=src --cov-report=term-missing
```

### Coverage Configuration

The coverage configuration is in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError"
]
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

## Debugging Tests

### Running Tests with Debugging

```bash
# Run with pdb on failure
pytest --pdb

# Run with verbose output
pytest -v -s

# Run specific test with debugging
pytest tests/unit/test_task_analyzer.py::test_find_todos -v -s --pdb
```

### Using Print Statements

```python
def test_debug_example():
    result = some_function()
    print(f"Debug: result = {result}")  # Will show in output with -s flag
    assert result is not None
```

### Logging in Tests

```python
import logging

def test_with_logging(caplog):
    with caplog.at_level(logging.INFO):
        logger = logging.getLogger(__name__)
        logger.info("Test message")
        
    assert "Test message" in caplog.text
```

## Test Best Practices

### 1. Test Structure (Arrange-Act-Assert)

```python
def test_example():
    # Arrange
    input_data = {"key": "value"}
    expected_result = "expected"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_result
```

### 2. Use Descriptive Test Names

```python
# Good
def test_task_analyzer_finds_todo_comments_in_python_files():
    pass

# Bad
def test_analyzer():
    pass
```

### 3. One Assertion Per Test (Generally)

```python
# Good
def test_task_creation_sets_correct_title():
    task = create_task(title="Test Task")
    assert task.title == "Test Task"

def test_task_creation_sets_pending_status():
    task = create_task()
    assert task.status == "pending"

# Acceptable for related assertions
def test_task_creation_sets_default_values():
    task = create_task()
    assert task.status == "pending"
    assert task.priority == "medium"
    assert task.created_at is not None
```

### 4. Use Fixtures for Common Setup

```python
@pytest.fixture
def configured_task_analyzer():
    config = load_test_config()
    return TaskAnalyzer(config)

def test_analyze_repository(configured_task_analyzer):
    # Test uses the fixture
    pass
```

### 5. Test Edge Cases

```python
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_empty_list_handling():
    result = process_list([])
    assert result == []

def test_none_input_handling():
    result = process_input(None)
    assert result is None
```

## Troubleshooting

### Common Issues

1. **Tests fail due to missing environment variables**
   - Solution: Use `mock_env_vars` fixture or set up `.env.test`

2. **Async tests hang**
   - Solution: Ensure proper use of `@pytest.mark.asyncio` and await statements

3. **Import errors in tests**
   - Solution: Check PYTHONPATH and ensure proper package structure

4. **Flaky tests**
   - Solution: Use proper waiting mechanisms and avoid time-dependent assertions

### Getting Help

- Check the [pytest documentation](https://docs.pytest.org/)
- Review existing tests for examples
- Use `pytest --help` for command-line options
- Check the test output and error messages carefully

## Contributing Tests

When contributing new tests:

1. Follow the existing test structure and naming conventions
2. Include both positive and negative test cases
3. Add appropriate markers (`@pytest.mark.unit`, etc.)
4. Update this documentation if adding new test types or patterns
5. Ensure tests are fast and reliable
6. Include docstrings for complex test scenarios