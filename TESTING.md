# Testing Guide for Claude Manager Service

## Overview

This document describes the testing framework and practices for the Claude Manager Service. The project uses pytest for unit testing with comprehensive coverage reporting.

## Test Framework Setup

### Dependencies
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **unittest.mock**: Built-in Python mocking

### Configuration
- **pytest.ini**: Main pytest configuration
- **Coverage target**: 80% minimum
- **Test discovery**: Automatic via `test_*.py` pattern

## Running Tests

### Using Virtual Environment
```bash
# Setup (one time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
source venv/bin/activate
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
```

### Using Test Runner Script
```bash
# Make executable (one time)
chmod +x test_runner.py

# Run tests
./test_runner.py
```

## Current Test Coverage

### Coverage Summary (as of implementation)
| Module | Statements | Missing | Coverage |
|--------|------------|---------|----------|
| github_api.py | 45 | 2 | 96% |
| prompt_builder.py | 25 | 0 | 100% |
| orchestrator.py | 85 | 24 | 72% |
| task_analyzer.py | 71 | 21 | 70% |
| **TOTAL** | **226** | **47** | **79%** |

### Test Suite Statistics
- **Total Tests**: 47
- **Pass Rate**: 100%
- **Test Files**: 4
- **Test Classes**: 4

## Test Structure

### Directory Layout
```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_github_api.py      (12 tests)
│   ├── test_prompt_builder.py  (15 tests)
│   ├── test_orchestrator.py    (10 tests)
│   └── test_task_analyzer.py   (10 tests)
└── integration/                (future)
```

## Test Categories

### Unit Tests

#### 1. GitHub API Tests (`test_github_api.py`)
- **Scope**: GitHubAPI class functionality
- **Coverage**: Authentication, repository operations, issue management
- **Key Tests**:
  - Token validation and initialization
  - Repository retrieval (success/failure)
  - Issue creation with duplicate prevention
  - Comment posting with error handling

#### 2. Prompt Builder Tests (`test_prompt_builder.py`)
- **Scope**: Template processing and prompt generation
- **Coverage**: Template loading, placeholder replacement, label-based selection
- **Key Tests**:
  - Template file handling (existing/missing)
  - Placeholder substitution with edge cases
  - Label-based template selection logic
  - Error handling for file operations

#### 3. Orchestrator Tests (`test_orchestrator.py`)
- **Scope**: Task execution orchestration
- **Coverage**: Terragon and Claude Flow task triggering
- **Key Tests**:
  - Terragon task comment generation
  - Claude Flow subprocess execution
  - Repository cloning and cleanup
  - Timeout and error handling
  - Repository extraction from issue body

#### 4. Task Analyzer Tests (`test_task_analyzer.py`)
- **Scope**: Repository scanning and issue analysis
- **Coverage**: TODO discovery, stale issue detection
- **Key Tests**:
  - TODO/FIXME comment scanning
  - Stale issue identification (>30 days)
  - Label-based filtering
  - API error handling

## Test Patterns and Best Practices

### 1. Mocking Strategy
```python
# External dependencies are mocked
@patch('module.external_dependency')
def test_function(self, mock_dependency):
    # Setup mock behavior
    mock_dependency.return_value = expected_value
    
    # Execute test
    result = function_under_test()
    
    # Verify interactions
    mock_dependency.assert_called_with(expected_args)
    assert result == expected_result
```

### 2. Error Handling Tests
```python
def test_function_with_error(self):
    # Test that functions handle errors gracefully
    mock_api.operation.side_effect = Exception("API Error")
    
    # Should not raise, but handle gracefully
    function_under_test(mock_api)
    
    # Verify error was handled appropriately
    assert_no_exceptions_raised()
```

### 3. Environment Variable Testing
```python
@patch.dict(os.environ, {'GITHUB_TOKEN': 'test_token'})
def test_with_environment(self):
    # Test with controlled environment variables
    api = GitHubAPI()
    assert api.token == 'test_token'
```

## Coverage Gaps and Improvement Areas

### 1. Integration Testing
- **Gap**: No end-to-end workflow tests
- **Priority**: High
- **Implementation**: Test complete scan→issue→execution cycle

### 2. Orchestrator Module
- **Gap**: 28% of statements not covered
- **Missing**: Error recovery, advanced subprocess handling
- **Priority**: High

### 3. Task Analyzer Module
- **Gap**: 30% of statements not covered
- **Missing**: Complex GitHub API interactions, edge cases
- **Priority**: Medium

### 4. Performance Testing
- **Gap**: No performance benchmarks
- **Implementation**: Load testing for multiple repository scanning
- **Priority**: Low

## Test Maintenance

### Adding New Tests
1. Follow existing naming conventions (`test_*`)
2. Use descriptive test names explaining the scenario
3. Include both positive and negative test cases
4. Mock external dependencies consistently
5. Maintain >80% coverage threshold

### Test Data Management
- Use Mock objects for GitHub API responses
- Create reusable test fixtures for common scenarios
- Avoid hardcoded values where possible

### Continuous Integration
- All tests must pass before merge
- Coverage threshold enforced at 80%
- Consider adding integration tests in CI pipeline

## Common Testing Scenarios

### 1. GitHub API Rate Limiting
```python
def test_rate_limit_handling(self):
    mock_api.operation.side_effect = GithubException(403, "Rate limited")
    # Test graceful degradation
```

### 2. Network Connectivity Issues
```python
def test_network_failure(self):
    mock_subprocess.side_effect = subprocess.TimeoutExpired("git", 30)
    # Test timeout handling
```

### 3. Malformed Configuration
```python
def test_invalid_config(self):
    with patch('builtins.open', mock_open(read_data="invalid json")):
        # Test configuration validation
```

## Future Testing Enhancements

### 1. Property-Based Testing
- Use hypothesis for edge case discovery
- Test with random but valid inputs
- Verify invariants across input space

### 2. Contract Testing
- Verify GitHub API contract compliance
- Test with different API versions
- Mock realistic API responses

### 3. Load Testing
- Multi-repository scanning performance
- Concurrent execution limits
- Memory usage patterns

### 4. Security Testing
- Token handling validation
- Input sanitization verification
- Subprocess injection prevention

---

## Quick Reference

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test File
```bash
python -m pytest tests/unit/test_github_api.py -v
```

### Run Specific Test
```bash
python -m pytest tests/unit/test_github_api.py::TestGitHubAPI::test_create_issue_success -v
```

### Coverage Report
```bash
# View in terminal
python -m pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report
python -m pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

---

*Last Updated: 2025-07-20*
*Test Suite Version: 1.0*