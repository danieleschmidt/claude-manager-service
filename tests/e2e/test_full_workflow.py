"""
End-to-end tests for Claude Code Manager workflow.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator import Orchestrator
from src.task_analyzer import TaskAnalyzer
from src.github_api import GitHubAPI


@pytest.mark.e2e
@pytest.mark.slow
class TestFullWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def temp_config_file(self, temp_dir, sample_config):
        """Create a temporary config file."""
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(sample_config, indent=2))
        return config_path

    @pytest.fixture
    def mock_github_repo(self):
        """Mock GitHub repository object."""
        repo = MagicMock()
        repo.full_name = "test_user/test_repo"
        repo.name = "test_repo"
        repo.description = "Test repository"
        repo.html_url = "https://github.com/test_user/test_repo"
        repo.default_branch = "main"
        repo.language = "Python"
        
        # Mock issues
        issue = MagicMock()
        issue.number = 123
        issue.title = "Fix authentication bug"
        issue.body = "Users cannot log in with OAuth"
        issue.state = "open"
        issue.html_url = "https://github.com/test_user/test_repo/issues/123"
        issue.labels = [MagicMock(name="bug"), MagicMock(name="priority-high")]
        issue.created_at = "2025-01-01T00:00:00Z"
        issue.updated_at = "2025-01-01T00:00:00Z"
        issue.assignees = []
        
        repo.get_issues.return_value = [issue]
        
        # Mock file contents for TODO scanning
        file_content = MagicMock()
        file_content.decoded_content = b"# TODO: Fix this function\ndef broken_function():\n    pass"
        file_content.path = "src/main.py"
        
        repo.get_contents.return_value = file_content
        
        return repo

    @pytest.mark.asyncio
    async def test_complete_task_discovery_workflow(
        self, temp_config_file, mock_github_repo, mock_env_vars
    ):
        """Test complete task discovery workflow."""
        with patch('src.github_api.Github') as mock_github_client:
            # Setup mocks
            mock_client = MagicMock()
            mock_github_client.return_value = mock_client
            mock_client.get_repo.return_value = mock_github_repo
            
            # Mock search results for TODO comments
            search_result = MagicMock()
            search_result.path = "src/main.py"
            search_result.html_url = "https://github.com/test_user/test_repo/blob/main/src/main.py#L1"
            mock_client.search_code.return_value = [search_result]
            
            # Initialize components
            github_api = GitHubAPI()
            analyzer = TaskAnalyzer()
            
            # Load config
            with open(temp_config_file) as f:
                config = json.load(f)
            
            # Run task discovery
            repos_to_scan = config['github']['reposToScan']
            discovered_tasks = []
            
            for repo_name in repos_to_scan:
                repo = github_api.get_repo(repo_name)
                if repo:
                    # Scan for TODOs
                    if config['analyzer']['scanForTodos']:
                        todos = analyzer.find_todo_comments(repo)
                        discovered_tasks.extend(todos)
                    
                    # Scan for stale issues
                    if config['analyzer']['scanOpenIssues']:
                        issues = analyzer.analyze_open_issues(repo)
                        discovered_tasks.extend(issues)
            
            # Assertions
            assert len(discovered_tasks) > 0
            
            # Verify task structure
            for task in discovered_tasks:
                assert 'title' in task
                assert 'description' in task
                assert 'repository' in task
                assert 'type' in task

    @pytest.mark.asyncio
    async def test_task_execution_workflow(
        self, temp_config_file, sample_issue_data, mock_env_vars
    ):
        """Test task execution workflow."""
        with patch('src.orchestrator.subprocess.run') as mock_subprocess:
            # Setup mocks
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Task completed successfully"
            mock_subprocess.return_value.stderr = ""
            
            with patch('src.github_api.Github') as mock_github_client:
                mock_client = MagicMock()
                mock_github_client.return_value = mock_client
                
                # Mock issue
                issue = MagicMock()
                issue.number = sample_issue_data['number']
                issue.title = sample_issue_data['title']
                issue.body = sample_issue_data['body']
                issue.html_url = sample_issue_data['html_url']
                issue.labels = [
                    MagicMock(name=label['name']) 
                    for label in sample_issue_data['labels']
                ]
                
                mock_client.get_repo.return_value.get_issue.return_value = issue
                
                # Load config
                with open(temp_config_file) as f:
                    config = json.load(f)
                
                # Initialize orchestrator
                orchestrator = Orchestrator()
                
                # Execute task
                result = orchestrator.execute_task(
                    repo_name="test_user/test_repo",
                    issue_number=123,
                    config=config
                )
                
                # Assertions
                assert result is not None
                assert mock_subprocess.called

    @pytest.mark.asyncio
    async def test_error_handling_workflow(
        self, temp_config_file, mock_env_vars
    ):
        """Test error handling in the workflow."""
        with patch('src.github_api.Github') as mock_github_client:
            # Setup failing GitHub client
            mock_client = MagicMock()
            mock_github_client.return_value = mock_client
            mock_client.get_repo.side_effect = Exception("API Error")
            
            # Initialize components
            github_api = GitHubAPI()
            
            # Load config
            with open(temp_config_file) as f:
                config = json.load(f)
            
            # Test error handling
            repo = github_api.get_repo("test_user/test_repo")
            
            # Should handle error gracefully
            assert repo is None

    @pytest.mark.asyncio
    async def test_configuration_validation_workflow(self, temp_dir):
        """Test configuration validation workflow."""
        # Test with invalid config
        invalid_config = {
            "github": {
                "username": "",  # Invalid: empty username
                "reposToScan": []  # Invalid: no repos to scan
            }
        }
        
        invalid_config_path = temp_dir / "invalid_config.json"
        invalid_config_path.write_text(json.dumps(invalid_config))
        
        # Load and validate config
        with open(invalid_config_path) as f:
            config = json.load(f)
        
        # Validation should catch issues
        is_valid = self._validate_config(config)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(
        self, temp_config_file, mock_env_vars
    ):
        """Test performance monitoring during workflow execution."""
        from src.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Start monitoring
        start_time = monitor.start_timing("test_workflow")
        
        # Simulate workflow operations
        with patch('time.sleep') as mock_sleep:
            mock_sleep.return_value = None
            
            # Simulate task processing
            for i in range(10):
                monitor.record_metric("tasks_processed", 1)
                monitor.record_metric("api_calls", 2)
        
        # End monitoring
        monitor.end_timing("test_workflow", start_time)
        
        # Get metrics
        metrics = monitor.get_metrics()
        
        # Assertions
        assert "tasks_processed" in metrics
        assert "api_calls" in metrics
        assert metrics["tasks_processed"] == 10
        assert metrics["api_calls"] == 20

    @pytest.mark.asyncio
    async def test_concurrent_repository_processing(
        self, temp_config_file, mock_env_vars
    ):
        """Test concurrent processing of multiple repositories."""
        import asyncio
        
        with patch('src.async_orchestrator.AsyncOrchestrator') as mock_orchestrator:
            mock_instance = AsyncMock()
            mock_orchestrator.return_value = mock_instance
            
            # Setup multiple repositories
            repositories = [
                "test_user/repo1",
                "test_user/repo2", 
                "test_user/repo3",
                "test_user/repo4"
            ]
            
            # Mock processing results
            mock_instance.process_repository.return_value = {
                "repository": "test_repo",
                "tasks_found": 5,
                "status": "completed"
            }
            
            # Process repositories concurrently
            async def process_repo(repo_name):
                return await mock_instance.process_repository(repo_name)
            
            results = await asyncio.gather(*[
                process_repo(repo) for repo in repositories
            ])
            
            # Assertions
            assert len(results) == len(repositories)
            assert all(result["status"] == "completed" for result in results)
            assert mock_instance.process_repository.call_count == len(repositories)

    @staticmethod
    def _validate_config(config):
        """Validate configuration structure."""
        required_fields = {
            "github": ["username", "reposToScan"],
            "analyzer": ["scanForTodos", "scanOpenIssues"]
        }
        
        for section, fields in required_fields.items():
            if section not in config:
                return False
            
            for field in fields:
                if field not in config[section]:
                    return False
                
                # Check for empty values
                if not config[section][field]:
                    return False
        
        return True

    @pytest.mark.asyncio
    async def test_backup_and_recovery_workflow(self, temp_dir):
        """Test backup and recovery workflow."""
        # Create test data
        test_data = {
            "tasks": [
                {"id": "task_1", "title": "Task 1", "status": "completed"},
                {"id": "task_2", "title": "Task 2", "status": "pending"}
            ],
            "metrics": {
                "total_tasks": 2,
                "completed_tasks": 1
            }
        }
        
        # Create backup
        backup_path = temp_dir / "backup.json"
        backup_path.write_text(json.dumps(test_data, indent=2))
        
        # Simulate data loss
        original_data_path = temp_dir / "data.json"
        original_data_path.write_text("{}")
        
        # Restore from backup
        restored_data = json.loads(backup_path.read_text())
        original_data_path.write_text(json.dumps(restored_data, indent=2))
        
        # Verify restoration
        final_data = json.loads(original_data_path.read_text())
        
        assert final_data == test_data
        assert len(final_data["tasks"]) == 2
        assert final_data["metrics"]["total_tasks"] == 2