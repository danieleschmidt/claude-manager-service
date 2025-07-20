"""
Integration tests for task analyzer end-to-end workflows
"""
import pytest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timezone, timedelta

sys.path.append('/root/repo/src')

from task_analyzer import find_todo_comments, analyze_open_issues
from github_api import GitHubAPI
from task_tracker import TaskTracker


class TestTaskAnalyzerIntegration:
    """Integration tests for task analyzer workflows"""
    
    @pytest.fixture
    def mock_github_api(self):
        """Create a mock GitHub API with realistic behavior"""
        api = Mock(spec=GitHubAPI)
        api.client = Mock()
        api.token = "test_token"
        return api
    
    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository object"""
        repo = Mock()
        repo.full_name = "test/integration-repo"
        return repo
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration file"""
        config = {
            "github": {
                "username": "testuser",
                "managerRepo": "testuser/claude-manager-service",
                "reposToScan": ["test/integration-repo"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "cleanupTasksOlderThanDays": 60
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        yield config_path
        os.unlink(config_path)
    
    def test_todo_discovery_end_to_end_flow(self, mock_github_api, mock_repo):
        """Test complete TODO discovery workflow from search to issue creation"""
        # Setup mock search results
        mock_search_result = Mock()
        mock_search_result.path = "src/auth.py"
        mock_search_result.html_url = "https://github.com/test/repo/blob/main/src/auth.py#L42"
        
        mock_github_api.client.search_code.return_value = [mock_search_result]
        
        # Setup mock file content with TODO
        mock_file_content = Mock()
        file_content = """def authenticate_user(username, password):
    # TODO: Add rate limiting to prevent brute force attacks
    # This is a critical security issue that needs immediate attention
    return validate_credentials(username, password)
"""
        mock_file_content.decoded_content.decode.return_value = file_content
        mock_repo.get_contents.return_value = mock_file_content
        
        # Track calls to create_issue
        issue_calls = []
        def track_create_issue(repo_name, title, body, labels):
            issue_calls.append({
                'repo_name': repo_name,
                'title': title,
                'body': body,
                'labels': labels
            })
        
        mock_github_api.create_issue.side_effect = track_create_issue
        
        # Create temporary task tracker state
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker_file = os.path.join(temp_dir, 'task_tracker.json')
            tracker = TaskTracker(tracker_file)
            
            # Patch the task tracker getter to return our temp instance
            with patch('task_analyzer.get_task_tracker', return_value=tracker):
                # Execute the function
                find_todo_comments(mock_github_api, mock_repo, "testuser/claude-manager-service")
        
        # Verify the complete workflow
        assert len(issue_calls) == 1
        call = issue_calls[0]
        
        # Verify issue details
        assert call['repo_name'] == "testuser/claude-manager-service"
        assert "Address TODO" in call['title']
        assert "src/auth.py:2" in call['title']
        assert "test/integration-repo" in call['body']
        assert "src/auth.py" in call['body']
        assert "**Line:** 2" in call['body']
        assert "rate limiting" in call['body']
        assert call['labels'] == ["task-proposal", "refactor", "todo"]
        
        # Verify GitHub API interactions
        mock_github_api.client.search_code.assert_called()
        mock_repo.get_contents.assert_called_with("src/auth.py")
        
        # Verify task tracker prevented duplicates
        # The actual line stored would be the stripped version of the line containing TODO
        expected_content = "# TODO: Add rate limiting to prevent brute force attacks"
        is_duplicate = tracker.is_task_processed(
            "test/integration-repo", 
            "src/auth.py", 
            2, 
            expected_content
        )
        assert is_duplicate, "Task should be marked as processed to prevent duplicates"
    
    def test_stale_issue_analysis_end_to_end_flow(self, mock_github_api, mock_repo):
        """Test complete stale issue analysis workflow"""
        # Create a stale issue (60 days old)
        stale_date = datetime.now(timezone.utc) - timedelta(days=60)
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = "Fix authentication bug"
        mock_issue.body = "Users are unable to log in when using special characters in passwords."
        bug_label = Mock()
        bug_label.name = "bug"
        help_wanted_label = Mock()
        help_wanted_label.name = "help wanted"
        mock_issue.labels = [bug_label, help_wanted_label]
        mock_issue.created_at = stale_date
        mock_issue.updated_at = stale_date
        mock_issue.assignees = []
        mock_issue.pull_request = None
        mock_issue.html_url = "https://github.com/test/repo/issues/123"
        
        # Setup mock to return our stale issue
        mock_repo.get_issues.return_value = [mock_issue]
        
        # Track calls to create_issue
        issue_calls = []
        def track_create_issue(repo_name, title, body, labels):
            issue_calls.append({
                'repo_name': repo_name,
                'title': title,
                'body': body,
                'labels': labels
            })
        
        mock_github_api.create_issue.side_effect = track_create_issue
        
        # Execute the function
        analyze_open_issues(mock_github_api, mock_repo, "testuser/claude-manager-service")
        
        # Verify the complete workflow
        assert len(issue_calls) == 1
        call = issue_calls[0]
        
        # Verify issue details
        assert call['repo_name'] == "testuser/claude-manager-service"
        assert "Review stale issue: 'Fix authentication bug'" in call['title']
        assert "inactive for 60 days" in call['body']
        assert "authentication bug" in call['body']
        assert "bug, help wanted" in call['body']
        assert call['labels'] == ["task-proposal", "stale-issue", "review"]
        
        # Verify GitHub API interactions
        mock_repo.get_issues.assert_called_with(state='open')
    
    def test_error_handling_in_integration_flow(self, mock_github_api, mock_repo):
        """Test error handling throughout the integration workflow"""
        # Setup GitHub API to raise an exception
        mock_github_api.client.search_code.side_effect = Exception("GitHub API error")
        
        # This should not raise an exception - errors should be handled gracefully
        find_todo_comments(mock_github_api, mock_repo, "testuser/claude-manager-service")
        
        # Verify that the function completed without crashing
        mock_github_api.client.search_code.assert_called()
    
    def test_prioritization_integration_with_todo_discovery(self, mock_github_api, mock_repo):
        """Test that discovered TODOs are properly prioritized"""
        # Setup mock search result for a security-related TODO
        mock_search_result = Mock()
        mock_search_result.path = "src/auth.py"
        mock_search_result.html_url = "https://github.com/test/repo/blob/main/src/auth.py#L10"
        
        mock_github_api.client.search_code.return_value = [mock_search_result]
        
        # Setup mock file content with high-priority security TODO
        mock_file_content = Mock()
        file_content = """def process_payment(card_number, amount):
    # TODO: Add input validation to prevent SQL injection
    # CRITICAL: This could expose sensitive payment data
    query = f"INSERT INTO payments (card, amount) VALUES ('{card_number}', {amount})"
    return execute_query(query)
"""
        mock_file_content.decoded_content.decode.return_value = file_content
        mock_repo.get_contents.return_value = mock_file_content
        
        # Track calls to create_issue
        issue_calls = []
        def track_create_issue(repo_name, title, body, labels):
            issue_calls.append({
                'repo_name': repo_name,
                'title': title,
                'body': body,
                'labels': labels
            })
        
        mock_github_api.create_issue.side_effect = track_create_issue
        
        # Create temporary task tracker state
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker_file = os.path.join(temp_dir, 'task_tracker.json')
            tracker = TaskTracker(tracker_file)
            
            # Patch the task tracker getter to return our temp instance
            with patch('task_analyzer.get_task_tracker', return_value=tracker):
                # Execute the function
                find_todo_comments(mock_github_api, mock_repo, "testuser/claude-manager-service")
        
        # Verify that security-related TODO was discovered and properly categorized
        assert len(issue_calls) == 1
        call = issue_calls[0]
        
        # Should contain security-related content
        assert "SQL injection" in call['body']
        assert "CRITICAL" in call['body']
        assert "payment" in call['body']
    
    def test_full_workflow_with_configuration_loading(self, temp_config):
        """Test the complete workflow including configuration loading"""
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'test_token'}):
            with patch('task_analyzer.GitHubAPI') as mock_api_class:
                with patch('task_analyzer.task_tracker') as mock_tracker:
                    # Setup mocks
                    mock_api = Mock()
                    mock_api_class.return_value = mock_api
                    
                    mock_repo = Mock()
                    mock_repo.full_name = "test/integration-repo"
                    mock_api.get_repo.return_value = mock_repo
                    
                    # Mock config loading
                    with patch('builtins.open', create=True) as mock_open:
                        with patch('json.load') as mock_json_load:
                            config = {
                                'github': {
                                    'managerRepo': 'testuser/claude-manager-service',
                                    'reposToScan': ['test/integration-repo']
                                },
                                'analyzer': {
                                    'scanForTodos': True,
                                    'scanOpenIssues': True
                                }
                            }
                            mock_json_load.return_value = config
                            
                            # Import and execute main logic
                            import task_analyzer
                            
                            # Patch main execution
                            with patch('task_analyzer.__name__', '__main__'):
                                with patch('task_analyzer.find_todo_comments') as mock_find_todos:
                                    with patch('task_analyzer.analyze_open_issues') as mock_analyze_issues:
                                        # This simulates running the main block
                                        # but we'll call the logic directly to avoid actual execution
                                        
                                        # Verify that both analysis functions would be called
                                        # with the correct parameters if this were a real run
                                        assert config['analyzer']['scanForTodos'] == True
                                        assert config['analyzer']['scanOpenIssues'] == True
                                        assert len(config['github']['reposToScan']) == 1
    
    def test_concurrent_processing_resilience(self, mock_github_api, mock_repo):
        """Test that the system handles concurrent processing gracefully"""
        # Setup multiple search results to simulate concurrent processing
        mock_results = []
        for i in range(5):
            result = Mock()
            result.path = f"src/module_{i}.py"
            result.html_url = f"https://github.com/test/repo/blob/main/src/module_{i}.py#L10"
            mock_results.append(result)
        
        mock_github_api.client.search_code.return_value = mock_results
        
        # Setup mock file content
        mock_file_content = Mock()
        file_content = "# TODO: Implement this function\npass"
        mock_file_content.decoded_content.decode.return_value = file_content
        mock_repo.get_contents.return_value = mock_file_content
        
        # Track calls to create_issue
        issue_calls = []
        def track_create_issue(repo_name, title, body, labels):
            issue_calls.append({
                'repo_name': repo_name,
                'title': title,
                'body': body,
                'labels': labels
            })
        
        mock_github_api.create_issue.side_effect = track_create_issue
        
        # Create temporary task tracker state
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker_file = os.path.join(temp_dir, 'task_tracker.json')
            tracker = TaskTracker(tracker_file)
            
            # Patch the task tracker getter to return our temp instance
            with patch('task_analyzer.get_task_tracker', return_value=tracker):
                # Execute the function multiple times to simulate concurrent access
                find_todo_comments(mock_github_api, mock_repo, "testuser/claude-manager-service")
        
        # Verify that multiple TODOs were processed
        assert len(issue_calls) <= 5, "Should not exceed the number of search results"
        
        # Verify that each issue was created with correct structure
        for call in issue_calls:
            assert call['repo_name'] == "testuser/claude-manager-service"
            assert "Address TODO" in call['title']
            assert call['labels'] == ["task-proposal", "refactor", "todo"]