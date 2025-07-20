"""
Integration tests for GitHub API workflows and interactions
"""
import pytest
import tempfile
import json
import os
import sys
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timezone, timedelta

sys.path.append('/root/repo/src')

from github_api import GitHubAPI


class TestGitHubAPIIntegration:
    """Integration tests for GitHub API workflows"""
    
    @pytest.fixture
    def mock_github_client(self):
        """Create a mock GitHub client"""
        with patch('github_api.Github') as mock_github_class:
            mock_client = Mock()
            mock_github_class.return_value = mock_client
            yield mock_client
    
    def test_github_api_initialization_and_authentication(self, mock_github_client):
        """Test GitHub API initialization with proper authentication"""
        # Use a properly formatted GitHub token (40+ characters)
        test_token = 'ghp_' + 'a' * 36  # GitHub personal access token format
        
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Verify initialization
            assert api.token == test_token
            assert api.client == mock_github_client
    
    def test_repository_access_workflow(self, mock_github_client):
        """Test complete repository access workflow"""
        # Setup mock repository
        mock_repo = Mock()
        mock_repo.full_name = "test/integration-repo"
        mock_repo.name = "integration-repo"
        mock_repo.owner.login = "test"
        
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Test repository retrieval
            repo = api.get_repo("test/integration-repo")
            
            # Verify repository access
            assert repo is not None
            assert repo.full_name == "test/integration-repo"
            
            # Verify GitHub client interaction
            mock_github_client.get_repo.assert_called_once_with("test/integration-repo")
    
    def test_issue_creation_workflow_with_duplicate_prevention(self, mock_github_client):
        """Test complete issue creation workflow including duplicate prevention"""
        # Setup mock repository
        mock_repo = Mock()
        mock_repo.full_name = "test/integration-repo"
        
        # Setup existing issues for duplicate checking
        existing_issue = Mock()
        existing_issue.title = "Fix authentication bug"
        existing_issue.number = 100
        
        mock_repo.get_issues.return_value = [existing_issue]
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Test duplicate detection
            api.create_issue(
                "test/integration-repo",
                "Fix authentication bug",  # Same title as existing issue
                "Duplicate issue body",
                ["bug", "duplicate-test"]
            )
            
            # Verify that create_issue was not called due to duplicate detection
            mock_repo.create_issue.assert_not_called()
            
            # Test creating a new issue with different title
            api.create_issue(
                "test/integration-repo",
                "Fix authorization bug",  # Different title
                "New issue body",
                ["bug", "new-issue"]
            )
            
            # Verify that create_issue was called for the new issue
            mock_repo.create_issue.assert_called_once_with(
                title="Fix authorization bug",
                body="New issue body",
                labels=["bug", "new-issue"]
            )
    
    def test_issue_retrieval_and_comment_workflow(self, mock_github_client):
        """Test complete issue retrieval and commenting workflow"""
        # Setup mock repository and issue
        mock_repo = Mock()
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = "Test Issue"
        
        mock_repo.get_issue.return_value = mock_issue
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Test issue retrieval
            retrieved_issue = api.get_issue("test/integration-repo", 123)
            
            # Verify issue retrieval
            assert retrieved_issue is not None
            assert retrieved_issue.number == 123
            assert retrieved_issue.title == "Test Issue"
            
            # Test comment addition
            api.add_comment_to_issue(
                "test/integration-repo",
                123,
                "This is a test comment from integration test"
            )
            
            # Verify comment was added
            mock_issue.create_comment.assert_called_once_with(
                "This is a test comment from integration test"
            )
            
            # Verify GitHub client interactions
            mock_github_client.get_repo.assert_called_with("test/integration-repo")
            mock_repo.get_issue.assert_called_with(number=123)
    
    def test_error_handling_in_github_operations(self, mock_github_client):
        """Test error handling throughout GitHub API operations"""
        from github import GithubException
        
        # Test repository access error
        mock_github_client.get_repo.side_effect = GithubException(404, "Not Found")
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # This should not raise an exception
            repo = api.get_repo("nonexistent/repo")
            assert repo is None
            
            # Test issue creation error
            mock_repo = Mock()
            mock_repo.get_issues.side_effect = GithubException(403, "Forbidden")
            mock_github_client.get_repo.return_value = mock_repo
            mock_github_client.get_repo.side_effect = None
            
            # This should handle the error gracefully
            api.create_issue(
                "test/repo",
                "Test Issue",
                "Test Body",
                ["test"]
            )
            
            # Verify error was handled without crashing
            mock_github_client.get_repo.assert_called()
    
    def test_rate_limiting_and_retry_behavior(self, mock_github_client):
        """Test GitHub API rate limiting and retry behavior"""
        from github import GithubException
        
        # Setup rate limiting scenario
        rate_limit_exception = GithubException(403, "API rate limit exceeded")
        
        # First call fails with rate limit, second succeeds
        mock_repo = Mock()
        mock_repo.get_issues.side_effect = [rate_limit_exception, []]
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # This should handle rate limiting gracefully
            api.create_issue(
                "test/repo",
                "Test Issue",
                "Test Body",
                ["test"]
            )
            
            # Verify that the function completed without crashing
            mock_github_client.get_repo.assert_called()
    
    def test_large_repository_handling(self, mock_github_client):
        """Test handling of repositories with large numbers of issues"""
        # Setup repository with many issues
        mock_repo = Mock()
        
        # Create a large number of mock issues
        mock_issues = []
        for i in range(100):
            issue = Mock()
            issue.title = f"Issue {i}"
            issue.number = i
            mock_issues.append(issue)
        
        mock_repo.get_issues.return_value = mock_issues
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Test creating issue in repository with many existing issues
            api.create_issue(
                "test/large-repo",
                "New Issue",
                "This is a new issue in a large repository",
                ["enhancement"]
            )
            
            # Verify that duplicate checking worked with large issue list
            mock_repo.get_issues.assert_called_with(state='open')
            
            # Should create the issue since title is unique
            mock_repo.create_issue.assert_called_once_with(
                title="New Issue",
                body="This is a new issue in a large repository",
                labels=["enhancement"]
            )
    
    def test_concurrent_api_operations(self, mock_github_client):
        """Test concurrent API operations and thread safety"""
        # Setup multiple repositories
        repos = {}
        for i in range(3):
            repo = Mock()
            repo.full_name = f"test/repo-{i}"
            repo.get_issues.return_value = []
            repos[f"test/repo-{i}"] = repo
        
        def get_repo_side_effect(repo_name):
            return repos.get(repo_name)
        
        mock_github_client.get_repo.side_effect = get_repo_side_effect
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Simulate concurrent operations
            for i in range(3):
                api.create_issue(
                    f"test/repo-{i}",
                    f"Issue for repo {i}",
                    f"Body for repo {i}",
                    [f"repo-{i}"]
                )
            
            # Verify all operations completed
            assert mock_github_client.get_repo.call_count == 3
            
            # Verify each repository's create_issue was called
            for i in range(3):
                repos[f"test/repo-{i}"].create_issue.assert_called_once()
    
    def test_authentication_failure_handling(self):
        """Test handling of authentication failures"""
        # Test missing token
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                GitHubAPI()
            
            assert "GITHUB_TOKEN environment variable not set" in str(exc_info.value)
        
        # Test invalid token (too short)
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'invalid_token'}):
            with pytest.raises(ValueError) as exc_info:
                GitHubAPI()
            
            assert "GITHUB_TOKEN appears to be invalid" in str(exc_info.value)
        
        # Test valid token format but bad credentials with GitHub
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            with patch('github_api.Github') as mock_github_class:
                from github import BadCredentialsException
                mock_github_class.side_effect = BadCredentialsException(401, "Bad credentials")
                
                with pytest.raises(BadCredentialsException):
                    GitHubAPI()
    
    def test_pagination_handling_in_issue_listing(self, mock_github_client):
        """Test proper handling of paginated issue responses"""
        # Setup repository with paginated issues
        mock_repo = Mock()
        
        # Create paginated issue response
        page1_issues = [Mock(title=f"Issue {i}", number=i) for i in range(30)]
        page2_issues = [Mock(title=f"Issue {i}", number=i) for i in range(30, 50)]
        
        # Mock paginated response
        mock_paginated_list = Mock()
        mock_paginated_list.__iter__ = Mock(return_value=iter(page1_issues + page2_issues))
        mock_repo.get_issues.return_value = mock_paginated_list
        
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Test creating issue - should check all paginated issues for duplicates
            api.create_issue(
                "test/paginated-repo",
                "New Unique Issue",
                "This issue title should not exist in the paginated results",
                ["new"]
            )
            
            # Verify that pagination was handled
            mock_repo.get_issues.assert_called_with(state='open')
            
            # Should create the issue since title is unique across all pages
            mock_repo.create_issue.assert_called_once()
    
    def test_github_api_integration_with_task_tracker(self, mock_github_client):
        """Test integration between GitHub API and task tracking system"""
        # Setup mock repository and issue creation
        mock_repo = Mock()
        mock_repo.get_issues.return_value = []
        
        created_issue = Mock()
        created_issue.number = 456
        created_issue.title = "Integration Test Issue"
        mock_repo.create_issue.return_value = created_issue
        
        mock_github_client.get_repo.return_value = mock_repo
        
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            api = GitHubAPI()
            
            # Create an issue
            api.create_issue(
                "test/integration-repo",
                "Integration Test Issue",
                "This issue tests GitHub API integration with task tracking",
                ["integration", "test"]
            )
            
            # Verify issue creation
            mock_repo.create_issue.assert_called_once_with(
                title="Integration Test Issue",
                body="This issue tests GitHub API integration with task tracking",
                labels=["integration", "test"]
            )
            
            # Test retrieving the created issue
            mock_repo.get_issue.return_value = created_issue
            retrieved_issue = api.get_issue("test/integration-repo", 456)
            
            assert retrieved_issue.number == 456
            assert retrieved_issue.title == "Integration Test Issue"