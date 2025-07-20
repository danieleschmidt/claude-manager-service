"""
Unit tests for github_api.py module (with security mocking)
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from github import Github, GithubException

import sys
sys.path.append('/root/repo/src')


class TestGitHubAPI:
    """Test cases for GitHubAPI class with proper security mocking"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create reusable mocks
        self.mock_config = Mock()
        self.mock_config.get_github_token.return_value = 'ghp_' + 'x' * 36
        
        self.mock_github_client = Mock()
        self.mock_user = Mock()
        self.mock_user.login = 'testuser'
        self.mock_github_client.get_user.return_value = self.mock_user

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    def test_init_with_token(self, mock_validate, mock_github, mock_get_config):
        """Test GitHubAPI initialization with token"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        
        assert api.token == 'ghp_' + 'x' * 36
        mock_github.assert_called_once_with('ghp_' + 'x' * 36)
        mock_get_config.assert_called_once()

    @patch('github_api.get_secure_config')
    def test_init_without_token_raises_error(self, mock_get_config):
        """Test GitHubAPI initialization without token raises ValueError"""
        mock_config = Mock()
        mock_config.get_github_token.side_effect = ValueError("Missing required environment variables")
        mock_get_config.return_value = mock_config
        
        from github_api import GitHubAPI
        with pytest.raises(ValueError, match="Missing required environment variables"):
            GitHubAPI()

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    def test_get_repo_success(self, mock_validate, mock_github, mock_get_config):
        """Test successful repository retrieval"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        mock_repo = Mock()
        self.mock_github_client.get_repo.return_value = mock_repo
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        result = api.get_repo('test/repo')
        
        assert result == mock_repo
        self.mock_github_client.get_repo.assert_called_once_with('test/repo')
        mock_validate.assert_called_once_with('test/repo')

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=False)
    def test_get_repo_invalid_name(self, mock_validate, mock_github, mock_get_config):
        """Test repository retrieval with invalid name"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        result = api.get_repo('invalid/repo/name')
        
        assert result is None
        mock_validate.assert_called_once_with('invalid/repo/name')
        self.mock_github_client.get_repo.assert_not_called()

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    def test_get_repo_failure(self, mock_validate, mock_github, mock_get_config):
        """Test repository retrieval failure"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        self.mock_github_client.get_repo.side_effect = GithubException(404, "Not found", None)
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        result = api.get_repo('nonexistent/repo')
        
        assert result is None

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    @patch('github_api.sanitize_issue_content')
    def test_create_issue_success(self, mock_sanitize, mock_validate, mock_github, mock_get_config):
        """Test successful issue creation"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        # Mock sanitization
        mock_sanitize.side_effect = lambda x: x or ""
        
        # Mock repository and issue creation
        mock_repo = Mock()
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = 'Test Issue'
        
        # Mock existing issues (empty)
        mock_repo.get_issues.return_value = []
        mock_repo.create_issue.return_value = mock_issue
        self.mock_github_client.get_repo.return_value = mock_repo
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        api.create_issue('test/repo', 'Test Issue', 'Test body', ['bug'])
        
        mock_repo.create_issue.assert_called_once_with(
            title='Test Issue',
            body='Test body',
            labels=['bug']
        )

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    @patch('github_api.sanitize_issue_content')
    def test_create_issue_duplicate_prevention(self, mock_sanitize, mock_validate, mock_github, mock_get_config):
        """Test duplicate issue prevention"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        # Mock sanitization
        mock_sanitize.side_effect = lambda x: x or ""
        
        # Mock repository
        mock_repo = Mock()
        mock_existing_issue = Mock()
        mock_existing_issue.title = 'Test Issue'
        mock_existing_issue.number = 456
        
        # Mock existing issues with duplicate title
        mock_repo.get_issues.return_value = [mock_existing_issue]
        self.mock_github_client.get_repo.return_value = mock_repo
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        api.create_issue('test/repo', 'Test Issue', 'Test body', ['bug'])
        
        # Should not create new issue
        mock_repo.create_issue.assert_not_called()

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    def test_get_issue_success(self, mock_validate, mock_github, mock_get_config):
        """Test successful issue retrieval"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        mock_repo = Mock()
        mock_issue = Mock()
        mock_repo.get_issue.return_value = mock_issue
        self.mock_github_client.get_repo.return_value = mock_repo
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        result = api.get_issue('test/repo', 123)
        
        assert result == mock_issue
        mock_repo.get_issue.assert_called_once_with(number=123)

    @patch('github_api.get_secure_config')
    @patch('github_api.Github')
    @patch('github_api.validate_repo_name', return_value=True)
    def test_add_comment_to_issue_success(self, mock_validate, mock_github, mock_get_config):
        """Test successful comment addition"""
        mock_get_config.return_value = self.mock_config
        mock_github.return_value = self.mock_github_client
        
        mock_repo = Mock()
        mock_issue = Mock()
        mock_repo.get_issue.return_value = mock_issue
        self.mock_github_client.get_repo.return_value = mock_repo
        
        from github_api import GitHubAPI
        api = GitHubAPI()
        api.add_comment_to_issue('test/repo', 123, 'Test comment')
        
        mock_issue.create_comment.assert_called_once_with('Test comment')