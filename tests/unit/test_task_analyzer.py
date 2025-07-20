"""
Unit tests for task_analyzer.py module
"""
import datetime
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('/root/repo/src')
from task_analyzer import find_todo_comments, analyze_open_issues


class TestTaskAnalyzer:
    """Test cases for task analyzer functions"""

    @patch('task_analyzer.get_task_tracker')
    def test_find_todo_comments_success(self, mock_get_tracker):
        """Test successful TODO comment discovery"""
        # Mock task tracker
        mock_tracker = Mock()
        mock_tracker.is_task_processed.return_value = False  # New task, not processed
        mock_tracker.mark_task_processed.return_value = "test_hash"
        mock_get_tracker.return_value = mock_tracker
        
        # Mock GitHub API and repository
        mock_github_api = Mock()
        mock_client = Mock()
        mock_github_api.client = mock_client
        
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Mock search results
        mock_search_result = Mock()
        mock_search_result.path = "src/test.py"
        mock_search_result.html_url = "https://github.com/test/repo/blob/main/src/test.py#L5"
        
        mock_client.search_code.return_value = [mock_search_result]
        
        # Mock file content
        mock_file_content = Mock()
        file_content = "def function():\n    # TODO: Implement this\n    pass\n"
        mock_file_content.decoded_content.decode.return_value = file_content
        mock_repo.get_contents.return_value = mock_file_content
        
        # Mock create_issue
        mock_github_api.create_issue = Mock()
        
        # Execute function
        find_todo_comments(mock_github_api, mock_repo, "test/manager")
        
        # Verify issue creation was called
        mock_github_api.create_issue.assert_called()
        call_args = mock_github_api.create_issue.call_args
        assert "Address TODO in src/test.py:2" in call_args[0][1]  # title
        assert "test/manager" in call_args[0][0]  # repo name
        
        # Verify task was marked as processed (should be called for each matching query)
        assert mock_tracker.mark_task_processed.call_count >= 1

    def test_find_todo_comments_no_results(self):
        """Test TODO comment discovery with no results"""
        mock_github_api = Mock()
        mock_client = Mock()
        mock_github_api.client = mock_client
        
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Mock empty search results
        mock_client.search_code.return_value = []
        mock_github_api.create_issue = Mock()
        
        # Execute function
        find_todo_comments(mock_github_api, mock_repo, "test/manager")
        
        # Verify no issues were created
        mock_github_api.create_issue.assert_not_called()

    def test_find_todo_comments_file_access_error(self):
        """Test TODO comment discovery with file access error"""
        mock_github_api = Mock()
        mock_client = Mock()
        mock_github_api.client = mock_client
        
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Mock search results
        mock_search_result = Mock()
        mock_search_result.path = "src/test.py"
        mock_client.search_code.return_value = [mock_search_result]
        
        # Mock file access error
        mock_repo.get_contents.side_effect = Exception("File not found")
        mock_github_api.create_issue = Mock()
        
        # Execute function - should handle error gracefully
        find_todo_comments(mock_github_api, mock_repo, "test/manager")
        
        # Function should not crash and not create issues
        mock_github_api.create_issue.assert_not_called()

    def test_find_todo_comments_search_error(self):
        """Test TODO comment discovery with search API error"""
        mock_github_api = Mock()
        mock_client = Mock()
        mock_github_api.client = mock_client
        
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Mock search error
        mock_client.search_code.side_effect = Exception("Search API error")
        mock_github_api.create_issue = Mock()
        
        # Execute function - should handle error gracefully
        find_todo_comments(mock_github_api, mock_repo, "test/manager")
        
        # Function should not crash
        mock_github_api.create_issue.assert_not_called()

    def test_analyze_open_issues_stale_bug(self):
        """Test analysis of stale bug issues"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Create mock stale issue
        mock_issue = Mock()
        mock_issue.pull_request = None  # Not a PR
        mock_issue.title = "Critical bug in login"
        mock_issue.number = 123
        mock_issue.body = "Users cannot log in due to authentication error"
        mock_issue.html_url = "https://github.com/test/repo/issues/123"
        mock_issue.assignees = []
        
        # Mock labels
        mock_label = Mock()
        mock_label.name = "bug"
        mock_issue.labels = [mock_label]
        
        # Mock dates (issue is 45 days old)
        now = datetime.datetime.now(datetime.timezone.utc)
        old_date = now - datetime.timedelta(days=45)
        mock_issue.created_at = old_date
        mock_issue.updated_at = old_date
        
        mock_repo.get_issues.return_value = [mock_issue]
        mock_github_api.create_issue = Mock()
        
        # Execute function
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Verify issue creation was called
        mock_github_api.create_issue.assert_called()
        call_args = mock_github_api.create_issue.call_args
        assert "Review stale issue:" in call_args[0][1]  # title
        assert "45 days" in call_args[0][2]  # body should mention days

    def test_analyze_open_issues_recent_issue(self):
        """Test analysis skips recent issues"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Create mock recent issue
        mock_issue = Mock()
        mock_issue.pull_request = None
        mock_issue.title = "Recent bug"
        
        # Mock labels
        mock_label = Mock()
        mock_label.name = "bug"
        mock_issue.labels = [mock_label]
        
        # Mock recent date (5 days old)
        now = datetime.datetime.now(datetime.timezone.utc)
        recent_date = now - datetime.timedelta(days=5)
        mock_issue.updated_at = recent_date
        
        mock_repo.get_issues.return_value = [mock_issue]
        mock_github_api.create_issue = Mock()
        
        # Execute function
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Verify no issue was created for recent issue
        mock_github_api.create_issue.assert_not_called()

    def test_analyze_open_issues_skip_pull_requests(self):
        """Test analysis skips pull requests"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Create mock PR (has pull_request attribute)
        mock_pr = Mock()
        mock_pr.pull_request = Mock()  # This indicates it's a PR
        
        mock_repo.get_issues.return_value = [mock_pr]
        mock_github_api.create_issue = Mock()
        
        # Execute function
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Verify no issue was created for PR
        mock_github_api.create_issue.assert_not_called()

    def test_analyze_open_issues_irrelevant_labels(self):
        """Test analysis skips issues without relevant labels"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Create mock issue with irrelevant labels
        mock_issue = Mock()
        mock_issue.pull_request = None
        mock_issue.title = "Documentation update"
        
        # Mock irrelevant labels
        mock_label = Mock()
        mock_label.name = "documentation"
        mock_issue.labels = [mock_label]
        
        # Mock old date
        now = datetime.datetime.now(datetime.timezone.utc)
        old_date = now - datetime.timedelta(days=45)
        mock_issue.updated_at = old_date
        
        mock_repo.get_issues.return_value = [mock_issue]
        mock_github_api.create_issue = Mock()
        
        # Execute function
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Verify no issue was created for irrelevant labels
        mock_github_api.create_issue.assert_not_called()

    def test_analyze_open_issues_help_wanted_label(self):
        """Test analysis processes 'help wanted' labels"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Create mock issue with help wanted label
        mock_issue = Mock()
        mock_issue.pull_request = None
        mock_issue.title = "Need help with feature"
        mock_issue.number = 456
        mock_issue.body = "Could use assistance implementing this feature"
        mock_issue.html_url = "https://github.com/test/repo/issues/456"
        mock_issue.assignees = []
        
        # Mock help wanted label
        mock_label = Mock()
        mock_label.name = "help wanted"
        mock_issue.labels = [mock_label]
        
        # Mock old date
        now = datetime.datetime.now(datetime.timezone.utc)
        old_date = now - datetime.timedelta(days=35)
        mock_issue.created_at = old_date
        mock_issue.updated_at = old_date
        
        mock_repo.get_issues.return_value = [mock_issue]
        mock_github_api.create_issue = Mock()
        
        # Execute function
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Verify issue creation was called
        mock_github_api.create_issue.assert_called()

    def test_analyze_open_issues_with_assignees(self):
        """Test analysis includes assignee information"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Create mock issue with assignees
        mock_issue = Mock()
        mock_issue.pull_request = None
        mock_issue.title = "Assigned bug"
        mock_issue.number = 789
        mock_issue.body = "This bug has assignees"
        mock_issue.html_url = "https://github.com/test/repo/issues/789"
        
        # Mock assignees
        mock_assignee = Mock()
        mock_assignee.login = "developer1"
        mock_issue.assignees = [mock_assignee]
        
        # Mock bug label
        mock_label = Mock()
        mock_label.name = "bug"
        mock_issue.labels = [mock_label]
        
        # Mock old date
        now = datetime.datetime.now(datetime.timezone.utc)
        old_date = now - datetime.timedelta(days=40)
        mock_issue.created_at = old_date
        mock_issue.updated_at = old_date
        
        mock_repo.get_issues.return_value = [mock_issue]
        mock_github_api.create_issue = Mock()
        
        # Execute function
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Verify issue creation was called with assignee info
        mock_github_api.create_issue.assert_called()
        call_args = mock_github_api.create_issue.call_args
        assert "developer1" in call_args[0][2]  # body should mention assignee

    def test_analyze_open_issues_exception_handling(self):
        """Test analysis handles exceptions gracefully"""
        mock_github_api = Mock()
        mock_repo = Mock()
        mock_repo.full_name = "test/repo"
        
        # Mock exception during get_issues
        mock_repo.get_issues.side_effect = Exception("API error")
        mock_github_api.create_issue = Mock()
        
        # Execute function - should not crash
        analyze_open_issues(mock_github_api, mock_repo, "test/manager")
        
        # Function should handle error gracefully
        mock_github_api.create_issue.assert_not_called()