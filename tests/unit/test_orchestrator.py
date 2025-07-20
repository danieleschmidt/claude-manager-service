"""
Unit tests for orchestrator.py module (with security mocking)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
sys.path.append('/root/repo/src')


class TestOrchestrator:
    """Test cases for orchestrator functions with proper security mocking"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create reusable mocks
        self.mock_config = Mock()
        self.mock_config.get_github_token.return_value = 'ghp_' + 'x' * 36
        
        self.mock_secure_subprocess = Mock()

    @patch('orchestrator.build_prompt')
    @patch('orchestrator.get_template_for_labels')
    def test_trigger_terragon_task_success(self, mock_get_template, mock_build_prompt):
        """Test successful Terragon task triggering"""
        from orchestrator import trigger_terragon_task
        
        # Mock dependencies
        mock_api = Mock()
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = "Test Issue"
        mock_issue.body = "Test description"
        mock_issue.html_url = "https://github.com/test/repo/issues/123"
        
        # Mock labels
        mock_label = Mock()
        mock_label.name = "bug"
        mock_issue.labels = [mock_label]
        
        mock_get_template.return_value = "prompts/fix_issue.txt"
        mock_build_prompt.return_value = "Built prompt content"
        
        config = {
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        # Execute function
        trigger_terragon_task(mock_api, "test/repo", mock_issue, config)
        
        # Verify API calls
        mock_get_template.assert_called_once_with(["bug"])
        mock_build_prompt.assert_called_once()
        mock_api.add_comment_to_issue.assert_called_once()
        
        # Verify comment content
        call_args = mock_api.add_comment_to_issue.call_args
        comment = call_args[0][2]  # Third argument is the comment body
        assert "@terragon-labs" in comment
        assert "Built prompt content" in comment

    @patch('orchestrator.get_secure_subprocess')
    @patch('orchestrator.SecureTempDir')
    @patch('orchestrator.validate_repo_name', return_value=True)
    def test_trigger_claude_flow_task_success(self, mock_validate, mock_temp_dir, mock_get_subprocess):
        """Test successful Claude Flow task triggering"""
        from orchestrator import trigger_claude_flow_task
        
        # Mock dependencies
        mock_api = Mock()
        mock_api.token = "test_token"
        
        mock_issue = Mock()
        mock_issue.number = 456
        mock_issue.title = "Implement feature"
        mock_issue.body = "**Repository:** test/target-repo\nPlease implement this feature"
        
        # Mock secure subprocess
        mock_subprocess = Mock()
        mock_get_subprocess.return_value = mock_subprocess
        
        # Mock successful subprocess calls
        clone_result = Mock()
        clone_result.returncode = 0
        
        claude_result = Mock()
        claude_result.returncode = 0
        claude_result.stdout = "Claude Flow completed successfully"
        claude_result.stderr = ""
        
        mock_subprocess.run_git_clone.return_value = clone_result
        mock_subprocess.run_with_sanitized_logging.return_value = claude_result
        
        # Mock temporary directory
        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = lambda self, x: Path("/tmp/test_dir/cloned_repo")
        mock_temp_dir.return_value.__enter__.return_value = mock_temp_path
        mock_temp_dir.return_value.__exit__.return_value = None
        
        # Execute function
        trigger_claude_flow_task(mock_api, "test/manager", mock_issue)
        
        # Verify subprocess calls
        mock_subprocess.run_git_clone.assert_called_once()
        mock_subprocess.run_with_sanitized_logging.assert_called_once()
        
        # Verify success comment
        mock_api.add_comment_to_issue.assert_called()
        comment_call = mock_api.add_comment_to_issue.call_args
        assert "✅ Claude Flow task completed successfully" in comment_call[0][2]

    @patch('orchestrator.get_secure_subprocess')
    @patch('orchestrator.SecureTempDir')
    @patch('orchestrator.validate_repo_name', return_value=True)
    def test_trigger_claude_flow_task_clone_failure(self, mock_validate, mock_temp_dir, mock_get_subprocess):
        """Test Claude Flow task with clone failure"""
        from orchestrator import trigger_claude_flow_task
        
        # Mock dependencies
        mock_api = Mock()
        mock_api.token = "test_token"
        
        mock_issue = Mock()
        mock_issue.number = 456
        mock_issue.title = "Implement feature"
        mock_issue.body = "**Repository:** test/target-repo\nDescription"
        
        # Mock secure subprocess
        mock_subprocess = Mock()
        mock_get_subprocess.return_value = mock_subprocess
        
        # Mock failed clone
        clone_result = Mock()
        clone_result.returncode = 1
        clone_result.stderr = "Repository not found"
        
        mock_subprocess.run_git_clone.return_value = clone_result
        
        # Mock temporary directory
        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = lambda self, x: Path("/tmp/test_dir/cloned_repo")
        mock_temp_dir.return_value.__enter__.return_value = mock_temp_path
        mock_temp_dir.return_value.__exit__.return_value = None
        
        # Execute function
        trigger_claude_flow_task(mock_api, "test/manager", mock_issue)
        
        # Verify only git clone was called
        mock_subprocess.run_git_clone.assert_called_once()
        mock_subprocess.run_with_sanitized_logging.assert_not_called()

    @patch('orchestrator.get_secure_subprocess')
    @patch('orchestrator.SecureTempDir')
    @patch('orchestrator.validate_repo_name', return_value=True)
    def test_trigger_claude_flow_task_execution_failure(self, mock_validate, mock_temp_dir, mock_get_subprocess):
        """Test Claude Flow task with execution failure"""
        from orchestrator import trigger_claude_flow_task
        
        # Mock dependencies
        mock_api = Mock()
        mock_api.token = "test_token"
        
        mock_issue = Mock()
        mock_issue.number = 456
        mock_issue.title = "Implement feature"
        mock_issue.body = "Description without repo"
        
        # Mock secure subprocess
        mock_subprocess = Mock()
        mock_get_subprocess.return_value = mock_subprocess
        
        # Mock successful clone, failed Claude Flow
        clone_result = Mock()
        clone_result.returncode = 0
        
        claude_result = Mock()
        claude_result.returncode = 1
        claude_result.stdout = "Partial output"
        claude_result.stderr = "Claude Flow error"
        
        mock_subprocess.run_git_clone.return_value = clone_result
        mock_subprocess.run_with_sanitized_logging.return_value = claude_result
        
        # Mock temporary directory
        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = lambda self, x: Path("/tmp/test_dir/cloned_repo")
        mock_temp_dir.return_value.__enter__.return_value = mock_temp_path
        mock_temp_dir.return_value.__exit__.return_value = None
        
        # Execute function
        trigger_claude_flow_task(mock_api, "test/manager", mock_issue)
        
        # Verify both subprocess calls
        mock_subprocess.run_git_clone.assert_called_once()
        mock_subprocess.run_with_sanitized_logging.assert_called_once()
        
        # Verify error comment
        mock_api.add_comment_to_issue.assert_called()
        comment_call = mock_api.add_comment_to_issue.call_args
        assert "❌ Claude Flow task failed" in comment_call[0][2]
        assert "Claude Flow error" in comment_call[0][2]