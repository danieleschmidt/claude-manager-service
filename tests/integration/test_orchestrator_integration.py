"""
Integration tests for orchestrator task execution workflows
"""
import pytest
import tempfile
import json
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timezone

sys.path.append('/root/repo/src')

from orchestrator import trigger_terragon_task, trigger_claude_flow_task
from github_api import GitHubAPI


class TestOrchestratorIntegration:
    """Integration tests for orchestrator task execution workflows"""
    
    @pytest.fixture
    def mock_github_api(self):
        """Create a mock GitHub API with realistic behavior"""
        api = Mock(spec=GitHubAPI)
        api.token = "test_token"
        return api
    
    @pytest.fixture
    def mock_issue(self):
        """Create a mock GitHub issue"""
        issue = Mock()
        issue.number = 123
        issue.title = "Fix authentication vulnerability"
        issue.body = "Critical security issue in user authentication system"
        issue.html_url = "https://github.com/test/repo/issues/123"
        
        # Mock labels
        label1 = Mock()
        label1.name = "bug"
        label2 = Mock()
        label2.name = "security"
        issue.labels = [label1, label2]
        
        return issue
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return {
            "github": {
                "username": "testuser",
                "managerRepo": "testuser/claude-manager-service",
                "reposToScan": ["testuser/test-repo"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
    
    def test_terragon_task_trigger_end_to_end(self, mock_github_api, mock_issue, test_config):
        """Test complete Terragon task triggering workflow"""
        # Track calls to add_comment_to_issue
        comment_calls = []
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        
        # Execute the function
        trigger_terragon_task(
            mock_github_api, 
            "testuser/test-repo", 
            mock_issue, 
            test_config
        )
        
        # Verify the complete workflow
        assert len(comment_calls) == 1
        call = comment_calls[0]
        
        # Verify comment details
        assert call['repo_name'] == "testuser/test-repo"
        assert call['issue_number'] == 123
        assert "@terragon-labs" in call['comment_body']
        assert "Fix authentication vulnerability" in call['comment_body']
        assert "Critical security issue" in call['comment_body']
        assert "automatically generated" in call['comment_body']
        
        # Verify GitHub API interaction
        mock_github_api.add_comment_to_issue.assert_called_once()
    
    def test_claude_flow_task_execution_end_to_end(self, mock_github_api, mock_issue):
        """Test complete Claude Flow task execution workflow"""
        # Track calls to add_comment_to_issue
        comment_calls = []
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        
        # Set required environment variables
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            # Mock successful git clone
            with patch('subprocess.run') as mock_subprocess:
                # Mock successful git clone
                clone_result = Mock()
                clone_result.returncode = 0
                clone_result.stderr = ""
                
                # Mock successful Claude Flow execution
                flow_result = Mock()
                flow_result.returncode = 0
                flow_result.stdout = "Task completed successfully"
                flow_result.stderr = ""
                
                # Configure subprocess.run to return different results based on command
                def subprocess_side_effect(*args, **kwargs):
                    if 'git' in args[0]:
                        return clone_result
                    elif 'claude-flow' in args[0] or 'npx' in args[0]:
                        return flow_result
                    return Mock(returncode=1)
                
                mock_subprocess.side_effect = subprocess_side_effect
                
                # Execute the function
                trigger_claude_flow_task(
                    mock_github_api,
                    "testuser/test-repo",
                    mock_issue
                )
        
        # Verify the complete workflow
        assert len(comment_calls) == 1
        call = comment_calls[0]
        
        # Verify success comment details
        assert call['repo_name'] == "testuser/test-repo"
        assert call['issue_number'] == 123
        assert "✅ Claude Flow task completed successfully" in call['comment_body']
        assert "Task completed successfully" in call['comment_body']
        assert "automatically executed" in call['comment_body']
        
        # Verify subprocess calls
        assert mock_subprocess.call_count == 2  # git clone + claude-flow
        
        # Verify git clone was called
        clone_call = mock_subprocess.call_args_list[0]
        assert 'git' in clone_call[0][0]
        assert 'clone' in clone_call[0][0]
        
        # Verify claude-flow was called
        flow_call = mock_subprocess.call_args_list[1]
        assert any('claude-flow' in str(arg) or 'npx' in str(arg) for arg in flow_call[0][0])
    
    def test_claude_flow_task_failure_handling(self, mock_github_api, mock_issue):
        """Test Claude Flow task failure handling"""
        # Track calls to add_comment_to_issue
        comment_calls = []
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        
        # Set required environment variables  
        test_token = 'ghp_' + 'a' * 36
        
        # Mock failed git clone
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            with patch('subprocess.run') as mock_subprocess:
                clone_result = Mock()
                clone_result.returncode = 1
                clone_result.stderr = "Repository not found"
                
                mock_subprocess.return_value = clone_result
                
                # Execute the function
                trigger_claude_flow_task(
                    mock_github_api,
                    "testuser/test-repo",
                    mock_issue
                )
        
        # Should handle the error gracefully without raising exceptions
        # The function should complete execution
        mock_subprocess.assert_called_once()
    
    def test_claude_flow_timeout_handling(self, mock_github_api, mock_issue):
        """Test Claude Flow task timeout handling"""
        # Track calls to add_comment_to_issue
        comment_calls = []
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        
        # Set required environment variables
        test_token = 'ghp_' + 'a' * 36
        
        # Mock timeout during execution
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            with patch('subprocess.run') as mock_subprocess:
                # Mock successful git clone
                clone_result = Mock()
                clone_result.returncode = 0
                clone_result.stderr = ""
                
                # Mock timeout on Claude Flow execution
                def subprocess_side_effect(*args, **kwargs):
                    if 'git' in args[0]:
                        return clone_result
                    elif 'claude-flow' in args[0] or 'npx' in args[0]:
                        raise subprocess.TimeoutExpired(args[0], timeout=300)
                    return Mock(returncode=1)
                
                mock_subprocess.side_effect = subprocess_side_effect
                
                # Execute the function
                trigger_claude_flow_task(
                    mock_github_api,
                    "testuser/test-repo",
                    mock_issue
                )
        
        # Verify timeout comment was posted
        assert len(comment_calls) == 1
        call = comment_calls[0]
        
        assert call['repo_name'] == "testuser/test-repo"
        assert call['issue_number'] == 123
        assert "⏰ Claude Flow task timed out" in call['comment_body']
        assert "5 minutes" in call['comment_body']
    
    def test_orchestrator_label_based_routing(self, mock_github_api, test_config):
        """Test that orchestrator correctly routes tasks based on labels"""
        # Create issues with different labels
        terragon_issue = Mock()
        terragon_issue.number = 1
        terragon_issue.title = "Terragon Task"
        terragon_issue.body = "Task for Terragon"
        terragon_issue.html_url = "https://github.com/test/repo/issues/1"
        
        terragon_label = Mock()
        terragon_label.name = "terragon-task"
        terragon_issue.labels = [terragon_label]
        
        claude_flow_issue = Mock()
        claude_flow_issue.number = 2
        claude_flow_issue.title = "Claude Flow Task"
        claude_flow_issue.body = "Task for Claude Flow"
        claude_flow_issue.html_url = "https://github.com/test/repo/issues/2"
        
        claude_flow_label = Mock()
        claude_flow_label.name = "claude-flow-task"
        claude_flow_issue.labels = [claude_flow_label]
        
        # Track function calls
        comment_calls = []
        subprocess_calls = []
        
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body,
                'task_type': 'terragon' if '@terragon-labs' in comment_body else 'unknown'
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        
        # Test Terragon task routing
        trigger_terragon_task(
            mock_github_api,
            "testuser/test-repo",
            terragon_issue,
            test_config
        )
        
        # Test Claude Flow task routing
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            with patch('subprocess.run') as mock_subprocess:
                def track_subprocess(*args, **kwargs):
                    subprocess_calls.append(args[0])
                    if 'git' in args[0]:
                        result = Mock()
                        result.returncode = 0
                        result.stderr = ""
                        return result
                    elif 'claude-flow' in args[0] or 'npx' in args[0]:
                        result = Mock()
                        result.returncode = 0
                        result.stdout = "Success"
                        result.stderr = ""
                        return result
                    return Mock(returncode=1)
                
                mock_subprocess.side_effect = track_subprocess
                
                trigger_claude_flow_task(
                    mock_github_api,
                    "testuser/test-repo",
                    claude_flow_issue
                )
        
        # Verify routing worked correctly
        assert len(comment_calls) == 2
        
        # Verify Terragon task
        terragon_call = next(call for call in comment_calls if call['issue_number'] == 1)
        assert terragon_call['task_type'] == 'terragon'
        assert '@terragon-labs' in terragon_call['comment_body']
        
        # Verify Claude Flow task
        claude_flow_call = next(call for call in comment_calls if call['issue_number'] == 2)
        assert '✅ Claude Flow task completed successfully' in claude_flow_call['comment_body']
        
        # Verify subprocess was called for Claude Flow
        assert len(subprocess_calls) >= 2  # git clone + claude-flow
    
    def test_prompt_template_integration(self, mock_github_api, mock_issue, test_config):
        """Test integration with prompt template system"""
        # Create issue with specific labels that should trigger template selection
        security_label = Mock()
        security_label.name = "security"
        bug_label = Mock()
        bug_label.name = "bug"
        mock_issue.labels = [security_label, bug_label]
        
        # Track calls to add_comment_to_issue
        comment_calls = []
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        
        # Mock prompt template file
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_file.read.return_value = """Task: Address GitHub Issue

**Title:** {issue_title}
**Issue Number:** #{issue_number}
**Repository:** {repository}
**Labels:** {labels}

**Description:**
{issue_body}

Please analyze and fix this issue following security best practices."""
            
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Execute the function
            trigger_terragon_task(
                mock_github_api,
                "testuser/test-repo",
                mock_issue,
                test_config
            )
        
        # Verify the complete workflow with template
        assert len(comment_calls) == 1
        call = comment_calls[0]
        
        # Verify template variables were replaced
        assert "Fix authentication vulnerability" in call['comment_body']
        assert "#123" in call['comment_body']
        assert "testuser/test-repo" in call['comment_body']
        assert "security, bug" in call['comment_body']
        assert "Critical security issue" in call['comment_body']
        assert "security best practices" in call['comment_body']
    
    def test_error_resilience_in_orchestration(self, mock_github_api, mock_issue, test_config):
        """Test error handling and resilience in orchestration workflows"""
        # Test GitHub API failure
        mock_github_api.add_comment_to_issue.side_effect = Exception("GitHub API error")
        
        # Set required environment variables for the test
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            # This should not raise an exception - errors should be handled gracefully
            try:
                trigger_terragon_task(
                    mock_github_api,
                    "testuser/test-repo",
                    mock_issue,
                    test_config
                )
            except Exception as e:
                pytest.fail(f"Function should handle errors gracefully, but raised: {e}")
        
        # Verify that the function attempted to call the API
        mock_github_api.add_comment_to_issue.assert_called_once()
    
    def test_security_token_handling_in_orchestration(self, mock_github_api, mock_issue):
        """Test that security tokens are handled properly during orchestration"""
        # Track calls to add_comment_to_issue
        comment_calls = []
        def track_add_comment(repo_name, issue_number, comment_body):
            comment_calls.append({
                'repo_name': repo_name,
                'issue_number': issue_number,
                'comment_body': comment_body
            })
        
        mock_github_api.add_comment_to_issue.side_effect = track_add_comment
        mock_github_api.token = "sensitive_token_value"
        
        # Mock subprocess to capture the command
        subprocess_calls = []
        test_token = 'ghp_' + 'a' * 36
        with patch.dict(os.environ, {'GITHUB_TOKEN': test_token}):
            with patch('subprocess.run') as mock_subprocess:
                def track_subprocess(*args, **kwargs):
                    subprocess_calls.append({
                        'command': args[0],
                        'kwargs': kwargs
                    })
                    
                    if 'git' in args[0]:
                        result = Mock()
                        result.returncode = 0
                        result.stderr = ""
                        return result
                    elif 'claude-flow' in args[0] or 'npx' in args[0]:
                        result = Mock()
                        result.returncode = 0
                        result.stdout = "Success"
                        result.stderr = ""
                        return result
                    return Mock(returncode=1)
                
                mock_subprocess.side_effect = track_subprocess
                
                # Execute the function
                trigger_claude_flow_task(
                    mock_github_api,
                    "testuser/test-repo",
                    mock_issue
                )
        
        # Verify that sensitive token is properly handled in git clone URL
        git_call = next(call for call in subprocess_calls if 'git' in call['command'])
        git_command = ' '.join(git_call['command'])
        
        # Token should be present in clone URL but properly formatted
        assert "x-access-token:" in git_command
        assert mock_github_api.token in git_command
        
        # Verify that no sensitive data appears in comments
        for call in comment_calls:
            assert "sensitive_token_value" not in call['comment_body']
            assert mock_github_api.token not in call['comment_body']