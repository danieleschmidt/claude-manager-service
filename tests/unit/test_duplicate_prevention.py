"""
Unit tests for duplicate task prevention functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import hashlib
import json
from pathlib import Path

import sys
sys.path.append('/root/repo/src')


class TestDuplicatePrevention:
    """Test cases for duplicate task prevention"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_repo = Mock()
        self.mock_repo.full_name = "test/repo"
        
    def test_generate_task_hash(self):
        """Test generating consistent hash for tasks"""
        from task_tracker import generate_task_hash
        
        # Same inputs should produce same hash
        hash1 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        hash2 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        assert hash1 == hash2
        
        # Different inputs should produce different hashes
        hash3 = generate_task_hash("test/repo", "src/file.py", 43, "TODO: Fix this")
        assert hash1 != hash3
        
        # Hash should be string and reasonable length
        assert isinstance(hash1, str)
        assert len(hash1) > 8

    def test_task_tracker_initialization(self):
        """Test TaskTracker initialization"""
        from task_tracker import TaskTracker
        
        with patch('task_tracker.Path.exists', return_value=False):
            with patch('task_tracker.Path.mkdir'):
                tracker = TaskTracker()
                assert tracker.tracker_file.name == 'task_tracker.json'

    def test_is_task_processed_new_task(self):
        """Test checking if a new task has been processed"""
        from task_tracker import TaskTracker
        
        with patch('task_tracker.Path.exists', return_value=False):
            with patch('task_tracker.Path.mkdir'):
                with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                    tracker = TaskTracker()
                    
                    result = tracker.is_task_processed("test/repo", "src/file.py", 42, "TODO: Fix this")
                    assert result is False

    def test_is_task_processed_existing_task(self):
        """Test checking if an existing task has been processed"""
        from task_tracker import TaskTracker
        
        task_hash = "abc123"
        existing_data = {
            task_hash: {
                "repo": "test/repo",
                "file": "src/file.py", 
                "line": 42,
                "content": "TODO: Fix this",
                "created_at": "2025-07-20T10:00:00Z",
                "issue_number": 123
            }
        }
        
        with patch('task_tracker.Path.exists', return_value=True):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                with patch('task_tracker.generate_task_hash', return_value=task_hash):
                    tracker = TaskTracker()
                    
                    result = tracker.is_task_processed("test/repo", "src/file.py", 42, "TODO: Fix this")
                    assert result is True

    def test_mark_task_processed(self):
        """Test marking a task as processed"""
        from task_tracker import TaskTracker
        
        with patch('task_tracker.Path.exists', return_value=False):
            with patch('task_tracker.Path.mkdir'):
                with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                    with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                        tracker = TaskTracker()
                        
                        tracker.mark_task_processed("test/repo", "src/file.py", 42, "TODO: Fix this", 123)
                        
                        # Verify save was called
                        mock_save.assert_called_once()
                        
                        # Check the data structure
                        call_args = mock_save.call_args[0][0]
                        assert len(call_args) == 1
                        
                        # Get the task entry
                        task_entry = list(call_args.values())[0]
                        assert task_entry["repo"] == "test/repo"
                        assert task_entry["file"] == "src/file.py"
                        assert task_entry["line"] == 42
                        assert task_entry["content"] == "TODO: Fix this"
                        assert task_entry["issue_number"] == 123
                        assert "created_at" in task_entry

    def test_cleanup_old_tasks(self):
        """Test cleaning up old task entries"""
        from task_tracker import TaskTracker
        
        # Create old and new task data
        old_date = "2025-06-01T10:00:00Z"  # 2 months ago
        recent_date = "2025-07-15T10:00:00Z"  # 5 days ago
        
        existing_data = {
            "old_task": {
                "repo": "test/repo",
                "file": "old_file.py",
                "line": 1,
                "content": "TODO: Old task",
                "created_at": old_date,
                "issue_number": 100
            },
            "recent_task": {
                "repo": "test/repo", 
                "file": "new_file.py",
                "line": 2,
                "content": "TODO: Recent task",
                "created_at": recent_date,
                "issue_number": 200
            }
        }
        
        with patch('task_tracker.Path.exists', return_value=True):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    tracker.cleanup_old_tasks(days=30)  # Remove tasks older than 30 days
                    
                    # Verify save was called
                    mock_save.assert_called_once()
                    
                    # Check that only recent task remains
                    saved_data = mock_save.call_args[0][0]
                    assert len(saved_data) == 1
                    assert "recent_task" in saved_data
                    assert "old_task" not in saved_data

    def test_get_task_statistics(self):
        """Test getting task statistics"""
        from task_tracker import TaskTracker
        
        existing_data = {
            "task1": {"repo": "test/repo1", "created_at": "2025-07-20T10:00:00Z"},
            "task2": {"repo": "test/repo1", "created_at": "2025-07-19T10:00:00Z"},
            "task3": {"repo": "test/repo2", "created_at": "2025-07-18T10:00:00Z"}
        }
        
        with patch('task_tracker.Path.exists', return_value=True):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                stats = tracker.get_task_statistics()
                
                assert stats["total_tasks"] == 3
                assert stats["repositories"] == 2
                assert "test/repo1" in stats["tasks_by_repo"]
                assert stats["tasks_by_repo"]["test/repo1"] == 2
                assert stats["tasks_by_repo"]["test/repo2"] == 1

    @patch('task_tracker.TaskTracker')
    def test_find_todo_comments_with_duplicate_prevention(self, mock_tracker_class):
        """Test find_todo_comments with duplicate prevention enabled"""
        from task_analyzer import find_todo_comments_with_tracking
        
        # Mock tracker instance
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Mock GitHub API
        mock_github_api = Mock()
        mock_search_result = Mock()
        mock_search_result.path = "src/test.py"
        mock_search_result.html_url = "https://github.com/test/repo/blob/main/src/test.py#L5"
        mock_github_api.client.search_code.return_value = [mock_search_result]
        
        # Mock file content
        mock_file_content = Mock()
        file_content = "def function():\n    # TODO: Implement this\n    pass\n"
        mock_file_content.decoded_content.decode.return_value = file_content
        self.mock_repo.get_contents.return_value = mock_file_content
        
        # Test case 1: New task (not processed)
        mock_tracker.is_task_processed.return_value = False
        mock_github_api.create_issue = Mock()
        
        find_todo_comments_with_tracking(mock_github_api, self.mock_repo, "test/manager")
        
        # Should create issue and mark as processed
        mock_github_api.create_issue.assert_called()
        mock_tracker.mark_task_processed.assert_called()
        
        # Reset mocks
        mock_github_api.create_issue.reset_mock()
        mock_tracker.mark_task_processed.reset_mock()
        
        # Test case 2: Already processed task
        mock_tracker.is_task_processed.return_value = True
        
        find_todo_comments_with_tracking(mock_github_api, self.mock_repo, "test/manager")
        
        # Should not create issue or mark as processed
        mock_github_api.create_issue.assert_not_called()
        mock_tracker.mark_task_processed.assert_not_called()

    def test_task_tracker_file_operations(self):
        """Test TaskTracker file read/write operations"""
        from task_tracker import TaskTracker
        
        test_data = {"test": "data"}
        
        with patch('task_tracker.Path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_data)
                
                tracker = TaskTracker()
                data = tracker._load_tracker_data()
                
                assert data == test_data

    def test_task_tracker_corrupted_file_handling(self):
        """Test handling of corrupted tracker file"""
        from task_tracker import TaskTracker
        
        with patch('task_tracker.Path.exists', return_value=True):
            with patch('task_tracker.Path.mkdir'):  # Mock directory creation
                with patch('builtins.open', create=True) as mock_open:
                    with patch.object(Path, 'rename'):  # Mock file rename operation
                        # Simulate corrupted JSON
                        mock_open.return_value.__enter__.return_value.read.return_value = "invalid json"
                        
                        tracker = TaskTracker()
                        
                        # Should have empty data after handling corruption
                        assert tracker._task_data == {}