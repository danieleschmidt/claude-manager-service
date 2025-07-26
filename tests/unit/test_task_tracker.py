"""
Comprehensive unit tests for task_tracker module

This module provides complete test coverage for the TaskTracker class,
including edge cases, error handling, and all public methods.
"""
import pytest
import json
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, call
from typing import Dict, Any

import sys
sys.path.append('/root/repo/src')

from task_tracker import TaskTracker, generate_task_hash, get_task_tracker


class TestGenerateTaskHash:
    """Test cases for the generate_task_hash function"""
    
    def test_hash_consistency(self):
        """Test that same inputs produce same hash"""
        hash1 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        hash2 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        assert hash1 == hash2
    
    def test_hash_uniqueness(self):
        """Test that different inputs produce different hashes"""
        base_hash = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        
        # Different repo
        diff_repo = generate_task_hash("other/repo", "src/file.py", 42, "TODO: Fix this")
        assert base_hash != diff_repo
        
        # Different file
        diff_file = generate_task_hash("test/repo", "src/other.py", 42, "TODO: Fix this")
        assert base_hash != diff_file
        
        # Different line
        diff_line = generate_task_hash("test/repo", "src/file.py", 43, "TODO: Fix this")
        assert base_hash != diff_line
        
        # Different content
        diff_content = generate_task_hash("test/repo", "src/file.py", 42, "FIXME: Fix this")
        assert base_hash != diff_content
    
    def test_hash_format(self):
        """Test hash format and length"""
        task_hash = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        
        # Should be string
        assert isinstance(task_hash, str)
        
        # Should be 16 characters (first 16 of SHA-256 hex)
        assert len(task_hash) == 16
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in task_hash.lower())
    
    def test_hash_with_unicode_content(self):
        """Test hash generation with unicode content"""
        unicode_content = "TODO: Fix this 测试 🚀"
        task_hash = generate_task_hash("test/repo", "src/file.py", 42, unicode_content)
        
        assert isinstance(task_hash, str)
        assert len(task_hash) == 16
    
    def test_hash_with_whitespace_content(self):
        """Test hash generation strips whitespace consistently"""
        content1 = "TODO: Fix this"
        content2 = "  TODO: Fix this  "
        content3 = "\tTODO: Fix this\n"
        
        hash1 = generate_task_hash("test/repo", "src/file.py", 42, content1)
        hash2 = generate_task_hash("test/repo", "src/file.py", 42, content2)
        hash3 = generate_task_hash("test/repo", "src/file.py", 42, content3)
        
        # All should be the same due to strip()
        assert hash1 == hash2 == hash3


class TestTaskTrackerInitialization:
    """Test cases for TaskTracker initialization"""
    
    def test_default_initialization(self):
        """Test TaskTracker initialization with default parameters"""
        with patch('task_tracker.Path.mkdir') as mock_mkdir:
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                tracker = TaskTracker()
                
                # Should create directory
                mock_mkdir.assert_called_once()
                
                # Should have correct file path
                assert tracker.tracker_file.name == 'task_tracker.json'
                assert '.claude_manager' in str(tracker.tracker_file)
    
    def test_custom_directory_initialization(self):
        """Test TaskTracker initialization with custom directory"""
        custom_dir = Path("/tmp/test_tracker")
        
        with patch('task_tracker.Path.mkdir') as mock_mkdir:
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                tracker = TaskTracker(tracker_dir=custom_dir)
                
                # Should use custom directory
                assert tracker.tracker_dir == custom_dir
                assert tracker.tracker_file == custom_dir / 'task_tracker.json'
                mock_mkdir.assert_called_once()
    
    def test_initialization_with_existing_data(self):
        """Test initialization when tracker file already exists"""
        existing_data = {
            "hash123": {
                "repo": "test/repo",
                "file": "src/test.py",
                "line": 10,
                "content": "TODO: Test",
                "created_at": "2025-07-20T10:00:00+00:00",
                "issue_number": 42
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                assert len(tracker._task_data) == 1
                assert "hash123" in tracker._task_data


class TestTaskTrackerDataOperations:
    """Test cases for TaskTracker data loading and saving"""
    
    def test_load_tracker_data_nonexistent_file(self):
        """Test loading when tracker file doesn't exist"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(Path, 'exists', return_value=False):
                tracker = TaskTracker()
                data = tracker._load_tracker_data()
                
                assert data == {}
    
    def test_load_tracker_data_valid_file(self):
        """Test loading valid tracker file"""
        test_data = {
            "hash123": {
                "repo": "test/repo",
                "file": "src/test.py",
                "line": 10,
                "content": "TODO: Test",
                "created_at": "2025-07-20T10:00:00+00:00",
                "issue_number": 42
            }
        }
        
        mock_file_content = json.dumps(test_data)
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(Path, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=mock_file_content)):
                    tracker = TaskTracker()
                    data = tracker._load_tracker_data()
                    
                    assert data == test_data
    
    def test_load_tracker_data_corrupted_file(self):
        """Test loading corrupted JSON file"""
        corrupted_content = "invalid json content"
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(Path, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=corrupted_content)):
                    with patch.object(Path, 'rename') as mock_rename:
                        tracker = TaskTracker()
                        data = tracker._load_tracker_data()
                        
                        # Should return empty dict and backup corrupted file
                        assert data == {}
                        # rename should be called at least once (possibly twice - during init and explicit call)
                        assert mock_rename.call_count >= 1
    
    def test_load_tracker_data_file_error(self):
        """Test loading when file read raises exception"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(Path, 'exists', return_value=True):
                with patch('builtins.open', side_effect=OSError("File error")):
                    tracker = TaskTracker()
                    data = tracker._load_tracker_data()
                    
                    assert data == {}
    
    def test_save_tracker_data_success(self):
        """Test successful saving of tracker data"""
        test_data = {"test": "data"}
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch.object(Path, 'rename') as mock_rename:
                        tracker = TaskTracker()
                        tracker._save_tracker_data(test_data)
                        
                        # Verify file operations
                        mock_file.assert_called()
                        mock_rename.assert_called_once()
    
    def test_save_tracker_data_error(self):
        """Test error handling during save"""
        test_data = {"test": "data"}
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch('builtins.open', side_effect=OSError("Write error")):
                    with patch.object(Path, 'exists', return_value=True):
                        with patch.object(Path, 'unlink') as mock_unlink:
                            tracker = TaskTracker()
                            tracker._save_tracker_data(test_data)
                            
                            # Should attempt cleanup
                            mock_unlink.assert_called_once()


class TestTaskTrackerProcessing:
    """Test cases for task processing operations"""
    
    def setup_method(self):
        """Setup for each test method (pytest style)"""
        self.test_repo = "test/repo"
        self.test_file = "src/test.py"
        self.test_line = 42
        self.test_content = "TODO: Fix this"
    
    def test_is_task_processed_new_task(self):
        """Test checking if new task is processed"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                tracker = TaskTracker()
                
                result = tracker.is_task_processed(
                    self.test_repo, self.test_file, self.test_line, self.test_content
                )
                
                assert result is False
    
    def test_is_task_processed_existing_task(self):
        """Test checking if existing task is processed"""
        task_hash = generate_task_hash(self.test_repo, self.test_file, self.test_line, self.test_content)
        existing_data = {
            task_hash: {
                "repo": self.test_repo,
                "file": self.test_file,
                "line": self.test_line,
                "content": self.test_content,
                "created_at": "2025-07-20T10:00:00+00:00",
                "issue_number": 123
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                result = tracker.is_task_processed(
                    self.test_repo, self.test_file, self.test_line, self.test_content
                )
                
                assert result is True
    
    def test_mark_task_processed_without_issue(self):
        """Test marking task as processed without issue number"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    
                    task_hash = tracker.mark_task_processed(
                        self.test_repo, self.test_file, self.test_line, self.test_content
                    )
                    
                    # Verify task was added
                    assert task_hash in tracker._task_data
                    task_entry = tracker._task_data[task_hash]
                    
                    assert task_entry["repo"] == self.test_repo
                    assert task_entry["file"] == self.test_file
                    assert task_entry["line"] == self.test_line
                    assert task_entry["content"] == self.test_content.strip()
                    assert task_entry["issue_number"] is None
                    assert "created_at" in task_entry
                    
                    # Verify save was called
                    mock_save.assert_called_once()
    
    def test_mark_task_processed_with_issue(self):
        """Test marking task as processed with issue number"""
        issue_number = 456
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    
                    task_hash = tracker.mark_task_processed(
                        self.test_repo, self.test_file, self.test_line, self.test_content, issue_number
                    )
                    
                    task_entry = tracker._task_data[task_hash]
                    assert task_entry["issue_number"] == issue_number
                    mock_save.assert_called_once()
    
    def test_mark_task_processed_strips_whitespace(self):
        """Test that content whitespace is stripped when marking processed"""
        content_with_whitespace = "  TODO: Fix this  \n"
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch.object(TaskTracker, '_save_tracker_data'):
                    tracker = TaskTracker()
                    
                    task_hash = tracker.mark_task_processed(
                        self.test_repo, self.test_file, self.test_line, content_with_whitespace
                    )
                    
                    task_entry = tracker._task_data[task_hash]
                    assert task_entry["content"] == "TODO: Fix this"


class TestTaskTrackerCleanup:
    """Test cases for task cleanup operations"""
    
    def test_cleanup_old_tasks_removes_old(self):
        """Test that old tasks are removed"""
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        
        existing_data = {
            "old_task": {
                "repo": "test/repo",
                "file": "old.py",
                "line": 1,
                "content": "TODO: Old",
                "created_at": old_date,
                "issue_number": 100
            },
            "recent_task": {
                "repo": "test/repo",
                "file": "recent.py",
                "line": 2,
                "content": "TODO: Recent",
                "created_at": recent_date,
                "issue_number": 200
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    
                    removed_count = tracker.cleanup_old_tasks(days=30)
                    
                    assert removed_count == 1
                    assert "old_task" not in tracker._task_data
                    assert "recent_task" in tracker._task_data
                    mock_save.assert_called_once()
    
    def test_cleanup_old_tasks_no_old_tasks(self):
        """Test cleanup when no old tasks exist"""
        recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        
        existing_data = {
            "recent_task": {
                "repo": "test/repo",
                "file": "recent.py",
                "line": 1,
                "content": "TODO: Recent",
                "created_at": recent_date,
                "issue_number": 100
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    
                    removed_count = tracker.cleanup_old_tasks(days=30)
                    
                    assert removed_count == 0
                    assert len(tracker._task_data) == 1
                    # Should not save if no changes
                    mock_save.assert_not_called()
    
    def test_cleanup_old_tasks_invalid_dates(self):
        """Test cleanup handles invalid date formats"""
        existing_data = {
            "invalid_date_task": {
                "repo": "test/repo",
                "file": "test.py",
                "line": 1,
                "content": "TODO: Test",
                "created_at": "invalid-date-format",
                "issue_number": 100
            },
            "missing_date_task": {
                "repo": "test/repo",
                "file": "test2.py",
                "line": 2,
                "content": "TODO: Test2",
                "issue_number": 200
                # No created_at field
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    
                    removed_count = tracker.cleanup_old_tasks(days=30)
                    
                    # Both invalid tasks should be removed
                    assert removed_count == 2
                    assert len(tracker._task_data) == 0
                    mock_save.assert_called_once()
    
    def test_cleanup_old_tasks_custom_days(self):
        """Test cleanup with custom days parameter"""
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        
        existing_data = {
            "old_task": {
                "repo": "test/repo",
                "file": "test.py",
                "line": 1,
                "content": "TODO: Test",
                "created_at": old_date,
                "issue_number": 100
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                with patch.object(TaskTracker, '_save_tracker_data') as mock_save:
                    tracker = TaskTracker()
                    
                    # With 120 days, task should not be removed
                    removed_count = tracker.cleanup_old_tasks(days=120)
                    assert removed_count == 0
                    
                    # With 50 days, task should be removed
                    removed_count = tracker.cleanup_old_tasks(days=50)
                    assert removed_count == 1


class TestTaskTrackerStatistics:
    """Test cases for task statistics"""
    
    def test_get_task_statistics_empty(self):
        """Test statistics with no tasks"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                tracker = TaskTracker()
                
                stats = tracker.get_task_statistics()
                
                assert stats["total_tasks"] == 0
                assert stats["repositories"] == 0
                assert stats["tasks_by_repo"] == {}
                assert "tracker_file" in stats
    
    def test_get_task_statistics_multiple_repos(self):
        """Test statistics with multiple repositories"""
        existing_data = {
            "task1": {"repo": "test/repo1"},
            "task2": {"repo": "test/repo1"},
            "task3": {"repo": "test/repo2"},
            "task4": {"repo": "test/repo2"},
            "task5": {"repo": "test/repo2"},
            "task6": {"repo": "test/repo3"}
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                stats = tracker.get_task_statistics()
                
                assert stats["total_tasks"] == 6
                assert stats["repositories"] == 3
                assert stats["tasks_by_repo"]["test/repo1"] == 2
                assert stats["tasks_by_repo"]["test/repo2"] == 3
                assert stats["tasks_by_repo"]["test/repo3"] == 1
    
    def test_get_task_statistics_missing_repo(self):
        """Test statistics with missing repo field"""
        existing_data = {
            "task1": {"repo": "test/repo1"},
            "task2": {},  # Missing repo field
            "task3": {"repo": "test/repo1"}
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                stats = tracker.get_task_statistics()
                
                assert stats["total_tasks"] == 3
                assert stats["repositories"] == 2  # "test/repo1" and "unknown"
                assert stats["tasks_by_repo"]["test/repo1"] == 2
                assert stats["tasks_by_repo"]["unknown"] == 1


class TestTaskTrackerRepoSpecificOperations:
    """Test cases for repository-specific operations"""
    
    def test_get_processed_tasks_for_repo_empty(self):
        """Test getting tasks for repo when none exist"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                tracker = TaskTracker()
                
                repo_tasks = tracker.get_processed_tasks_for_repo("test/repo")
                
                assert repo_tasks == {}
    
    def test_get_processed_tasks_for_repo_with_tasks(self):
        """Test getting tasks for specific repository"""
        existing_data = {
            "task1": {"repo": "test/repo1", "file": "file1.py"},
            "task2": {"repo": "test/repo2", "file": "file2.py"},
            "task3": {"repo": "test/repo1", "file": "file3.py"},
            "task4": {"repo": "test/repo3", "file": "file4.py"}
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                repo1_tasks = tracker.get_processed_tasks_for_repo("test/repo1")
                
                assert len(repo1_tasks) == 2
                assert "task1" in repo1_tasks
                assert "task3" in repo1_tasks
                assert repo1_tasks["task1"]["file"] == "file1.py"
                assert repo1_tasks["task3"]["file"] == "file3.py"
    
    def test_get_processed_tasks_for_repo_nonexistent(self):
        """Test getting tasks for repository that doesn't exist"""
        existing_data = {
            "task1": {"repo": "test/repo1", "file": "file1.py"}
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                repo_tasks = tracker.get_processed_tasks_for_repo("nonexistent/repo")
                
                assert repo_tasks == {}


class TestGlobalTaskTracker:
    """Test cases for global task tracker functionality"""
    
    def test_get_task_tracker_singleton(self):
        """Test that global task tracker returns same instance"""
        with patch.object(TaskTracker, '__init__', return_value=None) as mock_init:
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                # Reset global instance
                import task_tracker
                task_tracker._task_tracker = None
                
                tracker1 = get_task_tracker()
                tracker2 = get_task_tracker()
                
                # Should be same instance
                assert tracker1 is tracker2
                
                # Should only initialize once
                mock_init.assert_called_once()
    
    def test_get_task_tracker_reuses_existing(self):
        """Test that global task tracker reuses existing instance"""
        existing_tracker = Mock(spec=TaskTracker)
        
        import task_tracker
        task_tracker._task_tracker = existing_tracker
        
        tracker = get_task_tracker()
        
        assert tracker is existing_tracker


class TestTaskTrackerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_task_processing_with_unicode(self):
        """Test task processing with unicode characters"""
        unicode_repo = "测试/repo"
        unicode_file = "src/测试.py"
        unicode_content = "TODO: 修复这个问题 🚀"
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch.object(TaskTracker, '_save_tracker_data'):
                    tracker = TaskTracker()
                    
                    # Should handle unicode without errors
                    task_hash = tracker.mark_task_processed(
                        unicode_repo, unicode_file, 42, unicode_content, 123
                    )
                    
                    assert task_hash in tracker._task_data
                    task_entry = tracker._task_data[task_hash]
                    assert task_entry["repo"] == unicode_repo
                    assert task_entry["file"] == unicode_file
                    assert task_entry["content"] == unicode_content.strip()
    
    def test_task_processing_with_empty_content(self):
        """Test task processing with empty content"""
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch.object(TaskTracker, '_save_tracker_data'):
                    tracker = TaskTracker()
                    
                    # Should handle empty content
                    task_hash = tracker.mark_task_processed(
                        "test/repo", "src/file.py", 42, "   ", 123
                    )
                    
                    task_entry = tracker._task_data[task_hash]
                    assert task_entry["content"] == ""
    
    def test_task_processing_with_very_long_content(self):
        """Test task processing with very long content"""
        long_content = "TODO: " + "x" * 10000  # Very long content
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value={}):
                with patch.object(TaskTracker, '_save_tracker_data'):
                    tracker = TaskTracker()
                    
                    # Should handle long content without errors
                    task_hash = tracker.mark_task_processed(
                        "test/repo", "src/file.py", 42, long_content, 123
                    )
                    
                    assert task_hash in tracker._task_data
                    task_entry = tracker._task_data[task_hash]
                    assert task_entry["content"] == long_content.strip()
    
    def test_cleanup_with_timezone_aware_dates(self):
        """Test cleanup with timezone-aware dates"""
        # Date with different timezone
        tz_date = datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat()
        
        existing_data = {
            "tz_task": {
                "repo": "test/repo",
                "file": "test.py",
                "line": 1,
                "content": "TODO: Test",
                "created_at": tz_date,
                "issue_number": 100
            }
        }
        
        with patch('task_tracker.Path.mkdir'):
            with patch.object(TaskTracker, '_load_tracker_data', return_value=existing_data):
                tracker = TaskTracker()
                
                # Should handle timezone-aware dates correctly
                removed_count = tracker.cleanup_old_tasks(days=1)
                assert removed_count == 0  # Recent date should not be removed


if __name__ == "__main__":
    pytest.main([__file__])