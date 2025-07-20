"""
Task tracking module for duplicate prevention

This module provides functionality to track processed tasks and prevent
creating duplicate issues for the same TODO/FIXME comments.
"""
import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from logger import get_logger

logger = get_logger(__name__)


def generate_task_hash(repo_name: str, file_path: str, line_number: int, content: str) -> str:
    """
    Generate a unique hash for a task based on its identifying characteristics
    
    Args:
        repo_name (str): Repository name (e.g., "owner/repo")
        file_path (str): Path to the file containing the task
        line_number (int): Line number of the task
        content (str): Content of the TODO/FIXME comment
        
    Returns:
        str: Unique hash identifying this task
    """
    # Create a unique identifier from the task characteristics
    identifier = f"{repo_name}:{file_path}:{line_number}:{content.strip()}"
    
    # Generate SHA-256 hash and return first 16 characters for readability
    hash_object = hashlib.sha256(identifier.encode('utf-8'))
    return hash_object.hexdigest()[:16]


class TaskTracker:
    """
    Manages tracking of processed tasks to prevent duplicates
    
    Features:
    - Persistent storage of processed tasks
    - Hash-based task identification
    - Cleanup of old task entries
    - Statistics and reporting
    """
    
    def __init__(self, tracker_dir: Optional[Path] = None):
        """
        Initialize TaskTracker
        
        Args:
            tracker_dir (Optional[Path]): Directory to store tracker file.
                                         Defaults to project root/.claude_manager/
        """
        self.logger = get_logger(f"{__name__}.TaskTracker")
        
        # Set up tracker directory and file
        if tracker_dir is None:
            project_root = Path(__file__).parent.parent
            tracker_dir = project_root / '.claude_manager'
        
        self.tracker_dir = Path(tracker_dir)
        self.tracker_file = self.tracker_dir / 'task_tracker.json'
        
        # Ensure directory exists
        self.tracker_dir.mkdir(exist_ok=True)
        self.logger.debug(f"Task tracker directory: {self.tracker_dir}")
        
        # Load existing data
        self._task_data = self._load_tracker_data()
        self.logger.info(f"Task tracker initialized with {len(self._task_data)} existing tasks")
    
    def _load_tracker_data(self) -> Dict[str, Any]:
        """
        Load task tracking data from file
        
        Returns:
            Dict[str, Any]: Task tracking data
        """
        if not self.tracker_file.exists():
            self.logger.debug("No existing tracker file found, starting fresh")
            return {}
        
        try:
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.debug(f"Loaded {len(data)} tasks from tracker file")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupted tracker file {self.tracker_file}: {e}")
            # Backup corrupted file and start fresh
            backup_file = self.tracker_file.with_suffix('.json.corrupted')
            self.tracker_file.rename(backup_file)
            self.logger.warning(f"Corrupted file backed up to {backup_file}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading tracker file {self.tracker_file}: {e}")
            return {}
    
    def _save_tracker_data(self, data: Dict[str, Any]) -> None:
        """
        Save task tracking data to file
        
        Args:
            data (Dict[str, Any]): Task tracking data to save
        """
        try:
            # Write to temporary file first, then rename for atomicity
            temp_file = self.tracker_file.with_suffix('.json.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.rename(self.tracker_file)
            self.logger.debug(f"Saved {len(data)} tasks to tracker file")
            
        except Exception as e:
            self.logger.error(f"Error saving tracker file {self.tracker_file}: {e}")
            # Clean up temp file if it exists
            temp_file = self.tracker_file.with_suffix('.json.tmp')
            if temp_file.exists():
                temp_file.unlink()
    
    def is_task_processed(self, repo_name: str, file_path: str, line_number: int, content: str) -> bool:
        """
        Check if a task has already been processed
        
        Args:
            repo_name (str): Repository name
            file_path (str): File path
            line_number (int): Line number
            content (str): Task content
            
        Returns:
            bool: True if task has been processed, False otherwise
        """
        task_hash = generate_task_hash(repo_name, file_path, line_number, content)
        
        is_processed = task_hash in self._task_data
        
        if is_processed:
            task_info = self._task_data[task_hash]
            self.logger.debug(f"Task already processed: {file_path}:{line_number} (issue #{task_info.get('issue_number', 'unknown')})")
        
        return is_processed
    
    def mark_task_processed(self, repo_name: str, file_path: str, line_number: int, 
                          content: str, issue_number: Optional[int] = None) -> str:
        """
        Mark a task as processed
        
        Args:
            repo_name (str): Repository name
            file_path (str): File path
            line_number (int): Line number
            content (str): Task content
            issue_number (Optional[int]): GitHub issue number if created
            
        Returns:
            str: Task hash for the processed task
        """
        task_hash = generate_task_hash(repo_name, file_path, line_number, content)
        
        task_entry = {
            "repo": repo_name,
            "file": file_path,
            "line": line_number,
            "content": content.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "issue_number": issue_number
        }
        
        self._task_data[task_hash] = task_entry
        self._save_tracker_data(self._task_data)
        
        self.logger.info(f"Marked task as processed: {file_path}:{line_number} -> issue #{issue_number}")
        return task_hash
    
    def cleanup_old_tasks(self, days: int = 90) -> int:
        """
        Remove task entries older than specified days
        
        Args:
            days (int): Remove tasks older than this many days
            
        Returns:
            int: Number of tasks removed
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        tasks_to_remove = []
        for task_hash, task_data in self._task_data.items():
            try:
                created_at = datetime.fromisoformat(task_data["created_at"])
                if created_at < cutoff_date:
                    tasks_to_remove.append(task_hash)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid date in task {task_hash}: {e}")
                # Remove tasks with invalid dates too
                tasks_to_remove.append(task_hash)
        
        # Remove old tasks
        for task_hash in tasks_to_remove:
            del self._task_data[task_hash]
        
        if tasks_to_remove:
            self._save_tracker_data(self._task_data)
            self.logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks (older than {days} days)")
        
        return len(tasks_to_remove)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tracked tasks
        
        Returns:
            Dict[str, Any]: Statistics including total tasks, repositories, etc.
        """
        repositories = set()
        tasks_by_repo = {}
        
        for task_data in self._task_data.values():
            repo = task_data.get("repo", "unknown")
            repositories.add(repo)
            tasks_by_repo[repo] = tasks_by_repo.get(repo, 0) + 1
        
        return {
            "total_tasks": len(self._task_data),
            "repositories": len(repositories),
            "tasks_by_repo": tasks_by_repo,
            "tracker_file": str(self.tracker_file)
        }
    
    def get_processed_tasks_for_repo(self, repo_name: str) -> Dict[str, Any]:
        """
        Get all processed tasks for a specific repository
        
        Args:
            repo_name (str): Repository name
            
        Returns:
            Dict[str, Any]: Processed tasks for the repository
        """
        repo_tasks = {}
        
        for task_hash, task_data in self._task_data.items():
            if task_data.get("repo") == repo_name:
                repo_tasks[task_hash] = task_data
        
        self.logger.debug(f"Found {len(repo_tasks)} processed tasks for {repo_name}")
        return repo_tasks


# Global instance for easy access
_task_tracker = None

def get_task_tracker() -> TaskTracker:
    """
    Get global TaskTracker instance with lazy initialization
    
    Returns:
        TaskTracker: Global task tracker instance
    """
    global _task_tracker
    if _task_tracker is None:
        _task_tracker = TaskTracker()
    return _task_tracker


# Example usage and testing
if __name__ == "__main__":
    # Test the task tracker
    tracker = TaskTracker()
    
    # Test hash generation
    hash1 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
    hash2 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
    print(f"Hash consistency: {hash1 == hash2}")
    print(f"Generated hash: {hash1}")
    
    # Test task processing
    print(f"Is task processed: {tracker.is_task_processed('test/repo', 'src/file.py', 42, 'TODO: Fix this')}")
    
    task_hash = tracker.mark_task_processed('test/repo', 'src/file.py', 42, 'TODO: Fix this', 123)
    print(f"Marked task as processed: {task_hash}")
    
    print(f"Is task processed now: {tracker.is_task_processed('test/repo', 'src/file.py', 42, 'TODO: Fix this')}")
    
    # Test statistics
    stats = tracker.get_task_statistics()
    print(f"Task statistics: {stats}")
    
    print("Task tracker test completed")