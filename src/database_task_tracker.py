"""
Database-backed Task Tracker for Claude Manager Service

This module provides a database-backed implementation of the TaskTracker
that replaces the JSON file-based storage with SQLite database operations.

Features:
- Database persistence instead of JSON files
- Async operations for better performance
- Enhanced querying capabilities
- Data integrity and ACID compliance
- Backward compatibility with existing TaskTracker interface
"""

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import asyncio

from src.logger import get_logger
from src.services.database_service import get_database_service
from src.error_handler import DatabaseError


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


class DatabaseTaskTracker:
    """
    Database-backed task tracker for duplicate prevention
    
    This class provides the same interface as the original TaskTracker
    but uses a SQLite database for persistence instead of JSON files.
    
    Features:
    - Database persistence with ACID compliance
    - Async operations for better performance
    - Enhanced querying and filtering capabilities
    - Automatic data cleanup and maintenance
    - Statistics and reporting with SQL aggregations
    """
    
    def __init__(self):
        """Initialize DatabaseTaskTracker"""
        self.logger = get_logger(f"{__name__}.DatabaseTaskTracker")
        self._db_service = None
        self._initialized = False
        
        self.logger.info("Database task tracker initialized")
    
    async def _get_db_service(self):
        """Get database service instance (lazy initialization)"""
        if self._db_service is None:
            self._db_service = await get_database_service()
        return self._db_service
    
    async def _ensure_initialized(self):
        """Ensure the tracker is initialized"""
        if not self._initialized:
            await self._get_db_service()
            self._initialized = True
    
    async def is_task_processed(self, repo_name: str, file_path: str, line_number: int, content: str) -> bool:
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
        await self._ensure_initialized()
        
        task_hash = generate_task_hash(repo_name, file_path, line_number, content)
        
        try:
            db_service = await self._get_db_service()
            is_processed = await db_service.is_task_processed(task_hash)
            
            if is_processed:
                self.logger.debug(f"Task already processed: {file_path}:{line_number}")
            
            return is_processed
            
        except Exception as e:
            self.logger.error(f"Failed to check task processing status: {e}")
            # If check fails, assume not processed to avoid blocking new tasks
            return False
    
    async def mark_task_processed(self, repo_name: str, file_path: str, line_number: int, 
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
        await self._ensure_initialized()
        
        task_hash = generate_task_hash(repo_name, file_path, line_number, content)
        
        try:
            db_service = await self._get_db_service()
            success = await db_service.save_task_tracking(
                task_hash=task_hash,
                repo_name=repo_name,
                file_path=file_path,
                line_number=line_number,
                content=content.strip(),
                issue_number=issue_number
            )
            
            if success:
                self.logger.info(f"Marked task as processed: {file_path}:{line_number} -> issue #{issue_number}")
            else:
                self.logger.warning(f"Failed to mark task as processed: {task_hash}")
            
            return task_hash
            
        except Exception as e:
            self.logger.error(f"Failed to mark task as processed: {e}")
            raise DatabaseError(f"Task marking failed: {str(e)}", "mark_task_processed", e)
    
    async def cleanup_old_tasks(self, days: int = 90) -> int:
        """
        Remove task entries older than specified days
        
        Args:
            days (int): Remove tasks older than this many days
            
        Returns:
            int: Number of tasks removed
        """
        await self._ensure_initialized()
        
        try:
            db_service = await self._get_db_service()
            cleanup_stats = await db_service.cleanup_old_data(days)
            
            tasks_removed = cleanup_stats.get('task_tracking', 0)
            
            if tasks_removed > 0:
                self.logger.info(f"Cleaned up {tasks_removed} old tasks (older than {days} days)")
            
            return tasks_removed
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old tasks: {e}")
            return 0
    
    async def get_task_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tracked tasks
        
        Returns:
            Dict[str, Any]: Statistics including total tasks, repositories, etc.
        """
        await self._ensure_initialized()
        
        try:
            db_service = await self._get_db_service()
            conn = await db_service.get_connection()
            
            try:
                # Get total task count
                async with conn.execute("SELECT COUNT(*) FROM task_tracking") as cursor:
                    total_tasks = (await cursor.fetchone())[0]
                
                # Get repository count and tasks by repo
                async with conn.execute("""
                    SELECT repo_name, COUNT(*) as task_count 
                    FROM task_tracking 
                    GROUP BY repo_name
                    ORDER BY task_count DESC
                """) as cursor:
                    repo_stats = await cursor.fetchall()
                
                repositories = len(repo_stats)
                tasks_by_repo = {repo: count for repo, count in repo_stats}
                
                # Get recent activity (last 30 days)
                async with conn.execute("""
                    SELECT COUNT(*) FROM task_tracking 
                    WHERE created_at > datetime('now', '-30 days')
                """) as cursor:
                    recent_tasks = (await cursor.fetchone())[0]
                
                # Get success rate (tasks with issue numbers)
                async with conn.execute("""
                    SELECT COUNT(*) FROM task_tracking 
                    WHERE issue_number IS NOT NULL
                """) as cursor:
                    successful_tasks = (await cursor.fetchone())[0]
                
                success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
                
                return {
                    "total_tasks": total_tasks,
                    "repositories": repositories,
                    "tasks_by_repo": tasks_by_repo,
                    "recent_tasks_30_days": recent_tasks,
                    "successful_tasks": successful_tasks,
                    "success_rate_percent": round(success_rate, 2),
                    "most_active_repository": repo_stats[0][0] if repo_stats else None,
                    "tracker_type": "database"
                }
                
            finally:
                await db_service.return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Failed to get task statistics: {e}")
            return {
                "total_tasks": 0,
                "repositories": 0,
                "tasks_by_repo": {},
                "error": str(e),
                "tracker_type": "database"
            }
    
    async def get_processed_tasks_for_repo(self, repo_name: str) -> List[Dict[str, Any]]:
        """
        Get all processed tasks for a specific repository
        
        Args:
            repo_name (str): Repository name
            
        Returns:
            List[Dict[str, Any]]: Processed tasks for the repository
        """
        await self._ensure_initialized()
        
        try:
            db_service = await self._get_db_service()
            conn = await db_service.get_connection()
            
            try:
                async with conn.execute("""
                    SELECT task_hash, repo_name, file_path, line_number, content, 
                           issue_number, created_at, processed_at
                    FROM task_tracking 
                    WHERE repo_name = ?
                    ORDER BY processed_at DESC
                """, (repo_name,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    tasks = []
                    for row in rows:
                        task_data = dict(zip(columns, row))
                        tasks.append(task_data)
                    
                    self.logger.debug(f"Found {len(tasks)} processed tasks for {repo_name}")
                    return tasks
                    
            finally:
                await db_service.return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Failed to get processed tasks for {repo_name}: {e}")
            return []
    
    async def get_tasks_by_status(self, with_issues: bool = None) -> List[Dict[str, Any]]:
        """
        Get tasks filtered by whether they have associated issues
        
        Args:
            with_issues (bool): If True, return only tasks with issues.
                              If False, return only tasks without issues.
                              If None, return all tasks.
            
        Returns:
            List[Dict[str, Any]]: Filtered tasks
        """
        await self._ensure_initialized()
        
        try:
            db_service = await self._get_db_service()
            conn = await db_service.get_connection()
            
            try:
                if with_issues is True:
                    query = """
                        SELECT * FROM task_tracking 
                        WHERE issue_number IS NOT NULL
                        ORDER BY processed_at DESC
                    """
                    params = ()
                elif with_issues is False:
                    query = """
                        SELECT * FROM task_tracking 
                        WHERE issue_number IS NULL
                        ORDER BY processed_at DESC
                    """
                    params = ()
                else:
                    query = """
                        SELECT * FROM task_tracking 
                        ORDER BY processed_at DESC
                    """
                    params = ()
                
                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    return [dict(zip(columns, row)) for row in rows]
                    
            finally:
                await db_service.return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Failed to get tasks by status: {e}")
            return []
    
    async def export_tasks(self, repo_name: str = None, since_days: int = None) -> List[Dict[str, Any]]:
        """
        Export task data for backup or analysis
        
        Args:
            repo_name (str): Specific repository to export (optional)
            since_days (int): Export tasks from last N days (optional)
            
        Returns:
            List[Dict[str, Any]]: Exported task data
        """
        await self._ensure_initialized()
        
        try:
            db_service = await self._get_db_service()
            conn = await db_service.get_connection()
            
            try:
                query = "SELECT * FROM task_tracking WHERE 1=1"
                params = []
                
                if repo_name:
                    query += " AND repo_name = ?"
                    params.append(repo_name)
                
                if since_days:
                    query += " AND created_at > datetime('now', '-{} days')".format(since_days)
                
                query += " ORDER BY created_at DESC"
                
                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    exported_data = []
                    for row in rows:
                        task_data = dict(zip(columns, row))
                        exported_data.append(task_data)
                    
                    self.logger.info(f"Exported {len(exported_data)} tasks")
                    return exported_data
                    
            finally:
                await db_service.return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Failed to export tasks: {e}")
            return []

    # Synchronous compatibility methods for backward compatibility
    def is_task_processed_sync(self, repo_name: str, file_path: str, line_number: int, content: str) -> bool:
        """Synchronous version of is_task_processed for backward compatibility"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.is_task_processed(repo_name, file_path, line_number, content))
        except RuntimeError:
            # If no event loop is running, create a new one
            return asyncio.run(self.is_task_processed(repo_name, file_path, line_number, content))
    
    def mark_task_processed_sync(self, repo_name: str, file_path: str, line_number: int, 
                               content: str, issue_number: Optional[int] = None) -> str:
        """Synchronous version of mark_task_processed for backward compatibility"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.mark_task_processed(repo_name, file_path, line_number, content, issue_number))
        except RuntimeError:
            return asyncio.run(self.mark_task_processed(repo_name, file_path, line_number, content, issue_number))


# Global instance for easy access
_database_task_tracker = None


def get_database_task_tracker() -> DatabaseTaskTracker:
    """
    Get global DatabaseTaskTracker instance with lazy initialization
    
    Returns:
        DatabaseTaskTracker: Global database task tracker instance
    """
    global _database_task_tracker
    if _database_task_tracker is None:
        _database_task_tracker = DatabaseTaskTracker()
    return _database_task_tracker


# Async version of the global function
async def get_database_task_tracker_async() -> DatabaseTaskTracker:
    """
    Get global DatabaseTaskTracker instance (async version)
    
    Returns:
        DatabaseTaskTracker: Initialized global database task tracker instance
    """
    tracker = get_database_task_tracker()
    await tracker._ensure_initialized()
    return tracker


# Example usage and testing
async def example_database_task_tracker():
    """Example of using database task tracker"""
    try:
        # Get database task tracker
        tracker = await get_database_task_tracker_async()
        
        # Test hash generation
        hash1 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        hash2 = generate_task_hash("test/repo", "src/file.py", 42, "TODO: Fix this")
        print(f"Hash consistency: {hash1 == hash2}")
        print(f"Generated hash: {hash1}")
        
        # Test task processing
        is_processed_before = await tracker.is_task_processed('test/repo', 'src/file.py', 42, 'TODO: Fix this')
        print(f"Is task processed (before): {is_processed_before}")
        
        task_hash = await tracker.mark_task_processed('test/repo', 'src/file.py', 42, 'TODO: Fix this', 123)
        print(f"Marked task as processed: {task_hash}")
        
        is_processed_after = await tracker.is_task_processed('test/repo', 'src/file.py', 42, 'TODO: Fix this')
        print(f"Is task processed (after): {is_processed_after}")
        
        # Test statistics
        stats = await tracker.get_task_statistics()
        print(f"Task statistics: {stats}")
        
        print("Database task tracker test completed")
        
    except Exception as e:
        logger.error(f"Database task tracker example failed: {e}")
        raise


if __name__ == "__main__":
    # Test database task tracker
    asyncio.run(example_database_task_tracker())