"""
Database Service for Claude Manager Service

This service provides centralized database operations with SQLite support,
replacing file-based JSON storage with proper relational database persistence.

Features:
- SQLite database with async support
- Schema migration and versioning
- Task tracking with relational integrity
- Performance metrics storage
- Configuration persistence
- Query optimization and indexing
"""

import asyncio
import aiosqlite
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

from src.logger import get_logger
from src.error_handler import with_enhanced_error_handling, NetworkError

# Define DatabaseError for database-specific errors
class DatabaseError(Exception):
    """Database-specific error"""
    def __init__(self, message: str, operation: str, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.original_error = original_error


logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_path: str = "claude_manager.db"
    connection_timeout: int = 30
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    vacuum_interval_days: int = 7
    max_connections: int = 10


class DatabaseService:
    """
    Centralized database service for persistent storage
    
    This service provides a unified interface for all database operations,
    replacing file-based JSON storage with SQLite for better performance,
    querying capabilities, and data integrity.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        """
        Initialize database service
        
        Args:
            config: Database configuration
        """
        self.logger = get_logger(__name__)
        self.config = config or DatabaseConfig()
        
        # Database connection pool
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        
        # Schema version tracking
        self.current_schema_version = 3
        
        self.logger.info(f"Database service initialized: {self.config.database_path}")
    
    async def initialize(self) -> None:
        """Initialize database with schema creation and migrations"""
        if self._initialized:
            return
        
        try:
            # Ensure database directory exists
            db_path = Path(self.config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create initial connection and schema
            async with aiosqlite.connect(
                self.config.database_path,
                timeout=self.config.connection_timeout
            ) as conn:
                await self._create_schema(conn)
                await self._run_migrations(conn)
                await conn.commit()
            
            # Initialize connection pool
            await self._initialize_connection_pool()
            
            self._initialized = True
            self.logger.info("Database service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {str(e)}", "initialize", e)
    
    async def _create_schema(self, conn: aiosqlite.Connection) -> None:
        """Create database schema"""
        
        # Metadata table for schema versioning
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Task tracking table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_tracking (
                task_hash TEXT PRIMARY KEY,
                repo_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                issue_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(repo_name, file_path, line_number, content)
            )
        """)
        
        # Performance metrics table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                module_name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                memory_before REAL,
                memory_after REAL,
                memory_delta REAL,
                args_count INTEGER DEFAULT 0,
                kwargs_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Configuration storage table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS configuration_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                value_type TEXT NOT NULL DEFAULT 'string',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Issue metadata table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS issue_metadata (
                issue_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                repository TEXT NOT NULL,
                github_issue_number INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                labels TEXT, -- JSON array
                priority_score REAL,
                source_type TEXT DEFAULT 'manual',
                source_metadata TEXT, -- JSON object
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Task execution results table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration REAL,
                result_data TEXT, -- JSON object
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                performance_metrics TEXT, -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better query performance
        await self._create_indexes(conn)
    
    async def _create_indexes(self, conn: aiosqlite.Connection) -> None:
        """Create database indexes for query optimization"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_task_tracking_repo ON task_tracking(repo_name)",
            "CREATE INDEX IF NOT EXISTS idx_task_tracking_created ON task_tracking(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_performance_function ON performance_metrics(function_name, module_name)",
            "CREATE INDEX IF NOT EXISTS idx_performance_created ON performance_metrics(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_issue_repository ON issue_metadata(repository)",
            "CREATE INDEX IF NOT EXISTS idx_issue_status ON issue_metadata(status)",
            "CREATE INDEX IF NOT EXISTS idx_task_results_type ON task_results(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_task_results_status ON task_results(status)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
    
    async def _run_migrations(self, conn: aiosqlite.Connection) -> None:
        """Run database schema migrations"""
        # Get current schema version
        current_version = await self._get_schema_version(conn)
        
        if current_version < self.current_schema_version:
            self.logger.info(f"Running database migrations from version {current_version} to {self.current_schema_version}")
            
            # Migration v1 -> v2: Add performance metrics enhancements
            if current_version < 2:
                await conn.execute("""
                    ALTER TABLE performance_metrics 
                    ADD COLUMN memory_delta REAL DEFAULT 0
                """)
            
            # Migration v2 -> v3: Add task results table
            if current_version < 3:
                # Task results table was added in _create_schema, so just update version
                pass
            
            # Update schema version
            await self._set_schema_version(conn, self.current_schema_version)
    
    async def _get_schema_version(self, conn: aiosqlite.Connection) -> int:
        """Get current schema version"""
        try:
            async with conn.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ) as cursor:
                row = await cursor.fetchone()
                return int(row[0]) if row else 1
        except Exception:
            return 1
    
    async def _set_schema_version(self, conn: aiosqlite.Connection, version: int) -> None:
        """Set schema version"""
        await conn.execute("""
            INSERT OR REPLACE INTO schema_metadata (key, value, updated_at)
            VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
        """, (str(version),))
    
    async def _initialize_connection_pool(self) -> None:
        """Initialize database connection pool"""
        async with self._pool_lock:
            for _ in range(min(3, self.config.max_connections)):
                conn = await aiosqlite.connect(
                    self.config.database_path,
                    timeout=self.config.connection_timeout
                )
                # Enable foreign keys and WAL mode for better performance
                await conn.execute("PRAGMA foreign_keys = ON")
                await conn.execute("PRAGMA journal_mode = WAL")
                self._connection_pool.append(conn)
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Get database connection from pool"""
        if not self._initialized:
            await self.initialize()
        
        async with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            else:
                # Create new connection if pool is empty
                conn = await aiosqlite.connect(
                    self.config.database_path,
                    timeout=self.config.connection_timeout
                )
                await conn.execute("PRAGMA foreign_keys = ON")
                await conn.execute("PRAGMA journal_mode = WAL")
                return conn
    
    async def return_connection(self, conn: aiosqlite.Connection) -> None:
        """Return connection to pool"""
        async with self._pool_lock:
            if len(self._connection_pool) < self.config.max_connections:
                self._connection_pool.append(conn)
            else:
                await conn.close()
    
    @with_enhanced_error_handling("save_task_tracking", use_circuit_breaker=True)
    async def save_task_tracking(self, task_hash: str, repo_name: str, file_path: str,
                               line_number: int, content: str, issue_number: int = None) -> bool:
        """
        Save task tracking information
        
        Args:
            task_hash: Unique task hash
            repo_name: Repository name
            file_path: File path
            line_number: Line number
            content: Task content
            issue_number: GitHub issue number if created
            
        Returns:
            True if saved successfully
        """
        conn = await self.get_connection()
        try:
            await conn.execute("""
                INSERT OR REPLACE INTO task_tracking 
                (task_hash, repo_name, file_path, line_number, content, issue_number, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (task_hash, repo_name, file_path, line_number, content, issue_number))
            
            await conn.commit()
            self.logger.debug(f"Saved task tracking: {task_hash}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save task tracking {task_hash}: {e}")
            await conn.rollback()
            raise DatabaseError(f"Task tracking save failed: {str(e)}", "save_task_tracking", e)
        finally:
            await self.return_connection(conn)
    
    @with_enhanced_error_handling("is_task_processed", use_circuit_breaker=True)
    async def is_task_processed(self, task_hash: str) -> bool:
        """
        Check if task has been processed
        
        Args:
            task_hash: Unique task hash
            
        Returns:
            True if task has been processed
        """
        conn = await self.get_connection()
        try:
            async with conn.execute(
                "SELECT 1 FROM task_tracking WHERE task_hash = ?", (task_hash,)
            ) as cursor:
                result = await cursor.fetchone()
                return result is not None
        except Exception as e:
            self.logger.error(f"Failed to check task processing status {task_hash}: {e}")
            raise DatabaseError(f"Task status check failed: {str(e)}", "is_task_processed", e)
        finally:
            await self.return_connection(conn)
    
    @with_enhanced_error_handling("save_performance_metric", use_circuit_breaker=True)
    async def save_performance_metric(self, metric_data: Dict[str, Any]) -> bool:
        """
        Save performance metric
        
        Args:
            metric_data: Performance metric data
            
        Returns:
            True if saved successfully
        """
        conn = await self.get_connection()
        try:
            await conn.execute("""
                INSERT INTO performance_metrics 
                (function_name, module_name, start_time, end_time, duration, success,
                 error_message, memory_before, memory_after, memory_delta, args_count, kwargs_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric_data['function_name'],
                metric_data['module_name'],
                metric_data['start_time'],
                metric_data['end_time'],
                metric_data['duration'],
                metric_data['success'],
                metric_data.get('error_message'),
                metric_data.get('memory_before'),
                metric_data.get('memory_after'),
                metric_data.get('memory_delta'),
                metric_data.get('args_count', 0),
                metric_data.get('kwargs_count', 0)
            ))
            
            await conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save performance metric: {e}")
            await conn.rollback()
            raise DatabaseError(f"Performance metric save failed: {str(e)}", "save_performance_metric", e)
        finally:
            await self.return_connection(conn)
    
    @with_enhanced_error_handling("get_performance_metrics", use_circuit_breaker=True)
    async def get_performance_metrics(self, function_name: str = None, 
                                    since_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get performance metrics
        
        Args:
            function_name: Specific function name to filter
            since_hours: Hours to look back
            
        Returns:
            List of performance metrics
        """
        conn = await self.get_connection()
        try:
            query = """
                SELECT * FROM performance_metrics 
                WHERE created_at > datetime('now', '-{} hours')
            """.format(since_hours)
            
            params = []
            if function_name:
                query += " AND function_name = ?"
                params.append(function_name)
            
            query += " ORDER BY created_at DESC"
            
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            raise DatabaseError(f"Performance metrics retrieval failed: {str(e)}", "get_performance_metrics", e)
        finally:
            await self.return_connection(conn)
    
    async def save_configuration(self, key: str, value: Any, value_type: str = None) -> bool:
        """
        Save configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            value_type: Type of the value (string, json, etc.)
            
        Returns:
            True if saved successfully
        """
        conn = await self.get_connection()
        try:
            # Determine value type and serialize if needed
            if value_type is None:
                if isinstance(value, (dict, list)):
                    value_type = 'json'
                    value = json.dumps(value)
                else:
                    value_type = 'string'
                    value = str(value)
            
            await conn.execute("""
                INSERT OR REPLACE INTO configuration_store
                (key, value, value_type, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (key, value, value_type))
            
            await conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration {key}: {e}")
            await conn.rollback()
            return False
        finally:
            await self.return_connection(conn)
    
    async def get_configuration(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        conn = await self.get_connection()
        try:
            async with conn.execute(
                "SELECT value, value_type FROM configuration_store WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row is None:
                    return default
                
                value, value_type = row
                
                # Deserialize based on type
                if value_type == 'json':
                    return json.loads(value)
                else:
                    return value
                    
        except Exception as e:
            self.logger.error(f"Failed to get configuration {key}: {e}")
            return default
        finally:
            await self.return_connection(conn)
    
    async def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """
        Clean up old data from database
        
        Args:
            days: Number of days to retain
            
        Returns:
            Dictionary with cleanup statistics
        """
        conn = await self.get_connection()
        cleanup_stats = {}
        
        try:
            # Clean up old task tracking
            async with conn.execute("""
                DELETE FROM task_tracking 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days)) as cursor:
                cleanup_stats['task_tracking'] = cursor.rowcount
            
            # Clean up old performance metrics
            async with conn.execute("""
                DELETE FROM performance_metrics 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days)) as cursor:
                cleanup_stats['performance_metrics'] = cursor.rowcount
            
            # Clean up old task results
            async with conn.execute("""
                DELETE FROM task_results 
                WHERE created_at < datetime('now', '-{} days')
                AND status IN ('completed', 'failed')
            """.format(days)) as cursor:
                cleanup_stats['task_results'] = cursor.rowcount
            
            await conn.commit()
            
            # Vacuum database to reclaim space
            await conn.execute("VACUUM")
            
            self.logger.info(f"Database cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Database cleanup failed: {e}")
            await conn.rollback()
            return {}
        finally:
            await self.return_connection(conn)
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        conn = await self.get_connection()
        try:
            stats = {}
            
            # Table row counts
            tables = ['task_tracking', 'performance_metrics', 'configuration_store', 
                     'issue_metadata', 'task_results']
            
            for table in tables:
                async with conn.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                    row = await cursor.fetchone()
                    stats[f"{table}_count"] = row[0] if row else 0
            
            # Database size
            db_path = Path(self.config.database_path)
            if db_path.exists():
                stats['database_size_bytes'] = db_path.stat().st_size
                stats['database_size_mb'] = round(stats['database_size_bytes'] / (1024 * 1024), 2)
            
            # Schema version
            stats['schema_version'] = await self._get_schema_version(conn)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database statistics: {e}")
            return {}
        finally:
            await self.return_connection(conn)
    
    async def close(self) -> None:
        """Close all database connections"""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()
        
        self.logger.info("Database service closed")


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service(config: DatabaseConfig = None) -> DatabaseService:
    """
    Get global database service instance
    
    Args:
        config: Database configuration
        
    Returns:
        Initialized database service
    """
    global _database_service
    
    if _database_service is None:
        _database_service = DatabaseService(config)
        await _database_service.initialize()
    
    return _database_service


# Example usage and testing
async def example_database_service():
    """Example of using database service"""
    try:
        # Get database service
        db_service = await get_database_service()
        
        # Save some test data
        await db_service.save_task_tracking(
            task_hash="test_hash_123",
            repo_name="example/repo",
            file_path="src/test.py",
            line_number=42,
            content="TODO: Fix this",
            issue_number=123
        )
        
        # Check if task is processed
        is_processed = await db_service.is_task_processed("test_hash_123")
        
        # Save performance metric
        await db_service.save_performance_metric({
            'function_name': 'test_function',
            'module_name': 'test_module',
            'start_time': time.time(),
            'end_time': time.time() + 1.5,
            'duration': 1.5,
            'success': True
        })
        
        # Get statistics
        stats = await db_service.get_database_statistics()
        
        logger.info(f"Database service example completed")
        logger.info(f"Task processed: {is_processed}")
        logger.info(f"Database statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Database service example failed: {e}")
        raise


if __name__ == "__main__":
    import time
    # Test database service
    asyncio.run(example_database_service())