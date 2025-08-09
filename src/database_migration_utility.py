"""
Database Migration Utility for Claude Manager Service

This utility helps migrate from file-based JSON storage to database-backed storage.
It can import existing data from JSON files into the new SQLite database.

Features:
- Migrate task tracking data from JSON to database
- Import performance metrics from JSON files
- Migrate configuration from JSON to database
- Validate migration integrity
- Backup existing files before migration
"""

import json
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from .logger import get_logger
from .services.database_service import get_database_service
from .database_task_tracker import DatabaseTaskTracker, generate_task_hash
from .database_performance_monitor import DatabasePerformanceMonitor


logger = get_logger(__name__)


class DatabaseMigrationUtility:
    """
    Utility for migrating from file-based to database storage
    
    This class provides methods to migrate existing JSON data to the new
    SQLite database while preserving data integrity and creating backups.
    """
    
    def __init__(self, backup_dir: str = None):
        """
        Initialize migration utility
        
        Args:
            backup_dir: Directory to store backup files (optional)
        """
        self.logger = get_logger(__name__)
        self.backup_dir = Path(backup_dir) if backup_dir else Path('.') / 'migration_backup'
        self.db_service = None
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Migration utility initialized, backup dir: {self.backup_dir}")
    
    async def _get_db_service(self):
        """Get database service instance"""
        if self.db_service is None:
            self.db_service = await get_database_service()
        return self.db_service
    
    async def migrate_task_tracker_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Migrate task tracking data from JSON file to database
        
        Args:
            json_file_path: Path to the task_tracker.json file
            
        Returns:
            Migration results dictionary
        """
        json_path = Path(json_file_path)
        
        if not json_path.exists():
            self.logger.warning(f"Task tracker JSON file not found: {json_path}")
            return {
                'status': 'skipped',
                'reason': 'file_not_found',
                'migrated_count': 0
            }
        
        try:
            # Backup original file
            backup_path = self.backup_dir / f"task_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(json_path, backup_path)
            self.logger.info(f"Backed up task tracker data to: {backup_path}")
            
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Get database service
            db_service = await self._get_db_service()
            migrated_count = 0
            errors = []
            
            # Migrate each task
            for task_hash, task_data in json_data.items():
                try:
                    success = await db_service.save_task_tracking(
                        task_hash=task_hash,
                        repo_name=task_data.get('repo', ''),
                        file_path=task_data.get('file', ''),
                        line_number=task_data.get('line', 0),
                        content=task_data.get('content', ''),
                        issue_number=task_data.get('issue_number')
                    )
                    
                    if success:
                        migrated_count += 1
                    else:
                        errors.append(f"Failed to migrate task {task_hash}")
                        
                except Exception as e:
                    errors.append(f"Error migrating task {task_hash}: {str(e)}")
            
            self.logger.info(f"Migrated {migrated_count} task tracking entries")
            
            return {
                'status': 'completed',
                'migrated_count': migrated_count,
                'total_count': len(json_data),
                'errors': errors,
                'backup_path': str(backup_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to migrate task tracker data: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'migrated_count': 0
            }
    
    async def migrate_performance_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Migrate performance metrics from JSON file to database
        
        Args:
            json_file_path: Path to performance metrics JSON file
            
        Returns:
            Migration results dictionary
        """
        json_path = Path(json_file_path)
        
        if not json_path.exists():
            self.logger.warning(f"Performance metrics JSON file not found: {json_path}")
            return {
                'status': 'skipped',
                'reason': 'file_not_found',
                'migrated_count': 0
            }
        
        try:
            # Backup original file
            backup_path = self.backup_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(json_path, backup_path)
            self.logger.info(f"Backed up performance data to: {backup_path}")
            
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Get database service
            db_service = await self._get_db_service()
            migrated_count = 0
            errors = []
            
            # Handle different JSON formats
            if isinstance(json_data, dict):
                # Check if it's the format with function names as keys
                if 'operations' in json_data or 'metrics' in json_data:
                    # Extract operations list
                    operations = json_data.get('operations', json_data.get('metrics', []))
                else:
                    # Assume the dict contains function names as keys
                    operations = []
                    for func_name, func_data in json_data.items():
                        if isinstance(func_data, list):
                            operations.extend(func_data)
                        elif isinstance(func_data, dict):
                            operations.append(func_data)
            elif isinstance(json_data, list):
                operations = json_data
            else:
                operations = []
            
            # Migrate each metric
            for metric_data in operations:
                try:
                    # Convert old format to new format if needed
                    normalized_metric = self._normalize_performance_metric(metric_data)
                    
                    success = await db_service.save_performance_metric(normalized_metric)
                    
                    if success:
                        migrated_count += 1
                    else:
                        errors.append(f"Failed to migrate metric: {metric_data.get('function_name', 'unknown')}")
                        
                except Exception as e:
                    errors.append(f"Error migrating metric: {str(e)}")
            
            self.logger.info(f"Migrated {migrated_count} performance metrics")
            
            return {
                'status': 'completed',
                'migrated_count': migrated_count,
                'total_count': len(operations),
                'errors': errors,
                'backup_path': str(backup_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to migrate performance data: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'migrated_count': 0
            }
    
    def _normalize_performance_metric(self, metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize performance metric data to expected format"""
        return {
            'function_name': metric_data.get('function_name', 'unknown'),
            'module_name': metric_data.get('module_name', 'unknown'),
            'start_time': metric_data.get('start_time', 0.0),
            'end_time': metric_data.get('end_time', 0.0),
            'duration': metric_data.get('duration', 0.0),
            'success': metric_data.get('success', True),
            'error_message': metric_data.get('error_message'),
            'memory_before': metric_data.get('memory_before'),
            'memory_after': metric_data.get('memory_after'),
            'memory_delta': metric_data.get('memory_delta'),
            'args_count': metric_data.get('args_count', 0),
            'kwargs_count': metric_data.get('kwargs_count', 0)
        }
    
    async def migrate_configuration_data(self, config_file_path: str) -> Dict[str, Any]:
        """
        Migrate configuration data from JSON file to database
        
        Args:
            config_file_path: Path to configuration JSON file
            
        Returns:
            Migration results dictionary
        """
        config_path = Path(config_file_path)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration JSON file not found: {config_path}")
            return {
                'status': 'skipped',
                'reason': 'file_not_found',
                'migrated_count': 0
            }
        
        try:
            # Backup original file
            backup_path = self.backup_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(config_path, backup_path)
            self.logger.info(f"Backed up configuration data to: {backup_path}")
            
            # Load JSON data
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Get database service
            db_service = await self._get_db_service()
            migrated_count = 0
            errors = []
            
            # Migrate configuration values
            for key, value in config_data.items():
                try:
                    success = await db_service.save_configuration(key, value)
                    
                    if success:
                        migrated_count += 1
                    else:
                        errors.append(f"Failed to migrate config key: {key}")
                        
                except Exception as e:
                    errors.append(f"Error migrating config key {key}: {str(e)}")
            
            self.logger.info(f"Migrated {migrated_count} configuration entries")
            
            return {
                'status': 'completed',
                'migrated_count': migrated_count,
                'total_count': len(config_data),
                'errors': errors,
                'backup_path': str(backup_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to migrate configuration data: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'migrated_count': 0
            }
    
    async def auto_discover_and_migrate(self, project_root: str = '.') -> Dict[str, Any]:
        """
        Auto-discover and migrate all JSON files in the project
        
        Args:
            project_root: Root directory to search for JSON files
            
        Returns:
            Complete migration results
        """
        project_path = Path(project_root)
        migration_results = {
            'task_tracking': None,
            'performance_metrics': None,
            'configuration': None,
            'start_time': datetime.now().isoformat(),
            'total_migrated': 0
        }
        
        self.logger.info(f"Auto-discovering JSON files in: {project_path}")
        
        # Look for task tracker files
        task_tracker_patterns = [
            '.claude_manager/task_tracker.json',
            'task_tracker.json',
            '.task_tracker.json'
        ]
        
        for pattern in task_tracker_patterns:
            file_path = project_path / pattern
            if file_path.exists():
                self.logger.info(f"Found task tracker file: {file_path}")
                migration_results['task_tracking'] = await self.migrate_task_tracker_data(str(file_path))
                break
        
        # Look for performance metrics files
        performance_patterns = [
            'performance_metrics.json',
            '.performance_metrics.json',
            'metrics.json'
        ]
        
        for pattern in performance_patterns:
            file_path = project_path / pattern
            if file_path.exists():
                self.logger.info(f"Found performance metrics file: {file_path}")
                migration_results['performance_metrics'] = await self.migrate_performance_data(str(file_path))
                break
        
        # Look for configuration files
        config_patterns = [
            'config.json',
            'claude_config.json',
            '.config.json'
        ]
        
        for pattern in config_patterns:
            file_path = project_path / pattern
            if file_path.exists():
                self.logger.info(f"Found configuration file: {file_path}")
                migration_results['configuration'] = await self.migrate_configuration_data(str(file_path))
                break
        
        # Calculate totals
        for key, result in migration_results.items():
            if isinstance(result, dict) and 'migrated_count' in result:
                migration_results['total_migrated'] += result['migrated_count']
        
        migration_results['end_time'] = datetime.now().isoformat()
        
        self.logger.info(f"Migration completed: {migration_results['total_migrated']} total items migrated")
        
        return migration_results
    
    async def validate_migration(self) -> Dict[str, Any]:
        """
        Validate that migration was successful by checking database contents
        
        Returns:
            Validation results
        """
        try:
            db_service = await self._get_db_service()
            stats = await db_service.get_database_statistics()
            
            validation_results = {
                'database_accessible': True,
                'database_statistics': stats,
                'validation_time': datetime.now().isoformat()
            }
            
            # Check if we have data in key tables
            validation_results['has_task_tracking'] = stats.get('task_tracking_count', 0) > 0
            validation_results['has_performance_metrics'] = stats.get('performance_metrics_count', 0) > 0
            validation_results['has_configuration'] = stats.get('configuration_store_count', 0) > 0
            
            self.logger.info(f"Migration validation completed: {validation_results}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Migration validation failed: {e}")
            return {
                'database_accessible': False,
                'error': str(e),
                'validation_time': datetime.now().isoformat()
            }
    
    def generate_migration_report(self, migration_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable migration report
        
        Args:
            migration_results: Results from migration operations
            
        Returns:
            Formatted migration report
        """
        report_lines = [
            "=" * 60,
            "DATABASE MIGRATION REPORT",
            "=" * 60,
            f"Migration Time: {migration_results.get('start_time', 'unknown')} - {migration_results.get('end_time', 'unknown')}",
            f"Total Items Migrated: {migration_results.get('total_migrated', 0)}",
            "",
            "DETAILS:",
            ""
        ]
        
        # Task tracking
        task_result = migration_results.get('task_tracking')
        if task_result:
            report_lines.extend([
                f"Task Tracking: {task_result['status'].upper()}",
                f"  - Migrated: {task_result.get('migrated_count', 0)} items",
                f"  - Total: {task_result.get('total_count', 0)} items",
                f"  - Errors: {len(task_result.get('errors', []))}",
                ""
            ])
        else:
            report_lines.append("Task Tracking: NOT FOUND")
            report_lines.append("")
        
        # Performance metrics
        perf_result = migration_results.get('performance_metrics')
        if perf_result:
            report_lines.extend([
                f"Performance Metrics: {perf_result['status'].upper()}",
                f"  - Migrated: {perf_result.get('migrated_count', 0)} items",
                f"  - Total: {perf_result.get('total_count', 0)} items",
                f"  - Errors: {len(perf_result.get('errors', []))}",
                ""
            ])
        else:
            report_lines.append("Performance Metrics: NOT FOUND")
            report_lines.append("")
        
        # Configuration
        config_result = migration_results.get('configuration')
        if config_result:
            report_lines.extend([
                f"Configuration: {config_result['status'].upper()}",
                f"  - Migrated: {config_result.get('migrated_count', 0)} items",
                f"  - Total: {config_result.get('total_count', 0)} items",
                f"  - Errors: {len(config_result.get('errors', []))}",
                ""
            ])
        else:
            report_lines.append("Configuration: NOT FOUND")
            report_lines.append("")
        
        report_lines.extend([
            "=" * 60,
            "MIGRATION COMPLETE",
            "=" * 60
        ])
        
        return "\n".join(report_lines)


# Convenience functions
async def run_full_migration(project_root: str = '.', backup_dir: str = None) -> Dict[str, Any]:
    """
    Run a complete migration from JSON files to database
    
    Args:
        project_root: Root directory to search for JSON files
        backup_dir: Directory for backup files
        
    Returns:
        Migration results
    """
    migrator = DatabaseMigrationUtility(backup_dir)
    
    # Run auto-discovery migration
    results = await migrator.auto_discover_and_migrate(project_root)
    
    # Validate migration
    validation = await migrator.validate_migration()
    results['validation'] = validation
    
    # Generate report
    report = migrator.generate_migration_report(results)
    results['report'] = report
    
    # Print report
    print(report)
    
    return results


# Example usage and testing
async def example_migration():
    """Example of using migration utility"""
    try:
        logger.info("Starting database migration example")
        
        # Run full migration
        results = await run_full_migration()
        
        logger.info("Migration example completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Migration example failed: {e}")
        raise


if __name__ == "__main__":
    # Run migration
    asyncio.run(example_migration())