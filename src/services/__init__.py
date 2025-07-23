"""
Service Layer for Claude Manager Service

This package provides a clean service layer architecture with clear
separation of concerns and dependency injection capabilities.

Services:
- RepositoryService: Repository scanning and management
- IssueService: Issue creation and tracking
- TaskService: Task orchestration and execution
- ConfigurationService: Centralized configuration management
- DatabaseService: Database operations and persistence
"""

from .repository_service import RepositoryService
from .issue_service import IssueService, get_issue_service
from .task_service import TaskService, get_task_service
from .configuration_service import ConfigurationService, get_configuration_service
from .database_service import DatabaseService, get_database_service

__all__ = [
    'RepositoryService',
    'IssueService', 
    'TaskService',
    'ConfigurationService',
    'DatabaseService',
    'get_issue_service',
    'get_task_service',
    'get_configuration_service',
    'get_database_service'
]