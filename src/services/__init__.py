"""
Service Layer for Claude Manager Service

This package provides a clean service layer architecture with clear
separation of concerns and dependency injection capabilities.

Services:
- RepositoryService: Repository scanning and management
- IssueService: Issue creation and tracking
- TaskService: Task orchestration and execution
- ConfigurationService: Centralized configuration management
"""

from .repository_service import RepositoryService
# from .issue_service import IssueService  # TODO: Create IssueService
# from .task_service import TaskService    # TODO: Create TaskService
from .configuration_service import ConfigurationService

__all__ = [
    'RepositoryService',
    # 'IssueService', 
    # 'TaskService',
    'ConfigurationService'
]