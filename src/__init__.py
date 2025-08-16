"""Claude Manager Service Package

An autonomous software development lifecycle management system that scans repositories,
proposes tasks via GitHub issues, and executes them using AI agents.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "dev@terragon.ai"
__license__ = "MIT"

# Core components that should always be available
from src.logger import get_logger
try:
    from src.performance_monitor import get_monitor
except ImportError:
    get_monitor = None  # Disabled due to missing psutil dependency

# Core services with error handling for missing dependencies
_available_services = []
_available_components = []

# Try importing core components
try:
    from src.github_api import GitHubAPI
    _available_components.append('GitHubAPI')
except ImportError:
    GitHubAPI = None

try:
    from src.orchestrator import Orchestrator
    _available_components.append('Orchestrator')
except ImportError:
    Orchestrator = None

try:
    from src.task_analyzer import analyze_open_issues, find_todo_comments
    _available_components.extend(['analyze_open_issues', 'find_todo_comments'])
except ImportError:
    analyze_open_issues = None
    find_todo_comments = None

try:
    from src.core_system import CoreSystem
    _available_components.append('CoreSystem')
except ImportError:
    CoreSystem = None

# Try importing services (these may fail due to external dependencies like aiofiles)
try:
    from src.services.configuration_service import ConfigurationService
    from src.services.database_service import DatabaseService
    from src.services.issue_service import IssueService
    from src.services.repository_service import RepositoryService
    from src.services.task_service import TaskService
    _available_services.extend([
        'ConfigurationService', 'DatabaseService', 'IssueService',
        'RepositoryService', 'TaskService'
    ])
except ImportError:
    # Services require external dependencies, mark as None
    ConfigurationService = None
    DatabaseService = None
    IssueService = None
    RepositoryService = None
    TaskService = None

# Build dynamic __all__ based on what's actually available
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "get_logger",
    "get_monitor",
] + _available_components + _available_services