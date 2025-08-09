"""
Configuration validation module for Claude Manager Service

This module provides functionality to validate the config.json file
structure and contents to ensure proper operation of the service.
"""
import json
import os
import re
from typing import Dict, Any, List
from .logger import get_logger

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails"""
    pass


def validate_repo_name(repo_name: str, field_name: str = "repository") -> None:
    """
    Validate repository name format (owner/repo)
    
    Args:
        repo_name (str): Repository name to validate
        field_name (str): Name of the field being validated (for error messages)
        
    Raises:
        ConfigValidationError: If repository name format is invalid
    """
    repo_pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$'
    if not re.match(repo_pattern, repo_name):
        raise ConfigValidationError(
            f"Invalid {field_name} format: '{repo_name}'. "
            f"Expected format: 'owner/repository'"
        )


def validate_github_section(github_config: Dict[str, Any]) -> None:
    """
    Validate the github section of configuration
    
    Args:
        github_config (Dict[str, Any]): GitHub configuration section
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Required fields
    required_fields = ['username', 'managerRepo', 'reposToScan']
    
    for field in required_fields:
        if field not in github_config:
            raise ConfigValidationError(
                f"Missing required field 'github.{field}' in configuration"
            )
    
    # Validate username
    username = github_config['username']
    if not isinstance(username, str) or not username.strip():
        raise ConfigValidationError(
            "github.username must be a non-empty string"
        )
    
    # Validate managerRepo
    manager_repo = github_config['managerRepo']
    if not isinstance(manager_repo, str):
        raise ConfigValidationError(
            "github.managerRepo must be a string"
        )
    validate_repo_name(manager_repo, "github.managerRepo")
    
    # Validate reposToScan
    repos_to_scan = github_config['reposToScan']
    if not isinstance(repos_to_scan, list):
        raise ConfigValidationError(
            "github.reposToScan must be an array"
        )
    
    if len(repos_to_scan) == 0:
        raise ConfigValidationError(
            "github.reposToScan cannot be empty"
        )
    
    for i, repo in enumerate(repos_to_scan):
        if not isinstance(repo, str):
            raise ConfigValidationError(
                f"github.reposToScan[{i}] must be a string"
            )
        validate_repo_name(repo, f"github.reposToScan[{i}]")


def validate_analyzer_section(analyzer_config: Dict[str, Any]) -> None:
    """
    Validate the analyzer section of configuration
    
    Args:
        analyzer_config (Dict[str, Any]): Analyzer configuration section
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Required fields
    required_fields = ['scanForTodos', 'scanOpenIssues']
    
    for field in required_fields:
        if field not in analyzer_config:
            raise ConfigValidationError(
                f"Missing required field 'analyzer.{field}' in configuration"
            )
    
    # Validate boolean fields
    for field in ['scanForTodos', 'scanOpenIssues']:
        value = analyzer_config[field]
        if not isinstance(value, bool):
            raise ConfigValidationError(
                f"analyzer.{field} must be a boolean (true/false)"
            )
    
    # Validate optional fields
    if 'cleanupTasksOlderThanDays' in analyzer_config:
        cleanup_days = analyzer_config['cleanupTasksOlderThanDays']
        if not isinstance(cleanup_days, int):
            raise ConfigValidationError(
                "analyzer.cleanupTasksOlderThanDays must be an integer"
            )
        if cleanup_days < 1:
            raise ConfigValidationError(
                "analyzer.cleanupTasksOlderThanDays must be a positive integer"
            )


def validate_executor_section(executor_config: Dict[str, Any]) -> None:
    """
    Validate the executor section of configuration
    
    Args:
        executor_config (Dict[str, Any]): Executor configuration section
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Required fields
    required_fields = ['terragonUsername']
    
    for field in required_fields:
        if field not in executor_config:
            raise ConfigValidationError(
                f"Missing required field 'executor.{field}' in configuration"
            )
    
    # Validate terragonUsername
    terragon_username = executor_config['terragonUsername']
    if not isinstance(terragon_username, str) or not terragon_username.strip():
        raise ConfigValidationError(
            "executor.terragonUsername must be a non-empty string"
        )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the complete configuration structure and contents
    
    Args:
        config (Dict[str, Any]): Complete configuration dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Required top-level sections
    required_sections = ['github', 'analyzer', 'executor']
    
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(
                f"Missing required configuration section: '{section}'"
            )
        
        if not isinstance(config[section], dict):
            raise ConfigValidationError(
                f"Configuration section '{section}' must be an object"
            )
    
    # Validate individual sections
    try:
        validate_github_section(config['github'])
        validate_analyzer_section(config['analyzer'])
        validate_executor_section(config['executor'])
        
        logger.info("Configuration validation successful")
        
    except ConfigValidationError:
        logger.error("Configuration validation failed")
        raise


def load_and_validate_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Load configuration from file and validate it
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Validated configuration dictionary
        
    Raises:
        ConfigValidationError: If file doesn't exist, JSON is invalid, or validation fails
    """
    logger.debug(f"Loading configuration from {config_path}")
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise ConfigValidationError(
            f"Configuration file '{config_path}' not found"
        )
    
    # Load and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigValidationError(
            f"Invalid JSON in configuration file '{config_path}': {e}"
        )
    except Exception as e:
        raise ConfigValidationError(
            f"Error reading configuration file '{config_path}': {e}"
        )
    
    # Validate configuration
    validate_config(config)
    
    logger.info(f"Successfully loaded and validated configuration from {config_path}")
    return config


def get_validated_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Get validated configuration with helpful error handling
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Validated configuration dictionary
        
    Raises:
        SystemExit: If configuration validation fails (for production use)
    """
    try:
        return load_and_validate_config(config_path)
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error("Please check your configuration file and try again")
        
        # Provide helpful guidance
        logger.info("Example configuration structure:")
        logger.info("""
{
  "github": {
    "username": "your-github-username",
    "managerRepo": "username/manager-repo",
    "reposToScan": ["username/repo1", "username/repo2"]
  },
  "analyzer": {
    "scanForTodos": true,
    "scanOpenIssues": true,
    "cleanupTasksOlderThanDays": 90
  },
  "executor": {
    "terragonUsername": "@terragon-labs"
  }
}
        """)
        
        raise SystemExit(1)


# Example usage and testing
if __name__ == "__main__":
    try:
        config = load_and_validate_config('config.json')
        print("✅ Configuration is valid!")
        print(f"Found {len(config['github']['reposToScan'])} repositories to scan")
        print(f"Manager repository: {config['github']['managerRepo']}")
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
    except SystemExit:
        pass