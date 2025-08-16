"""
Configuration Service for Claude Manager Service

This service provides centralized configuration management with
validation, environment variable integration, and hot-reloading support.

Features:
- Centralized configuration access
- Environment variable integration
- Configuration validation and defaults
- Hot-reloading capabilities
- Async configuration operations
"""

import asyncio
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from src.logger import get_logger
from src.async_file_operations import load_json_async, save_json_async, file_exists_async
from src.validation import ConfigurationValidationError
from src.error_handler import FileOperationError, JsonParsingError
from src.config_env import get_env_config


logger = get_logger(__name__)


@dataclass
class ConfigurationSchema:
    """Schema definition for configuration validation"""
    required_fields: List[str]
    optional_fields: Dict[str, Any]  # field_name -> default_value
    nested_schemas: Dict[str, 'ConfigurationSchema']


class ConfigurationService:
    """
    Centralized configuration service with validation and environment integration
    
    This service manages all configuration needs for the application including
    file-based config, environment variables, and runtime configuration.
    """
    
    def __init__(self, config_path: str = 'config.json', auto_reload: bool = False):
        """
        Initialize configuration service
        
        Args:
            config_path: Path to main configuration file
            auto_reload: Enable automatic configuration reloading
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path
        self.auto_reload = auto_reload
        
        # Configuration data
        self._config_data: Dict[str, Any] = {}
        self._env_config = None
        self._last_modified: Optional[datetime] = None
        
        # Configuration schema
        self._schema = self._define_schema()
        
        # Initialization flag
        self._initialized = False
        
        self.logger.info(f"Configuration service created for: {config_path}")
    
    async def initialize(self) -> None:
        """Initialize the configuration service asynchronously"""
        if self._initialized:
            return
        
        try:
            # Load environment configuration
            self._env_config = get_env_config()
            self.logger.debug("Environment configuration loaded")
            
            # Load file-based configuration
            await self._load_config_file()
            
            # Merge configurations
            await self._merge_configurations()
            
            # Validate final configuration
            self._validate_configuration()
            
            self._initialized = True
            self.logger.info("Configuration service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration service: {e}")
            raise ConfigurationValidationError(f"Configuration initialization failed: {str(e)}")
    
    async def get_config(self, key_path: str = None) -> Union[Dict[str, Any], Any]:
        """
        Get configuration value by key path
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'github.managerRepo')
                     If None, returns entire configuration
                     
        Returns:
            Configuration value or dictionary
            
        Raises:
            ConfigurationValidationError: If key path not found
        """
        if not self._initialized:
            await self.initialize()
        
        if key_path is None:
            return self._config_data.copy()
        
        # Navigate nested configuration
        keys = key_path.split('.')
        current = self._config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise ConfigurationValidationError(f"Configuration key not found: {key_path}")
        
        return current
    
    async def set_config(self, key_path: str, value: Any, persist: bool = False) -> None:
        """
        Set configuration value
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
            persist: Whether to persist to configuration file
            
        Raises:
            ConfigurationValidationError: If validation fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Navigate to parent and set value
        keys = key_path.split('.')
        current = self._config_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        
        # Validate updated configuration
        self._validate_configuration()
        
        if persist:
            await self._save_config_file()
        
        self.logger.debug(f"Configuration updated: {key_path} = {value}")
    
    async def get_github_config(self) -> Dict[str, Any]:
        """Get GitHub-specific configuration"""
        return await self.get_config('github')
    
    async def get_manager_repo(self) -> str:
        """Get manager repository name"""
        return await self.get_config('github.managerRepo')
    
    async def get_repos_to_scan(self) -> List[str]:
        """Get list of repositories to scan"""
        return await self.get_config('github.reposToScan')
    
    async def get_analyzer_config(self) -> Dict[str, Any]:
        """Get analyzer configuration"""
        try:
            return await self.get_config('analyzer')
        except ConfigurationValidationError:
            # Return defaults if analyzer config not found
            return {
                'maxConcurrentScans': 5,
                'scanTimeoutSeconds': 300,
                'maxTodosPerFile': 3
            }
    
    async def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration"""
        try:
            return await self.get_config('performance')
        except ConfigurationValidationError:
            # Return defaults
            return {
                'enabled': True,
                'retentionDays': 30,
                'alertThresholds': {
                    'apiCallFailureRate': 0.1,
                    'memoryUsageMB': 512
                }
            }
    
    async def reload_configuration(self) -> bool:
        """
        Reload configuration from file
        
        Returns:
            True if configuration was reloaded, False if no changes
        """
        try:
            # Check if file was modified
            if await self._is_config_file_modified():
                await self._load_config_file()
                await self._merge_configurations()
                self._validate_configuration()
                
                self.logger.info("Configuration reloaded successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            raise ConfigurationValidationError(f"Configuration reload failed: {str(e)}")
    
    async def export_configuration(self, export_path: str) -> None:
        """
        Export current configuration to file
        
        Args:
            export_path: Path to export configuration to
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'config_source': self.config_path,
                'configuration': self._config_data
            }
            
            await save_json_async(export_path, export_data)
            self.logger.info(f"Configuration exported to: {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            raise
    
    def get_environment_config(self) -> Any:
        """Get environment configuration object"""
        return self._env_config
    
    async def _load_config_file(self) -> None:
        """Load configuration from file"""
        if not await file_exists_async(self.config_path):
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            self._config_data = {}
            return
        
        try:
            self._config_data = await load_json_async(self.config_path)
            
            # Update last modified timestamp
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(None, lambda: Path(self.config_path).stat())
            self._last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            self.logger.debug(f"Configuration loaded from: {self.config_path}")
            
        except (FileOperationError, JsonParsingError) as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            raise ConfigurationValidationError(f"Configuration file error: {str(e)}")
    
    async def _save_config_file(self) -> None:
        """Save configuration to file"""
        try:
            await save_json_async(self.config_path, self._config_data)
            
            # Update last modified timestamp
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(None, lambda: Path(self.config_path).stat())
            self._last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            self.logger.debug(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration file: {e}")
            raise
    
    async def _merge_configurations(self) -> None:
        """Merge file and environment configurations"""
        # Environment configuration takes precedence
        if self._env_config:
            # Add environment-specific overrides
            env_overrides = {
                'performance': {
                    'enabled': self._env_config.enable_performance_monitoring,
                    'retentionDays': self._env_config.perf_retention_days
                },
                'security': {
                    'maxContentLength': self._env_config.security_max_content_length
                },
                'rateLimiting': {
                    'maxRequests': self._env_config.rate_limit_max_requests,
                    'timeWindow': self._env_config.rate_limit_time_window
                }
            }
            
            # Deep merge environment overrides
            self._config_data = self._deep_merge(self._config_data, env_overrides)
        
        self.logger.debug("Configuration merged with environment variables")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _is_config_file_modified(self) -> bool:
        """Check if configuration file was modified since last load"""
        if not self._last_modified or not await file_exists_async(self.config_path):
            return True
        
        try:
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(None, lambda: Path(self.config_path).stat())
            current_modified = datetime.fromtimestamp(stat.st_mtime)
            
            return current_modified > self._last_modified
            
        except Exception:
            return True
    
    def _define_schema(self) -> ConfigurationSchema:
        """Define configuration schema for validation"""
        github_schema = ConfigurationSchema(
            required_fields=['managerRepo'],
            optional_fields={
                'reposToScan': [],
                'apiTimeout': 30
            },
            nested_schemas={}
        )
        
        return ConfigurationSchema(
            required_fields=['github'],
            optional_fields={
                'analyzer': {
                    'maxConcurrentScans': 5,
                    'scanTimeoutSeconds': 300
                },
                'performance': {
                    'enabled': True,
                    'retentionDays': 30
                }
            },
            nested_schemas={
                'github': github_schema
            }
        )
    
    def _validate_configuration(self) -> None:
        """Validate configuration against schema"""
        try:
            self._validate_schema(self._config_data, self._schema, "root")
            self.logger.debug("Configuration validation passed")
            
        except Exception as e:
            raise ConfigurationValidationError(f"Configuration validation failed: {str(e)}")
    
    def _validate_schema(self, data: Dict[str, Any], schema: ConfigurationSchema, path: str) -> None:
        """Validate data against schema recursively"""
        # Check required fields
        for field in schema.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {path}.{field}")
        
        # Validate nested schemas
        for field, nested_schema in schema.nested_schemas.items():
            if field in data:
                self._validate_schema(data[field], nested_schema, f"{path}.{field}")


# Global configuration service instance
_config_service: Optional[ConfigurationService] = None


async def get_configuration_service(config_path: str = 'config.json') -> ConfigurationService:
    """
    Get global configuration service instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized configuration service
    """
    global _config_service
    
    if _config_service is None:
        _config_service = ConfigurationService(config_path)
        await _config_service.initialize()
    
    return _config_service


# Convenience functions
async def get_config(key_path: str = None) -> Union[Dict[str, Any], Any]:
    """Convenience function to get configuration"""
    service = await get_configuration_service()
    return await service.get_config(key_path)


async def get_github_config() -> Dict[str, Any]:
    """Convenience function to get GitHub configuration"""
    service = await get_configuration_service()
    return await service.get_github_config()


# Example usage and testing
async def example_configuration_service():
    """Example of using configuration service"""
    try:
        # Get configuration service
        config_service = await get_configuration_service()
        
        # Get various configuration values
        github_config = await config_service.get_github_config()
        manager_repo = await config_service.get_manager_repo()
        analyzer_config = await config_service.get_analyzer_config()
        
        logger.info(f"Configuration service example completed")
        logger.info(f"Manager repo: {manager_repo}")
        logger.info(f"Analyzer config: {analyzer_config}")
        
    except Exception as e:
        logger.error(f"Configuration service example failed: {e}")
        raise


if __name__ == "__main__":
    # Test configuration service
    asyncio.run(example_configuration_service())