"""
Environment variable configuration utilities

This module provides utilities for managing environment variable configuration
with validation, defaults, and proper error handling.
"""
import os
import sys
from typing import Dict, Any, Union, Optional, Callable
from src.logger import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails"""
    pass


def get_env_int(key: str, default: int, min_value: int = None, max_value: int = None) -> int:
    """
    Get integer value from environment variable with validation
    
    Args:
        key: Environment variable key
        default: Default value if not set
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        int: Validated integer value
        
    Raises:
        ConfigurationError: If value is invalid
    """
    value_str = os.getenv(key)
    
    if value_str is None:
        logger.debug(f"Environment variable {key} not set, using default: {default}")
        return default
    
    try:
        value = int(value_str)
    except ValueError:
        raise ConfigurationError(f"Environment variable {key}='{value_str}' is not a valid integer")
    
    if min_value is not None and value < min_value:
        raise ConfigurationError(f"Environment variable {key}={value} is below minimum value {min_value}")
    
    if max_value is not None and value > max_value:
        raise ConfigurationError(f"Environment variable {key}={value} exceeds maximum value {max_value}")
    
    logger.debug(f"Environment variable {key} set to: {value}")
    return value


def get_env_float(key: str, default: float, min_value: float = None, max_value: float = None) -> float:
    """
    Get float value from environment variable with validation
    
    Args:
        key: Environment variable key
        default: Default value if not set
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        float: Validated float value
        
    Raises:
        ConfigurationError: If value is invalid
    """
    value_str = os.getenv(key)
    
    if value_str is None:
        logger.debug(f"Environment variable {key} not set, using default: {default}")
        return default
    
    try:
        value = float(value_str)
    except ValueError:
        raise ConfigurationError(f"Environment variable {key}='{value_str}' is not a valid float")
    
    if min_value is not None and value < min_value:
        raise ConfigurationError(f"Environment variable {key}={value} is below minimum value {min_value}")
    
    if max_value is not None and value > max_value:
        raise ConfigurationError(f"Environment variable {key}={value} exceeds maximum value {max_value}")
    
    logger.debug(f"Environment variable {key} set to: {value}")
    return value


def get_env_bool(key: str, default: bool) -> bool:
    """
    Get boolean value from environment variable
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        bool: Boolean value
    """
    value_str = os.getenv(key)
    
    if value_str is None:
        logger.debug(f"Environment variable {key} not set, using default: {default}")
        return default
    
    value_str = value_str.lower()
    if value_str in ('true', '1', 'yes', 'on'):
        value = True
    elif value_str in ('false', '0', 'no', 'off'):
        value = False
    else:
        raise ConfigurationError(f"Environment variable {key}='{value_str}' is not a valid boolean")
    
    logger.debug(f"Environment variable {key} set to: {value}")
    return value


def get_env_string(key: str, default: str, allowed_values: Optional[list] = None) -> str:
    """
    Get string value from environment variable with validation
    
    Args:
        key: Environment variable key
        default: Default value if not set
        allowed_values: List of allowed values (optional)
        
    Returns:
        str: Validated string value
        
    Raises:
        ConfigurationError: If value is invalid
    """
    value = os.getenv(key, default)
    
    if allowed_values is not None and value not in allowed_values:
        raise ConfigurationError(
            f"Environment variable {key}='{value}' must be one of: {allowed_values}"
        )
    
    if value == default:
        logger.debug(f"Environment variable {key} not set, using default: '{default}'")
    else:
        logger.debug(f"Environment variable {key} set to: '{value}'")
    
    return value


class EnvironmentConfig:
    """
    Centralized environment configuration management
    """
    
    def __init__(self):
        """Initialize configuration with validation"""
        self.logger = get_logger(f"{__name__}.EnvironmentConfig")
        self._load_configuration()
    
    def _load_configuration(self):
        """Load and validate all environment configuration"""
        try:
            # Performance monitoring configuration
            self.perf_max_operations = get_env_int('PERF_MAX_OPERATIONS', 10000, min_value=1000, max_value=100000)
            self.perf_retention_days = get_env_int('PERF_RETENTION_DAYS', 30, min_value=1, max_value=365)
            self.perf_alert_duration = get_env_float('PERF_ALERT_DURATION', 30.0, min_value=0.1, max_value=3600.0)
            self.perf_alert_error_rate = get_env_float('PERF_ALERT_ERROR_RATE', 0.1, min_value=0.0, max_value=1.0)
            
            # Rate limiting configuration
            self.rate_limit_max_requests = get_env_int('RATE_LIMIT_MAX_REQUESTS', 5000, min_value=100, max_value=50000)
            self.rate_limit_time_window = get_env_float('RATE_LIMIT_TIME_WINDOW', 3600.0, min_value=60.0, max_value=86400.0)
            
            # Security configuration
            self.security_max_content_length = get_env_int('SECURITY_MAX_CONTENT_LENGTH', 50000, min_value=1000, max_value=100000)
            self.security_enhanced_max_content_length = get_env_int('SECURITY_ENHANCED_MAX_CONTENT_LENGTH', 60000, min_value=1000, max_value=100000)
            
            # Log level configuration
            self.log_level = get_env_string('LOG_LEVEL', 'INFO', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
            
            # Feature flags
            self.enable_performance_monitoring = get_env_bool('ENABLE_PERFORMANCE_MONITORING', True)
            self.enable_rate_limiting = get_env_bool('ENABLE_RATE_LIMITING', True)
            self.enable_enhanced_security = get_env_bool('ENABLE_ENHANCED_SECURITY', True)
            
            self.logger.info("Environment configuration loaded successfully")
            
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e}")
            sys.exit(1)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration
        
        Returns:
            Dict[str, Any]: Configuration summary
        """
        return {
            'performance': {
                'max_operations': self.perf_max_operations,
                'retention_days': self.perf_retention_days,
                'alert_duration': self.perf_alert_duration,
                'alert_error_rate': self.perf_alert_error_rate,
                'monitoring_enabled': self.enable_performance_monitoring
            },
            'rate_limiting': {
                'max_requests': self.rate_limit_max_requests,
                'time_window': self.rate_limit_time_window,
                'enabled': self.enable_rate_limiting
            },
            'security': {
                'max_content_length': self.security_max_content_length,
                'enhanced_max_content_length': self.security_enhanced_max_content_length,
                'enhanced_security_enabled': self.enable_enhanced_security
            },
            'logging': {
                'level': self.log_level
            }
        }
    
    def validate_runtime_config(self) -> bool:
        """
        Perform runtime configuration validation
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate inter-dependencies
            if self.security_enhanced_max_content_length <= self.security_max_content_length:
                self.logger.warning(
                    "SECURITY_ENHANCED_MAX_CONTENT_LENGTH should be greater than SECURITY_MAX_CONTENT_LENGTH"
                )
            
            # Validate reasonable rate limiting
            if self.rate_limit_max_requests / (self.rate_limit_time_window / 60) > 1000:
                self.logger.warning(
                    f"Rate limit is very high: {self.rate_limit_max_requests} requests per "
                    f"{self.rate_limit_time_window/60:.1f} minutes"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Runtime configuration validation failed: {e}")
            return False


# Global configuration instance
_env_config = None


def get_env_config() -> EnvironmentConfig:
    """
    Get global environment configuration instance
    
    Returns:
        EnvironmentConfig: Global configuration instance
    """
    global _env_config
    if _env_config is None:
        _env_config = EnvironmentConfig()
    return _env_config


if __name__ == "__main__":
    # Test configuration loading
    config = EnvironmentConfig()
    print("Configuration loaded successfully!")
    
    import json
    summary = config.get_config_summary()
    print(json.dumps(summary, indent=2))
    
    # Test validation
    is_valid = config.validate_runtime_config()
    print(f"Configuration is valid: {is_valid}")