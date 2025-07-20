"""
Centralized logging configuration for Claude Manager Service

This module provides structured logging with configurable levels,
formatters, and handlers for all components of the service.
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
            record.levelname = colored_levelname
        
        return super().format(record)


class ServiceLogger:
    """
    Centralized logger for the Claude Manager Service
    
    Features:
    - Configurable log levels via environment variable
    - Structured JSON logging for production
    - Colored console output for development
    - Automatic log rotation
    - Component-specific loggers
    """
    
    _instance: Optional['ServiceLogger'] = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.log_level = self._get_log_level()
        self.log_dir = self._setup_log_directory()
        self.is_production = os.getenv('ENVIRONMENT', 'development').lower() == 'production'
        
        # Configure root logger
        self._setup_root_logger()
        self._initialized = True
    
    def _get_log_level(self):
        """Get log level from environment variable"""
        level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
        return getattr(logging, level_name, logging.INFO)
    
    def _setup_log_directory(self):
        """Create logs directory if it doesn't exist"""
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        return log_dir
    
    def _setup_root_logger(self):
        """Configure the root logger with appropriate handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        if self.is_production:
            # Production: JSON structured logging
            console_formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"component": "%(name)s", "message": "%(message)s", '
                '"function": "%(funcName)s", "line": %(lineno)d}'
            )
        else:
            # Development: Human-readable with colors
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'claude_manager.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler for errors only
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name):
        """
        Get a logger for a specific component
        
        Args:
            name (str): Logger name, typically the module name
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            self._loggers[name] = logger
        
        return self._loggers[name]


# Global logger instance
_service_logger = ServiceLogger()


def get_logger(name=None):
    """
    Get a logger instance for the current module
    
    Args:
        name (str, optional): Logger name. If None, uses caller's module name.
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Operation completed successfully")
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return _service_logger.get_logger(name)


def configure_github_logging():
    """Configure PyGithub library logging to reduce noise"""
    github_logger = logging.getLogger('github')
    github_logger.setLevel(logging.WARNING)
    
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.WARNING)


def log_function_call(func):
    """
    Decorator to automatically log function entry and exit
    
    Example:
        @log_function_call
        def my_function(arg1, arg2):
            return result
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        args_str = ', '.join([repr(arg) for arg in args[:3]])  # Limit to first 3 args
        if len(args) > 3:
            args_str += ', ...'
        
        kwargs_str = ', '.join([f'{k}={repr(v)}' for k, v in list(kwargs.items())[:3]])
        if len(kwargs) > 3:
            kwargs_str += ', ...'
        
        logger.debug(f"Entering {func.__name__}({args_str}, {kwargs_str})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} -> {type(result).__name__}")
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function execution time
    
    Example:
        @log_performance
        def slow_function():
            time.sleep(1)
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


# Configure third-party library logging
configure_github_logging()


# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test decorators
    @log_function_call
    @log_performance
    def test_function(x, y=10):
        import time
        time.sleep(0.1)
        return x + y
    
    result = test_function(5, y=15)
    logger.info(f"Test function result: {result}")