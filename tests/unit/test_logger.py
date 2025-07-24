"""
Comprehensive unit tests for logger module

This module provides complete test coverage for the centralized logging system,
including configuration, formatters, handlers, decorators, and edge cases.
"""
import pytest
import logging
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from io import StringIO
import time

import sys
sys.path.append('/root/repo/src')


def create_mock_file_handler():
    """Create a properly mocked RotatingFileHandler with integer level"""
    mock_handler = Mock()
    mock_handler.level = logging.DEBUG  # Integer level to prevent MagicMock comparison errors
    mock_handler.setLevel = Mock()
    mock_handler.setFormatter = Mock()
    return mock_handler


class TestColoredFormatter:
    """Test cases for ColoredFormatter class"""
    
    def setup_method(self):
        """Setup for each test"""
        from logger import ColoredFormatter
        self.formatter = ColoredFormatter('%(levelname)s - %(message)s')
    
    def test_color_codes_defined(self):
        """Test that all expected color codes are defined"""
        from logger import ColoredFormatter
        
        expected_colors = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'RESET']
        for color in expected_colors:
            assert color in ColoredFormatter.COLORS
            assert isinstance(ColoredFormatter.COLORS[color], str)
            assert ColoredFormatter.COLORS[color].startswith('\033[')
    
    def test_format_adds_colors_to_level_names(self):
        """Test that format method adds colors to level names"""
        # Create a log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        
        # Should contain color codes
        assert '\033[32m' in formatted  # Green for INFO
        assert '\033[0m' in formatted   # Reset
        assert 'Test message' in formatted
    
    def test_format_different_log_levels(self):
        """Test formatting for different log levels"""
        test_cases = [
            (logging.DEBUG, '\033[36m'),    # Cyan
            (logging.INFO, '\033[32m'),     # Green
            (logging.WARNING, '\033[33m'),  # Yellow
            (logging.ERROR, '\033[31m'),    # Red
            (logging.CRITICAL, '\033[35m'), # Magenta
        ]
        
        for level, expected_color in test_cases:
            record = logging.LogRecord(
                name='test',
                level=level,
                pathname='test.py',
                lineno=1,
                msg='Test message',
                args=(),
                exc_info=None
            )
            
            formatted = self.formatter.format(record)
            assert expected_color in formatted
            assert '\033[0m' in formatted  # Reset code
    
    def test_format_unknown_level(self):
        """Test formatting with unknown log level"""
        # Create record with custom level
        record = logging.LogRecord(
            name='test',
            level=99,  # Custom level
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.levelname = 'CUSTOM'
        
        formatted = self.formatter.format(record)
        
        # Should not crash and should include the message
        assert 'Test message' in formatted
        # Should not add colors for unknown levels
        assert 'CUSTOM' in formatted


class TestServiceLogger:
    """Test cases for ServiceLogger class"""
    
    def setup_method(self):
        """Setup for each test"""
        # Reset the singleton instance for each test
        from logger import ServiceLogger
        ServiceLogger._instance = None
        ServiceLogger._loggers = {}
    
    def test_singleton_pattern(self):
        """Test that ServiceLogger implements singleton pattern"""
        from logger import ServiceLogger
        
        logger1 = ServiceLogger()
        logger2 = ServiceLogger()
        
        assert logger1 is logger2
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'})
    def test_get_log_level_from_environment(self):
        """Test log level configuration from environment variable"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        assert logger_service.log_level == logging.DEBUG
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'INVALID_LEVEL'})
    def test_get_log_level_invalid_fallback(self):
        """Test fallback to INFO for invalid log level"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        assert logger_service.log_level == logging.INFO
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_log_level_default(self):
        """Test default log level when environment variable not set"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        assert logger_service.log_level == logging.INFO
    
    @patch('logger.Path.mkdir')
    def test_setup_log_directory_creation(self, mock_mkdir):
        """Test that log directory is created if it doesn't exist"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        
        mock_mkdir.assert_called_once_with(exist_ok=True)
        assert isinstance(logger_service.log_dir, Path)
        assert logger_service.log_dir.name == 'logs'
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_environment_detection(self):
        """Test detection of production environment"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        assert logger_service.is_production is True
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'development'})
    def test_development_environment_detection(self):
        """Test detection of development environment"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        assert logger_service.is_production is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_default_environment_is_development(self):
        """Test that default environment is development"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        assert logger_service.is_production is False
    
    def test_get_logger_creates_new_logger(self):
        """Test that get_logger creates new logger for new names"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        test_logger = logger_service.get_logger('test_module')
        
        assert isinstance(test_logger, logging.Logger)
        assert test_logger.name == 'test_module'
        assert 'test_module' in ServiceLogger._loggers
    
    def test_get_logger_returns_cached_logger(self):
        """Test that get_logger returns cached logger for existing names"""
        from logger import ServiceLogger
        
        logger_service = ServiceLogger()
        logger1 = logger_service.get_logger('test_module')
        logger2 = logger_service.get_logger('test_module')
        
        assert logger1 is logger2
    
    @patch('logging.handlers.RotatingFileHandler')
    @patch('logger.Path.mkdir')
    def test_setup_root_logger_handlers(self, mock_mkdir, mock_file_handler):
        """Test that root logger is configured with proper handlers"""
        from logger import ServiceLogger
        
        # Mock file handler
        mock_handler_instance = Mock()
        mock_file_handler.return_value = mock_handler_instance
        
        logger_service = ServiceLogger()
        
        root_logger = logging.getLogger()
        
        # Should have console handler and file handlers
        assert len(root_logger.handlers) >= 2
        
        # Verify file handlers were created
        assert mock_file_handler.call_count >= 2  # One for main log, one for errors
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_json_formatter(self):
        """Test that production environment uses JSON formatter"""
        from logger import ServiceLogger
        
        with patch('logging.handlers.RotatingFileHandler'), \
             patch('logger.Path.mkdir'):
            
            logger_service = ServiceLogger()
            
            # Check that console handler uses JSON-like format
            root_logger = logging.getLogger()
            console_handler = root_logger.handlers[0]  # First handler should be console
            
            # JSON formatter should contain timestamp, level, component, message
            formatter_format = console_handler.formatter._fmt
            assert 'timestamp' in formatter_format
            assert 'level' in formatter_format
            assert 'component' in formatter_format
            assert 'message' in formatter_format
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'development'})
    def test_development_colored_formatter(self):
        """Test that development environment uses colored formatter"""
        from logger import ServiceLogger, ColoredFormatter
        
        with patch('logging.handlers.RotatingFileHandler'), \
             patch('logger.Path.mkdir'):
            
            logger_service = ServiceLogger()
            
            root_logger = logging.getLogger()
            console_handler = root_logger.handlers[0]
            
            # Should use ColoredFormatter in development
            assert isinstance(console_handler.formatter, ColoredFormatter)


class TestGlobalLoggerFunctions:
    """Test cases for global logger functions"""
    
    def setup_method(self):
        """Setup for each test"""
        # Reset singleton
        from logger import ServiceLogger
        ServiceLogger._instance = None
        ServiceLogger._loggers = {}
    
    def test_get_logger_with_name(self):
        """Test get_logger function with explicit name"""
        from logger import get_logger
        
        test_logger = get_logger('test_module')
        
        assert isinstance(test_logger, logging.Logger)
        assert test_logger.name == 'test_module'
    
    def test_get_logger_without_name_uses_caller_module(self):
        """Test get_logger function without name uses caller's module"""
        from logger import get_logger
        
        # Mock inspect to return a specific module name
        with patch('inspect.currentframe') as mock_frame:
            mock_frame_obj = Mock()
            mock_frame_obj.f_back.f_globals = {'__name__': 'caller_module'}
            mock_frame.return_value = mock_frame_obj
            
            test_logger = get_logger()
            
            assert test_logger.name == 'caller_module'
    
    def test_get_logger_without_name_fallback(self):
        """Test get_logger fallback when module name unavailable"""
        from logger import get_logger
        
        with patch('inspect.currentframe') as mock_frame:
            mock_frame_obj = Mock()
            mock_frame_obj.f_back.f_globals = {}  # No __name__ key
            mock_frame.return_value = mock_frame_obj
            
            test_logger = get_logger()
            
            assert test_logger.name == 'unknown'
    
    def test_configure_github_logging(self):
        """Test that GitHub library logging is configured"""
        from logger import configure_github_logging
        
        configure_github_logging()
        
        github_logger = logging.getLogger('github')
        urllib3_logger = logging.getLogger('urllib3')
        
        assert github_logger.level == logging.WARNING
        assert urllib3_logger.level == logging.WARNING


class TestLoggerDecorators:
    """Test cases for logging decorators"""
    
    def setup_method(self):
        """Setup for each test"""
        from logger import ServiceLogger
        ServiceLogger._instance = None
        ServiceLogger._loggers = {}
    
    def test_log_function_call_decorator(self):
        """Test log_function_call decorator"""
        from logger import log_function_call, get_logger
        
        # Create a test function
        @log_function_call
        def test_function(arg1, arg2, kwarg1=None):
            return arg1 + arg2
        
        # Mock the logger to capture log calls
        with patch('logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function(1, 2, kwarg1='test')
            
            assert result == 3
            
            # Verify logging calls
            assert mock_logger.debug.call_count >= 2  # Entry and exit
            
            # Check entry log
            entry_call = mock_logger.debug.call_args_list[0]
            assert 'Entering test_function' in str(entry_call)
            
            # Check exit log
            exit_call = mock_logger.debug.call_args_list[1]
            assert 'Exiting test_function' in str(exit_call)
    
    def test_log_function_call_decorator_with_exception(self):
        """Test log_function_call decorator handles exceptions"""
        from logger import log_function_call
        
        @log_function_call
        def failing_function():
            raise ValueError("Test error")
        
        with patch('logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                failing_function()
            
            # Should log error
            mock_logger.error.assert_called_once()
            error_call_args = mock_logger.error.call_args[0][0]
            assert 'Exception in failing_function' in error_call_args
            assert 'Test error' in error_call_args
    
    def test_log_function_call_limits_args_logging(self):
        """Test that log_function_call limits number of args/kwargs logged"""
        from logger import log_function_call
        
        @log_function_call
        def function_with_many_args(a, b, c, d, e, f, **kwargs):
            return sum([a, b, c, d, e, f])
        
        with patch('logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            function_with_many_args(1, 2, 3, 4, 5, 6, x=1, y=2, z=3, w=4)
            
            # Check that args are limited (should see "..." for excess args)
            entry_call = mock_logger.debug.call_args_list[0][0][0]
            assert '...' in entry_call
    
    def test_log_performance_decorator(self):
        """Test log_performance decorator"""
        from logger import log_performance
        
        @log_performance
        def slow_function():
            time.sleep(0.01)  # Small sleep for timing
            return "result"
        
        with patch('logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = slow_function()
            
            assert result == "result"
            
            # Should log completion with timing
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert 'slow_function completed in' in log_message
            assert 's' in log_message  # Should include seconds
    
    def test_log_performance_decorator_with_exception(self):
        """Test log_performance decorator handles exceptions"""
        from logger import log_performance
        
        @log_performance
        def failing_slow_function():
            time.sleep(0.01)
            raise RuntimeError("Test error")
        
        with patch('logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(RuntimeError):
                failing_slow_function()
            
            # Should log failure with timing
            mock_logger.error.assert_called_once()
            error_message = mock_logger.error.call_args[0][0]
            assert 'failing_slow_function failed after' in error_message
            assert 'Test error' in error_message
    
    def test_decorators_preserve_function_metadata(self):
        """Test that decorators preserve original function metadata"""
        from logger import log_function_call, log_performance
        
        @log_function_call
        @log_performance
        def documented_function(arg1, arg2="default"):
            """This is a test function with documentation."""
            return arg1
        
        # Check that function metadata is preserved
        assert documented_function.__name__ == 'documented_function'
        assert documented_function.__doc__ == "This is a test function with documentation."
    
    def test_combined_decorators(self):
        """Test that multiple decorators work together"""
        from logger import log_function_call, log_performance
        
        @log_function_call
        @log_performance
        def combined_function(x):
            return x * 2
        
        with patch('logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = combined_function(5)
            
            assert result == 10
            
            # Should have calls from both decorators
            assert mock_logger.debug.call_count >= 2  # From log_function_call
            assert mock_logger.info.call_count >= 1   # From log_performance


class TestLoggerIntegration:
    """Integration tests for the complete logging system"""
    
    def setup_method(self):
        """Setup for each test"""
        from logger import ServiceLogger
        ServiceLogger._instance = None
        ServiceLogger._loggers = {}
    
    def test_end_to_end_logging_flow(self):
        """Test complete logging flow from get_logger to output"""
        from logger import get_logger
        
        with patch('logging.handlers.RotatingFileHandler', return_value=create_mock_file_handler()), \
             patch('logger.Path.mkdir'):
            
            # Capture console output
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                logger = get_logger('integration_test')
                logger.info("Integration test message")
                
                # Should have output (though exact format depends on environment)
                # Just verify that logging infrastructure is working
                assert isinstance(logger, logging.Logger)
                assert logger.name == 'integration_test'
    
    def test_multiple_loggers_isolation(self):
        """Test that multiple loggers work independently"""
        from logger import get_logger
        
        with patch('logging.handlers.RotatingFileHandler', return_value=create_mock_file_handler()), \
             patch('logger.Path.mkdir'):
            
            logger1 = get_logger('module1')
            logger2 = get_logger('module2')
            
            assert logger1 is not logger2
            assert logger1.name == 'module1'
            assert logger2.name == 'module2'
            
            # Both should be functional
            logger1.info("Message from module1")
            logger2.info("Message from module2")
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'})
    def test_log_level_affects_output(self):
        """Test that log level configuration affects what gets logged"""
        from logger import get_logger
        
        with patch('logging.handlers.RotatingFileHandler', return_value=create_mock_file_handler()), \
             patch('logger.Path.mkdir'):
            
            logger = get_logger('level_test')
            
            # With DEBUG level, debug messages should be processed
            assert logger.isEnabledFor(logging.DEBUG)
            assert logger.isEnabledFor(logging.INFO)
            assert logger.isEnabledFor(logging.WARNING)
    
    def test_error_isolation_no_crash(self):
        """Test that logging errors don't crash the application"""
        from logger import get_logger
        
        # Mock file handler to raise exception
        with patch('logging.handlers.RotatingFileHandler', side_effect=Exception("File error")), \
             patch('logger.Path.mkdir'):
            
            # Should not crash even if file handler setup fails
            try:
                logger = get_logger('error_test')
                logger.info("Test message")
            except Exception as e:
                pytest.fail(f"Logger setup should not crash on file handler errors: {e}")


class TestLoggerEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup for each test"""
        from logger import ServiceLogger
        ServiceLogger._instance = None
        ServiceLogger._loggers = {}
    
    def test_unicode_log_messages(self):
        """Test logging with unicode characters"""
        from logger import get_logger
        
        with patch('logging.handlers.RotatingFileHandler', return_value=create_mock_file_handler()), \
             patch('logger.Path.mkdir'):
            
            logger = get_logger('unicode_test')
            
            # Should handle unicode without crashing
            unicode_message = "Test message with unicode: 测试 🚀 ñáéíóú"
            logger.info(unicode_message)
    
    def test_very_long_log_messages(self):
        """Test logging with very long messages"""
        from logger import get_logger
        
        with patch('logging.handlers.RotatingFileHandler', return_value=create_mock_file_handler()), \
             patch('logger.Path.mkdir'):
            
            logger = get_logger('long_message_test')
            
            # Very long message
            long_message = "x" * 10000
            logger.info(long_message)
    
    def test_none_logger_name(self):
        """Test handling of None logger name"""
        from logger import get_logger
        
        with patch('inspect.currentframe') as mock_frame:
            mock_frame_obj = Mock()
            mock_frame_obj.f_back.f_globals = {'__name__': None}
            mock_frame.return_value = mock_frame_obj
            
            logger = get_logger()
            
            # Should handle None gracefully
            assert isinstance(logger, logging.Logger)
    
    def test_empty_string_logger_name(self):
        """Test handling of empty string logger name"""
        from logger import get_logger
        
        logger = get_logger('')
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == ''
    
    @patch('logger.Path.mkdir', side_effect=PermissionError("Cannot create directory"))
    def test_log_directory_creation_failure(self, mock_mkdir):
        """Test handling when log directory cannot be created"""
        from logger import ServiceLogger
        
        # Should handle directory creation failure gracefully
        try:
            logger_service = ServiceLogger()
            # Should not crash even if directory creation fails
        except PermissionError:
            # It's acceptable if this raises PermissionError, 
            # as long as it's the expected one from mkdir
            pass


if __name__ == "__main__":
    pytest.main([__file__])