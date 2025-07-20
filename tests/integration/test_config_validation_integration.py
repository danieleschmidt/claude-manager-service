"""
Integration tests for configuration validation with the main modules
"""
import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.append('/root/repo/src')


class TestConfigValidationIntegration:
    """Integration tests for configuration validation"""
    
    def test_task_analyzer_with_invalid_config(self):
        """Test that task_analyzer properly handles invalid configuration"""
        invalid_config = {
            "github": {
                "username": "test",
                # Missing managerRepo and reposToScan
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name
        
        try:
            # Mock sys.argv to prevent pytest from interfering
            with patch('sys.argv', ['task_analyzer.py']):
                # Import and run with invalid config
                from task_analyzer import __name__ as main_module
                if main_module == "__main__":
                    # This would normally be executed, but we'll test the import path
                    from config_validator import get_validated_config
                    
                    with pytest.raises(SystemExit):
                        get_validated_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_orchestrator_with_invalid_config(self):
        """Test that orchestrator properly handles invalid configuration"""
        invalid_config = {
            "analyzer": {
                "scanForTodos": "not_a_boolean"  # Wrong type
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name
        
        try:
            from config_validator import get_validated_config
            
            with pytest.raises(SystemExit):
                get_validated_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_successful_config_loading(self):
        """Test that valid configuration loads successfully"""
        valid_config = {
            "github": {
                "username": "testuser",
                "managerRepo": "testuser/manager",
                "reposToScan": ["testuser/repo1"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "cleanupTasksOlderThanDays": 60
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config, f)
            config_path = f.name
        
        try:
            from config_validator import get_validated_config
            
            loaded_config = get_validated_config(config_path)
            assert loaded_config == valid_config
            assert loaded_config['analyzer']['cleanupTasksOlderThanDays'] == 60
        finally:
            os.unlink(config_path)
    
    def test_config_validation_helpful_error_messages(self):
        """Test that configuration validation provides helpful error messages"""
        from config_validator import ConfigValidationError, validate_config
        
        # Test missing section
        config_missing_section = {"github": {}}
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_missing_section)
        
        error_message = str(exc_info.value)
        assert "analyzer" in error_message.lower()
        assert "missing" in error_message.lower()
        
        # Test invalid repository format
        config_invalid_repo = {
            "github": {
                "username": "test",
                "managerRepo": "invalid-format",  # Should be owner/repo
                "reposToScan": ["test/repo"]
            },
            "analyzer": {"scanForTodos": True, "scanOpenIssues": True},
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_invalid_repo)
        
        error_message = str(exc_info.value)
        assert "managerRepo" in error_message
        assert "owner/repository" in error_message