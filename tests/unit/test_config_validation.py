"""
Unit tests for configuration validation functionality
"""
import pytest
from unittest.mock import Mock, patch, mock_open
import json
import tempfile
import os
from pathlib import Path

import sys
sys.path.append('/root/repo/src')


class TestConfigValidation:
    """Test cases for configuration validation"""
    
    def test_valid_config_validation(self):
        """Test validation of a completely valid configuration"""
        from config_validator import validate_config
        
        valid_config = {
            "github": {
                "username": "testuser",
                "managerRepo": "testuser/manager-repo",
                "reposToScan": ["testuser/repo1", "testuser/repo2"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "cleanupTasksOlderThanDays": 90
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        # Should not raise any exceptions
        validate_config(valid_config)

    def test_missing_required_sections(self):
        """Test validation with missing required sections"""
        from config_validator import validate_config, ConfigValidationError
        
        # Missing github section
        config_missing_github = {
            "analyzer": {"scanForTodos": True, "scanOpenIssues": True},
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_missing_github)
        assert "github" in str(exc_info.value).lower()
        
        # Missing analyzer section  
        config_missing_analyzer = {
            "github": {
                "username": "test",
                "managerRepo": "test/repo",
                "reposToScan": ["test/repo"]
            },
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_missing_analyzer)
        assert "analyzer" in str(exc_info.value).lower()

    def test_missing_required_fields(self):
        """Test validation with missing required fields within sections"""
        from config_validator import validate_config, ConfigValidationError
        
        # Missing username in github section
        config_missing_username = {
            "github": {
                "managerRepo": "test/repo",
                "reposToScan": ["test/repo"]
            },
            "analyzer": {"scanForTodos": True, "scanOpenIssues": True},
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_missing_username)
        assert "username" in str(exc_info.value).lower()

    def test_invalid_repository_names(self):
        """Test validation of repository name formats"""
        from config_validator import validate_config, ConfigValidationError
        
        # Invalid manager repo format
        config_invalid_manager = {
            "github": {
                "username": "test",
                "managerRepo": "invalid-repo-name",  # Missing owner/repo format
                "reposToScan": ["test/repo"]
            },
            "analyzer": {"scanForTodos": True, "scanOpenIssues": True},
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_invalid_manager)
        assert "managerRepo" in str(exc_info.value)
        
        # Invalid repos to scan format
        config_invalid_repos = {
            "github": {
                "username": "test", 
                "managerRepo": "test/manager",
                "reposToScan": ["valid/repo", "invalid-format"]  # Mixed valid/invalid
            },
            "analyzer": {"scanForTodos": True, "scanOpenIssues": True},
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_invalid_repos)
        assert "reposToScan" in str(exc_info.value)

    def test_invalid_data_types(self):
        """Test validation of correct data types"""
        from config_validator import validate_config, ConfigValidationError
        
        # Boolean fields with wrong type
        config_wrong_types = {
            "github": {
                "username": "test",
                "managerRepo": "test/repo", 
                "reposToScan": ["test/repo"]
            },
            "analyzer": {
                "scanForTodos": "true",  # Should be boolean, not string
                "scanOpenIssues": 1      # Should be boolean, not int
            },
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_wrong_types)
        assert "scanForTodos" in str(exc_info.value) or "scanOpenIssues" in str(exc_info.value)

    def test_empty_repos_to_scan(self):
        """Test validation with empty reposToScan array"""
        from config_validator import validate_config, ConfigValidationError
        
        config_empty_repos = {
            "github": {
                "username": "test",
                "managerRepo": "test/repo",
                "reposToScan": []  # Empty array
            },
            "analyzer": {"scanForTodos": True, "scanOpenIssues": True},
            "executor": {"terragonUsername": "@terragon-labs"}
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_empty_repos)
        assert "reposToScan" in str(exc_info.value)
        assert "empty" in str(exc_info.value).lower()

    def test_load_and_validate_config_success(self):
        """Test successful loading and validation from file"""
        from config_validator import load_and_validate_config
        
        valid_config = {
            "github": {
                "username": "test",
                "managerRepo": "test/repo",
                "reposToScan": ["test/repo"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(valid_config))):
            with patch('os.path.exists', return_value=True):
                config = load_and_validate_config('config.json')
                assert config == valid_config

    def test_load_and_validate_config_file_not_found(self):
        """Test loading config when file doesn't exist"""
        from config_validator import load_and_validate_config, ConfigValidationError
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(ConfigValidationError) as exc_info:
                load_and_validate_config('missing.json')
            assert "not found" in str(exc_info.value).lower()

    def test_load_and_validate_config_invalid_json(self):
        """Test loading config with invalid JSON"""
        from config_validator import load_and_validate_config, ConfigValidationError
        
        invalid_json = '{"github": {"username": "test"'  # Missing closing braces
        
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with patch('os.path.exists', return_value=True):
                with pytest.raises(ConfigValidationError) as exc_info:
                    load_and_validate_config('config.json')
                assert "json" in str(exc_info.value).lower()

    def test_config_with_optional_fields(self):
        """Test configuration with optional fields populated"""
        from config_validator import validate_config
        
        config_with_optionals = {
            "github": {
                "username": "test",
                "managerRepo": "test/repo",
                "reposToScan": ["test/repo"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "cleanupTasksOlderThanDays": 60  # Optional field
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        # Should not raise any exceptions
        validate_config(config_with_optionals)

    def test_invalid_cleanup_days_value(self):
        """Test validation of cleanupTasksOlderThanDays field"""
        from config_validator import validate_config, ConfigValidationError
        
        config_invalid_cleanup = {
            "github": {
                "username": "test",
                "managerRepo": "test/repo",
                "reposToScan": ["test/repo"]
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "cleanupTasksOlderThanDays": -5  # Invalid negative value
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config_invalid_cleanup)
        assert "cleanupTasksOlderThanDays" in str(exc_info.value)