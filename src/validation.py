"""
Enhanced Validation System for Claude Manager Service

This module provides schema-based validation and enhanced input validation
to replace and improve upon basic validation patterns throughout the codebase.

Features:
- JSON Schema-based configuration validation
- API parameter validation with type checking
- Enhanced input validation with context-aware sanitization
- Validation result caching for performance
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging

from .logger import get_logger


# Custom Validation Exceptions
class ValidationError(Exception):
    """Base class for validation errors"""
    pass


class ConfigurationValidationError(ValidationError):
    """Configuration validation specific error"""
    def __init__(self, message: str, field_path: str = None, invalid_value: Any = None):
        super().__init__(message)
        self.field_path = field_path
        self.invalid_value = invalid_value


class ParameterValidationError(ValidationError):
    """API parameter validation error"""
    def __init__(self, message: str, parameter_name: str = None, expected_type: str = None):
        super().__init__(message)
        self.parameter_name = parameter_name
        self.expected_type = expected_type


class SchemaValidationError(ValidationError):
    """Schema validation error with detailed information"""
    def __init__(self, message: str, schema_path: str = None, validation_errors: List[str] = None):
        super().__init__(message)
        self.schema_path = schema_path
        self.validation_errors = validation_errors or []


# Configuration Schema Definition
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["github", "analyzer", "executor"],
    "properties": {
        "github": {
            "type": "object",
            "required": ["username", "managerRepo", "reposToScan"],
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 39,
                    "pattern": "^[a-zA-Z0-9-]+$"
                },
                "managerRepo": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"
                },
                "reposToScan": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "items": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"
                    }
                }
            },
            "additionalProperties": False
        },
        "analyzer": {
            "type": "object",
            "required": ["scanForTodos", "scanOpenIssues"],
            "properties": {
                "scanForTodos": {"type": "boolean"},
                "scanOpenIssues": {"type": "boolean"},
                "maxIssuesPerRepo": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "todoPatterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["TODO:", "FIXME:", "HACK:", "BUG:"]
                },
                "excludePatterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["test_", "mock_", ".git/"]
                }
            },
            "additionalProperties": False
        },
        "executor": {
            "type": "object",
            "required": ["terragonUsername"],
            "properties": {
                "terragonUsername": {
                    "type": "string",
                    "pattern": "^@[a-zA-Z0-9_-]+$"
                },
                "defaultExecutor": {
                    "type": "string",
                    "enum": ["terragon", "claude-flow"],
                    "default": "terragon"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 60,
                    "maximum": 3600,
                    "default": 300
                }
            },
            "additionalProperties": False
        },
        "performance": {
            "type": "object",
            "properties": {
                "monitoringEnabled": {"type": "boolean", "default": True},
                "retentionDays": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30
                },
                "alertThresholds": {
                    "type": "object",
                    "properties": {
                        "durationSeconds": {"type": "number", "minimum": 0.1},
                        "errorRate": {"type": "number", "minimum": 0.01, "maximum": 1.0}
                    }
                }
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

# API Parameter Schemas
API_PARAMETER_SCHEMAS = {
    "create_issue": {
        "type": "object",
        "required": ["repo_name", "title"],
        "properties": {
            "repo_name": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$",
                "maxLength": 100
            },
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256
            },
            "body": {
                "type": "string",
                "maxLength": 65536
            },
            "labels": {
                "type": "array",
                "maxItems": 100,
                "items": {
                    "type": "string",
                    "maxLength": 50,
                    "pattern": "^[a-zA-Z0-9._-]+$"
                }
            }
        }
    },
    "get_repository": {
        "type": "object",
        "required": ["repo_name"],
        "properties": {
            "repo_name": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$",
                "maxLength": 100
            }
        }
    },
    "add_comment": {
        "type": "object",
        "required": ["repo_name", "issue_number", "comment_body"],
        "properties": {
            "repo_name": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$",
                "maxLength": 100
            },
            "issue_number": {
                "type": "integer",
                "minimum": 1
            },
            "comment_body": {
                "type": "string",
                "minLength": 1,
                "maxLength": 65536
            }
        }
    }
}


class JsonSchemaValidator:
    """
    Simple JSON schema validator implementation
    
    This is a lightweight implementation for basic validation needs.
    For production use, consider using jsonschema library.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate(self, data: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
        """
        Validate data against schema
        
        Args:
            data: Data to validate
            schema: Schema definition
            path: Current path in data structure
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Type validation
        if "type" in schema:
            expected_type = schema["type"]
            if not self._check_type(data, expected_type):
                errors.append(f"At {path}: Expected {expected_type}, got {type(data).__name__}")
                return errors  # Stop validation if type is wrong
        
        # Specific type validations
        if isinstance(data, dict):
            errors.extend(self._validate_object(data, schema, path))
        elif isinstance(data, list):
            errors.extend(self._validate_array(data, schema, path))
        elif isinstance(data, str):
            errors.extend(self._validate_string(data, schema, path))
        elif isinstance(data, (int, float)):
            errors.extend(self._validate_number(data, schema, path))
        
        return errors
    
    def _check_type(self, data: Any, expected_type: str) -> bool:
        """Check if data matches expected type"""
        type_mapping = {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return False
        
        return isinstance(data, expected_python_type)
    
    def _validate_object(self, data: Dict, schema: Dict, path: str) -> List[str]:
        """Validate object/dictionary"""
        errors = []
        
        # Check required properties
        if "required" in schema:
            for required_prop in schema["required"]:
                if required_prop not in data:
                    errors.append(f"At {path}: Missing required property '{required_prop}'")
        
        # Validate properties
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if prop in data:
                    prop_path = f"{path}.{prop}" if path else prop
                    errors.extend(self.validate(data[prop], prop_schema, prop_path))
        
        # Check additional properties
        if schema.get("additionalProperties", True) is False:
            allowed_props = set(schema.get("properties", {}).keys())
            for prop in data:
                if prop not in allowed_props:
                    errors.append(f"At {path}: Additional property '{prop}' not allowed")
        
        return errors
    
    def _validate_array(self, data: List, schema: Dict, path: str) -> List[str]:
        """Validate array/list"""
        errors = []
        
        # Check length constraints
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"At {path}: Array too short (minimum {schema['minItems']} items)")
        
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"At {path}: Array too long (maximum {schema['maxItems']} items)")
        
        # Validate items
        if "items" in schema:
            item_schema = schema["items"]
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"
                errors.extend(self.validate(item, item_schema, item_path))
        
        return errors
    
    def _validate_string(self, data: str, schema: Dict, path: str) -> List[str]:
        """Validate string"""
        errors = []
        
        # Length constraints
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(f"At {path}: String too short (minimum {schema['minLength']} characters)")
        
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(f"At {path}: String too long (maximum {schema['maxLength']} characters)")
        
        # Pattern validation
        if "pattern" in schema:
            pattern = re.compile(schema["pattern"])
            if not pattern.match(data):
                errors.append(f"At {path}: String does not match pattern '{schema['pattern']}'")
        
        return errors
    
    def _validate_number(self, data: Union[int, float], schema: Dict, path: str) -> List[str]:
        """Validate number (integer or float)"""
        errors = []
        
        # Range constraints
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"At {path}: Number too small (minimum {schema['minimum']})")
        
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"At {path}: Number too large (maximum {schema['maximum']})")
        
        # Enum validation
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"At {path}: Value must be one of {schema['enum']}")
        
        return errors


def validate_config_schema(config: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ConfigurationValidationError: If validation fails
    """
    validator = JsonSchemaValidator()
    errors = validator.validate(config, CONFIG_SCHEMA)
    
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(errors)
        raise ConfigurationValidationError(error_message)
    
    return True


def validate_api_parameters(parameters: Dict[str, Any], operation: str) -> bool:
    """
    Validate API parameters against schema
    
    Args:
        parameters: Parameters to validate
        operation: API operation name
        
    Returns:
        True if valid
        
    Raises:
        ParameterValidationError: If validation fails
    """
    if operation not in API_PARAMETER_SCHEMAS:
        raise ParameterValidationError(f"Unknown API operation: {operation}")
    
    schema = API_PARAMETER_SCHEMAS[operation]
    validator = JsonSchemaValidator()
    errors = validator.validate(parameters, schema)
    
    if errors:
        error_message = f"Parameter validation failed for {operation}:\n" + "\n".join(errors)
        raise ParameterValidationError(error_message)
    
    return True


def validate_repo_list(repos: List[str]) -> List[str]:
    """
    Validate list of repository names
    
    Args:
        repos: List of repository names
        
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    if not isinstance(repos, list):
        errors.append("Repository list must be an array")
        return errors
    
    if len(repos) == 0:
        errors.append("Repository list cannot be empty")
        return errors
    
    if len(repos) > 50:
        errors.append("Repository list too long (maximum 50 repositories)")
    
    for i, repo in enumerate(repos):
        if not isinstance(repo, str):
            errors.append(f"Repository at index {i} must be a string")
            continue
        
        # Validate repo name format
        repo_pattern = re.compile(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$')
        if not repo_pattern.match(repo):
            errors.append(f"Invalid repository name format at index {i}: {repo}")
        
        if len(repo) > 100:
            errors.append(f"Repository name too long at index {i}: {repo}")
    
    return errors


def validate_labels(labels: List[str]) -> List[str]:
    """
    Validate GitHub issue labels
    
    Args:
        labels: List of label names
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not isinstance(labels, list):
        errors.append("Labels must be an array")
        return errors
    
    if len(labels) > 100:
        errors.append("Too many labels (maximum 100)")
    
    for i, label in enumerate(labels):
        if not isinstance(label, str):
            errors.append(f"Label at index {i} must be a string")
            continue
        
        if len(label) == 0:
            errors.append(f"Label at index {i} cannot be empty")
        elif len(label) > 50:
            errors.append(f"Label at index {i} too long (maximum 50 characters)")
        
        # Check for invalid characters
        if not re.match(r'^[a-zA-Z0-9._\s-]+$', label):
            errors.append(f"Label at index {i} contains invalid characters: {label}")
    
    return errors


class AdvancedValidator:
    """
    Advanced validation with caching and custom rules
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_cache: Dict[str, Dict] = {}
        self._schema_validator = JsonSchemaValidator()
    
    def register_validator(self, name: str, validator_func: Callable[[Any], List[str]]):
        """
        Register custom validator function
        
        Args:
            name: Validator name
            validator_func: Function that takes data and returns list of errors
        """
        self.custom_validators[name] = validator_func
        self.logger.debug(f"Registered custom validator: {name}")
    
    def validate_with_cache(self, data: Any, schema: Dict[str, Any], cache_key: str = None) -> bool:
        """
        Validate with result caching
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            cache_key: Optional cache key (auto-generated if not provided)
            
        Returns:
            True if valid
        """
        if cache_key is None:
            import hashlib
            data_str = json.dumps(data, sort_keys=True, default=str)
            schema_str = json.dumps(schema, sort_keys=True)
            cache_key = hashlib.md5(f"{data_str}:{schema_str}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            if datetime.fromisoformat(cached_result["timestamp"]) > (
                datetime.now() - timedelta(minutes=5)
            ):
                return cached_result["valid"]
        
        # Perform validation
        errors = self._schema_validator.validate(data, schema)
        is_valid = len(errors) == 0
        
        # Cache result
        self.validation_cache[cache_key] = {
            "valid": is_valid,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
        if not is_valid:
            raise SchemaValidationError("Validation failed")
        
        return True
    
    def validate_custom(self, data: Any, validator_name: str) -> bool:
        """
        Use custom validator
        
        Args:
            data: Data to validate
            validator_name: Name of registered validator
            
        Returns:
            True if valid
        """
        if validator_name not in self.custom_validators:
            raise ValidationError(f"Unknown custom validator: {validator_name}")
        
        validator_func = self.custom_validators[validator_name]
        errors = validator_func(data)
        
        if errors:
            raise ValidationError(f"Custom validation failed: {'; '.join(errors)}")
        
        return True
    
    def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
        self.logger.debug("Validation cache cleared")


# Default validators for common patterns
def validate_github_username(username: str) -> List[str]:
    """Validate GitHub username format"""
    errors = []
    
    if not isinstance(username, str):
        errors.append("Username must be a string")
        return errors
    
    if len(username) < 1 or len(username) > 39:
        errors.append(f"Username length invalid: {len(username)} (must be 1-39 characters)")
    
    if not re.match(r'^[a-zA-Z0-9-]+$', username):
        errors.append("Username contains invalid characters (only letters, numbers, and hyphens allowed)")
    
    if username.startswith('-') or username.endswith('-'):
        errors.append("Username cannot start or end with hyphen")
    
    return errors


def validate_issue_title(title: str) -> List[str]:
    """Validate GitHub issue title"""
    errors = []
    
    if not isinstance(title, str):
        errors.append("Title must be a string")
        return errors
    
    if len(title.strip()) == 0:
        errors.append("Title cannot be empty")
    elif len(title) > 256:
        errors.append(f"Title too long: {len(title)} characters (maximum 256)")
    
    # Check for potential security issues
    if re.search(r'<script\b', title, re.IGNORECASE):
        errors.append("Title contains potentially dangerous content")
    
    return errors


# Global validator instance
_advanced_validator = AdvancedValidator()

# Register default custom validators
_advanced_validator.register_validator("github_username", validate_github_username)
_advanced_validator.register_validator("issue_title", validate_issue_title)
_advanced_validator.register_validator("repo_list", validate_repo_list)
_advanced_validator.register_validator("labels", validate_labels)


def get_validator() -> AdvancedValidator:
    """Get global validator instance"""
    return _advanced_validator