#!/usr/bin/env python3
"""
Advanced Input Validation and Sanitization System

Provides comprehensive validation and sanitization for all inputs in the autonomous SDLC system:
- Schema-based validation with Pydantic
- Content sanitization and filtering
- Rate limiting and input size controls
- Data format validation
- Security-focused input checks
"""

import re
import json
import html
import urllib.parse
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Pattern, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator, ValidationError
# from pydantic.validators import str_validator  # Not needed in Pydantic v2

# Import existing security utilities
from src.security import sanitize_issue_content, validate_repo_name, InputSanitizationError

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"

class ContentType(Enum):
    """Content type classification"""
    GITHUB_REPO = "github_repo"
    ISSUE_TITLE = "issue_title"
    ISSUE_BODY = "issue_body"
    FILE_PATH = "file_path"
    CODE_CONTENT = "code_content"
    URL = "url"
    EMAIL = "email"
    USERNAME = "username"
    JSON_DATA = "json_data"
    PLAIN_TEXT = "plain_text"

@dataclass
class ValidationRule:
    """Validation rule configuration"""
    name: str
    pattern: Optional[Pattern] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_chars: Optional[Set[str]] = None
    forbidden_chars: Optional[Set[str]] = None
    allowed_values: Optional[Set[str]] = None
    custom_validator: Optional[callable] = None
    sanitize_before_validation: bool = True

@dataclass
class ValidationConfig:
    """Validation configuration"""
    level: ValidationLevel = ValidationLevel.MODERATE
    max_content_size: int = 1048576  # 1MB
    enable_content_filtering: bool = True
    enable_xss_protection: bool = True
    enable_sql_injection_protection: bool = True
    enable_path_traversal_protection: bool = True
    custom_rules: Dict[str, ValidationRule] = field(default_factory=dict)

class GitHubRepositoryModel(BaseModel):
    """Pydantic model for GitHub repository validation"""
    name: str = Field(..., min_length=1, max_length=100)
    full_name: str = Field(..., pattern=r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$')
    private: bool = False
    description: Optional[str] = Field(None, max_length=500)
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Repository name contains invalid characters')
        return v
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if not validate_repo_name(v):
            raise ValueError('Invalid repository full name format')
        return v

class IssueModel(BaseModel):
    """Pydantic model for GitHub issue validation"""
    title: str = Field(..., min_length=1, max_length=500)
    body: Optional[str] = Field(None, max_length=65536)  # 64KB
    number: int = Field(..., gt=0)
    state: str = Field(..., pattern=r'^(open|closed)$')
    labels: List[str] = Field(default_factory=list)
    assignees: List[str] = Field(default_factory=list)
    
    @validator('title')
    def validate_title(cls, v):
        # Remove potential XSS and sanitize
        sanitized = sanitize_issue_content(v)
        if len(sanitized.strip()) < 1:
            raise ValueError('Issue title cannot be empty after sanitization')
        return sanitized
    
    @validator('body')
    def validate_body(cls, v):
        if v is None:
            return v
        # Sanitize but allow markdown
        return sanitize_issue_content(v)
    
    @validator('labels')
    def validate_labels(cls, v):
        if len(v) > 20:  # Reasonable limit
            raise ValueError('Too many labels (max 20)')
        for label in v:
            if len(label) > 50:
                raise ValueError('Label too long (max 50 characters)')
        return v

class FilePathModel(BaseModel):
    """Pydantic model for file path validation"""
    path: str = Field(..., min_length=1, max_length=4096)
    
    @validator('path')
    def validate_path(cls, v):
        # Check for path traversal attacks
        if '..' in v or v.startswith('/'):
            raise ValueError('Invalid file path: potential path traversal')
        
        # Normalize path
        normalized = str(Path(v).as_posix())
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9._/-]+$', normalized):
            raise ValueError('File path contains invalid characters')
        
        return normalized

class ConfigurationModel(BaseModel):
    """Pydantic model for configuration validation"""
    github_username: str = Field(..., min_length=1, max_length=39)
    manager_repo: str = Field(..., pattern=r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$')
    repos_to_scan: List[str] = Field(..., min_items=1, max_items=100)
    scan_for_todos: bool = True
    scan_open_issues: bool = True
    terragon_username: str = Field(..., pattern=r'^@[a-zA-Z0-9._-]+$')
    
    @validator('github_username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$', v):
            raise ValueError('Invalid GitHub username format')
        return v
    
    @validator('repos_to_scan')
    def validate_repos(cls, v):
        for repo in v:
            if not validate_repo_name(repo):
                raise ValueError(f'Invalid repository name: {repo}')
        return v

class AdvancedValidator:
    """Advanced validation system with multiple strategies"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validation_rules = self._load_default_rules()
        self.validation_rules.update(self.config.custom_rules)
        
        # Compile regex patterns for performance
        self._compiled_patterns = {}
        for name, rule in self.validation_rules.items():
            if rule.pattern:
                self._compiled_patterns[name] = re.compile(rule.pattern)
    
    def _load_default_rules(self) -> Dict[str, ValidationRule]:
        """Load default validation rules"""
        return {
            'github_repo': ValidationRule(
                name='github_repo',
                pattern=re.compile(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'),
                min_length=3,
                max_length=100
            ),
            'issue_title': ValidationRule(
                name='issue_title',
                min_length=1,
                max_length=500,
                forbidden_chars={'<', '>', '"', "'"}
            ),
            'file_path': ValidationRule(
                name='file_path',
                pattern=re.compile(r'^[a-zA-Z0-9._/-]+$'),
                max_length=4096,
                custom_validator=self._validate_path_traversal
            ),
            'url': ValidationRule(
                name='url',
                pattern=re.compile(r'^https?://[^\s<>"\']+$'),
                max_length=2048
            ),
            'email': ValidationRule(
                name='email',
                pattern=re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
                max_length=254
            ),
            'username': ValidationRule(
                name='username',
                pattern=re.compile(r'^[a-zA-Z0-9._-]{1,39}$'),
                min_length=1,
                max_length=39
            )
        }
    
    def validate_content(self, content: Any, content_type: ContentType, 
                        strict: bool = None) -> Tuple[bool, str, Any]:
        """
        Validate content based on type
        
        Returns:
            Tuple of (is_valid, error_message, sanitized_content)
        """
        if strict is None:
            strict = self.config.level == ValidationLevel.STRICT
        
        try:
            # Size check
            if self._get_content_size(content) > self.config.max_content_size:
                return False, "Content exceeds maximum size limit", None
            
            # Type-specific validation
            if content_type == ContentType.GITHUB_REPO:
                return self._validate_github_repo(content, strict)
            elif content_type == ContentType.ISSUE_TITLE:
                return self._validate_issue_title(content, strict)
            elif content_type == ContentType.ISSUE_BODY:
                return self._validate_issue_body(content, strict)
            elif content_type == ContentType.FILE_PATH:
                return self._validate_file_path(content, strict)
            elif content_type == ContentType.CODE_CONTENT:
                return self._validate_code_content(content, strict)
            elif content_type == ContentType.URL:
                return self._validate_url(content, strict)
            elif content_type == ContentType.EMAIL:
                return self._validate_email(content, strict)
            elif content_type == ContentType.USERNAME:
                return self._validate_username(content, strict)
            elif content_type == ContentType.JSON_DATA:
                return self._validate_json(content, strict)
            else:  # PLAIN_TEXT
                return self._validate_plain_text(content, strict)
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def validate_with_schema(self, data: Dict[str, Any], schema_type: str) -> Tuple[bool, str, Any]:
        """Validate data using Pydantic schemas"""
        try:
            if schema_type == "github_repository":
                validated = GitHubRepositoryModel(**data)
                return True, "", validated.dict()
            elif schema_type == "issue":
                validated = IssueModel(**data)
                return True, "", validated.dict()
            elif schema_type == "file_path":
                validated = FilePathModel(**data)
                return True, "", validated.dict()
            elif schema_type == "configuration":
                validated = ConfigurationModel(**data)
                return True, "", validated.dict()
            else:
                return False, f"Unknown schema type: {schema_type}", None
                
        except ValidationError as e:
            error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            return False, "; ".join(error_messages), None
        except Exception as e:
            return False, f"Schema validation error: {str(e)}", None
    
    def sanitize_and_validate(self, content: str, content_type: ContentType) -> str:
        """Sanitize content and validate it's safe"""
        if not isinstance(content, str):
            content = str(content)
        
        # Basic sanitization
        sanitized = html.escape(content)
        
        # Content-specific sanitization
        if self.config.enable_xss_protection:
            sanitized = self._remove_xss_patterns(sanitized)
        
        if self.config.enable_sql_injection_protection:
            sanitized = self._remove_sql_injection_patterns(sanitized)
        
        if self.config.enable_path_traversal_protection:
            sanitized = self._remove_path_traversal_patterns(sanitized)
        
        # Validate the sanitized content
        is_valid, error_msg, _ = self.validate_content(sanitized, content_type)
        if not is_valid:
            raise InputSanitizationError(f"Content failed validation after sanitization: {error_msg}")
        
        return sanitized
    
    def _validate_github_repo(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate GitHub repository name"""
        if not validate_repo_name(content):
            return False, "Invalid GitHub repository format", None
        return True, "", content
    
    def _validate_issue_title(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate issue title"""
        rule = self.validation_rules.get('issue_title')
        return self._apply_rule(content, rule, strict)
    
    def _validate_issue_body(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate issue body"""
        if content is None:
            return True, "", content
        
        # Allow markdown but sanitize dangerous content
        sanitized = sanitize_issue_content(content)
        
        if len(sanitized) > 65536:  # 64KB limit
            return False, "Issue body too long", None
        
        return True, "", sanitized
    
    def _validate_file_path(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate file path"""
        rule = self.validation_rules.get('file_path')
        return self._apply_rule(content, rule, strict)
    
    def _validate_code_content(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate code content"""
        # Basic checks for code content
        if len(content) > 1048576:  # 1MB limit
            return False, "Code content too large", None
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'format\s+c:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.call',
            r'os\.system'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                if strict:
                    return False, f"Potentially dangerous code pattern detected", None
        
        return True, "", content
    
    def _validate_url(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate URL"""
        rule = self.validation_rules.get('url')
        return self._apply_rule(content, rule, strict)
    
    def _validate_email(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate email address"""
        rule = self.validation_rules.get('email')
        return self._apply_rule(content, rule, strict)
    
    def _validate_username(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate username"""
        rule = self.validation_rules.get('username')
        return self._apply_rule(content, rule, strict)
    
    def _validate_json(self, content: Union[str, dict], strict: bool) -> Tuple[bool, str, dict]:
        """Validate JSON data"""
        try:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            
            # Check depth to prevent deeply nested attacks
            max_depth = 10 if strict else 20
            if self._get_json_depth(data) > max_depth:
                return False, "JSON too deeply nested", None
            
            return True, "", data
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", None
    
    def _validate_plain_text(self, content: str, strict: bool) -> Tuple[bool, str, str]:
        """Validate plain text"""
        if len(content) > self.config.max_content_size:
            return False, "Content too long", None
        
        # Basic sanitization for plain text
        sanitized = html.escape(content)
        return True, "", sanitized
    
    def _apply_rule(self, content: str, rule: ValidationRule, strict: bool) -> Tuple[bool, str, str]:
        """Apply a validation rule to content"""
        if rule is None:
            return True, "", content
        
        original_content = content
        
        # Sanitize first if requested
        if rule.sanitize_before_validation:
            content = sanitize_issue_content(content)
        
        # Length checks
        if rule.min_length is not None and len(content) < rule.min_length:
            return False, f"Content too short (min {rule.min_length})", None
        
        if rule.max_length is not None and len(content) > rule.max_length:
            return False, f"Content too long (max {rule.max_length})", None
        
        # Pattern matching
        if rule.pattern and not rule.pattern.match(content):
            return False, f"Content doesn't match required pattern", None
        
        # Character restrictions
        if rule.allowed_chars:
            invalid_chars = set(content) - rule.allowed_chars
            if invalid_chars:
                return False, f"Contains forbidden characters: {invalid_chars}", None
        
        if rule.forbidden_chars:
            found_forbidden = set(content) & rule.forbidden_chars
            if found_forbidden:
                return False, f"Contains forbidden characters: {found_forbidden}", None
        
        # Allowed values
        if rule.allowed_values and content not in rule.allowed_values:
            return False, f"Content not in allowed values", None
        
        # Custom validator
        if rule.custom_validator:
            try:
                if not rule.custom_validator(content):
                    return False, "Failed custom validation", None
            except Exception as e:
                return False, f"Custom validation error: {str(e)}", None
        
        return True, "", content
    
    def _validate_path_traversal(self, path: str) -> bool:
        """Custom validator for path traversal"""
        return '..' not in path and not path.startswith('/')
    
    def _get_content_size(self, content: Any) -> int:
        """Get size of content in bytes"""
        if isinstance(content, str):
            return len(content.encode('utf-8'))
        elif isinstance(content, bytes):
            return len(content)
        else:
            return len(str(content).encode('utf-8'))
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Get maximum depth of JSON object"""
        if depth > 50:  # Prevent infinite recursion
            return depth
        
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._get_json_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._get_json_depth(item, depth + 1) for item in obj)
        else:
            return depth
    
    def _remove_xss_patterns(self, content: str) -> str:
        """Remove common XSS patterns"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'expression\s*\(',
            r'@import',
            r'<iframe[^>]*>.*?</iframe>',
        ]
        
        for pattern in xss_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content
    
    def _remove_sql_injection_patterns(self, content: str) -> str:
        """Remove common SQL injection patterns"""
        sql_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'[\'";](\s*|\s*--|\s*/\*)',
        ]
        
        for pattern in sql_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _remove_path_traversal_patterns(self, content: str) -> str:
        """Remove path traversal patterns"""
        # Remove ../ and ..\\ patterns
        content = re.sub(r'\.\.[\\/]', '', content)
        # Remove encoded versions
        content = re.sub(r'%2e%2e%2f', '', content, flags=re.IGNORECASE)
        content = re.sub(r'%2e%2e%5c', '', content, flags=re.IGNORECASE)
        
        return content

# Utility functions
def create_validator(level: ValidationLevel = ValidationLevel.MODERATE,
                    max_size: int = 1048576) -> AdvancedValidator:
    """Create a validator with common configuration"""
    config = ValidationConfig(
        level=level,
        max_content_size=max_size
    )
    return AdvancedValidator(config)

def validate_github_data(data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Quick validation for GitHub-related data"""
    validator = create_validator()
    
    # Validate repository if present
    if 'repository' in data:
        is_valid, error, _ = validator.validate_content(
            data['repository'], ContentType.GITHUB_REPO
        )
        if not is_valid:
            return False, f"Invalid repository: {error}", None
    
    # Validate issue data if present
    if any(key in data for key in ['title', 'body', 'number']):
        return validator.validate_with_schema(data, "issue")
    
    return True, "", data

def validate_and_sanitize_input(content: str, content_type: ContentType) -> str:
    """Quick validation and sanitization"""
    validator = create_validator()
    return validator.sanitize_and_validate(content, content_type)

# Example usage
if __name__ == "__main__":
    # Create validator
    validator = create_validator(ValidationLevel.STRICT)
    
    # Test different content types
    test_cases = [
        ("user/repo", ContentType.GITHUB_REPO),
        ("Fix critical bug in authentication", ContentType.ISSUE_TITLE),
        ("src/main.py", ContentType.FILE_PATH),
        ("https://github.com/user/repo", ContentType.URL),
        ("user@example.com", ContentType.EMAIL),
    ]
    
    for content, content_type in test_cases:
        is_valid, error, sanitized = validator.validate_content(content, content_type)
        print(f"{content_type.value}: {content} -> Valid: {is_valid}, Error: {error}")