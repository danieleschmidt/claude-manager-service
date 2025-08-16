"""
Enhanced Configuration Validation System for Generation 2

This module provides comprehensive configuration validation with:
- Schema-based validation with detailed error reporting
- Environment-specific configuration validation
- Configuration hot-reloading with validation
- Security-focused configuration checks
- Performance optimization validation
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import logging

from src.enhanced_logger import get_enhanced_logger
from src.validation import (
    ValidationError, ConfigurationValidationError, 
    JsonSchemaValidator, CONFIG_SCHEMA
)


@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    security_issues: List[str] = None
    performance_issues: List[str] = None


@dataclass
class ConfigSection:
    """Configuration section metadata"""
    name: str
    description: str
    required: bool
    schema: Dict[str, Any]
    validator_func: Optional[Callable] = None
    security_sensitive: bool = False


class EnhancedConfigValidator:
    """Enhanced configuration validator with comprehensive checks"""
    
    def __init__(self):
        self.logger = get_enhanced_logger(__name__)
        self.schema_validator = JsonSchemaValidator()
        
        # Define configuration sections
        self.config_sections = {
            'github': ConfigSection(
                name='github',
                description='GitHub integration settings',
                required=True,
                schema=CONFIG_SCHEMA['properties']['github'],
                validator_func=self._validate_github_config,
                security_sensitive=True
            ),
            'analyzer': ConfigSection(
                name='analyzer',
                description='Task analyzer settings',
                required=True,
                schema=CONFIG_SCHEMA['properties']['analyzer'],
                validator_func=self._validate_analyzer_config
            ),
            'executor': ConfigSection(
                name='executor',
                description='Task executor settings',
                required=True,
                schema=CONFIG_SCHEMA['properties']['executor'],
                validator_func=self._validate_executor_config
            ),
            'performance': ConfigSection(
                name='performance',
                description='Performance monitoring settings',
                required=False,
                schema=CONFIG_SCHEMA['properties']['performance'],
                validator_func=self._validate_performance_config
            ),
            'security': ConfigSection(
                name='security',
                description='Security settings',
                required=False,
                schema=self._get_security_schema(),
                validator_func=self._validate_security_config,
                security_sensitive=True
            ),
            'logging': ConfigSection(
                name='logging',
                description='Logging configuration',
                required=False,
                schema=self._get_logging_schema(),
                validator_func=self._validate_logging_config
            ),
            'database': ConfigSection(
                name='database',
                description='Database configuration',
                required=False,
                schema=self._get_database_schema(),
                validator_func=self._validate_database_config,
                security_sensitive=True
            )
        }
    
    def _get_security_schema(self) -> Dict[str, Any]:
        """Get security configuration schema"""
        return {
            "type": "object",
            "properties": {
                "rate_limiting": {
                    "type": "object",
                    "properties": {
                        "max_requests_per_minute": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 60},
                        "burst_capacity": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                        "block_duration_minutes": {"type": "integer", "minimum": 1, "maximum": 1440, "default": 30}
                    }
                },
                "authentication": {
                    "type": "object",
                    "properties": {
                        "session_timeout_minutes": {"type": "integer", "minimum": 5, "maximum": 1440, "default": 480},
                        "max_login_attempts": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                        "require_2fa": {"type": "boolean", "default": False}
                    }
                },
                "input_validation": {
                    "type": "object",
                    "properties": {
                        "max_input_length": {"type": "integer", "minimum": 100, "maximum": 100000, "default": 65536},
                        "allow_html": {"type": "boolean", "default": False},
                        "strict_mode": {"type": "boolean", "default": True}
                    }
                }
            },
            "additionalProperties": False
        }
    
    def _get_logging_schema(self) -> Dict[str, Any]:
        """Get logging configuration schema"""
        return {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    "default": "INFO"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "text"],
                    "default": "json"
                },
                "file_rotation": {
                    "type": "object",
                    "properties": {
                        "max_size_mb": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
                        "backup_count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
                    }
                },
                "structured_logging": {"type": "boolean", "default": True},
                "performance_logging": {"type": "boolean", "default": True},
                "security_logging": {"type": "boolean", "default": True}
            },
            "additionalProperties": False
        }
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """Get database configuration schema"""
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["sqlite", "postgresql", "mysql"],
                    "default": "sqlite"
                },
                "connection_string": {"type": "string"},
                "pool_settings": {
                    "type": "object",
                    "properties": {
                        "min_connections": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
                        "max_connections": {"type": "integer", "minimum": 5, "maximum": 1000, "default": 20},
                        "connection_timeout": {"type": "integer", "minimum": 5, "maximum": 300, "default": 30}
                    }
                },
                "backup_settings": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "frequency_hours": {"type": "integer", "minimum": 1, "maximum": 168, "default": 24},
                        "retention_days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30}
                    }
                }
            },
            "additionalProperties": False
        }
    
    async def validate_configuration(self, config: Dict[str, Any], environment: str = "development") -> ConfigValidationResult:
        """
        Comprehensive configuration validation
        
        Args:
            config: Configuration dictionary to validate
            environment: Target environment (development, staging, production)
            
        Returns:
            Detailed validation result
        """
        errors = []
        warnings = []
        suggestions = []
        security_issues = []
        performance_issues = []
        
        try:
            # 1. Schema validation
            schema_errors = self._validate_schema(config)
            errors.extend(schema_errors)
            
            # 2. Section-specific validation
            for section_name, section_config in self.config_sections.items():
                if section_name in config:
                    section_result = await self._validate_section(
                        section_name, config[section_name], environment
                    )
                    errors.extend(section_result.get('errors', []))
                    warnings.extend(section_result.get('warnings', []))
                    suggestions.extend(section_result.get('suggestions', []))
                    security_issues.extend(section_result.get('security_issues', []))
                    performance_issues.extend(section_result.get('performance_issues', []))
                elif section_config.required:
                    errors.append(f"Required configuration section missing: {section_name}")
            
            # 3. Cross-section validation
            cross_validation_result = self._validate_cross_sections(config, environment)
            errors.extend(cross_validation_result.get('errors', []))
            warnings.extend(cross_validation_result.get('warnings', []))
            
            # 4. Environment-specific validation
            env_result = self._validate_environment_specific(config, environment)
            errors.extend(env_result.get('errors', []))
            warnings.extend(env_result.get('warnings', []))
            suggestions.extend(env_result.get('suggestions', []))
            
            # 5. Security validation
            security_result = self._validate_security_requirements(config, environment)
            security_issues.extend(security_result.get('security_issues', []))
            
            # 6. Performance validation
            perf_result = self._validate_performance_requirements(config, environment)
            performance_issues.extend(perf_result.get('performance_issues', []))
            
            is_valid = len(errors) == 0 and len(security_issues) == 0
            
            return ConfigValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                security_issues=security_issues,
                performance_issues=performance_issues
            )
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Validation process failed: {str(e)}"]
            )
    
    def _validate_schema(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema"""
        try:
            return self.schema_validator.validate(config, CONFIG_SCHEMA)
        except Exception as e:
            return [f"Schema validation error: {str(e)}"]
    
    async def _validate_section(self, section_name: str, section_data: Any, environment: str) -> Dict[str, List[str]]:
        """Validate a specific configuration section"""
        result = {
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'security_issues': [],
            'performance_issues': []
        }
        
        section_config = self.config_sections.get(section_name)
        if not section_config:
            result['warnings'].append(f"Unknown configuration section: {section_name}")
            return result
        
        # Schema validation for section
        schema_errors = self.schema_validator.validate(section_data, section_config.schema)
        result['errors'].extend(schema_errors)
        
        # Custom validation function
        if section_config.validator_func:
            try:
                custom_result = await section_config.validator_func(section_data, environment)
                for key in result.keys():
                    if key in custom_result:
                        result[key].extend(custom_result[key])
            except Exception as e:
                result['errors'].append(f"Custom validation failed for {section_name}: {str(e)}")
        
        return result
    
    async def _validate_github_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate GitHub configuration"""
        result = {'errors': [], 'warnings': [], 'suggestions': [], 'security_issues': []}
        
        # Check for GitHub token in environment
        if not os.getenv('GITHUB_TOKEN'):
            result['security_issues'].append("GITHUB_TOKEN environment variable not set")
        
        # Validate repository format
        if 'reposToScan' in config:
            for repo in config['reposToScan']:
                if not self._is_valid_github_repo(repo):
                    result['errors'].append(f"Invalid GitHub repository format: {repo}")
        
        # Check manager repo
        if 'managerRepo' in config:
            if not self._is_valid_github_repo(config['managerRepo']):
                result['errors'].append(f"Invalid manager repository format: {config['managerRepo']}")
        
        # Environment-specific checks
        if environment == 'production':
            if len(config.get('reposToScan', [])) > 20:
                result['warnings'].append("Large number of repositories to scan may impact performance")
            
            if not config.get('apiTimeout', 30) >= 60:
                result['suggestions'].append("Consider increasing API timeout for production environment")
        
        return result
    
    async def _validate_analyzer_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate analyzer configuration"""
        result = {'errors': [], 'warnings': [], 'performance_issues': []}
        
        max_issues = config.get('maxIssuesPerRepo', 10)
        if environment == 'production' and max_issues > 50:
            result['performance_issues'].append(f"maxIssuesPerRepo ({max_issues}) may be too high for production")
        
        if not config.get('scanForTodos') and not config.get('scanOpenIssues'):
            result['warnings'].append("Both TODO and issue scanning are disabled")
        
        return result
    
    async def _validate_executor_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate executor configuration"""
        result = {'errors': [], 'suggestions': []}
        
        if 'terragonUsername' in config:
            username = config['terragonUsername']
            if not username.startswith('@'):
                result['errors'].append("terragonUsername must start with @")
        
        timeout = config.get('timeout', 300)
        if environment == 'production' and timeout < 600:
            result['suggestions'].append("Consider increasing timeout for production environment")
        
        return result
    
    async def _validate_performance_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate performance configuration"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        if not config.get('monitoringEnabled', True):
            if environment in ['staging', 'production']:
                result['warnings'].append("Performance monitoring disabled in non-development environment")
        
        retention_days = config.get('retentionDays', 30)
        if environment == 'production' and retention_days < 7:
            result['suggestions'].append("Consider longer retention period for production")
        
        return result
    
    async def _validate_security_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate security configuration"""
        result = {'security_issues': [], 'warnings': []}
        
        # Rate limiting validation
        if 'rate_limiting' in config:
            rate_config = config['rate_limiting']
            max_requests = rate_config.get('max_requests_per_minute', 60)
            
            if environment == 'production' and max_requests > 1000:
                result['security_issues'].append("High rate limit may allow abuse")
            
            if max_requests < 10:
                result['warnings'].append("Very low rate limit may impact usability")
        
        # Authentication validation
        if 'authentication' in config:
            auth_config = config['authentication']
            session_timeout = auth_config.get('session_timeout_minutes', 480)
            
            if environment == 'production' and session_timeout > 1440:  # 24 hours
                result['security_issues'].append("Long session timeout may pose security risk")
        
        return result
    
    async def _validate_logging_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate logging configuration"""
        result = {'warnings': [], 'suggestions': []}
        
        if config.get('level') == 'DEBUG' and environment == 'production':
            result['warnings'].append("DEBUG logging enabled in production may impact performance")
        
        if not config.get('structured_logging', True):
            result['suggestions'].append("Structured logging recommended for better log analysis")
        
        return result
    
    async def _validate_database_config(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate database configuration"""
        result = {'errors': [], 'warnings': [], 'security_issues': []}
        
        db_type = config.get('type', 'sqlite')
        
        if environment == 'production' and db_type == 'sqlite':
            result['warnings'].append("SQLite may not be suitable for production workloads")
        
        if 'connection_string' in config:
            conn_str = config['connection_string']
            if 'password=' in conn_str.lower():
                result['security_issues'].append("Database password in connection string - use environment variables")
        
        if 'pool_settings' in config:
            pool_config = config['pool_settings']
            max_conn = pool_config.get('max_connections', 20)
            
            if environment == 'production' and max_conn < 10:
                result['warnings'].append("Low connection pool size may impact performance")
        
        return result
    
    def _validate_cross_sections(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate cross-section dependencies and consistency"""
        result = {'errors': [], 'warnings': []}
        
        # Check GitHub and executor consistency
        if 'github' in config and 'executor' in config:
            github_config = config['github']
            executor_config = config['executor']
            
            # If scanning repos, need executor config
            if github_config.get('reposToScan') and not executor_config.get('terragonUsername'):
                result['warnings'].append("Repository scanning enabled but no Terragon username configured")
        
        # Check performance and logging consistency
        if 'performance' in config and 'logging' in config:
            perf_config = config['performance']
            log_config = config['logging']
            
            if perf_config.get('monitoringEnabled') and not log_config.get('performance_logging', True):
                result['warnings'].append("Performance monitoring enabled but performance logging disabled")
        
        return result
    
    def _validate_environment_specific(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate environment-specific requirements"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        if environment == 'production':
            # Production-specific validations
            if not config.get('performance', {}).get('monitoringEnabled', True):
                result['suggestions'].append("Enable performance monitoring for production")
            
            if not config.get('logging', {}).get('security_logging', True):
                result['warnings'].append("Security logging should be enabled in production")
            
            # Check for secure defaults
            if config.get('analyzer', {}).get('maxIssuesPerRepo', 10) > 100:
                result['warnings'].append("High maxIssuesPerRepo may impact production performance")
        
        elif environment == 'development':
            # Development-specific suggestions
            if config.get('logging', {}).get('level') != 'DEBUG':
                result['suggestions'].append("Consider DEBUG logging for development environment")
        
        return result
    
    def _validate_security_requirements(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate security requirements"""
        result = {'security_issues': []}
        
        # Check for sensitive data in configuration
        sensitive_keys = ['password', 'token', 'secret', 'key']
        self._check_sensitive_data(config, sensitive_keys, result['security_issues'])
        
        # Environment-specific security checks
        if environment == 'production':
            # Check for security configurations
            if 'security' not in config:
                result['security_issues'].append("Security configuration missing for production")
            
            # Check rate limiting
            if not config.get('security', {}).get('rate_limiting'):
                result['security_issues'].append("Rate limiting not configured for production")
        
        return result
    
    def _validate_performance_requirements(self, config: Dict[str, Any], environment: str) -> Dict[str, List[str]]:
        """Validate performance requirements"""
        result = {'performance_issues': []}
        
        if environment == 'production':
            # Check for performance-critical configurations
            analyzer_config = config.get('analyzer', {})
            max_concurrent = analyzer_config.get('maxConcurrentScans', 5)
            
            if max_concurrent > 20:
                result['performance_issues'].append(f"High concurrent scans ({max_concurrent}) may overload system")
            
            # Check database configuration
            db_config = config.get('database', {})
            if db_config.get('type') == 'sqlite':
                result['performance_issues'].append("SQLite may not provide adequate performance for production")
        
        return result
    
    def _check_sensitive_data(self, config: Any, sensitive_keys: List[str], issues: List[str], path: str = ""):
        """Recursively check for sensitive data in configuration"""
        if isinstance(config, dict):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if key contains sensitive information
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    issues.append(f"Sensitive data found in configuration: {current_path}")
                
                # Recurse into nested structures
                self._check_sensitive_data(value, sensitive_keys, issues, current_path)
        
        elif isinstance(config, list):
            for i, item in enumerate(config):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._check_sensitive_data(item, sensitive_keys, issues, current_path)
    
    def _is_valid_github_repo(self, repo_name: str) -> bool:
        """Check if repository name has valid GitHub format"""
        if not isinstance(repo_name, str):
            return False
        
        import re
        return bool(re.match(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$', repo_name))
    
    async def validate_config_file(self, config_path: str, environment: str = "development") -> ConfigValidationResult:
        """Validate configuration from file"""
        try:
            if not Path(config_path).exists():
                return ConfigValidationResult(
                    is_valid=False,
                    errors=[f"Configuration file not found: {config_path}"]
                )
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return await self.validate_configuration(config, environment)
            
        except json.JSONDecodeError as e:
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON in configuration file: {str(e)}"]
            )
        except Exception as e:
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Failed to validate configuration file: {str(e)}"]
            )
    
    def generate_validation_report(self, result: ConfigValidationResult) -> str:
        """Generate human-readable validation report"""
        report = []
        
        if result.is_valid:
            report.append("✅ Configuration validation PASSED")
        else:
            report.append("❌ Configuration validation FAILED")
        
        report.append("")
        
        if result.errors:
            report.append("🔴 ERRORS (must be fixed):")
            for error in result.errors:
                report.append(f"  • {error}")
            report.append("")
        
        if result.security_issues:
            report.append("🔒 SECURITY ISSUES (critical):")
            for issue in result.security_issues:
                report.append(f"  • {issue}")
            report.append("")
        
        if result.warnings:
            report.append("⚠️  WARNINGS:")
            for warning in result.warnings:
                report.append(f"  • {warning}")
            report.append("")
        
        if result.performance_issues:
            report.append("⚡ PERFORMANCE ISSUES:")
            for issue in result.performance_issues:
                report.append(f"  • {issue}")
            report.append("")
        
        if result.suggestions:
            report.append("💡 SUGGESTIONS:")
            for suggestion in result.suggestions:
                report.append(f"  • {suggestion}")
        
        return "\n".join(report)


# Global validator instance
_config_validator = EnhancedConfigValidator()


async def validate_configuration(config: Dict[str, Any], environment: str = "development") -> ConfigValidationResult:
    """Validate configuration using global validator"""
    return await _config_validator.validate_configuration(config, environment)


async def validate_config_file(config_path: str, environment: str = "development") -> ConfigValidationResult:
    """Validate configuration file using global validator"""
    return await _config_validator.validate_config_file(config_path, environment)


def generate_validation_report(result: ConfigValidationResult) -> str:
    """Generate validation report using global validator"""
    return _config_validator.generate_validation_report(result)