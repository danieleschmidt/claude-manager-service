"""
Sample data fixtures for testing various components.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List


class SampleData:
    """Collection of sample data for testing."""

    @staticmethod
    def task_data(
        task_id: str = "task_001",
        status: str = "pending",
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """Sample task data."""
        return {
            "id": task_id,
            "title": "Fix authentication bug",
            "description": "Users are unable to authenticate using OAuth2",
            "status": status,
            "priority": priority,
            "repository": "test-user/test-repo",
            "issue_number": 123,
            "labels": ["bug", "authentication", f"priority-{priority}"],
            "assignee": "test-user",
            "created_at": datetime(2024, 12, 1, tzinfo=timezone.utc).isoformat(),
            "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
            "due_date": datetime(2025, 1, 15, tzinfo=timezone.utc).isoformat(),
            "estimated_hours": 8,
            "actual_hours": None,
            "tags": ["backend", "security"],
            "metadata": {
                "complexity": "medium",
                "business_impact": "high",
                "technical_debt": False
            }
        }

    @staticmethod
    def performance_metrics() -> Dict[str, Any]:
        """Sample performance metrics data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_metrics": {
                "cpu_usage_percent": 45.2,
                "memory_usage_mb": 512.8,
                "disk_usage_gb": 23.4,
                "network_io_mbps": 12.7
            },
            "application_metrics": {
                "response_time_ms": 125.6,
                "throughput_rps": 850.3,
                "error_rate_percent": 0.15,
                "active_connections": 42
            },
            "database_metrics": {
                "query_time_ms": 45.2,
                "connections_active": 8,
                "connections_max": 20,
                "slow_queries_count": 2
            },
            "github_api_metrics": {
                "requests_made": 145,
                "requests_remaining": 4855,
                "rate_limit_reset": datetime.now(timezone.utc).isoformat(),
                "average_response_time_ms": 230.5
            }
        }

    @staticmethod
    def config_data() -> Dict[str, Any]:
        """Sample configuration data."""
        return {
            "github": {
                "username": "test-user",
                "managerRepo": "test-user/claude-manager-service",
                "reposToScan": [
                    "test-user/project-alpha",
                    "test-user/project-beta",
                    "test-user/project-gamma"
                ],
                "api_timeout": 30,
                "max_retries": 3
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True,
                "scanSecurityIssues": True,
                "scanPerformanceIssues": False,
                "excludePatterns": [
                    "*.md",
                    "test_*",
                    "node_modules/*"
                ],
                "includePatterns": [
                    "*.py",
                    "*.js",
                    "*.ts"
                ]
            },
            "executor": {
                "terragonUsername": "@terragon-labs",
                "maxConcurrentTasks": 5,
                "taskTimeoutMinutes": 60,
                "retryFailedTasks": True,
                "notifyOnCompletion": True
            },
            "performance": {
                "enableMonitoring": True,
                "alertThresholds": {
                    "cpuUsagePercent": 80,
                    "memoryUsageMB": 1024,
                    "responseTimeMs": 1000,
                    "errorRatePercent": 5
                },
                "metricsRetentionDays": 30
            },
            "security": {
                "enableSecurityScanning": True,
                "maxContentLengthKB": 1024,
                "allowedFileTypes": [".py", ".js", ".ts", ".json", ".yaml"],
                "blockedPatterns": ["password", "secret", "token"],
                "encryptSensitiveData": True
            }
        }

    @staticmethod
    def repository_scan_results() -> Dict[str, Any]:
        """Sample repository scan results."""
        return {
            "repository": "test-user/test-repo",
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "scan_duration_seconds": 45.3,
            "files_scanned": 127,
            "issues_found": {
                "todo_comments": [
                    {
                        "file": "src/auth.py",
                        "line": 45,
                        "type": "TODO",
                        "message": "Implement OAuth2 refresh token handling",
                        "priority": "high"
                    },
                    {
                        "file": "src/utils.py",
                        "line": 123,
                        "type": "FIXME",
                        "message": "This function has a memory leak",
                        "priority": "critical"
                    }
                ],
                "stale_issues": [
                    {
                        "issue_number": 78,
                        "title": "Update dependencies",
                        "last_activity": "2024-10-15T00:00:00Z",
                        "days_inactive": 78,
                        "labels": ["maintenance", "dependencies"]
                    }
                ],
                "security_issues": [
                    {
                        "file": "src/config.py",
                        "line": 12,
                        "severity": "medium",
                        "message": "Hardcoded secret detected",
                        "cwe_id": "CWE-798"
                    }
                ],
                "performance_issues": [
                    {
                        "file": "src/database.py",
                        "line": 67,
                        "type": "n_plus_one_query",
                        "message": "Potential N+1 query detected",
                        "impact": "high"
                    }
                ]
            },
            "quality_metrics": {
                "code_coverage_percent": 85.4,
                "technical_debt_ratio": 0.12,
                "maintainability_index": 78.9,
                "cyclomatic_complexity": 4.2
            },
            "recommendations": [
                "Address critical FIXME comments immediately",
                "Update stale issues or close if no longer relevant",
                "Review and resolve security vulnerabilities",
                "Optimize database queries to improve performance"
            ]
        }

    @staticmethod
    def workflow_execution_log() -> List[Dict[str, Any]]:
        """Sample workflow execution log."""
        return [
            {
                "timestamp": "2025-01-01T10:00:00Z",
                "level": "INFO",
                "component": "TaskAnalyzer",
                "message": "Starting repository scan",
                "metadata": {
                    "repository": "test-user/test-repo",
                    "scan_id": "scan_001"
                }
            },
            {
                "timestamp": "2025-01-01T10:01:30Z",
                "level": "DEBUG",
                "component": "GitHubAPI",
                "message": "Fetching repository contents",
                "metadata": {
                    "api_call": "GET /repos/test-user/test-repo/contents",
                    "response_time_ms": 245
                }
            },
            {
                "timestamp": "2025-01-01T10:02:15Z",
                "level": "WARNING",
                "component": "SecurityScanner",
                "message": "Potential security issue detected",
                "metadata": {
                    "file": "src/config.py",
                    "line": 12,
                    "issue_type": "hardcoded_secret"
                }
            },
            {
                "timestamp": "2025-01-01T10:03:45Z",
                "level": "INFO",
                "component": "TaskAnalyzer",
                "message": "Repository scan completed",
                "metadata": {
                    "repository": "test-user/test-repo",
                    "scan_id": "scan_001",
                    "duration_seconds": 225,
                    "issues_found": 5
                }
            },
            {
                "timestamp": "2025-01-01T10:04:00Z",
                "level": "INFO",
                "component": "Orchestrator",
                "message": "Creating GitHub issue for TODO comment",
                "metadata": {
                    "repository": "test-user/test-repo",
                    "file": "src/auth.py",
                    "line": 45,
                    "issue_title": "Address TODO in auth.py:45"
                }
            },
            {
                "timestamp": "2025-01-01T10:04:30Z",
                "level": "ERROR",
                "component": "GitHubAPI",
                "message": "Failed to create issue due to rate limiting",
                "metadata": {
                    "api_call": "POST /repos/test-user/test-repo/issues",
                    "error_code": 403,
                    "retry_after": 300
                }
            }
        ]

    @staticmethod
    def test_scenarios() -> Dict[str, Dict[str, Any]]:
        """Collection of test scenarios for various testing needs."""
        return {
            "successful_task_execution": {
                "input": {
                    "task": SampleData.task_data(status="approved"),
                    "config": SampleData.config_data()
                },
                "expected_output": {
                    "status": "completed",
                    "execution_time_seconds": 120,
                    "github_issue_created": True,
                    "ai_agent_triggered": True
                }
            },
            "failed_task_execution": {
                "input": {
                    "task": SampleData.task_data(status="failed"),
                    "config": SampleData.config_data()
                },
                "expected_output": {
                    "status": "failed",
                    "error_message": "GitHub API rate limit exceeded",
                    "retry_scheduled": True,
                    "retry_at": "2025-01-01T11:00:00Z"
                }
            },
            "high_priority_task": {
                "input": {
                    "task": SampleData.task_data(priority="critical"),
                    "config": SampleData.config_data()
                },
                "expected_output": {
                    "priority_escalation": True,
                    "notification_sent": True,
                    "execution_order": 1
                }
            },
            "rate_limited_scenario": {
                "input": {
                    "github_api_requests_remaining": 0,
                    "config": SampleData.config_data()
                },
                "expected_output": {
                    "execution_delayed": True,
                    "delay_until": "2025-01-01T11:00:00Z",
                    "fallback_strategy": "cache_only"
                }
            }
        }

    @staticmethod
    def mock_file_contents() -> Dict[str, str]:
        """Mock file contents for testing file analysis."""
        return {
            "python_with_todos": '''
"""
Example Python module with various TODO comments.
"""

import os
import sys

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user."""
    # TODO: Implement proper password hashing
    # FIXME: This is vulnerable to timing attacks
    return username == "admin" and password == "secret"

class UserManager:
    """Manages user operations."""
    
    def __init__(self):
        # TODO: Connect to actual database
        self.users = {}
    
    def create_user(self, username: str, email: str) -> bool:
        """Create a new user."""
        # HACK: Temporary implementation
        if username in self.users:
            return False
        
        # TODO: Validate email format
        # TODO: Send confirmation email
        self.users[username] = {"email": email}
        return True
    
    def get_user_profile(self, username: str) -> dict:
        """Get user profile."""
        # FIXME: Handle case when user doesn't exist
        return self.users[username]
''',
            "javascript_with_issues": '''
// Example JavaScript file with various issues

function processPayment(amount, cardNumber, cvv) {
    // TODO: Implement proper validation
    // FIXME: CVV should not be logged
    console.log("Processing payment:", amount, cardNumber, cvv);
    
    // HACK: Temporary fix for decimal precision
    const processedAmount = Math.round(amount * 100) / 100;
    
    // TODO: Connect to payment gateway
    return {
        success: true,
        transactionId: Math.random().toString(36).substr(2, 9)
    };
}

class PaymentProcessor {
    constructor(apiKey) {
        // FIXME: API key should not be stored in plain text
        this.apiKey = apiKey;
    }
    
    async processTransaction(transaction) {
        // TODO: Add retry logic
        // TODO: Implement proper error handling
        try {
            const result = await fetch('/api/payment', {
                method: 'POST',
                body: JSON.stringify(transaction),
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`
                }
            });
            return result.json();
        } catch (error) {
            // FIXME: Should not expose internal errors to user
            throw error;
        }
    }
}
''',
            "yaml_config": '''
# Configuration file with potential issues

database:
  host: localhost
  port: 5432
  name: myapp
  # TODO: Use environment variables for credentials
  username: admin
  password: supersecret123  # FIXME: Password should be encrypted

api:
  # TODO: Implement rate limiting
  rate_limit: null
  timeout: 30
  
  # HACK: Temporary fix for CORS issues
  cors:
    allow_all: true
    
security:
  # FIXME: JWT secret should be randomly generated
  jwt_secret: "my-super-secret-key"
  
  # TODO: Configure proper session timeout
  session_timeout: 86400
  
logging:
  level: DEBUG  # TODO: Change to INFO in production
  
  # FIXME: Log files can grow indefinitely
  file: /var/log/app.log
'''
        }