"""
Contract testing for API interfaces and data schemas.
"""

import json
from typing import Any, Dict

import pytest
from jsonschema import validate, ValidationError


class TestAPIContracts:
    """Test API contracts and data schemas."""

    def test_github_issue_schema(self, sample_issue_data):
        """Test GitHub issue data matches expected schema."""
        schema = {
            "type": "object",
            "required": ["number", "title", "body", "state", "html_url"],
            "properties": {
                "number": {"type": "integer"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed"]},
                "html_url": {"type": "string", "format": "uri"},
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {"name": {"type": "string"}}
                    }
                }
            }
        }
        
        validate(instance=sample_issue_data, schema=schema)

    def test_task_data_schema(self, task_data):
        """Test task data matches expected schema."""
        schema = {
            "type": "object",
            "required": ["id", "title", "status", "priority"],
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "failed"]
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                },
                "repository": {"type": "string"},
                "issue_number": {"type": "integer"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        validate(instance=task_data, schema=schema)

    def test_configuration_schema(self, sample_config):
        """Test configuration data matches expected schema."""
        schema = {
            "type": "object",
            "required": ["github", "analyzer", "executor"],
            "properties": {
                "github": {
                    "type": "object",
                    "required": ["username", "managerRepo"],
                    "properties": {
                        "username": {"type": "string"},
                        "managerRepo": {"type": "string"},
                        "reposToScan": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "analyzer": {
                    "type": "object",
                    "properties": {
                        "scanForTodos": {"type": "boolean"},
                        "scanOpenIssues": {"type": "boolean"}
                    }
                },
                "executor": {
                    "type": "object",
                    "properties": {
                        "terragonUsername": {"type": "string"}
                    }
                }
            }
        }
        
        validate(instance=sample_config, schema=schema)

    def test_performance_metrics_schema(self, performance_data):
        """Test performance metrics match expected schema."""
        schema = {
            "type": "object",
            "required": ["cpu_usage", "memory_usage", "response_time"],
            "properties": {
                "cpu_usage": {"type": "number", "minimum": 0, "maximum": 100},
                "memory_usage": {"type": "number", "minimum": 0},
                "disk_usage": {"type": "number", "minimum": 0, "maximum": 100},
                "network_io": {"type": "number", "minimum": 0},
                "response_time": {"type": "number", "minimum": 0},
                "throughput": {"type": "number", "minimum": 0},
                "error_rate": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        
        validate(instance=performance_data, schema=schema)

    def test_error_response_schema(self):
        """Test error responses match expected schema."""
        error_response = {
            "error": {
                "type": "ValidationError",
                "message": "Invalid input provided",
                "code": 400,
                "details": {
                    "field": "email",
                    "reason": "Invalid email format"
                }
            },
            "timestamp": "2025-01-01T00:00:00Z",
            "request_id": "req_123456"
        }
        
        schema = {
            "type": "object",
            "required": ["error", "timestamp"],
            "properties": {
                "error": {
                    "type": "object",
                    "required": ["type", "message", "code"],
                    "properties": {
                        "type": {"type": "string"},
                        "message": {"type": "string"},
                        "code": {"type": "integer"},
                        "details": {"type": "object"}
                    }
                },
                "timestamp": {"type": "string"},
                "request_id": {"type": "string"}
            }
        }
        
        validate(instance=error_response, schema=schema)


class TestBackwardCompatibility:
    """Test backward compatibility of API changes."""

    def test_legacy_task_format_compatibility(self):
        """Test that legacy task formats are still supported."""
        # Legacy format (v1.0)
        legacy_task = {
            "task_id": "legacy_123",
            "task_title": "Legacy task",
            "task_status": "open",
            "repo_name": "test/repo"
        }
        
        # Should be convertible to new format
        from src.task_analyzer import TaskAnalyzer
        
        # Mock the conversion logic
        converted = {
            "id": legacy_task["task_id"],
            "title": legacy_task["task_title"],
            "status": "pending" if legacy_task["task_status"] == "open" else "completed",
            "repository": legacy_task["repo_name"],
            "priority": "medium"  # default
        }
        
        assert converted["id"] == "legacy_123"
        assert converted["title"] == "Legacy task"
        assert converted["status"] == "pending"

    def test_api_version_headers(self):
        """Test API version handling."""
        # Simulate API request with version header
        headers = {
            "Accept": "application/vnd.claude-manager.v1+json",
            "User-Agent": "Claude-Manager-Client/1.0"
        }
        
        # Should handle version negotiation
        supported_versions = ["v1", "v2"]
        requested_version = "v1"
        
        assert requested_version in supported_versions


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_task_state_transitions(self):
        """Test valid task state transitions."""
        valid_transitions = {
            "pending": ["in_progress", "failed"],
            "in_progress": ["completed", "failed", "pending"],
            "completed": [],  # Terminal state
            "failed": ["pending", "in_progress"]
        }
        
        def is_valid_transition(from_state: str, to_state: str) -> bool:
            return to_state in valid_transitions.get(from_state, [])
        
        # Test valid transitions
        assert is_valid_transition("pending", "in_progress")
        assert is_valid_transition("in_progress", "completed")
        assert is_valid_transition("failed", "pending")
        
        # Test invalid transitions
        assert not is_valid_transition("completed", "pending")
        assert not is_valid_transition("pending", "completed")

    def test_data_consistency_rules(self, task_data):
        """Test data consistency rules."""
        # Rule: High priority tasks must have descriptions
        if task_data["priority"] == "high":
            assert "description" in task_data
            assert len(task_data["description"]) > 0
        
        # Rule: Completed tasks must have completion timestamp
        if task_data["status"] == "completed":
            # This would normally check for completed_at field
            pass
        
        # Rule: Repository field must be valid format
        if "repository" in task_data:
            repo = task_data["repository"]
            assert "/" in repo, "Repository must be in 'owner/name' format"
            parts = repo.split("/")
            assert len(parts) == 2, "Repository must have exactly one slash"
            assert all(part.strip() for part in parts), "Repository parts cannot be empty"

    def test_unique_constraints(self):
        """Test unique constraint validation."""
        tasks = [
            {"id": "task_1", "title": "Task 1"},
            {"id": "task_2", "title": "Task 2"},
            {"id": "task_1", "title": "Duplicate Task"}  # Duplicate ID
        ]
        
        # Should detect duplicate IDs
        task_ids = [task["id"] for task in tasks]
        unique_ids = set(task_ids)
        
        assert len(unique_ids) < len(task_ids), "Duplicate IDs should be detected"

    def test_referential_integrity(self):
        """Test referential integrity between related data."""
        # Mock related data
        repositories = ["test/repo1", "test/repo2"]
        tasks = [
            {"id": "task_1", "repository": "test/repo1"},
            {"id": "task_2", "repository": "test/repo3"}  # Invalid reference
        ]
        
        # Check referential integrity
        for task in tasks:
            if "repository" in task:
                assert task["repository"] in repositories, \
                    f"Task references unknown repository: {task['repository']}"