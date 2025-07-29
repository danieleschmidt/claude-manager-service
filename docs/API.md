# Claude Code Manager API Documentation

## Overview

The Claude Code Manager provides a RESTful API for autonomous task management and code generation. This document describes the available endpoints, authentication methods, and response formats.

## Base URL

```
http://localhost:5000
```

## Authentication

API access requires a valid GitHub Personal Access Token (PAT) with appropriate repository permissions.

### Headers

```http
Authorization: Bearer <your-github-token>
Content-Type: application/json
```

## Endpoints

### Health Check

#### `GET /health`

Returns the health status of the application.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-29T10:30:00Z",
  "version": "0.1.0",
  "uptime": 3600
}
```

#### `GET /status`

Returns detailed system status including component health.

**Response:**
```json
{
  "status": "operational",
  "components": {
    "database": "healthy",
    "github_api": "healthy", 
    "task_analyzer": "healthy",
    "orchestrator": "healthy"
  },
  "metrics": {
    "active_tasks": 5,
    "completed_tasks_today": 12,
    "error_rate": 0.02,
    "avg_response_time_ms": 150
  }
}
```

### Task Management

#### `GET /api/v1/tasks`

List all tasks with optional filtering.

**Query Parameters:**
- `status` (optional): Filter by task status (`pending`, `in_progress`, `completed`, `failed`)
- `repository` (optional): Filter by repository name
- `limit` (optional): Number of results to return (default: 50, max: 200)
- `offset` (optional): Number of results to skip (default: 0)

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-123",
      "title": "Fix authentication bug",
      "repository": "org/repo",
      "status": "in_progress",
      "priority": "high",
      "created_at": "2025-07-29T08:00:00Z",
      "updated_at": "2025-07-29T09:15:00Z",
      "assigned_to": "claude-agent",
      "labels": ["bug", "security"],
      "estimated_duration": 3600,
      "progress": 65
    }
  ],
  "pagination": {
    "total": 156,
    "limit": 50,
    "offset": 0,
    "has_next": true
  }
}
```

#### `POST /api/v1/tasks`

Create a new task.

**Request Body:**
```json
{
  "title": "Implement user authentication",
  "description": "Add OAuth2 authentication to the API",
  "repository": "org/repo", 
  "priority": "high",
  "labels": ["feature", "security"],
  "executor": "claude-flow"
}
```

**Response:**
```json
{
  "id": "task-124",
  "title": "Implement user authentication",
  "status": "pending",
  "created_at": "2025-07-29T10:30:00Z",
  "github_issue_url": "https://github.com/org/repo/issues/42"
}
```

#### `GET /api/v1/tasks/{task_id}`

Get details for a specific task.

**Response:**
```json
{
  "id": "task-123",
  "title": "Fix authentication bug",
  "description": "The OAuth2 flow is not properly validating tokens",
  "repository": "org/repo",
  "status": "in_progress",
  "priority": "high",
  "created_at": "2025-07-29T08:00:00Z",
  "updated_at": "2025-07-29T09:15:00Z",
  "assigned_to": "claude-agent",
  "labels": ["bug", "security"],
  "estimated_duration": 3600,
  "actual_duration": 2340,
  "progress": 65,
  "github_issue_url": "https://github.com/org/repo/issues/41",
  "pull_request_url": "https://github.com/org/repo/pull/42",
  "logs": [
    {
      "timestamp": "2025-07-29T08:15:00Z",
      "level": "info",
      "message": "Started analysis of authentication flow"
    }
  ]
}
```

#### `PUT /api/v1/tasks/{task_id}`

Update a task.

**Request Body:**
```json
{
  "status": "completed",
  "priority": "medium"
}
```

#### `DELETE /api/v1/tasks/{task_id}`

Cancel/delete a task.

### Repository Management

#### `GET /api/v1/repositories`

List monitored repositories.

**Response:**
```json
{
  "repositories": [
    {
      "name": "org/repo1",
      "status": "active",
      "last_scan": "2025-07-29T06:00:00Z",
      "task_count": {
        "pending": 3,
        "in_progress": 1,
        "completed": 25
      },
      "health_score": 8.5
    }
  ]
}
```

#### `POST /api/v1/repositories/{repo_name}/scan`

Trigger a manual scan of a repository.

**Response:**
```json
{
  "scan_id": "scan-456",
  "status": "started",
  "estimated_completion": "2025-07-29T10:45:00Z"
}
```

### Analytics

#### `GET /api/v1/metrics`

Get system performance metrics.

**Query Parameters:**
- `period` (optional): Time period (`hour`, `day`, `week`, `month`)
- `repository` (optional): Filter by repository

**Response:**
```json
{
  "period": "day",
  "metrics": {
    "tasks_created": 15,
    "tasks_completed": 12,
    "tasks_failed": 1,
    "avg_completion_time_minutes": 45,
    "success_rate": 0.92,
    "performance_score": 8.7
  },
  "trends": {
    "completion_rate_change": "+5%",
    "performance_change": "-2%"
  }
}
```

#### `GET /api/v1/reports/dora`

Get DORA (DevOps Research and Assessment) metrics.

**Response:**
```json
{
  "deployment_frequency": {
    "value": 2.5,
    "unit": "per_day",
    "classification": "high"
  },
  "lead_time": {
    "value": 4.2,
    "unit": "hours",
    "classification": "high"
  },
  "mean_time_to_recovery": {
    "value": 1.8,
    "unit": "hours", 
    "classification": "high"
  },
  "change_failure_rate": {
    "value": 0.05,
    "unit": "percentage",
    "classification": "low"
  }
}
```

## Error Responses

All endpoints may return these standard error responses:

### 400 Bad Request
```json
{
  "error": "validation_error",
  "message": "Invalid repository name format",
  "details": {
    "field": "repository",
    "expected": "owner/repo"
  }
}
```

### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Invalid or missing authentication token"
}
```

### 403 Forbidden
```json
{
  "error": "forbidden", 
  "message": "Insufficient permissions for repository access"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "Task not found",
  "resource_id": "task-123"
}
```

### 429 Too Many Requests
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests, please try again later",
  "retry_after": 60
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "request_id": "req-789"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Global**: 1000 requests per hour per API key
- **Per endpoint**: 100 requests per minute per endpoint
- **Burst**: Up to 10 requests per second

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1690623600
```

## Webhooks

The system can send webhooks for task events:

### Webhook Events

- `task.created` - New task created
- `task.started` - Task execution started
- `task.completed` - Task completed successfully
- `task.failed` - Task execution failed
- `repository.scanned` - Repository scan completed

### Webhook Payload Example

```json
{
  "event": "task.completed",
  "timestamp": "2025-07-29T10:30:00Z",
  "data": {
    "task_id": "task-123",
    "repository": "org/repo",
    "title": "Fix authentication bug",
    "duration_seconds": 2340,
    "pull_request_url": "https://github.com/org/repo/pull/42"
  }
}
```

## SDK and Client Libraries

Official SDKs are available for:

- **Python**: `pip install claude-code-manager-python`
- **JavaScript/Node.js**: `npm install claude-code-manager-js`
- **Go**: `go get github.com/terragon-labs/claude-code-manager-go`

### Python SDK Example

```python
from claude_code_manager import Client

client = Client(token="your-github-token")

# Create a task
task = client.tasks.create(
    title="Fix bug in authentication",
    repository="org/repo",
    priority="high"
)

# Monitor task progress
while task.status != "completed":
    task.refresh()
    print(f"Progress: {task.progress}%")
```

## API Versioning

The API uses URL-based versioning. The current version is `v1`. 

When breaking changes are introduced, a new version will be released with backward compatibility maintained for at least 12 months.

## Support

For API support and questions:

- **Documentation**: https://docs.terragon.ai/claude-code-manager
- **Issues**: https://github.com/terragon-labs/claude-code-manager/issues
- **Email**: api-support@terragon.ai