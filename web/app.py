"""
Claude Manager Service - Web Dashboard

This module provides a web interface for monitoring the Claude Manager Service,
including backlog status, task prioritization, performance metrics, and system health.

Features:
- Real-time backlog status display
- Task prioritization visualization
- Performance metrics dashboard
- System configuration overview
- Task execution history
"""

import os
import json
import time
import re
import html
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from functools import wraps
from collections import defaultdict

from flask import Flask, render_template, jsonify, request, Response, abort
from flask_cors import CORS

# Import Claude Manager Service modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from logger import get_logger
from performance_monitor import get_monitor
from config_validator import get_validated_config
from task_prioritization import TaskPrioritizer, prioritize_discovered_tasks

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API endpoints

# Configure logging
logger = get_logger(__name__)

# Initialize service components
performance_monitor = get_monitor()
task_prioritizer = TaskPrioritizer()


# Security classes and utilities
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        now = time.time()
        client_requests = self.clients[client_id]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests 
                             if now - req_time < self.window_seconds]
        
        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False
        
        # Record this request
        client_requests.append(now)
        return True


# Global rate limiter instance - configurable for testing
rate_limit_config = {
    'max_requests': int(os.getenv('RATE_LIMIT_MAX_REQUESTS', '10')),  # Lower default for testing
    'window_seconds': int(os.getenv('RATE_LIMIT_WINDOW_SECONDS', '60'))
}
rate_limiter = RateLimiter(max_requests=rate_limit_config['max_requests'], 
                          window_seconds=rate_limit_config['window_seconds'])


def rate_limit(f):
    """Decorator to add rate limiting to routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Use IP address as client identifier
        client_id = request.environ.get('REMOTE_ADDR', 'unknown')
        
        if not rate_limiter.is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        return f(*args, **kwargs)
    return decorated_function


def validate_limit_parameter(limit_str: str) -> int:
    """Validate and sanitize limit parameter"""
    try:
        limit = int(limit_str)
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        return limit
    except (ValueError, TypeError):
        raise ValueError("Invalid limit parameter")


def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent XSS and other attacks"""
    if not isinstance(user_input, str):
        return str(user_input)
    
    # HTML escape to prevent XSS
    sanitized = html.escape(user_input, quote=True)
    
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
        r'<script[^>]*>.*?</script>',
    ]
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized


def add_security_headers(response: Response) -> Response:
    """Add security headers to response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


@app.after_request
def after_request(response):
    """Add security headers to all responses"""
    return add_security_headers(response)


class DashboardAPI:
    """API handler for dashboard data"""
    
    def __init__(self) -> None:
        self.logger = get_logger(f"{__name__}.DashboardAPI")
    
    def get_backlog_status(self) -> Dict[str, Any]:
        """Get current backlog status from BACKLOG.md"""
        try:
            backlog_path = Path(__file__).parent.parent / "BACKLOG.md"
            if not backlog_path.exists():
                return {"error": "Backlog file not found", "items": []}
            
            # Parse backlog markdown file
            with open(backlog_path, 'r') as f:
                content = f.read()
            
            # Extract completed and pending items (simplified parsing)
            lines = content.split('\n')
            completed_count = len([line for line in lines if '✅' in line])
            total_items = len([line for line in lines if line.strip().startswith('###')])
            
            high_priority_completed = len([line for line in lines if '✅' in line and 'WSJF' in line and 'High Priority' in content[max(0, content.find(line)-500):content.find(line)+500]])
            
            return {
                "total_items": total_items,
                "completed_items": completed_count,
                "pending_items": total_items - completed_count,
                "completion_rate": round((completed_count / max(total_items, 1)) * 100, 1),
                "high_priority_completed": high_priority_completed,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error reading backlog status: {e}")
            # Return generic error message to avoid information disclosure
            return {"error": "Internal server error", "items": []}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            metrics = performance_monitor.get_current_metrics()
            
            # Format metrics for dashboard display
            return {
                "memory_usage": {
                    "current_mb": round(metrics.get("memory_usage_mb", 0), 2),
                    "peak_mb": round(metrics.get("peak_memory_mb", 0), 2)
                },
                "operation_metrics": metrics.get("operation_metrics", {}),
                "system_health": {
                    "uptime_hours": round(metrics.get("uptime_seconds", 0) / 3600, 2),
                    "last_operation": metrics.get("last_operation_time", "Never"),
                    "total_operations": metrics.get("total_operations", 0)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration summary"""
        try:
            config = get_validated_config()
            
            # Sanitize sensitive information
            safe_config = {
                "repositories": config.get("repositories", []),
                "analyzer": {
                    "scanForTodos": config.get("analyzer", {}).get("scanForTodos", False),
                    "scanOpenIssues": config.get("analyzer", {}).get("scanOpenIssues", False),
                    "maxFilesPerRepo": config.get("analyzer", {}).get("maxFilesPerRepo", 0)
                },
                "terragon": {
                    "enabled": bool(config.get("terragon", {}).get("apiKey"))
                },
                "manager_repo": config.get("managerRepo", "Not configured")
            }
            
            return safe_config
        except Exception as e:
            self.logger.error(f"Error getting system config: {e}")
            return {"error": str(e)}
    
    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task execution history"""
        # This would typically read from task execution logs or database
        # For now, return mock data with sanitized content
        tasks = []
        for i in range(limit):
            task = {
                "id": f"task_{i}",
                "type": ["security", "bug", "feature", "documentation"][i % 4],
                "title": f"Example task {i+1}",
                "priority_score": round(8.5 - (i * 0.3), 1),
                "status": ["completed", "in_progress", "pending"][i % 3],
                "created_at": (datetime.now() - timedelta(hours=i*2)).isoformat(),
                "repository": f"example/repo{i%3 + 1}"
            }
            
            # Sanitize all string fields
            for key, value in task.items():
                if isinstance(value, str):
                    task[key] = sanitize_user_input(value)
            
            tasks.append(task)
        
        return tasks


# Initialize dashboard API
dashboard_api = DashboardAPI()


@app.route('/')
def index() -> str:
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/backlog')
@rate_limit
def api_backlog() -> Response:
    """API endpoint for backlog status"""
    try:
        return jsonify(dashboard_api.get_backlog_status())
    except Exception as e:
        logger.error(f"Error in api_backlog: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/performance')
@rate_limit
def api_performance() -> Response:
    """API endpoint for performance metrics"""
    try:
        return jsonify(dashboard_api.get_performance_metrics())
    except Exception as e:
        logger.error(f"Error in api_performance: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/config')
@rate_limit
def api_config() -> Response:
    """API endpoint for system configuration"""
    try:
        return jsonify(dashboard_api.get_system_config())
    except Exception as e:
        logger.error(f"Error in api_config: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/tasks')
@rate_limit
def api_tasks() -> Response:
    """API endpoint for recent tasks"""
    try:
        # Validate and sanitize limit parameter
        limit_str = request.args.get('limit', '10')
        try:
            limit = validate_limit_parameter(limit_str)
        except ValueError as e:
            return jsonify({"error": "Invalid limit parameter"}), 400
        
        return jsonify(dashboard_api.get_recent_tasks(limit))
    except Exception as e:
        logger.error(f"Error in api_tasks: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/health')
@rate_limit
def api_health() -> Response:
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "performance_monitor": "active",
                "task_prioritizer": "active",
                "configuration": "loaded"
            }
        })
    except Exception as e:
        logger.error(f"Error in api_health: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Configure app
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Log startup
    logger.info("Starting Claude Manager Service Web Dashboard")
    logger.info(f"Debug mode: {app.config['DEBUG']}")
    
    # Run the app
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    host = os.getenv('DASHBOARD_HOST', '127.0.0.1')
    
    logger.info(f"Dashboard starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=app.config['DEBUG'])