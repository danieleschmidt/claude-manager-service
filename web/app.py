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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from flask import Flask, render_template, jsonify, request, Response
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
            return {"error": str(e), "items": []}
    
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
        # For now, return mock data
        return [
            {
                "id": f"task_{i}",
                "type": ["security", "bug", "feature", "documentation"][i % 4],
                "title": f"Example task {i+1}",
                "priority_score": round(8.5 - (i * 0.3), 1),
                "status": ["completed", "in_progress", "pending"][i % 3],
                "created_at": (datetime.now() - timedelta(hours=i*2)).isoformat(),
                "repository": f"example/repo{i%3 + 1}"
            }
            for i in range(limit)
        ]


# Initialize dashboard API
dashboard_api = DashboardAPI()


@app.route('/')
def index() -> str:
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/backlog')
def api_backlog() -> Response:
    """API endpoint for backlog status"""
    return jsonify(dashboard_api.get_backlog_status())


@app.route('/api/performance')
def api_performance() -> Response:
    """API endpoint for performance metrics"""
    return jsonify(dashboard_api.get_performance_metrics())


@app.route('/api/config')
def api_config() -> Response:
    """API endpoint for system configuration"""
    return jsonify(dashboard_api.get_system_config())


@app.route('/api/tasks')
def api_tasks() -> Response:
    """API endpoint for recent tasks"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(dashboard_api.get_recent_tasks(limit))


@app.route('/api/health')
def api_health() -> Response:
    """Health check endpoint"""
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