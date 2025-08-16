"""
DORA Metrics Implementation for Autonomous Backlog Management

Implements the four key DevOps Research and Assessment (DORA) metrics:
1. Deployment Frequency - How often we deploy code to production
2. Lead Time for Changes - Time from code commit to production deployment  
3. Change Failure Rate - Percentage of deployments causing failures
4. Time to Restore Service - How quickly we recover from failures

Additional metrics for autonomous management:
- Cycle Time - Time from task discovery to completion
- Conflict Resolution Rate - Percentage of merge conflicts auto-resolved
- Backlog Velocity - Tasks completed per time period
- Quality Gate Success Rate - Percentage of changes passing all quality checks
"""

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from src.logger import get_logger

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DeploymentEvent:
    """Represents a deployment event"""
    timestamp: float
    commit_sha: str
    branch: str
    environment: str
    success: bool
    duration_seconds: float
    artifacts_deployed: List[str]
    rollback_required: bool = False
    
    
@dataclass  
class ChangeEvent:
    """Represents a code change event"""
    commit_sha: str
    commit_timestamp: float
    merge_timestamp: Optional[float]
    deploy_timestamp: Optional[float]
    author: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    task_id: Optional[str] = None


@dataclass
class IncidentEvent:
    """Represents a production incident"""
    incident_id: str
    start_timestamp: float
    end_timestamp: Optional[float]
    severity: IncidentSeverity
    root_cause_commit: Optional[str]
    resolution_commit: Optional[str]
    affected_services: List[str]
    customer_impact: bool
    automated_resolution: bool = False


@dataclass
class ConflictEvent:
    """Represents a merge conflict resolution"""
    timestamp: float
    branch: str
    files_affected: List[str]
    resolution_method: str  # "rerere", "merge_driver", "manual"
    resolution_time_seconds: float
    success: bool


@dataclass
class DoraMetrics:
    """DORA metrics calculation results"""
    # Core DORA metrics
    deployment_frequency: float  # deployments per day
    lead_time_hours: float  # hours from commit to production
    change_failure_rate: float  # percentage (0-100)
    mttr_hours: float  # mean time to recovery in hours
    
    # Extended metrics for autonomous management
    cycle_time_hours: float  # hours from task discovery to completion
    conflict_resolution_rate: float  # percentage auto-resolved
    backlog_velocity: float  # tasks completed per day
    quality_gate_success_rate: float  # percentage passing all checks
    
    # Metadata
    measurement_period_days: int
    total_deployments: int
    total_commits: int
    total_incidents: int
    calculation_timestamp: float


class DoraMetricsCollector:
    """Collects and calculates DORA metrics for autonomous backlog management"""
    
    def __init__(self, metrics_dir: Path = Path("metrics")):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Event storage
        self.deployments: deque = deque(maxlen=1000)
        self.changes: deque = deque(maxlen=1000)
        self.incidents: deque = deque(maxlen=100)
        self.conflicts: deque = deque(maxlen=500)
        self.task_completions: deque = deque(maxlen=1000)
        
        # Load existing events
        self._load_events()
        
    def _load_events(self):
        """Load events from persistent storage"""
        try:
            events_file = self.metrics_dir / "dora_events.json"
            if events_file.exists():
                with open(events_file, 'r') as f:
                    data = json.load(f)
                    
                self.deployments.extend([
                    DeploymentEvent(**event) for event in data.get('deployments', [])
                ])
                self.changes.extend([
                    ChangeEvent(**event) for event in data.get('changes', [])  
                ])
                self.incidents.extend([
                    IncidentEvent(**{**event, 'severity': IncidentSeverity(event['severity'])}) 
                    for event in data.get('incidents', [])
                ])
                self.conflicts.extend([
                    ConflictEvent(**event) for event in data.get('conflicts', [])
                ])
                self.task_completions.extend(data.get('task_completions', []))
                
                logger.info("Loaded DORA metrics events from storage")
        except Exception as e:
            logger.warning(f"Failed to load DORA events: {e}")
            
    def _save_events(self):
        """Save events to persistent storage"""
        try:
            events_file = self.metrics_dir / "dora_events.json"
            data = {
                'deployments': [asdict(event) for event in self.deployments],
                'changes': [asdict(event) for event in self.changes],
                'incidents': [
                    {**asdict(event), 'severity': event.severity.value} 
                    for event in self.incidents
                ],
                'conflicts': [asdict(event) for event in self.conflicts],
                'task_completions': list(self.task_completions)
            }
            
            with open(events_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save DORA events: {e}")
            
    def record_deployment(
        self, 
        commit_sha: str,
        branch: str, 
        environment: str,
        success: bool,
        duration_seconds: float,
        artifacts: List[str] = None
    ):
        """Record a deployment event"""
        event = DeploymentEvent(
            timestamp=time.time(),
            commit_sha=commit_sha,
            branch=branch,
            environment=environment,
            success=success,
            duration_seconds=duration_seconds,
            artifacts_deployed=artifacts or []
        )
        self.deployments.append(event)
        self._save_events()
        logger.info(f"Recorded deployment: {commit_sha[:8]} to {environment}")
        
    def record_change(
        self,
        commit_sha: str,
        commit_timestamp: float,
        author: str,
        files_changed: int,
        lines_added: int,
        lines_deleted: int,
        task_id: str = None
    ):
        """Record a code change event"""
        event = ChangeEvent(
            commit_sha=commit_sha,
            commit_timestamp=commit_timestamp,
            merge_timestamp=None,
            deploy_timestamp=None,
            author=author,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            task_id=task_id
        )
        self.changes.append(event)
        self._save_events()
        logger.info(f"Recorded change: {commit_sha[:8]} by {author}")
        
    def record_incident(
        self,
        incident_id: str,
        severity: IncidentSeverity,
        affected_services: List[str],
        root_cause_commit: str = None,
        customer_impact: bool = True
    ):
        """Record a production incident"""
        event = IncidentEvent(
            incident_id=incident_id,
            start_timestamp=time.time(),
            end_timestamp=None,
            severity=severity,
            root_cause_commit=root_cause_commit,
            resolution_commit=None,
            affected_services=affected_services,
            customer_impact=customer_impact
        )
        self.incidents.append(event)
        self._save_events()
        logger.warning(f"Recorded incident: {incident_id} ({severity.value})")
        
    def resolve_incident(self, incident_id: str, resolution_commit: str = None):
        """Mark an incident as resolved"""
        for incident in reversed(self.incidents):
            if incident.incident_id == incident_id and incident.end_timestamp is None:
                incident.end_timestamp = time.time()
                incident.resolution_commit = resolution_commit
                self._save_events()
                duration = incident.end_timestamp - incident.start_timestamp
                logger.info(f"Resolved incident: {incident_id} in {duration/3600:.1f}h")
                return
        logger.warning(f"Incident not found for resolution: {incident_id}")
        
    def record_conflict_resolution(
        self,
        branch: str,
        files_affected: List[str],
        resolution_method: str,
        resolution_time_seconds: float,
        success: bool
    ):
        """Record merge conflict resolution"""
        event = ConflictEvent(
            timestamp=time.time(),
            branch=branch,
            files_affected=files_affected,
            resolution_method=resolution_method,
            resolution_time_seconds=resolution_time_seconds,
            success=success
        )
        self.conflicts.append(event)
        self._save_events()
        logger.info(f"Recorded conflict resolution: {resolution_method} on {branch}")
        
    def record_task_completion(
        self,
        task_id: str,
        start_timestamp: float,
        completion_timestamp: float,
        quality_gates_passed: bool
    ):
        """Record task completion for cycle time tracking"""
        completion_event = {
            'task_id': task_id,
            'start_timestamp': start_timestamp,
            'completion_timestamp': completion_timestamp,
            'cycle_time_hours': (completion_timestamp - start_timestamp) / 3600,
            'quality_gates_passed': quality_gates_passed
        }
        self.task_completions.append(completion_event)
        self._save_events()
        
    def calculate_metrics(self, days: int = 30) -> DoraMetrics:
        """Calculate DORA metrics for the specified period"""
        end_time = time.time()
        start_time = end_time - (days * 24 * 3600)
        
        # Filter events to the measurement period
        period_deployments = [
            d for d in self.deployments 
            if start_time <= d.timestamp <= end_time
        ]
        period_changes = [
            c for c in self.changes
            if start_time <= c.commit_timestamp <= end_time
        ]
        period_incidents = [
            i for i in self.incidents
            if start_time <= i.start_timestamp <= end_time
        ]
        period_conflicts = [
            c for c in self.conflicts
            if start_time <= c.timestamp <= end_time
        ]
        period_completions = [
            t for t in self.task_completions
            if start_time <= t['completion_timestamp'] <= end_time
        ]
        
        # Calculate core DORA metrics
        deployment_frequency = len(period_deployments) / days if days > 0 else 0
        
        # Lead time calculation (commit to deployment)
        lead_times = []
        for change in period_changes:
            for deployment in period_deployments:
                if deployment.commit_sha == change.commit_sha:
                    lead_time_hours = (deployment.timestamp - change.commit_timestamp) / 3600
                    lead_times.append(lead_time_hours)
                    break
        avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0
        
        # Change failure rate
        failed_deployments = [d for d in period_deployments if not d.success]
        change_failure_rate = (len(failed_deployments) / len(period_deployments) * 100) if period_deployments else 0
        
        # Mean time to recovery (MTTR)
        resolved_incidents = [i for i in period_incidents if i.end_timestamp is not None]
        recovery_times = [
            (i.end_timestamp - i.start_timestamp) / 3600 
            for i in resolved_incidents
        ]
        mttr = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Extended metrics
        cycle_times = [t['cycle_time_hours'] for t in period_completions]
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
        
        auto_resolved_conflicts = [c for c in period_conflicts if c.success and c.resolution_method != "manual"]
        conflict_resolution_rate = (len(auto_resolved_conflicts) / len(period_conflicts) * 100) if period_conflicts else 100
        
        backlog_velocity = len(period_completions) / days if days > 0 else 0
        
        quality_passed = [t for t in period_completions if t['quality_gates_passed']]
        quality_gate_success_rate = (len(quality_passed) / len(period_completions) * 100) if period_completions else 100
        
        return DoraMetrics(
            deployment_frequency=deployment_frequency,
            lead_time_hours=avg_lead_time,
            change_failure_rate=change_failure_rate,
            mttr_hours=mttr,
            cycle_time_hours=avg_cycle_time,
            conflict_resolution_rate=conflict_resolution_rate,
            backlog_velocity=backlog_velocity,
            quality_gate_success_rate=quality_gate_success_rate,
            measurement_period_days=days,
            total_deployments=len(period_deployments),
            total_commits=len(period_changes),
            total_incidents=len(period_incidents),
            calculation_timestamp=time.time()
        )
        
    def export_metrics_report(self, days: int = 30) -> Dict[str, Any]:
        """Export comprehensive metrics report"""
        metrics = self.calculate_metrics(days)
        
        # Performance benchmarks (based on DORA research)
        benchmarks = {
            "elite": {
                "deployment_frequency": "> 1 per day",
                "lead_time_hours": "< 24 hours", 
                "change_failure_rate": "< 5%",
                "mttr_hours": "< 1 hour"
            },
            "high": {
                "deployment_frequency": "Weekly to monthly",
                "lead_time_hours": "< 168 hours (1 week)",
                "change_failure_rate": "5-10%", 
                "mttr_hours": "< 24 hours"
            },
            "medium": {
                "deployment_frequency": "Monthly to bimonthly",
                "lead_time_hours": "168-720 hours (1-4 weeks)",
                "change_failure_rate": "10-15%",
                "mttr_hours": "24-168 hours"
            },
            "low": {
                "deployment_frequency": "< 6 months",
                "lead_time_hours": "> 720 hours (> 4 weeks)",
                "change_failure_rate": "> 15%",
                "mttr_hours": "> 168 hours"
            }
        }
        
        # Determine performance tier
        def get_performance_tier(metrics: DoraMetrics) -> str:
            if (metrics.deployment_frequency > 1 and 
                metrics.lead_time_hours < 24 and
                metrics.change_failure_rate < 5 and
                metrics.mttr_hours < 1):
                return "elite"
            elif (metrics.deployment_frequency > 0.25 and  # > weekly
                  metrics.lead_time_hours < 168 and
                  metrics.change_failure_rate < 10 and
                  metrics.mttr_hours < 24):
                return "high"
            elif (metrics.deployment_frequency > 0.125 and  # > biweekly
                  metrics.lead_time_hours < 720 and
                  metrics.change_failure_rate < 15 and
                  metrics.mttr_hours < 168):
                return "medium"
            else:
                return "low"
        
        performance_tier = get_performance_tier(metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "measurement_period_days": days,
            "dora_metrics": asdict(metrics),
            "performance_tier": performance_tier,
            "benchmarks": benchmarks,
            "trend_data": self._calculate_trends(),
            "recommendations": self._generate_recommendations(metrics, performance_tier)
        }
        
    def _calculate_trends(self) -> Dict[str, List[float]]:
        """Calculate metric trends over time"""
        # Calculate metrics for last 4 weeks to show trend
        trends = {}
        for weeks_back in range(4, 0, -1):
            end_time = time.time() - ((weeks_back - 1) * 7 * 24 * 3600)
            start_time = end_time - (7 * 24 * 3600)
            
            # Filter events for this week
            week_deployments = [
                d for d in self.deployments 
                if start_time <= d.timestamp <= end_time
            ]
            week_conflicts = [
                c for c in self.conflicts
                if start_time <= c.timestamp <= end_time
            ]
            week_completions = [
                t for t in self.task_completions
                if start_time <= t['completion_timestamp'] <= end_time
            ]
            
            if f"deployment_frequency" not in trends:
                trends["deployment_frequency"] = []
                trends["conflict_resolution_rate"] = []
                trends["backlog_velocity"] = []
                
            trends["deployment_frequency"].append(len(week_deployments))
            
            auto_resolved = [c for c in week_conflicts if c.success and c.resolution_method != "manual"]
            conflict_rate = (len(auto_resolved) / len(week_conflicts) * 100) if week_conflicts else 100
            trends["conflict_resolution_rate"].append(conflict_rate)
            
            trends["backlog_velocity"].append(len(week_completions))
            
        return trends
        
    def _generate_recommendations(self, metrics: DoraMetrics, tier: str) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        if metrics.deployment_frequency < 0.25:  # Less than weekly
            recommendations.append("Increase deployment frequency by implementing smaller, more frequent releases")
            
        if metrics.lead_time_hours > 48:
            recommendations.append("Reduce lead time by optimizing CI/CD pipeline and reducing batch sizes")
            
        if metrics.change_failure_rate > 10:
            recommendations.append("Improve change failure rate with better testing and feature flags")
            
        if metrics.mttr_hours > 24:
            recommendations.append("Implement faster incident response and automated rollback procedures")
            
        if metrics.conflict_resolution_rate < 80:
            recommendations.append("Enhance merge conflict automation with better rerere patterns")
            
        if metrics.cycle_time_hours > 72:
            recommendations.append("Optimize task breakdown to reduce cycle time")
            
        if tier == "low":
            recommendations.append("Consider implementing trunk-based development and continuous integration")
            
        return recommendations