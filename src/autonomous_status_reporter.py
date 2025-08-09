"""
Autonomous Status Reporter for Backlog Management

Generates comprehensive status reports including:
- Backlog status and progress
- DORA metrics 
- Quality metrics
- Security status
- Performance trends
- Recommendations for improvement
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .dora_metrics import DoraMetricsCollector
from .performance_monitor import PerformanceMonitor
from .logger import get_logger

logger = get_logger(__name__)


class AutonomousStatusReporter:
    """Generates comprehensive status reports for autonomous backlog management"""
    
    def __init__(
        self, 
        status_dir: Path = Path("docs/status"),
        metrics_dir: Path = Path("metrics")
    ):
        self.status_dir = status_dir
        self.metrics_dir = metrics_dir
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.dora_collector = DoraMetricsCollector(metrics_dir)
        self.performance_monitor = PerformanceMonitor()
        
    def generate_daily_report(self, completed_tasks: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive daily status report"""
        timestamp = datetime.now()
        
        # Load current backlog status
        backlog_status = self._get_backlog_status()
        
        # Get DORA metrics
        dora_report = self.dora_collector.export_metrics_report(days=30)
        
        # Get performance metrics
        performance_stats = self.performance_monitor.get_summary_stats()
        
        # Calculate conflict metrics from git
        conflict_metrics = self._calculate_conflict_metrics()
        
        # Get CI status
        ci_summary = self._get_ci_status()
        
        # Generate the report
        report = {
            "timestamp": timestamp.isoformat(),
            "report_type": "daily_autonomous_status",
            "completed_ids": completed_tasks or [],
            "backlog_summary": backlog_status,
            "dora_metrics": dora_report["dora_metrics"],
            "performance_tier": dora_report["performance_tier"],
            "quality_metrics": {
                "coverage_delta": self._calculate_coverage_delta(),
                "flaky_tests": self._detect_flaky_tests(),
                "quality_gate_success_rate": dora_report["dora_metrics"]["quality_gate_success_rate"],
                "security_scan_status": self._get_security_scan_status()
            },
            "operational_metrics": {
                "ci_summary": ci_summary,
                "open_prs": self._count_open_prs(),
                "conflict_metrics": conflict_metrics,
                "performance_stats": performance_stats
            },
            "risks_and_blocks": self._identify_risks_and_blocks(),
            "recommendations": dora_report["recommendations"],
            "autonomous_health": self._assess_autonomous_health(),
            "next_actions": self._generate_next_actions(backlog_status)
        }
        
        # Save the report
        self._save_report(report, timestamp)
        
        return report
        
    def _get_backlog_status(self) -> Dict[str, Any]:
        """Get current backlog status from discovery system"""
        try:
            # Load from the latest discovered backlog
            backlog_files = list(self.status_dir.glob("discovered_backlog_*.json"))
            if backlog_files:
                latest_backlog = max(backlog_files, key=lambda p: p.stat().st_mtime)
                with open(latest_backlog, 'r') as f:
                    backlog_data = json.load(f)
                    
                return {
                    "total_tasks": backlog_data.get("total_tasks", 0),
                    "tasks_by_type": backlog_data.get("tasks_by_type", {}),
                    "tasks_by_status": self._categorize_by_status(backlog_data.get("all_tasks", [])),
                    "avg_wsjf_score": self._calculate_avg_wsjf(backlog_data.get("all_tasks", [])),
                    "high_priority_count": len([
                        t for t in backlog_data.get("all_tasks", []) 
                        if t.get("wsjf_score", 0) > 10
                    ]),
                    "last_updated": backlog_data.get("timestamp", "")
                }
        except Exception as e:
            logger.warning(f"Failed to load backlog status: {e}")
            
        return {
            "total_tasks": 0,
            "tasks_by_type": {},
            "tasks_by_status": {},
            "avg_wsjf_score": 0,
            "high_priority_count": 0,
            "last_updated": ""
        }
        
    def _categorize_by_status(self, tasks: List[Dict]) -> Dict[str, int]:
        """Categorize tasks by their status"""
        status_counts = {}
        for task in tasks:
            status = task.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
        
    def _calculate_avg_wsjf(self, tasks: List[Dict]) -> float:
        """Calculate average WSJF score"""
        scores = [task.get("wsjf_score", 0) for task in tasks if task.get("wsjf_score")]
        return sum(scores) / len(scores) if scores else 0
        
    def _calculate_coverage_delta(self) -> str:
        """Calculate test coverage change"""
        # This would integrate with coverage tools
        # For now, return a placeholder
        return "stable"
        
    def _detect_flaky_tests(self) -> List[str]:
        """Detect flaky tests from recent runs"""
        # This would analyze test results over time
        # For now, return empty list
        return []
        
    def _get_security_scan_status(self) -> Dict[str, Any]:
        """Get security scan status"""
        return {
            "last_scan": datetime.now().isoformat(),
            "vulnerabilities_found": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "scan_tool": "placeholder"
        }
        
    def _get_ci_status(self) -> str:
        """Get CI pipeline status"""
        # This would integrate with CI system
        return "passing"
        
    def _count_open_prs(self) -> int:
        """Count open pull requests"""
        # This would integrate with GitHub API
        return 0
        
    def _calculate_conflict_metrics(self) -> Dict[str, Any]:
        """Calculate merge conflict resolution metrics"""
        try:
            # Get conflict events from DORA collector
            recent_conflicts = [
                c for c in self.dora_collector.conflicts
                if c.timestamp > time.time() - (7 * 24 * 3600)  # Last 7 days
            ]
            
            auto_resolved = [
                c for c in recent_conflicts 
                if c.success and c.resolution_method != "manual"
            ]
            
            return {
                "total_conflicts_7d": len(recent_conflicts),
                "auto_resolved_7d": len(auto_resolved),
                "auto_resolution_rate": (len(auto_resolved) / len(recent_conflicts) * 100) if recent_conflicts else 100,
                "avg_resolution_time_seconds": sum(c.resolution_time_seconds for c in recent_conflicts) / len(recent_conflicts) if recent_conflicts else 0,
                "common_resolution_methods": self._get_common_resolution_methods(recent_conflicts)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate conflict metrics: {e}")
            return {
                "total_conflicts_7d": 0,
                "auto_resolved_7d": 0,
                "auto_resolution_rate": 100,
                "avg_resolution_time_seconds": 0,
                "common_resolution_methods": []
            }
            
    def _get_common_resolution_methods(self, conflicts: List) -> List[str]:
        """Get most common conflict resolution methods"""
        methods = {}
        for conflict in conflicts:
            method = conflict.resolution_method
            methods[method] = methods.get(method, 0) + 1
            
        # Sort by frequency
        sorted_methods = sorted(methods.items(), key=lambda x: x[1], reverse=True)
        return [method for method, count in sorted_methods[:3]]
        
    def _identify_risks_and_blocks(self) -> List[str]:
        """Identify current risks and blocking issues"""
        risks = []
        
        # Check DORA metrics for risks
        dora_metrics = self.dora_collector.calculate_metrics(days=7)
        
        if dora_metrics.change_failure_rate > 15:
            risks.append("High change failure rate indicates quality issues")
            
        if dora_metrics.mttr_hours > 24:
            risks.append("Long MTTR indicates slow incident response")
            
        if dora_metrics.conflict_resolution_rate < 70:
            risks.append("Low conflict auto-resolution rate causing manual overhead")
            
        # Check for blocked tasks
        backlog_status = self._get_backlog_status()
        blocked_count = backlog_status["tasks_by_status"].get("BLOCKED", 0)
        if blocked_count > 5:
            risks.append(f"{blocked_count} tasks are blocked and need attention")
            
        return risks
        
    def _assess_autonomous_health(self) -> Dict[str, Any]:
        """Assess the health of the autonomous system"""
        dora_metrics = self.dora_collector.calculate_metrics(days=7)
        
        # Health score calculation (0-100)
        health_score = 100
        
        # Deduct points for poor metrics
        if dora_metrics.change_failure_rate > 10:
            health_score -= 20
        if dora_metrics.mttr_hours > 12:
            health_score -= 15
        if dora_metrics.conflict_resolution_rate < 80:
            health_score -= 15
        if dora_metrics.backlog_velocity < 1:
            health_score -= 10
            
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"
            
        return {
            "overall_score": max(0, health_score),
            "status": status,
            "key_indicators": {
                "deployment_automation": "active" if dora_metrics.deployment_frequency > 0.1 else "inactive",
                "conflict_resolution": "effective" if dora_metrics.conflict_resolution_rate > 80 else "needs_improvement",
                "quality_gates": "passing" if dora_metrics.quality_gate_success_rate > 90 else "failing",
                "backlog_processing": "active" if dora_metrics.backlog_velocity > 0.5 else "slow"
            }
        }
        
    def _generate_next_actions(self, backlog_status: Dict[str, Any]) -> List[str]:
        """Generate recommended next actions"""
        actions = []
        
        ready_tasks = backlog_status["tasks_by_status"].get("READY", 0)
        if ready_tasks > 0:
            actions.append(f"Process {ready_tasks} ready tasks from backlog")
            
        blocked_tasks = backlog_status["tasks_by_status"].get("BLOCKED", 0)
        if blocked_tasks > 0:
            actions.append(f"Review and unblock {blocked_tasks} blocked tasks")
            
        high_priority = backlog_status.get("high_priority_count", 0)
        if high_priority > 5:
            actions.append(f"Prioritize {high_priority} high-WSJF tasks")
            
        # Add maintenance actions
        last_update = backlog_status.get("last_updated", "")
        if last_update:
            try:
                last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                if datetime.now() - last_update_time.replace(tzinfo=None) > timedelta(hours=24):
                    actions.append("Refresh backlog discovery - last update > 24h ago")
            except:
                actions.append("Refresh backlog discovery - timestamp parse error")
                
        return actions
        
    def _save_report(self, report: Dict[str, Any], timestamp: datetime):
        """Save report to both JSON and markdown formats"""
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # Save JSON report
        json_file = self.status_dir / f"autonomous_status_{date_str}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate markdown report
        md_content = self._generate_markdown_report(report)
        md_file = self.status_dir / f"autonomous_status_{date_str}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
            
        logger.info(f"Status report saved: {json_file.name} and {md_file.name}")
        
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown version of the status report"""
        md = f"""# Autonomous Backlog Management Status Report

**Generated**: {report['timestamp']}  
**Performance Tier**: {report['performance_tier']}  
**Health Status**: {report['autonomous_health']['status']} ({report['autonomous_health']['overall_score']}/100)

## Executive Summary

{len(report['completed_ids'])} tasks completed today. {report['backlog_summary']['total_tasks']} total tasks in backlog.

## DORA Metrics (30-day window)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Deployment Frequency | {report['dora_metrics']['deployment_frequency']:.2f}/day | > 1/day (Elite) |
| Lead Time | {report['dora_metrics']['lead_time_hours']:.1f}h | < 24h (Elite) |
| Change Failure Rate | {report['dora_metrics']['change_failure_rate']:.1f}% | < 5% (Elite) |
| MTTR | {report['dora_metrics']['mttr_hours']:.1f}h | < 1h (Elite) |

## Backlog Status

- **Total Tasks**: {report['backlog_summary']['total_tasks']}
- **High Priority (WSJF > 10)**: {report['backlog_summary']['high_priority_count']}
- **Average WSJF Score**: {report['backlog_summary']['avg_wsjf_score']:.1f}

### Tasks by Status
"""
        
        for status, count in report['backlog_summary']['tasks_by_status'].items():
            md += f"- **{status}**: {count}\n"
            
        md += f"""
### Tasks by Type
"""
        
        for task_type, count in report['backlog_summary']['tasks_by_type'].items():
            md += f"- **{task_type}**: {count}\n"
            
        md += f"""
## Quality Metrics

- **Coverage Delta**: {report['quality_metrics']['coverage_delta']}
- **Quality Gate Success Rate**: {report['quality_metrics']['quality_gate_success_rate']:.1f}%
- **Flaky Tests**: {len(report['quality_metrics']['flaky_tests'])}

## Operational Metrics

- **CI Status**: {report['operational_metrics']['ci_summary']}
- **Open PRs**: {report['operational_metrics']['open_prs']}
- **Conflict Auto-Resolution Rate**: {report['operational_metrics']['conflict_metrics']['auto_resolution_rate']:.1f}%

## Risks and Blockers
"""
        
        if report['risks_and_blocks']:
            for risk in report['risks_and_blocks']:
                md += f"- ⚠️ {risk}\n"
        else:
            md += "- ✅ No significant risks identified\n"
            
        md += f"""
## Recommendations
"""
        
        if report['recommendations']:
            for rec in report['recommendations']:
                md += f"- 💡 {rec}\n"
        else:
            md += "- ✅ System performing well, no immediate improvements needed\n"
            
        md += f"""
## Next Actions
"""
        
        for action in report['next_actions']:
            md += f"- 🎯 {action}\n"
            
        md += f"""
## Autonomous System Health

| Component | Status |
|-----------|---------|
| Deployment Automation | {report['autonomous_health']['key_indicators']['deployment_automation']} |
| Conflict Resolution | {report['autonomous_health']['key_indicators']['conflict_resolution']} |
| Quality Gates | {report['autonomous_health']['key_indicators']['quality_gates']} |
| Backlog Processing | {report['autonomous_health']['key_indicators']['backlog_processing']} |

---
*Report generated by Autonomous Backlog Management System*
"""
        
        return md