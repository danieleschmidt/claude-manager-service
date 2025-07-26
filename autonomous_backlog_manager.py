#!/usr/bin/env python3
"""
Autonomous Senior Coding Assistant - Complete Backlog Management System

This is the main orchestrator that implements the comprehensive autonomous backlog 
management system according to the specifications. It continuously discovers, 
prioritizes, and executes all actionable work until no tasks remain.

Key Features:
- Complete backlog discovery from multiple sources
- WSJF-based prioritization
- TDD micro-cycles with security validation
- Automated merge conflict resolution
- DORA metrics collection
- Comprehensive status reporting
- Safety constraints and human escalation
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import all the components we've built
from src.continuous_backlog_executor import ContinuousBacklogExecutor, TaskStatus
from src.dora_metrics import DoraMetricsCollector, IncidentSeverity
from src.autonomous_status_reporter import AutonomousStatusReporter
from src.security_scanner import SecurityScanner
from src.async_task_analyzer import AsyncTaskAnalyzer
from src.services.configuration_service import ConfigurationService
from src.logger import get_logger

logger = get_logger(__name__)


class AutonomousBacklogManager:
    """
    Main autonomous backlog management system orchestrator.
    
    Implements the complete autonomous senior coding assistant that:
    1. Discovers tasks from all sources
    2. Scores and prioritizes using WSJF
    3. Executes through TDD micro-cycles
    4. Handles merge conflicts automatically
    5. Tracks DORA metrics
    6. Reports comprehensive status
    """
    
    def __init__(self, config_path: str = "config.json", workspace_root: Path = Path(".")):
        self.workspace_root = workspace_root
        self.config_path = config_path
        self.running = False
        
        # Load automation scope configuration
        self.automation_scope = self._load_automation_scope()
        
        # Initialize components
        self.config_service = ConfigurationService()
        self.backlog_executor = ContinuousBacklogExecutor()
        self.dora_collector = DoraMetricsCollector()
        self.status_reporter = AutonomousStatusReporter()
        self.security_scanner = SecurityScanner()
        self.task_analyzer = AsyncTaskAnalyzer()
        
        # Execution state
        self.completed_tasks = []
        self.current_cycle = 0
        self.start_time = time.time()
        self.last_report_time = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _load_automation_scope(self) -> Dict[str, Any]:
        """Load automation scope configuration"""
        scope_file = self.workspace_root / ".automation-scope.yaml"
        if scope_file.exists():
            try:
                with open(scope_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load automation scope: {e}")
                
        # Default scope - workspace only
        return {
            "workspace_root": "./",
            "external_operations": {"allowed": False},
            "safety": {
                "max_files_per_operation": 10,
                "require_approval": ["public_api_changes", "security_sensitive_files"],
                "protected_files": [".github/workflows/*", "*.yml", "*.yaml"]
            }
        }
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        
    async def run_autonomous_cycle(self, max_cycles: int = None, max_duration_hours: int = None) -> Dict[str, Any]:
        """
        Run the complete autonomous backlog management cycle.
        
        This is the main macro execution loop that:
        1. Syncs repository and CI state
        2. Discovers new tasks from all sources
        3. Scores and sorts backlog using WSJF
        4. Executes highest priority ready tasks
        5. Reports status and metrics
        6. Continues until backlog is empty or constraints reached
        """
        self.running = True
        execution_summary = {
            "start_time": datetime.now().isoformat(),
            "cycles_completed": 0,
            "tasks_completed": [],
            "tasks_blocked": [],
            "errors_encountered": [],
            "final_status": "unknown"
        }
        
        logger.info("🚀 Starting Autonomous Backlog Management System")
        logger.info(f"Max cycles: {max_cycles or 'unlimited'}")
        logger.info(f"Max duration: {max_duration_hours or 'unlimited'} hours")
        
        try:
            while self.running:
                cycle_start_time = time.time()
                self.current_cycle += 1
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Starting Cycle {self.current_cycle}")
                logger.info(f"{'='*60}")
                
                # Check termination conditions
                if max_cycles and self.current_cycle > max_cycles:
                    logger.info(f"Reached maximum cycles ({max_cycles})")
                    execution_summary["final_status"] = "max_cycles_reached"
                    break
                    
                if max_duration_hours:
                    elapsed_hours = (time.time() - self.start_time) / 3600
                    if elapsed_hours > max_duration_hours:
                        logger.info(f"Reached maximum duration ({max_duration_hours}h)")
                        execution_summary["final_status"] = "max_duration_reached"
                        break
                
                try:
                    # Step 1: Sync repository and CI state
                    await self._sync_repo_and_ci()
                    
                    # Step 2: Comprehensive task discovery
                    discovered_tasks = await self._discover_all_tasks()
                    
                    if not discovered_tasks:
                        logger.info("✅ No tasks found in backlog - work is complete!")
                        execution_summary["final_status"] = "backlog_empty"
                        break
                    
                    # Step 3: Score and sort backlog using WSJF
                    prioritized_backlog = await self._score_and_sort_backlog(discovered_tasks)
                    
                    # Step 4: Find next ready task to execute
                    next_task = self._get_next_ready_task(prioritized_backlog)
                    
                    if not next_task:
                        logger.warning("⚠️ No ready tasks available - all are blocked or require approval")
                        
                        # Check if we should escalate blocked tasks
                        blocked_tasks = [t for t in prioritized_backlog if t.get("status") == "BLOCKED"]
                        if len(blocked_tasks) > 5:
                            await self._escalate_blocked_tasks(blocked_tasks)
                            
                        # Wait before next cycle
                        await asyncio.sleep(300)  # 5 minutes
                        continue
                    
                    # Step 5: Execute the task through micro-cycle
                    task_result = await self._execute_micro_cycle(next_task)
                    
                    if task_result["success"]:
                        self.completed_tasks.append(next_task["id"])
                        execution_summary["tasks_completed"].append(next_task["id"])
                        logger.info(f"✅ Completed task: {next_task['title']}")
                        
                        # Record metrics
                        await self._record_task_completion(next_task, task_result)
                        
                    else:
                        execution_summary["tasks_blocked"].append(next_task["id"])
                        logger.warning(f"❌ Task blocked: {next_task['title']}")
                        
                        # Check if task should be escalated
                        if self._should_escalate_task(next_task, task_result):
                            await self._escalate_task(next_task, task_result)
                    
                    # Step 6: Generate status report (every hour or after significant events)
                    if self._should_generate_report():
                        await self._generate_status_report()
                        
                except Exception as e:
                    logger.error(f"Error in cycle {self.current_cycle}: {e}")
                    execution_summary["errors_encountered"].append(str(e))
                    
                    # Implement exponential backoff for errors
                    await asyncio.sleep(min(300, 30 * len(execution_summary["errors_encountered"])))
                    
                # Log cycle completion
                cycle_duration = time.time() - cycle_start_time
                logger.info(f"Cycle {self.current_cycle} completed in {cycle_duration:.1f}s")
                execution_summary["cycles_completed"] = self.current_cycle
                
                # Brief pause between cycles
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            execution_summary["final_status"] = "user_interrupted"
            
        except Exception as e:
            logger.error(f"Fatal error in autonomous cycle: {e}")
            execution_summary["final_status"] = "fatal_error"
            execution_summary["errors_encountered"].append(str(e))
            
        finally:
            # Generate final report
            execution_summary["end_time"] = datetime.now().isoformat()
            execution_summary["total_duration_hours"] = (time.time() - self.start_time) / 3600
            
            await self._generate_final_report(execution_summary)
            logger.info("🏁 Autonomous Backlog Management System shutdown complete")
            
        return execution_summary
        
    async def _sync_repo_and_ci(self):
        """Sync repository state and CI status"""
        logger.info("🔄 Syncing repository and CI state...")
        
        try:
            # Git operations
            import subprocess
            
            # Fetch latest changes
            result = subprocess.run(["git", "fetch", "origin"], 
                                 capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode != 0:
                logger.warning(f"Git fetch failed: {result.stderr}")
                
            # Check if we're behind main
            result = subprocess.run(["git", "rev-list", "--count", "HEAD..origin/main"], 
                                 capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0 and result.stdout.strip() != "0":
                commits_behind = result.stdout.strip()
                logger.info(f"Branch is {commits_behind} commits behind main")
                
                # Auto-rebase if safe
                if int(commits_behind) <= 5:  # Only auto-rebase small changes
                    logger.info("Attempting auto-rebase...")
                    result = subprocess.run(["git", "rebase", "origin/main"], 
                                         capture_output=True, text=True, cwd=self.workspace_root)
                    if result.returncode == 0:
                        logger.info("Auto-rebase successful")
                    else:
                        logger.warning("Auto-rebase failed, manual intervention may be needed")
                        
        except Exception as e:
            logger.warning(f"Repository sync failed: {e}")
            
    async def _discover_all_tasks(self) -> List[Dict[str, Any]]:
        """Discover tasks from all sources"""
        logger.info("🔍 Discovering tasks from all sources...")
        
        all_tasks = []
        
        try:
            # 1. Discover TODO/FIXME comments
            todo_tasks = await self.task_analyzer.discover_todo_tasks()
            all_tasks.extend(todo_tasks)
            logger.info(f"Found {len(todo_tasks)} TODO/FIXME tasks")
            
            # 2. Discover from failing tests
            test_tasks = await self._discover_failing_test_tasks()
            all_tasks.extend(test_tasks)
            logger.info(f"Found {len(test_tasks)} test-related tasks")
            
            # 3. Discover from security scans
            security_tasks = await self._discover_security_tasks()
            all_tasks.extend(security_tasks)
            logger.info(f"Found {len(security_tasks)} security tasks")
            
            # 4. Discover from PR feedback (if available)
            pr_tasks = await self._discover_pr_feedback_tasks()
            all_tasks.extend(pr_tasks)
            logger.info(f"Found {len(pr_tasks)} PR feedback tasks")
            
            # 5. Discover from GitHub issues
            github_tasks = await self._discover_github_issues()
            all_tasks.extend(github_tasks)
            logger.info(f"Found {len(github_tasks)} GitHub issue tasks")
            
            # 6. Discover from dependency alerts
            dependency_tasks = await self._discover_dependency_alerts()
            all_tasks.extend(dependency_tasks)
            logger.info(f"Found {len(dependency_tasks)} dependency tasks")
            
            # Deduplicate tasks
            deduplicated_tasks = self._deduplicate_tasks(all_tasks)
            logger.info(f"Total unique tasks after deduplication: {len(deduplicated_tasks)}")
            
            return deduplicated_tasks
            
        except Exception as e:
            logger.error(f"Task discovery failed: {e}")
            return []
            
    async def _discover_failing_test_tasks(self) -> List[Dict[str, Any]]:
        """Discover tasks from failing tests"""
        tasks = []
        
        try:
            # Run tests and capture failures
            import subprocess
            result = subprocess.run(["python", "-m", "pytest", "--tb=short", "-v"], 
                                 capture_output=True, text=True, cwd=self.workspace_root)
            
            if result.returncode != 0:
                # Parse test failures and create tasks
                failures = self._parse_test_failures(result.stdout)
                for failure in failures:
                    task = {
                        "id": f"test-failure-{hash(failure['test_name'])}",
                        "title": f"Fix failing test: {failure['test_name']}",
                        "description": f"Test failure: {failure['error_message']}",
                        "type": "Bug",
                        "file_path": failure['file_path'],
                        "line_number": failure.get('line_number'),
                        "effort": 2,  # Medium effort for test fixes
                        "value": 4,   # High value for test stability
                        "time_criticality": 3,  # Medium urgency
                        "risk_reduction": 3,    # Medium risk reduction
                        "status": "READY"
                    }
                    tasks.append(task)
                    
        except Exception as e:
            logger.warning(f"Failed to discover test tasks: {e}")
            
        return tasks
        
    async def _discover_security_tasks(self) -> List[Dict[str, Any]]:
        """Discover tasks from security scans"""
        try:
            # Run comprehensive security scan
            scan_results = await self.security_scanner.run_comprehensive_scan()
            
            # Convert security vulnerabilities to backlog items
            security_tasks = self.security_scanner.create_security_backlog_items(max_items=20)
            
            return security_tasks
            
        except Exception as e:
            logger.warning(f"Failed to discover security tasks: {e}")
            return []
            
    async def _discover_pr_feedback_tasks(self) -> List[Dict[str, Any]]:
        """Discover tasks from PR review feedback"""
        # Placeholder - would integrate with GitHub API to parse PR comments
        return []
        
    async def _discover_github_issues(self) -> List[Dict[str, Any]]:
        """Discover tasks from GitHub issues"""
        # Placeholder - would integrate with GitHub API
        return []
        
    async def _discover_dependency_alerts(self) -> List[Dict[str, Any]]:
        """Discover tasks from dependency vulnerability alerts"""
        # This is covered by security scanner dependency scan
        return []
        
    def _parse_test_failures(self, test_output: str) -> List[Dict[str, Any]]:
        """Parse pytest output to extract test failures"""
        failures = []
        
        # Simple parsing - in production this would be more sophisticated
        lines = test_output.split('\n')
        current_failure = None
        
        for line in lines:
            if 'FAILED' in line and '::' in line:
                parts = line.split('::')
                if len(parts) >= 2:
                    current_failure = {
                        'file_path': parts[0].replace('FAILED ', '').strip(),
                        'test_name': '::'.join(parts[1:]).strip(),
                        'error_message': line.strip()
                    }
                    failures.append(current_failure)
                    
        return failures
        
    def _deduplicate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tasks based on ID and content similarity"""
        seen_ids = set()
        unique_tasks = []
        
        for task in tasks:
            task_id = task.get("id", "")
            if task_id not in seen_ids:
                seen_ids.add(task_id)
                unique_tasks.append(task)
                
        return unique_tasks
        
    async def _score_and_sort_backlog(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score tasks using WSJF and sort by priority"""
        logger.info("📊 Scoring and prioritizing backlog using WSJF...")
        
        try:
            # Use the existing task prioritization system
            from src.task_prioritization import TaskPrioritizer
            prioritizer = TaskPrioritizer()
            
            for task in tasks:
                # Calculate WSJF score if not already present
                if "wsjf_score" not in task:
                    task["wsjf_score"] = (
                        task.get("value", 3) + 
                        task.get("time_criticality", 2) + 
                        task.get("risk_reduction", 2)
                    ) / max(task.get("effort", 1), 1)
                    
                # Apply aging multiplier for old tasks
                task["wsjf_score"] = self._apply_aging_multiplier(task)
                
            # Sort by WSJF score (highest first)
            sorted_tasks = sorted(tasks, key=lambda t: t.get("wsjf_score", 0), reverse=True)
            
            logger.info(f"Top 5 priorities (WSJF scores): {[t['wsjf_score'] for t in sorted_tasks[:5]]}")
            
            return sorted_tasks
            
        except Exception as e:
            logger.error(f"Backlog scoring failed: {e}")
            return tasks
            
    def _apply_aging_multiplier(self, task: Dict[str, Any]) -> float:
        """Apply aging multiplier to prevent stale tasks (max 2.0x)"""
        base_score = task.get("wsjf_score", 0)
        
        # Get task age (use created_at if available)
        task_age_days = 0
        if "created_at" in task:
            try:
                created_time = datetime.fromisoformat(task["created_at"].replace('Z', '+00:00'))
                task_age_days = (datetime.now() - created_time.replace(tzinfo=None)).days
            except:
                pass
                
        # Apply aging multiplier (increases linearly to 2.0x over 30 days)
        aging_multiplier = min(2.0, 1.0 + (task_age_days / 30.0))
        
        return base_score * aging_multiplier
        
    def _get_next_ready_task(self, backlog: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the next ready task from prioritized backlog"""
        for task in backlog:
            # Check if task is ready and within scope
            if (task.get("status") == "READY" and 
                self._is_task_in_scope(task) and
                not self._requires_human_approval(task)):
                return task
                
        return None
        
    def _is_task_in_scope(self, task: Dict[str, Any]) -> bool:
        """Check if task is within automation scope"""
        file_path = task.get("file_path", "")
        
        # Check if file is within workspace
        if not file_path.startswith("./") and not file_path.startswith("/"):
            file_path = "./" + file_path
            
        workspace_root = self.automation_scope.get("workspace_root", "./")
        if not file_path.startswith(workspace_root):
            return False
            
        # Check protected files
        protected_patterns = self.automation_scope.get("safety", {}).get("protected_files", [])
        for pattern in protected_patterns:
            if self._matches_pattern(file_path, pattern):
                return False
                
        return True
        
    def _requires_human_approval(self, task: Dict[str, Any]) -> bool:
        """Check if task requires human approval"""
        approval_triggers = self.automation_scope.get("safety", {}).get("require_approval", [])
        
        # Check effort threshold
        if task.get("effort", 0) > 5:
            return True
            
        # Check security sensitivity
        if task.get("type") == "Security" and "security_sensitive_files" in approval_triggers:
            return True
            
        # Check if it's a public API change
        file_path = task.get("file_path", "")
        if "api" in file_path.lower() and "public_api_changes" in approval_triggers:
            return True
            
        return False
        
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Simple pattern matching for file paths"""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern)
        
    async def _execute_micro_cycle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task through TDD micro-cycle"""
        logger.info(f"🔧 Executing task: {task['title']}")
        
        start_time = time.time()
        result = {
            "success": False,
            "task_id": task["id"],
            "start_time": start_time,
            "error_message": None,
            "security_checks_passed": False,
            "tests_passed": False,
            "lint_passed": False
        }
        
        try:
            # Use the existing continuous backlog executor
            execution_result = await self.backlog_executor.execute_task_with_tdd(task)
            
            result.update(execution_result)
            result["success"] = execution_result.get("status") == "completed"
            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            result["error_message"] = str(e)
            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - start_time
            return result
            
    async def _record_task_completion(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Record task completion in DORA metrics"""
        try:
            # Record in DORA metrics
            self.dora_collector.record_task_completion(
                task_id=task["id"],
                start_timestamp=result["start_time"],
                completion_timestamp=result["end_time"],
                quality_gates_passed=result.get("tests_passed", False) and result.get("lint_passed", False)
            )
            
            # Record change event if code was modified
            if result.get("files_modified"):
                self.dora_collector.record_change(
                    commit_sha="pending",  # Would get from git
                    commit_timestamp=result["end_time"],
                    author="autonomous-assistant",
                    files_changed=len(result.get("files_modified", [])),
                    lines_added=result.get("lines_added", 0),
                    lines_deleted=result.get("lines_deleted", 0),
                    task_id=task["id"]
                )
                
        except Exception as e:
            logger.warning(f"Failed to record task completion metrics: {e}")
            
    def _should_escalate_task(self, task: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Determine if a failed task should be escalated to humans"""
        # Escalate if security checks failed
        if not result.get("security_checks_passed", True):
            return True
            
        # Escalate if it's a high-value task that failed
        if task.get("value", 0) >= 4 and not result["success"]:
            return True
            
        # Escalate if it's a critical severity security issue
        if (task.get("type") == "Security" and 
            task.get("security_metadata", {}).get("severity") == "critical"):
            return True
            
        return False
        
    async def _escalate_task(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Escalate task to human review"""
        logger.warning(f"🚨 Escalating task to human review: {task['title']}")
        
        # Create escalation issue (would integrate with GitHub Issues API)
        escalation = {
            "title": f"ESCALATION: {task['title']}",
            "body": f"""
This task has been escalated for human review by the Autonomous Backlog Management System.

**Original Task:**
- ID: {task['id']}
- Type: {task.get('type', 'Unknown')}
- WSJF Score: {task.get('wsjf_score', 0)}
- File: {task.get('file_path', 'N/A')}

**Escalation Reason:**
{result.get('error_message', 'Task failed automated execution')}

**Security Status:**
- Security checks passed: {result.get('security_checks_passed', 'Unknown')}

**Recommendation:**
Please review this task manually and either:
1. Fix the blocking issue and retry
2. Modify the task scope to be more automatable
3. Mark as requiring human implementation

Labels: autonomous-escalation, needs-human-review
""",
            "labels": ["autonomous-escalation", "needs-human-review", task.get("type", "unknown").lower()]
        }
        
        # Save escalation locally
        escalation_file = Path("docs/escalations") / f"escalation_{task['id']}_{int(time.time())}.json"
        escalation_file.parent.mkdir(exist_ok=True)
        
        with open(escalation_file, 'w') as f:
            json.dump(escalation, f, indent=2)
            
        logger.info(f"Escalation saved: {escalation_file}")
        
    async def _escalate_blocked_tasks(self, blocked_tasks: List[Dict[str, Any]]):
        """Escalate multiple blocked tasks for batch review"""
        logger.warning(f"🚨 Escalating {len(blocked_tasks)} blocked tasks for batch review")
        
        # Group by type
        by_type = {}
        for task in blocked_tasks:
            task_type = task.get("type", "Unknown")
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(task)
            
        # Create batch escalation report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_blocked_tasks": len(blocked_tasks),
            "blocked_by_type": {k: len(v) for k, v in by_type.items()},
            "high_priority_blocked": [
                t for t in blocked_tasks if t.get("wsjf_score", 0) > 10
            ],
            "recommendations": [
                "Review blocked tasks to identify common patterns",
                "Consider updating automation scope configuration",
                "Evaluate if additional tooling or permissions are needed"
            ]
        }
        
        escalation_file = Path("docs/escalations") / f"blocked_tasks_batch_{int(time.time())}.json"
        escalation_file.parent.mkdir(exist_ok=True)
        
        with open(escalation_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Batch escalation saved: {escalation_file}")
        
    def _should_generate_report(self) -> bool:
        """Check if we should generate a status report"""
        # Generate report every hour
        return time.time() - self.last_report_time > 3600
        
    async def _generate_status_report(self):
        """Generate comprehensive status report"""
        try:
            logger.info("📊 Generating status report...")
            
            report = self.status_reporter.generate_daily_report(self.completed_tasks)
            self.last_report_time = time.time()
            
            logger.info(f"Status report generated: {len(self.completed_tasks)} tasks completed")
            
        except Exception as e:
            logger.error(f"Failed to generate status report: {e}")
            
    async def _generate_final_report(self, execution_summary: Dict[str, Any]):
        """Generate final execution report"""
        logger.info("📋 Generating final execution report...")
        
        try:
            # Generate comprehensive final report
            final_report = {
                **execution_summary,
                "dora_metrics": self.dora_collector.export_metrics_report(days=1),
                "system_health": {
                    "total_cycles": self.current_cycle,
                    "avg_cycle_time": execution_summary["total_duration_hours"] / max(self.current_cycle, 1) * 3600,
                    "task_completion_rate": len(execution_summary["tasks_completed"]) / max(self.current_cycle, 1),
                    "error_rate": len(execution_summary["errors_encountered"]) / max(self.current_cycle, 1)
                }
            }
            
            # Save final report
            report_file = Path("docs/status") / f"autonomous_execution_final_{datetime.now().strftime('%Y-%m-%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
                
            logger.info(f"Final report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")


async def main():
    """Main entry point for autonomous backlog management"""
    parser = argparse.ArgumentParser(description="Autonomous Backlog Management System")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--max-cycles", type=int, help="Maximum number of execution cycles")
    parser.add_argument("--max-duration", type=int, help="Maximum duration in hours")
    parser.add_argument("--workspace", default=".", help="Workspace root directory")
    parser.add_argument("--dry-run", action="store_true", help="Discover and prioritize only, no execution")
    
    args = parser.parse_args()
    
    # Initialize the autonomous manager
    manager = AutonomousBacklogManager(
        config_path=args.config,
        workspace_root=Path(args.workspace)
    )
    
    if args.dry_run:
        logger.info("🔍 Running in discovery mode (no execution)")
        tasks = await manager._discover_all_tasks()
        prioritized = await manager._score_and_sort_backlog(tasks)
        
        print(f"\nDiscovered {len(tasks)} tasks")
        print("Top 10 priorities:")
        for i, task in enumerate(prioritized[:10], 1):
            print(f"{i:2d}. [{task.get('wsjf_score', 0):5.1f}] {task['title']}")
            
    else:
        # Run the full autonomous cycle
        result = await manager.run_autonomous_cycle(
            max_cycles=args.max_cycles,
            max_duration_hours=args.max_duration
        )
        
        print(f"\nExecution Summary:")
        print(f"Status: {result['final_status']}")
        print(f"Cycles: {result['cycles_completed']}")
        print(f"Tasks completed: {len(result['tasks_completed'])}")
        print(f"Tasks blocked: {len(result['tasks_blocked'])}")
        print(f"Duration: {result['total_duration_hours']:.2f} hours")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown completed.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)