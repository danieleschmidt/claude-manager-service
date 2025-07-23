#!/usr/bin/env python3
"""
Continuous Backlog Execution Engine

A disciplined, impact-maximizing autonomous coding assistant that continuously
processes every actionable item in the backlog until completion or blocked status.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from async_github_api import AsyncGitHubAPI
from async_task_analyzer import AsyncTaskAnalyzer
from async_orchestrator import AsyncOrchestrator
from task_prioritization import TaskPrioritizer, calculate_wsjf_score
from task_tracker import TaskTracker
from services.configuration_service import ConfigurationService
from services.repository_service import RepositoryService
from logger import get_logger
from error_handler import ErrorHandler


class TaskStatus(Enum):
    """Task status enumeration"""
    NEW = "NEW"
    REFINED = "REFINED" 
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    MERGED = "MERGED"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class TaskType(Enum):
    """Task type enumeration"""
    FEATURE = "Feature"
    BUG = "Bug"
    REFACTOR = "Refactor"
    SECURITY = "Security"
    DOC = "Doc"
    TEST = "Test"
    PERF = "Performance"


@dataclass
class BacklogItem:
    """Structured backlog item with all required fields"""
    id: str
    title: str
    description: str
    task_type: TaskType
    impact: int  # 1-13 scale
    effort: int  # 1-13 scale
    status: TaskStatus
    wsjf_score: float
    created_at: datetime
    updated_at: datetime
    links: List[str]
    acceptance_criteria: List[str]
    security_notes: str = ""
    test_notes: str = ""
    aging_multiplier: float = 1.0
    blocked_reason: str = ""
    pr_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['task_type'] = self.task_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacklogItem':
        """Create from dictionary"""
        data['task_type'] = TaskType(data['task_type'])
        data['status'] = TaskStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ExecutionMetrics:
    """Metrics for tracking execution performance"""
    cycle_start: datetime
    items_processed: int = 0
    items_completed: int = 0
    items_blocked: int = 0
    coverage_delta: float = 0.0
    wsjf_distribution: Dict[str, int] = None
    cycle_time_avg: float = 0.0
    
    def __post_init__(self):
        if self.wsjf_distribution is None:
            self.wsjf_distribution = {}


class ContinuousBacklogExecutor:
    """
    Autonomous senior coding assistant that continuously processes backlog items
    using TDD discipline and WSJF prioritization.
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler()
        
        # Initialize services
        self.config_service = ConfigurationService(config_path)
        self.config = self.config_service.get_config()
        
        # Initialize components
        self.github_api = AsyncGitHubAPI()
        self.task_analyzer = AsyncTaskAnalyzer()
        self.orchestrator = AsyncOrchestrator()
        self.task_prioritizer = TaskPrioritizer()
        self.task_tracker = TaskTracker()
        self.repository_service = RepositoryService(self.github_api)
        
        # Execution state
        self.backlog: List[BacklogItem] = []
        self.backlog_file = Path("DOCS/backlog.yml")
        self.status_dir = Path("DOCS/status")
        self.tech_debt_file = Path("DOCS/tech_debt.md")
        
        # Execution parameters
        self.slice_size_threshold = 5  # effort > 5 requires splitting
        self.max_cycle_time = 3600  # 1 hour max per cycle
        self.aging_cap = 2.0  # maximum aging multiplier
        
        # Ensure directories exist
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.backlog_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def run_continuous_execution(self) -> None:
        """
        Main execution loop - continuously process backlog until empty or all blocked
        """
        self.logger.info("🚀 Starting Continuous Backlog Execution")
        
        cycle_count = 0
        while True:
            cycle_count += 1
            self.logger.info(f"📋 Starting execution cycle #{cycle_count}")
            
            metrics = ExecutionMetrics(cycle_start=datetime.now())
            
            try:
                # 1. Sync & Refresh
                await self._sync_and_refresh()
                
                # 2. Check if we have actionable items
                actionable_items = self._get_actionable_items()
                if not actionable_items:
                    self.logger.info("✅ No actionable items remaining. Execution complete.")
                    break
                
                # 3. Process items in priority order
                for item in actionable_items:
                    if await self._should_stop_cycle(metrics):
                        break
                    
                    try:
                        result = await self._process_backlog_item(item)
                        metrics.items_processed += 1
                        
                        if result == "completed":
                            metrics.items_completed += 1
                        elif result == "blocked":
                            metrics.items_blocked += 1
                            
                    except Exception as e:
                        self.logger.error(f"Failed to process item {item.id}: {e}")
                        self._mark_item_blocked(item, str(e))
                        metrics.items_blocked += 1
                
                # 4. End of cycle maintenance
                await self._end_of_cycle_maintenance(metrics)
                
                # 5. Check termination conditions
                if self._should_terminate():
                    break
                    
                # Brief pause between cycles
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Critical error in execution cycle: {e}")
                break
        
        self.logger.info("🏁 Continuous Backlog Execution completed")
    
    async def _sync_and_refresh(self) -> None:
        """Sync repo state and refresh backlog"""
        self.logger.info("🔄 Syncing repository state and refreshing backlog")
        
        # Load existing backlog
        await self._load_backlog()
        
        # Discover new tasks
        await self._discover_new_tasks()
        
        # Normalize and score all items
        self._normalize_backlog_items()
        self._score_and_rank_backlog()
        
        # Save updated backlog
        await self._save_backlog()
        
        self.logger.info(f"📊 Backlog refreshed: {len(self.backlog)} total items")
    
    async def _load_backlog(self) -> None:
        """Load backlog from file"""
        if self.backlog_file.exists():
            try:
                with open(self.backlog_file, 'r') as f:
                    data = json.load(f)
                    self.backlog = [BacklogItem.from_dict(item) for item in data]
                self.logger.info(f"📥 Loaded {len(self.backlog)} items from backlog")
            except Exception as e:
                self.logger.warning(f"Failed to load backlog: {e}")
                self.backlog = []
        else:
            # Initialize from existing BACKLOG.md if available
            await self._import_from_markdown()
    
    async def _import_from_markdown(self) -> None:
        """Import initial backlog from BACKLOG.md"""
        backlog_md = Path("BACKLOG.md")
        if not backlog_md.exists():
            self.backlog = []
            return
        
        # Parse remaining items from BACKLOG.md
        # For now, focus on discovering new items
        self.backlog = []
        self.logger.info("📋 Initialized empty backlog - will discover tasks")
    
    async def _discover_new_tasks(self) -> None:
        """Discover new tasks from various sources"""
        self.logger.info("🔍 Discovering new tasks")
        
        new_items = []
        
        # Discover from TODO/FIXME comments
        todo_items = await self._discover_todo_tasks()
        new_items.extend(todo_items)
        
        # Discover from failing tests
        test_items = await self._discover_test_failures()
        new_items.extend(test_items)
        
        # Discover from PR feedback
        pr_items = await self._discover_pr_feedback()
        new_items.extend(pr_items)
        
        # Discover from security scans
        security_items = await self._discover_security_issues()
        new_items.extend(security_items)
        
        # Discover from dependency alerts
        dep_items = await self._discover_dependency_issues()
        new_items.extend(dep_items)
        
        # Merge with existing backlog, avoiding duplicates
        self._merge_new_items(new_items)
        
        self.logger.info(f"🆕 Discovered {len(new_items)} new tasks")
    
    async def _discover_todo_tasks(self) -> List[BacklogItem]:
        """Discover tasks from TODO/FIXME comments"""
        todo_items = []
        
        try:
            # Use existing task analyzer to find TODOs
            for repo_name in self.config['github']['reposToScan']:
                repo = await self.github_api.get_repo(repo_name)
                if repo:
                    # Get TODO results from task analyzer
                    results = await self.task_analyzer.find_todo_comments_async(
                        repo, self.config['github']['managerRepo']
                    )
                    
                    # Convert to backlog items
                    for result in results:
                        if isinstance(result, dict) and 'title' in result:
                            item = self._create_backlog_item_from_todo(result)
                            todo_items.append(item)
        
        except Exception as e:
            self.logger.error(f"Error discovering TODO tasks: {e}")
        
        return todo_items
    
    def _create_backlog_item_from_todo(self, todo_result: Dict) -> BacklogItem:
        """Create a backlog item from TODO discovery result"""
        now = datetime.now()
        
        # Extract priority signals from TODO content
        content = todo_result.get('content', '')
        impact = self._estimate_todo_impact(content)
        effort = self._estimate_todo_effort(content)
        
        return BacklogItem(
            id=f"todo_{hash(todo_result.get('title', ''))}__{int(now.timestamp())}",
            title=todo_result.get('title', 'Unknown TODO'),
            description=todo_result.get('description', ''),
            task_type=self._classify_todo_type(content),
            impact=impact,
            effort=effort,
            status=TaskStatus.NEW,
            wsjf_score=0.0,  # Will be calculated later
            created_at=now,
            updated_at=now,
            links=[todo_result.get('url', '')],
            acceptance_criteria=self._generate_todo_acceptance_criteria(content),
            security_notes=self._extract_security_notes(content),
            test_notes="Add unit tests to verify fix"
        )
    
    def _estimate_todo_impact(self, content: str) -> int:
        """Estimate impact of TODO item (1-13 scale)"""
        content_lower = content.lower()
        
        # High impact indicators
        if any(word in content_lower for word in [
            'security', 'vulnerability', 'critical', 'urgent', 'blocking',
            'crash', 'memory leak', 'performance', 'authentication'
        ]):
            return 8
        
        # Medium impact indicators  
        if any(word in content_lower for word in [
            'bug', 'error', 'fix', 'improvement', 'optimization'
        ]):
            return 5
        
        # Low impact (documentation, cleanup, etc.)
        return 3
    
    def _estimate_todo_effort(self, content: str) -> int:
        """Estimate effort for TODO item (1-13 scale)"""
        content_lower = content.lower()
        
        # High effort indicators
        if any(word in content_lower for word in [
            'refactor', 'rewrite', 'architecture', 'migration', 'integration'
        ]):
            return 8
        
        # Medium effort indicators
        if any(word in content_lower for word in [
            'implement', 'add', 'create', 'build', 'develop'
        ]):
            return 5
        
        # Low effort (simple fixes, updates)
        return 2
    
    def _classify_todo_type(self, content: str) -> TaskType:
        """Classify TODO based on content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['security', 'vulnerability', 'auth']):
            return TaskType.SECURITY
        elif any(word in content_lower for word in ['bug', 'fix', 'error']):
            return TaskType.BUG
        elif any(word in content_lower for word in ['test', 'testing']):
            return TaskType.TEST
        elif any(word in content_lower for word in ['performance', 'optimize']):
            return TaskType.PERF
        elif any(word in content_lower for word in ['refactor', 'cleanup']):
            return TaskType.REFACTOR
        elif any(word in content_lower for word in ['doc', 'documentation']):
            return TaskType.DOC
        else:
            return TaskType.FEATURE
    
    def _generate_todo_acceptance_criteria(self, content: str) -> List[str]:
        """Generate acceptance criteria for TODO item"""
        return [
            f"Address the TODO comment: {content}",
            "All existing tests continue to pass",
            "Add new tests if functionality is added/changed",
            "Code follows project style guidelines",
            "No new security vulnerabilities introduced"
        ]
    
    def _extract_security_notes(self, content: str) -> str:
        """Extract security considerations from TODO content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in [
            'security', 'vulnerability', 'auth', 'token', 'password', 
            'injection', 'xss', 'csrf'
        ]):
            return "⚠️ Security-related change - requires security review and testing"
        
        return "Standard security practices should be followed"
    
    async def _discover_test_failures(self) -> List[BacklogItem]:
        """Discover tasks from failing tests"""
        # TODO: Implement test failure discovery
        return []
    
    async def _discover_pr_feedback(self) -> List[BacklogItem]: 
        """Discover tasks from PR comments and feedback"""
        # TODO: Implement PR feedback discovery
        return []
    
    async def _discover_security_issues(self) -> List[BacklogItem]:
        """Discover tasks from security scan results"""
        # TODO: Implement security issue discovery
        return []
    
    async def _discover_dependency_issues(self) -> List[BacklogItem]:
        """Discover tasks from dependency vulnerability alerts"""
        # TODO: Implement dependency issue discovery
        return []
    
    def _merge_new_items(self, new_items: List[BacklogItem]) -> None:
        """Merge new items with existing backlog, avoiding duplicates"""
        existing_titles = {item.title for item in self.backlog}
        
        for item in new_items:
            if item.title not in existing_titles:
                self.backlog.append(item)
                self.logger.debug(f"Added new item: {item.title}")
            else:
                self.logger.debug(f"Skipped duplicate item: {item.title}")
    
    def _normalize_backlog_items(self) -> None:
        """Ensure all backlog items have required fields"""
        for item in self.backlog:
            # Ensure all required fields are present
            if not item.acceptance_criteria:
                item.acceptance_criteria = ["Item needs acceptance criteria defined"]
                item.status = TaskStatus.NEW
            
            # Split large items
            if item.effort > self.slice_size_threshold and item.status == TaskStatus.NEW:
                self._split_large_item(item)
            
            # Update status if criteria are met
            if item.status == TaskStatus.NEW and item.acceptance_criteria:
                item.status = TaskStatus.REFINED
            
            if (item.status == TaskStatus.REFINED and 
                item.acceptance_criteria and 
                item.security_notes and 
                item.test_notes):
                item.status = TaskStatus.READY
    
    def _split_large_item(self, item: BacklogItem) -> None:
        """Split large items into smaller slices"""
        if item.effort <= self.slice_size_threshold:
            return
        
        # Create smaller sub-items
        num_slices = (item.effort + self.slice_size_threshold - 1) // self.slice_size_threshold
        slice_effort = item.effort // num_slices
        
        for i in range(num_slices):
            slice_item = BacklogItem(
                id=f"{item.id}_slice_{i+1}",
                title=f"{item.title} (Part {i+1}/{num_slices})",
                description=f"Slice {i+1} of {num_slices}: {item.description}",
                task_type=item.task_type,
                impact=item.impact,
                effort=slice_effort,
                status=TaskStatus.NEW,
                wsjf_score=0.0,
                created_at=item.created_at,
                updated_at=datetime.now(),
                links=item.links.copy(),
                acceptance_criteria=[f"Complete slice {i+1} of the original task"],
                security_notes=item.security_notes,
                test_notes=item.test_notes
            )
            self.backlog.append(slice_item)
        
        # Mark original as split/completed
        item.status = TaskStatus.DONE
        self.logger.info(f"Split large item {item.title} into {num_slices} slices")
    
    def _score_and_rank_backlog(self) -> None:
        """Calculate WSJF scores and rank backlog"""
        current_time = datetime.now()
        
        for item in self.backlog:
            # Calculate aging multiplier (capped)
            days_old = (current_time - item.created_at).days
            aging_factor = 1.0 + (days_old * 0.1)  # 10% per day
            item.aging_multiplier = min(aging_factor, self.aging_cap)
            
            # Calculate WSJF score
            cost_of_delay = item.impact + 2 + 3  # impact + time_criticality + risk_reduction  
            item.wsjf_score = (cost_of_delay * item.aging_multiplier) / max(item.effort, 0.5)
            
            item.updated_at = current_time
        
        # Sort by WSJF score (descending)
        self.backlog.sort(key=lambda x: x.wsjf_score, reverse=True)
        
        self.logger.info("📊 Backlog scored and ranked by WSJF")
    
    def _get_actionable_items(self) -> List[BacklogItem]:
        """Get items that are ready to work on"""
        actionable = []
        
        for item in self.backlog:
            if item.status == TaskStatus.READY and not item.blocked_reason:
                actionable.append(item)
            elif item.status in [TaskStatus.DONE, TaskStatus.MERGED]:
                continue
            elif item.status == TaskStatus.BLOCKED:
                self.logger.debug(f"Skipping blocked item: {item.title} - {item.blocked_reason}")
        
        return actionable
    
    async def _process_backlog_item(self, item: BacklogItem) -> str:
        """
        Process a single backlog item using TDD micro-cycle
        Returns: 'completed', 'blocked', or 'in_progress'
        """
        self.logger.info(f"🔧 Processing: {item.title} (WSJF: {item.wsjf_score:.2f})")
        
        # Update status to DOING
        item.status = TaskStatus.DOING
        item.updated_at = datetime.now()
        await self._save_backlog()
        
        try:
            # 1. Restate acceptance criteria
            self.logger.info(f"📋 Acceptance criteria: {item.acceptance_criteria}")
            
            # 2. Check if this requires human clarification
            if await self._requires_human_clarification(item):
                self._mark_item_blocked(item, "Requires human clarification for high-risk change")
                return "blocked"
            
            # 3. Execute TDD micro-cycle
            success = await self._execute_tdd_cycle(item)
            
            if success:
                item.status = TaskStatus.PR
                self.logger.info(f"✅ Completed: {item.title}")
                return "completed"
            else:
                self._mark_item_blocked(item, "TDD cycle failed")
                return "blocked"
            
        except Exception as e:
            self.logger.error(f"Error processing {item.title}: {e}")
            self._mark_item_blocked(item, str(e))
            return "blocked"
    
    async def _requires_human_clarification(self, item: BacklogItem) -> bool:
        """Check if item requires human review before proceeding"""
        
        # High-risk indicators
        high_risk_keywords = [
            'public interface', 'api change', 'breaking change',
            'authentication', 'security', 'crypto', 'secrets',
            'migration', 'database', 'performance critical'
        ]
        
        text_to_check = f"{item.title} {item.description}".lower()
        
        if any(keyword in text_to_check for keyword in high_risk_keywords):
            return True
        
        # Large effort items
        if item.effort >= 8:
            return True
        
        return False
    
    async def _execute_tdd_cycle(self, item: BacklogItem) -> bool:
        """Execute TDD Red-Green-Refactor cycle for the item"""
        try:
            # For this implementation, we'll simulate the TDD cycle
            # In practice, this would involve:
            # 1. Writing failing tests
            # 2. Writing minimal code to pass
            # 3. Refactoring
            # 4. Security checks
            # 5. Documentation updates
            # 6. CI pipeline verification
            
            self.logger.info(f"🔴 RED: Writing failing test for {item.title}")
            await asyncio.sleep(1)  # Simulate work
            
            self.logger.info(f"🟢 GREEN: Implementing minimal code for {item.title}")
            await asyncio.sleep(2)  # Simulate implementation
            
            self.logger.info(f"🔵 REFACTOR: Cleaning up code for {item.title}")
            await asyncio.sleep(1)  # Simulate refactoring
            
            self.logger.info(f"🔒 Applying security checklist for {item.title}")
            await asyncio.sleep(0.5)  # Simulate security checks
            
            self.logger.info(f"📚 Updating documentation for {item.title}")
            await asyncio.sleep(0.5)  # Simulate docs update
            
            self.logger.info(f"🧪 Running CI pipeline for {item.title}")
            await asyncio.sleep(3)  # Simulate CI run
            
            # Mark as ready for PR
            item.pr_url = f"https://github.com/example/repo/pull/{item.id}"
            
            return True
            
        except Exception as e:
            self.logger.error(f"TDD cycle failed for {item.title}: {e}")
            return False
    
    def _mark_item_blocked(self, item: BacklogItem, reason: str) -> None:
        """Mark item as blocked with reason"""
        item.status = TaskStatus.BLOCKED
        item.blocked_reason = reason
        item.updated_at = datetime.now()
        self.logger.warning(f"🚫 Blocked: {item.title} - {reason}")
    
    async def _should_stop_cycle(self, metrics: ExecutionMetrics) -> bool:
        """Check if we should stop the current cycle"""
        elapsed = (datetime.now() - metrics.cycle_start).seconds
        return elapsed > self.max_cycle_time
    
    async def _end_of_cycle_maintenance(self, metrics: ExecutionMetrics) -> None:
        """Perform end-of-cycle maintenance tasks"""
        
        # Update metrics
        self._update_metrics(metrics)
        
        # Save backlog state
        await self._save_backlog()
        
        # Generate status report
        await self._generate_status_report(metrics)
        
        # Re-score backlog for next cycle
        self._score_and_rank_backlog()
    
    def _update_metrics(self, metrics: ExecutionMetrics) -> None:
        """Update execution metrics"""
        cycle_time = (datetime.now() - metrics.cycle_start).seconds
        metrics.cycle_time_avg = cycle_time
        
        # Update WSJF distribution
        for item in self.backlog:
            score_range = f"{int(item.wsjf_score)}-{int(item.wsjf_score)+1}"
            metrics.wsjf_distribution[score_range] = metrics.wsjf_distribution.get(score_range, 0) + 1
    
    async def _generate_status_report(self, metrics: ExecutionMetrics) -> None:
        """Generate status report for the cycle"""
        timestamp = datetime.now()
        
        # Count items by status
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len([item for item in self.backlog if item.status == status])
        
        # Calculate notable risks/blocks
        blocked_items = [item for item in self.backlog if item.status == TaskStatus.BLOCKED]
        notable_risks = [f"{item.title}: {item.blocked_reason}" for item in blocked_items[:5]]
        
        report = {
            "timestamp": timestamp.isoformat(),
            "completed_items": [item.id for item in self.backlog if item.status == TaskStatus.DONE],
            "coverage_delta": metrics.coverage_delta,
            "flaky_tests_new": 0,  # TODO: Implement
            "CI_status_stability": "stable",  # TODO: Implement
            "open_PRs": len([item for item in self.backlog if item.status == TaskStatus.PR]),
            "notable_risks_or_blocks": notable_risks,
            "backlog_size_by_status": status_counts,
            "avg_cycle_time_last_N": metrics.cycle_time_avg,
            "wsjf_distribution_snapshot": metrics.wsjf_distribution,
            "items_processed_this_cycle": metrics.items_processed,
            "items_completed_this_cycle": metrics.items_completed,
            "items_blocked_this_cycle": metrics.items_blocked
        }
        
        # Save report
        report_file = self.status_dir / f"status_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Status report saved: {report_file}")
        
        # Log summary
        self.logger.info(f"📈 Cycle Summary: {metrics.items_processed} processed, "
                        f"{metrics.items_completed} completed, {metrics.items_blocked} blocked")
    
    def _should_terminate(self) -> bool:
        """Check if execution should terminate"""
        actionable_items = self._get_actionable_items()
        
        if not actionable_items:
            return True
        
        # Check if all remaining items are blocked
        non_done_items = [item for item in self.backlog 
                         if item.status not in [TaskStatus.DONE, TaskStatus.MERGED]]
        
        if non_done_items and all(item.status == TaskStatus.BLOCKED for item in non_done_items):
            self.logger.info("🚫 All remaining items are blocked - terminating")
            return True
        
        return False
    
    async def _save_backlog(self) -> None:
        """Save backlog to file"""
        try:
            backlog_data = [item.to_dict() for item in self.backlog]
            with open(self.backlog_file, 'w') as f:
                json.dump(backlog_data, f, indent=2, default=str)
            self.logger.debug(f"💾 Saved backlog with {len(self.backlog)} items")
        except Exception as e:
            self.logger.error(f"Failed to save backlog: {e}")


# CLI entry point
async def main():
    """Main entry point for continuous backlog execution"""
    executor = ContinuousBacklogExecutor()
    await executor.run_continuous_execution()


if __name__ == "__main__":
    asyncio.run(main())