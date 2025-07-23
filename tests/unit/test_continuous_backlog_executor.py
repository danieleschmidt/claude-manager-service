#!/usr/bin/env python3
"""
Unit tests for Continuous Backlog Executor
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.continuous_backlog_executor import (
    ContinuousBacklogExecutor, BacklogItem, TaskStatus, TaskType, ExecutionMetrics
)


class TestBacklogItem:
    """Test BacklogItem data class"""
    
    def test_backlog_item_creation(self):
        """Test creating a backlog item with all fields"""
        now = datetime.now()
        item = BacklogItem(
            id="test_1",
            title="Test Item",
            description="Test description",
            task_type=TaskType.FEATURE,
            impact=5,
            effort=3,
            status=TaskStatus.NEW,
            wsjf_score=1.67,
            created_at=now,
            updated_at=now,
            links=["http://example.com"],
            acceptance_criteria=["Criteria 1", "Criteria 2"],
            security_notes="Security note",
            test_notes="Test note"
        )
        
        assert item.id == "test_1"
        assert item.title == "Test Item"
        assert item.task_type == TaskType.FEATURE
        assert item.status == TaskStatus.NEW
        assert item.impact == 5
        assert item.effort == 3
        assert len(item.acceptance_criteria) == 2
    
    def test_backlog_item_to_dict(self):
        """Test converting backlog item to dictionary"""
        now = datetime.now()
        item = BacklogItem(
            id="test_1",
            title="Test Item",
            description="Test description",
            task_type=TaskType.BUG,
            impact=8,
            effort=2,
            status=TaskStatus.READY,
            wsjf_score=4.0,
            created_at=now,
            updated_at=now,
            links=[],
            acceptance_criteria=["Must fix bug"]
        )
        
        data = item.to_dict()
        
        assert data['id'] == "test_1"
        assert data['task_type'] == "Bug"
        assert data['status'] == "READY"
        assert data['impact'] == 8
        assert data['effort'] == 2
        assert isinstance(data['created_at'], str)
        assert isinstance(data['updated_at'], str)
    
    def test_backlog_item_from_dict(self):
        """Test creating backlog item from dictionary"""
        now = datetime.now()
        data = {
            'id': 'test_1',
            'title': 'Test Item',
            'description': 'Test description',
            'task_type': 'Security',
            'impact': 13,
            'effort': 1,
            'status': 'BLOCKED',
            'wsjf_score': 13.0,
            'created_at': now.isoformat(),
            'updated_at': now.isoformat(),
            'links': ['http://test.com'],
            'acceptance_criteria': ['Must be secure'],
            'security_notes': 'Critical security fix',
            'test_notes': 'Requires security tests',
            'aging_multiplier': 1.5,
            'blocked_reason': 'Waiting for approval',
            'pr_url': ''
        }
        
        item = BacklogItem.from_dict(data)
        
        assert item.id == 'test_1'
        assert item.task_type == TaskType.SECURITY
        assert item.status == TaskStatus.BLOCKED
        assert item.impact == 13
        assert item.effort == 1
        assert item.blocked_reason == 'Waiting for approval'
        assert isinstance(item.created_at, datetime)


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        'github': {
            'username': 'testuser',
            'managerRepo': 'testuser/test-repo',
            'reposToScan': ['testuser/test-repo']
        },
        'analyzer': {
            'scanForTodos': True,
            'scanOpenIssues': True
        },
        'executor': {
            'terragonUsername': '@test-bot'
        }
    }


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create necessary directories
        (temp_path / "DOCS").mkdir()
        (temp_path / "DOCS" / "status").mkdir()
        
        yield temp_path


class TestContinuousBacklogExecutor:
    """Test ContinuousBacklogExecutor class"""
    
    @patch('src.continuous_backlog_executor.ConfigurationService')
    @patch('src.continuous_backlog_executor.AsyncGitHubAPI')
    @patch('src.continuous_backlog_executor.AsyncTaskAnalyzer')
    def test_executor_initialization(self, mock_analyzer, mock_github, mock_config_service, temp_workspace):
        """Test executor initialization"""
        
        # Setup mocks
        mock_config_service.return_value.get_config.return_value = {
            'github': {'reposToScan': ['test/repo']}
        }
        
        with patch.object(Path, 'cwd', return_value=temp_workspace):
            executor = ContinuousBacklogExecutor()
            
            assert executor.backlog == []
            assert executor.slice_size_threshold == 5
            assert executor.max_cycle_time == 3600
            assert executor.aging_cap == 2.0
    
    def test_estimate_todo_impact(self, temp_workspace):
        """Test TODO impact estimation"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # High impact
                high_impact_content = "TODO: Fix critical security vulnerability"
                assert executor._estimate_todo_impact(high_impact_content) == 8
                
                # Medium impact
                medium_impact_content = "TODO: Fix bug in user interface"
                assert executor._estimate_todo_impact(medium_impact_content) == 5
                
                # Low impact
                low_impact_content = "TODO: Update documentation"
                assert executor._estimate_todo_impact(low_impact_content) == 3
    
    def test_estimate_todo_effort(self, temp_workspace):
        """Test TODO effort estimation"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # High effort
                high_effort_content = "TODO: Refactor entire authentication system"
                assert executor._estimate_todo_effort(high_effort_content) == 8
                
                # Medium effort
                medium_effort_content = "TODO: Implement new API endpoint"
                assert executor._estimate_todo_effort(medium_effort_content) == 5
                
                # Low effort
                low_effort_content = "TODO: Fix typo"
                assert executor._estimate_todo_effort(low_effort_content) == 2
    
    def test_classify_todo_type(self, temp_workspace):
        """Test TODO type classification"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # Security
                security_content = "TODO: Fix SQL injection vulnerability"
                assert executor._classify_todo_type(security_content) == TaskType.SECURITY
                
                # Bug
                bug_content = "TODO: Fix error in calculation"
                assert executor._classify_todo_type(bug_content) == TaskType.BUG
                
                # Test
                test_content = "TODO: Add unit tests for this function"
                assert executor._classify_todo_type(test_content) == TaskType.TEST
                
                # Performance
                perf_content = "TODO: Optimize database query performance"
                assert executor._classify_todo_type(perf_content) == TaskType.PERF
                
                # Refactor
                refactor_content = "TODO: Refactor this messy code"
                assert executor._classify_todo_type(refactor_content) == TaskType.REFACTOR
                
                # Documentation
                doc_content = "TODO: Update API documentation"
                assert executor._classify_todo_type(doc_content) == TaskType.DOC
                
                # Feature (default)
                feature_content = "TODO: Add new user management feature"
                assert executor._classify_todo_type(feature_content) == TaskType.FEATURE
    
    def test_generate_todo_acceptance_criteria(self, temp_workspace):
        """Test generation of acceptance criteria for TODOs"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                content = "Fix memory leak in authentication module"
                criteria = executor._generate_todo_acceptance_criteria(content)
                
                assert len(criteria) == 5
                assert content in criteria[0]
                assert "existing tests" in criteria[1]
                assert "new tests" in criteria[2]
                assert "style guidelines" in criteria[3]
                assert "security vulnerabilities" in criteria[4]
    
    def test_extract_security_notes(self, temp_workspace):
        """Test extraction of security notes from content"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # Security-related content
                security_content = "TODO: Fix XSS vulnerability in user input"
                security_notes = executor._extract_security_notes(security_content)
                assert "Security-related change" in security_notes
                assert "security review" in security_notes
                
                # Non-security content
                normal_content = "TODO: Add logging to this function"
                normal_notes = executor._extract_security_notes(normal_content)
                assert "Standard security practices" in normal_notes
    
    def test_create_backlog_item_from_todo(self, temp_workspace):
        """Test creating backlog item from TODO result"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                todo_result = {
                    'title': 'Fix TODO in authentication module',
                    'description': 'TODO: Add rate limiting to prevent brute force attacks',
                    'content': 'TODO: Add rate limiting to prevent brute force attacks',
                    'url': 'https://github.com/test/repo/blob/main/auth.py#L42'
                }
                
                item = executor._create_backlog_item_from_todo(todo_result)
                
                assert item.title == 'Fix TODO in authentication module'
                assert item.task_type == TaskType.SECURITY  # Due to "attacks" keyword
                assert item.status == TaskStatus.NEW
                assert item.impact == 8  # High impact due to security
                assert item.effort >= 2
                assert len(item.acceptance_criteria) == 5
                assert "Security-related change" in item.security_notes
                assert len(item.links) == 1
    
    def test_split_large_item(self, temp_workspace):
        """Test splitting large items into smaller slices"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                executor.slice_size_threshold = 5
                
                # Create large item
                large_item = BacklogItem(
                    id="large_1",
                    title="Large Task",
                    description="A very large task",
                    task_type=TaskType.FEATURE,
                    impact=8,
                    effort=12,  # Larger than threshold
                    status=TaskStatus.NEW,
                    wsjf_score=0.0,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    links=[],
                    acceptance_criteria=["Complete large task"]
                )
                
                initial_count = len(executor.backlog)
                executor._split_large_item(large_item)
                
                # Should create multiple slices
                assert len(executor.backlog) > initial_count
                assert large_item.status == TaskStatus.DONE
                
                # Check slice properties
                slices = [item for item in executor.backlog if item.id.startswith("large_1_slice_")]
                assert len(slices) > 1
                for slice_item in slices:
                    assert slice_item.effort <= executor.slice_size_threshold
                    assert "Part" in slice_item.title
    
    def test_score_and_rank_backlog(self, temp_workspace):
        """Test WSJF scoring and ranking"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # Create items with different characteristics
                old_item = BacklogItem(
                    id="old_1", title="Old Item", description="", task_type=TaskType.BUG,
                    impact=3, effort=2, status=TaskStatus.READY, wsjf_score=0.0,
                    created_at=datetime.now() - timedelta(days=10),
                    updated_at=datetime.now(), links=[], acceptance_criteria=["Fix bug"]
                )
                
                high_impact_item = BacklogItem(
                    id="high_1", title="High Impact Item", description="", task_type=TaskType.SECURITY,
                    impact=13, effort=3, status=TaskStatus.READY, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Fix security issue"]
                )
                
                executor.backlog = [old_item, high_impact_item]
                executor._score_and_rank_backlog()
                
                # Check that scores were calculated
                assert old_item.wsjf_score > 0
                assert high_impact_item.wsjf_score > 0
                
                # High impact item should have higher score
                assert high_impact_item.wsjf_score > old_item.wsjf_score
                
                # Check aging multiplier for old item
                assert old_item.aging_multiplier > 1.0
                assert old_item.aging_multiplier <= executor.aging_cap
                
                # Backlog should be sorted by WSJF score
                assert executor.backlog[0].wsjf_score >= executor.backlog[1].wsjf_score
    
    def test_get_actionable_items(self, temp_workspace):
        """Test getting actionable items from backlog"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                ready_item = BacklogItem(
                    id="ready_1", title="Ready Item", description="", task_type=TaskType.FEATURE,
                    impact=5, effort=3, status=TaskStatus.READY, wsjf_score=1.67,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete feature"]
                )
                
                blocked_item = BacklogItem(
                    id="blocked_1", title="Blocked Item", description="", task_type=TaskType.BUG,
                    impact=8, effort=2, status=TaskStatus.BLOCKED, wsjf_score=4.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Fix bug"], blocked_reason="Waiting for approval"
                )
                
                done_item = BacklogItem(
                    id="done_1", title="Done Item", description="", task_type=TaskType.REFACTOR,
                    impact=3, effort=1, status=TaskStatus.DONE, wsjf_score=3.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Refactor complete"]
                )
                
                executor.backlog = [ready_item, blocked_item, done_item]
                actionable = executor._get_actionable_items()
                
                assert len(actionable) == 1
                assert actionable[0].id == "ready_1"
    
    def test_normalize_backlog_items(self, temp_workspace):
        """Test backlog item normalization"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # Item without acceptance criteria
                incomplete_item = BacklogItem(
                    id="incomplete_1", title="Incomplete Item", description="", task_type=TaskType.FEATURE,
                    impact=5, effort=3, status=TaskStatus.NEW, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=[]  # Empty
                )
                
                # Item with all required fields
                complete_item = BacklogItem(
                    id="complete_1", title="Complete Item", description="", task_type=TaskType.BUG,
                    impact=8, effort=2, status=TaskStatus.REFINED, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Fix bug"],
                    security_notes="Security review required",
                    test_notes="Add unit tests"
                )
                
                # Large item that needs splitting
                large_item = BacklogItem(
                    id="large_1", title="Large Item", description="", task_type=TaskType.REFACTOR,
                    impact=13, effort=8, status=TaskStatus.NEW, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=[]
                )
                
                executor.backlog = [incomplete_item, complete_item, large_item]
                executor._normalize_backlog_items()
                
                # Incomplete item should have criteria added and status updated
                assert len(incomplete_item.acceptance_criteria) > 0
                assert incomplete_item.status == TaskStatus.NEW
                
                # Complete item should be marked as READY
                assert complete_item.status == TaskStatus.READY
                
                # Large item should be split (marked as DONE)
                assert large_item.status == TaskStatus.DONE
                
                # Should have additional slice items
                slices = [item for item in executor.backlog if item.id.startswith("large_1_slice_")]
                assert len(slices) > 1
    
    @pytest.mark.asyncio
    async def test_requires_human_clarification(self, temp_workspace):
        """Test detection of items requiring human clarification"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # High-risk security item
                security_item = BacklogItem(
                    id="sec_1", title="Fix Authentication Vulnerability", 
                    description="Critical security fix needed", task_type=TaskType.SECURITY,
                    impact=13, effort=5, status=TaskStatus.READY, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Fix vulnerability"]
                )
                
                # Large effort item
                large_effort_item = BacklogItem(
                    id="large_1", title="Major Refactor", description="Complete rewrite",
                    task_type=TaskType.REFACTOR, impact=8, effort=13, status=TaskStatus.READY,
                    wsjf_score=0.0, created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete refactor"]
                )
                
                # Normal item
                normal_item = BacklogItem(
                    id="normal_1", title="Add Logging", description="Add debug logging",
                    task_type=TaskType.FEATURE, impact=3, effort=2, status=TaskStatus.READY,
                    wsjf_score=0.0, created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Add logging"]
                )
                
                # Test results
                assert await executor._requires_human_clarification(security_item) == True
                assert await executor._requires_human_clarification(large_effort_item) == True
                assert await executor._requires_human_clarification(normal_item) == False
    
    def test_mark_item_blocked(self, temp_workspace):
        """Test marking item as blocked"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                item = BacklogItem(
                    id="test_1", title="Test Item", description="", task_type=TaskType.FEATURE,
                    impact=5, effort=3, status=TaskStatus.DOING, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete feature"]
                )
                
                reason = "Waiting for external dependency"
                executor._mark_item_blocked(item, reason)
                
                assert item.status == TaskStatus.BLOCKED
                assert item.blocked_reason == reason
                assert item.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_tdd_cycle(self, temp_workspace):
        """Test TDD cycle execution (simulated)"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                item = BacklogItem(
                    id="test_1", title="Test TDD Item", description="Test TDD",
                    task_type=TaskType.FEATURE, impact=5, effort=3, status=TaskStatus.DOING,
                    wsjf_score=0.0, created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Implement feature with TDD"]
                )
                
                # Mock the TDD cycle steps
                success = await executor._execute_tdd_cycle(item)
                
                assert success == True
                assert item.pr_url != ""
    
    def test_should_terminate(self, temp_workspace):
        """Test termination condition checking"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                
                # Empty backlog should terminate
                executor.backlog = []
                assert executor._should_terminate() == True
                
                # All items done should terminate
                done_item = BacklogItem(
                    id="done_1", title="Done Item", description="", task_type=TaskType.FEATURE,
                    impact=5, effort=3, status=TaskStatus.DONE, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete"]
                )
                executor.backlog = [done_item]
                assert executor._should_terminate() == True
                
                # All items blocked should terminate
                blocked_item = BacklogItem(
                    id="blocked_1", title="Blocked Item", description="", task_type=TaskType.FEATURE,
                    impact=5, effort=3, status=TaskStatus.BLOCKED, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete"], blocked_reason="Blocked"
                )
                executor.backlog = [blocked_item]
                assert executor._should_terminate() == True
                
                # Has ready items should not terminate
                ready_item = BacklogItem(
                    id="ready_1", title="Ready Item", description="", task_type=TaskType.FEATURE,
                    impact=5, effort=3, status=TaskStatus.READY, wsjf_score=0.0,
                    created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete"]
                )
                executor.backlog = [ready_item]
                assert executor._should_terminate() == False
    
    @pytest.mark.asyncio
    async def test_save_and_load_backlog(self, temp_workspace):
        """Test saving and loading backlog"""
        with patch('src.continuous_backlog_executor.ConfigurationService'):
            with patch.object(Path, 'cwd', return_value=temp_workspace):
                executor = ContinuousBacklogExecutor()
                executor.backlog_file = temp_workspace / "test_backlog.json"
                
                # Create test items
                item1 = BacklogItem(
                    id="test_1", title="Test Item 1", description="Description 1",
                    task_type=TaskType.FEATURE, impact=5, effort=3, status=TaskStatus.READY,
                    wsjf_score=1.67, created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Complete feature 1"]
                )
                
                item2 = BacklogItem(
                    id="test_2", title="Test Item 2", description="Description 2",
                    task_type=TaskType.BUG, impact=8, effort=2, status=TaskStatus.BLOCKED,
                    wsjf_score=4.0, created_at=datetime.now(), updated_at=datetime.now(),
                    links=[], acceptance_criteria=["Fix bug"], blocked_reason="Needs approval"
                )
                
                executor.backlog = [item1, item2]
                
                # Save backlog
                await executor._save_backlog()
                assert executor.backlog_file.exists()
                
                # Clear and reload
                executor.backlog = []
                await executor._load_backlog()
                
                assert len(executor.backlog) == 2
                
                # Check that items were loaded correctly
                loaded_item1 = next(item for item in executor.backlog if item.id == "test_1")
                assert loaded_item1.title == "Test Item 1"
                assert loaded_item1.task_type == TaskType.FEATURE
                assert loaded_item1.status == TaskStatus.READY
                
                loaded_item2 = next(item for item in executor.backlog if item.id == "test_2")
                assert loaded_item2.title == "Test Item 2"
                assert loaded_item2.task_type == TaskType.BUG
                assert loaded_item2.status == TaskStatus.BLOCKED
                assert loaded_item2.blocked_reason == "Needs approval"


class TestExecutionMetrics:
    """Test ExecutionMetrics data class"""
    
    def test_execution_metrics_creation(self):
        """Test creating execution metrics"""
        start_time = datetime.now()
        metrics = ExecutionMetrics(cycle_start=start_time)
        
        assert metrics.cycle_start == start_time
        assert metrics.items_processed == 0
        assert metrics.items_completed == 0
        assert metrics.items_blocked == 0
        assert metrics.coverage_delta == 0.0
        assert metrics.wsjf_distribution == {}
        assert metrics.cycle_time_avg == 0.0
    
    def test_execution_metrics_with_data(self):
        """Test execution metrics with data"""
        start_time = datetime.now()
        wsjf_dist = {"1-2": 3, "2-3": 5}
        
        metrics = ExecutionMetrics(
            cycle_start=start_time,
            items_processed=10,
            items_completed=7,
            items_blocked=2,
            coverage_delta=5.5,
            wsjf_distribution=wsjf_dist,
            cycle_time_avg=120.5
        )
        
        assert metrics.items_processed == 10
        assert metrics.items_completed == 7
        assert metrics.items_blocked == 2
        assert metrics.coverage_delta == 5.5
        assert metrics.wsjf_distribution == wsjf_dist
        assert metrics.cycle_time_avg == 120.5


# Integration test helpers
@pytest.mark.asyncio
async def test_full_cycle_integration(temp_workspace):
    """Integration test for a complete execution cycle"""
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI') as mock_github:
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer') as mock_analyzer:
                
                # Setup mocks
                mock_config.return_value.get_config.return_value = {
                    'github': {'reposToScan': ['test/repo'], 'managerRepo': 'test/manager'}
                }
                
                mock_repo = Mock()
                mock_repo.full_name = 'test/repo'
                mock_github.return_value.get_repo = AsyncMock(return_value=mock_repo)
                
                mock_analyzer.return_value.find_todo_comments_async = AsyncMock(return_value=[
                    {
                        'title': 'Fix TODO in auth module',
                        'description': 'TODO: Add rate limiting',
                        'content': 'TODO: Add rate limiting to prevent brute force attacks',
                        'url': 'https://github.com/test/repo/blob/main/auth.py#L42'
                    }
                ])
                
                with patch.object(Path, 'cwd', return_value=temp_workspace):
                    executor = ContinuousBacklogExecutor()
                    executor.backlog_file = temp_workspace / "test_backlog.json"
                    
                    # Run a single sync cycle
                    await executor._sync_and_refresh()
                    
                    # Should have discovered and processed TODO items
                    assert len(executor.backlog) > 0
                    
                    # Check that items were properly normalized and scored
                    for item in executor.backlog:
                        assert item.wsjf_score > 0
                        assert item.status in [TaskStatus.NEW, TaskStatus.REFINED, TaskStatus.READY]
                        assert len(item.acceptance_criteria) > 0


if __name__ == '__main__':
    pytest.main([__file__])