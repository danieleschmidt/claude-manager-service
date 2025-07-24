#!/usr/bin/env python3
"""
Integration tests for Continuous Backlog Executor
"""

import asyncio
import json
import os
import pytest
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

sys.path.append('/root/repo/src')

from continuous_backlog_executor import ContinuousBacklogExecutor, TaskStatus, TaskType


@pytest.fixture
def integration_workspace():
    """Create integration test workspace"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure
        (temp_path / "DOCS").mkdir()
        (temp_path / "DOCS" / "status").mkdir()
        (temp_path / "src").mkdir()
        
        # Create mock config file
        config = {
            'github': {
                'username': 'testuser',
                'managerRepo': 'testuser/test-manager',
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
        
        config_file = temp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        yield temp_path


@patch.dict('os.environ', {'GITHUB_TOKEN': 'ghp_' + 'x' * 36})
@pytest.mark.asyncio
async def test_full_discovery_and_processing_cycle(integration_workspace):
    """Test complete discovery and processing cycle"""
    
    # Mock GitHub API responses
    mock_search_results = [
        Mock(
            path="src/auth.py",
            html_url="https://github.com/test/repo/blob/main/src/auth.py"
        ),
        Mock(
            path="src/api.py", 
            html_url="https://github.com/test/repo/blob/main/src/api.py"
        )
    ]
    
    mock_file_content_auth = Mock()
    mock_file_content_auth.decoded_content = b"""
def authenticate_user(username, password):
    # TODO: Add rate limiting to prevent brute force attacks
    # This is a critical security vulnerability
    return check_credentials(username, password)
"""
    
    mock_file_content_api = Mock()
    mock_file_content_api.decoded_content = b"""
def process_request(data):
    # FIXME: Add input validation here
    # Current implementation is vulnerable to injection
    return execute_query(data)
"""
    
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config_service:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI') as mock_github:
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer') as mock_analyzer:
                with patch('src.continuous_backlog_executor.AsyncOrchestrator') as mock_orchestrator:
                    with patch('src.continuous_backlog_executor.RepositoryService') as mock_repo_service:
                        with patch('async_github_api.GitHubAPI') as mock_sync_github:
                        
                        # Setup configuration service
                        config_path = integration_workspace / "config.json"
                        mock_config_service.return_value.get_config.return_value = {
                            'github': {
                                'reposToScan': ['testuser/test-repo'],
                                'managerRepo': 'testuser/test-manager'
                            }
                        }
                        
                        # Setup GitHub API mock
                        mock_repo = Mock()
                        mock_repo.full_name = 'testuser/test-repo'
                        mock_github_instance = mock_github.return_value
                        mock_github_instance.get_repo = AsyncMock(return_value=mock_repo)
                        
                        # Setup task analyzer mock to return TODO discoveries
                        mock_analyzer_instance = mock_analyzer.return_value
                        mock_analyzer_instance.find_todo_comments_async = AsyncMock(return_value=[
                            {
                                'title': 'Address TODO in src/auth.py:3',
                                'description': 'Add rate limiting to prevent brute force attacks',
                                'content': 'TODO: Add rate limiting to prevent brute force attacks',
                                'url': 'https://github.com/test/repo/blob/main/src/auth.py#L3',
                                'file_path': 'src/auth.py',
                                'line_number': 3
                            },
                            {
                                'title': 'Address FIXME in src/api.py:2',
                                'description': 'Add input validation here',
                                'content': 'FIXME: Add input validation here',
                                'url': 'https://github.com/test/repo/blob/main/src/api.py#L2',
                                'file_path': 'src/api.py',
                                'line_number': 2
                            }
                        ])
                        
                        # Create executor with mocked workspace
                        with patch.object(Path, 'cwd', return_value=integration_workspace):
                            executor = ContinuousBacklogExecutor(str(config_path))
                            executor.backlog_file = integration_workspace / "DOCS" / "backlog.json"
                            executor.status_dir = integration_workspace / "DOCS" / "status"
                            
                            # Run discovery and initial processing
                            await executor._sync_and_refresh()
                            
                            # Verify backlog was populated
                            assert len(executor.backlog) >= 2
                            
                            # Find the security-related item (rate limiting)
                            security_item = None
                            validation_item = None
                            
                            for item in executor.backlog:
                                if "rate limiting" in item.description.lower():
                                    security_item = item
                                elif "validation" in item.description.lower():
                                    validation_item = item
                            
                            # Verify security item properties
                            assert security_item is not None
                            assert security_item.task_type == TaskType.SECURITY
                            assert security_item.impact >= 8  # High impact due to security
                            assert security_item.wsjf_score > 0
                            assert len(security_item.acceptance_criteria) >= 5
                            assert "Security-related change" in security_item.security_notes
                            
                            # Verify validation item properties
                            assert validation_item is not None
                            assert validation_item.task_type == TaskType.SECURITY  # Due to "injection" vulnerability
                            assert validation_item.impact >= 8
                            assert validation_item.wsjf_score > 0
                            
                            # Verify items are properly ranked by WSJF
                            scores = [item.wsjf_score for item in executor.backlog]
                            assert scores == sorted(scores, reverse=True)
                            
                            # Test getting actionable items
                            actionable = executor._get_actionable_items()
                            
                            # Should have actionable items if they're marked as READY
                            ready_items = [item for item in executor.backlog if item.status == TaskStatus.READY]
                            assert len(actionable) == len(ready_items)


@pytest.mark.asyncio
async def test_tdd_cycle_execution(integration_workspace):
    """Test TDD cycle execution with realistic scenario"""
    
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config_service:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI'):
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer'):
                with patch('src.continuous_backlog_executor.AsyncOrchestrator'):
                    with patch('src.continuous_backlog_executor.RepositoryService'):
                        
                        config_path = integration_workspace / "config.json"
                        mock_config_service.return_value.get_config.return_value = {
                            'github': {'reposToScan': ['test/repo']}
                        }
                        
                        with patch.object(Path, 'cwd', return_value=integration_workspace):
                            executor = ContinuousBacklogExecutor(str(config_path))
                            
                            # Create a test item ready for processing
                            from src.continuous_backlog_executor import BacklogItem
                            
                            test_item = BacklogItem(
                                id="test_security_fix",
                                title="Fix authentication rate limiting",
                                description="Add rate limiting to prevent brute force attacks",
                                task_type=TaskType.SECURITY,
                                impact=8,
                                effort=3,
                                status=TaskStatus.READY,
                                wsjf_score=10.67,
                                created_at=datetime.now(),
                                updated_at=datetime.now(),
                                links=["https://github.com/test/repo/issues/123"],
                                acceptance_criteria=[
                                    "Implement rate limiting middleware",
                                    "Add unit tests for rate limiting",
                                    "Ensure existing tests pass",
                                    "Update security documentation",
                                    "No new vulnerabilities introduced"
                                ],
                                security_notes="Critical security fix - requires thorough testing",
                                test_notes="Must include tests for rate limit scenarios"
                            )
                            
                            # Test processing the item
                            result = await executor._process_backlog_item(test_item)
                            
                            # Should complete successfully (simulated)
                            assert result == "completed"
                            assert test_item.status == TaskStatus.PR
                            assert test_item.pr_url != ""
                            
                            # Verify TDD cycle steps were followed
                            # (In a real implementation, this would verify actual code changes)


@pytest.mark.asyncio
async def test_blocked_item_handling(integration_workspace):
    """Test handling of blocked items"""
    
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config_service:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI'):
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer'):
                with patch('src.continuous_backlog_executor.AsyncOrchestrator'):
                    with patch('src.continuous_backlog_executor.RepositoryService'):
                        
                        config_path = integration_workspace / "config.json"
                        mock_config_service.return_value.get_config.return_value = {
                            'github': {'reposToScan': ['test/repo']}
                        }
                        
                        with patch.object(Path, 'cwd', return_value=integration_workspace):
                            executor = ContinuousBacklogExecutor(str(config_path))
                            
                            from src.continuous_backlog_executor import BacklogItem
                            
                            # High-risk item that should require human clarification
                            high_risk_item = BacklogItem(
                                id="high_risk_auth",
                                title="Redesign authentication system",
                                description="Complete overhaul of public authentication API",
                                task_type=TaskType.SECURITY,
                                impact=13,
                                effort=13,  # Very large effort
                                status=TaskStatus.READY,
                                wsjf_score=8.0,
                                created_at=datetime.now(),
                                updated_at=datetime.now(),
                                links=[],
                                acceptance_criteria=["Redesign auth system"]
                            )
                            
                            # Test processing - should be blocked for human review
                            result = await executor._process_backlog_item(high_risk_item)
                            
                            assert result == "blocked"
                            assert high_risk_item.status == TaskStatus.BLOCKED
                            assert "human clarification" in high_risk_item.blocked_reason
                            
                            # Test termination condition with all items blocked
                            executor.backlog = [high_risk_item]
                            assert executor._should_terminate() == True


@pytest.mark.asyncio
async def test_metrics_and_reporting(integration_workspace):
    """Test metrics collection and status reporting"""
    
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config_service:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI'):
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer'):
                with patch('src.continuous_backlog_executor.AsyncOrchestrator'):
                    with patch('src.continuous_backlog_executor.RepositoryService'):
                        
                        config_path = integration_workspace / "config.json"
                        mock_config_service.return_value.get_config.return_value = {
                            'github': {'reposToScan': ['test/repo']}
                        }
                        
                        with patch.object(Path, 'cwd', return_value=integration_workspace):
                            executor = ContinuousBacklogExecutor(str(config_path))
                            executor.status_dir = integration_workspace / "DOCS" / "status"
                            
                            from src.continuous_backlog_executor import BacklogItem, ExecutionMetrics
                            
                            # Create test backlog with various statuses
                            executor.backlog = [
                                BacklogItem(
                                    id="completed_1", title="Completed Item", description="",
                                    task_type=TaskType.BUG, impact=5, effort=2, status=TaskStatus.DONE,
                                    wsjf_score=2.5, created_at=datetime.now(), updated_at=datetime.now(),
                                    links=[], acceptance_criteria=["Fix completed"]
                                ),
                                BacklogItem(
                                    id="blocked_1", title="Blocked Item", description="",
                                    task_type=TaskType.SECURITY, impact=8, effort=3, status=TaskStatus.BLOCKED,
                                    wsjf_score=10.67, created_at=datetime.now(), updated_at=datetime.now(),
                                    links=[], acceptance_criteria=["Security fix"], 
                                    blocked_reason="Waiting for security review"
                                ),
                                BacklogItem(
                                    id="ready_1", title="Ready Item", description="",
                                    task_type=TaskType.FEATURE, impact=3, effort=1, status=TaskStatus.READY,
                                    wsjf_score=8.0, created_at=datetime.now(), updated_at=datetime.now(),
                                    links=[], acceptance_criteria=["Add feature"]
                                )
                            ]
                            
                            # Create test metrics
                            metrics = ExecutionMetrics(
                                cycle_start=datetime.now(),
                                items_processed=3,
                                items_completed=1,
                                items_blocked=1
                            )
                            
                            # Update metrics from backlog
                            executor._update_metrics(metrics)
                            
                            # Generate status report
                            await executor._generate_status_report(metrics)
                            
                            # Verify status report was created
                            status_files = list(executor.status_dir.glob("status_*.json"))
                            assert len(status_files) >= 1
                            
                            # Read and verify report content
                            with open(status_files[-1], 'r') as f:
                                report = json.load(f)
                            
                            assert "timestamp" in report
                            assert "completed_items" in report
                            assert report["items_processed_this_cycle"] == 3
                            assert report["items_completed_this_cycle"] == 1
                            assert report["items_blocked_this_cycle"] == 1
                            assert len(report["notable_risks_or_blocks"]) >= 1
                            assert "Waiting for security review" in report["notable_risks_or_blocks"][0]
                            
                            # Verify status counts
                            status_counts = report["backlog_size_by_status"]
                            assert status_counts["DONE"] == 1
                            assert status_counts["BLOCKED"] == 1
                            assert status_counts["READY"] == 1


@pytest.mark.asyncio
async def test_backlog_persistence(integration_workspace):
    """Test backlog saving and loading persistence"""
    
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config_service:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI'):
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer'):
                with patch('src.continuous_backlog_executor.AsyncOrchestrator'):
                    with patch('src.continuous_backlog_executor.RepositoryService'):
                        
                        config_path = integration_workspace / "config.json"
                        mock_config_service.return_value.get_config.return_value = {
                            'github': {'reposToScan': ['test/repo']}
                        }
                        
                        with patch.object(Path, 'cwd', return_value=integration_workspace):
                            # Create first executor instance
                            executor1 = ContinuousBacklogExecutor(str(config_path))
                            executor1.backlog_file = integration_workspace / "DOCS" / "backlog.json"
                            
                            from src.continuous_backlog_executor import BacklogItem
                            
                            # Add items to backlog
                            executor1.backlog = [
                                BacklogItem(
                                    id="persist_1", title="Persistent Item 1", 
                                    description="Test persistence", task_type=TaskType.FEATURE,
                                    impact=5, effort=3, status=TaskStatus.READY, wsjf_score=1.67,
                                    created_at=datetime.now(), updated_at=datetime.now(),
                                    links=["http://test.com"], acceptance_criteria=["Persist data"],
                                    security_notes="Standard security", test_notes="Add tests"
                                ),
                                BacklogItem(
                                    id="persist_2", title="Persistent Item 2",
                                    description="Test persistence 2", task_type=TaskType.BUG,
                                    impact=8, effort=2, status=TaskStatus.BLOCKED, wsjf_score=4.0,
                                    created_at=datetime.now(), updated_at=datetime.now(),
                                    links=[], acceptance_criteria=["Fix bug"],
                                    blocked_reason="External dependency"
                                )
                            ]
                            
                            # Save backlog
                            await executor1._save_backlog()
                            
                            # Verify file was created
                            assert executor1.backlog_file.exists()
                            
                            # Create second executor instance
                            executor2 = ContinuousBacklogExecutor(str(config_path))
                            executor2.backlog_file = integration_workspace / "DOCS" / "backlog.json"
                            
                            # Load backlog
                            await executor2._load_backlog()
                            
                            # Verify items were loaded correctly
                            assert len(executor2.backlog) == 2
                            
                            item1 = next(item for item in executor2.backlog if item.id == "persist_1")
                            assert item1.title == "Persistent Item 1"
                            assert item1.task_type == TaskType.FEATURE
                            assert item1.status == TaskStatus.READY
                            assert item1.impact == 5
                            assert item1.effort == 3
                            assert len(item1.acceptance_criteria) == 1
                            
                            item2 = next(item for item in executor2.backlog if item.id == "persist_2")
                            assert item2.title == "Persistent Item 2"
                            assert item2.task_type == TaskType.BUG
                            assert item2.status == TaskStatus.BLOCKED
                            assert item2.blocked_reason == "External dependency"


@pytest.mark.asyncio
async def test_large_item_splitting_integration(integration_workspace):
    """Test integration of large item splitting functionality"""
    
    with patch('src.continuous_backlog_executor.ConfigurationService') as mock_config_service:
        with patch('src.continuous_backlog_executor.AsyncGitHubAPI'):
            with patch('src.continuous_backlog_executor.AsyncTaskAnalyzer'):
                with patch('src.continuous_backlog_executor.AsyncOrchestrator'):
                    with patch('src.continuous_backlog_executor.RepositoryService'):
                        
                        config_path = integration_workspace / "config.json"
                        mock_config_service.return_value.get_config.return_value = {
                            'github': {'reposToScan': ['test/repo']}
                        }
                        
                        with patch.object(Path, 'cwd', return_value=integration_workspace):
                            executor = ContinuousBacklogExecutor(str(config_path))
                            executor.slice_size_threshold = 5  # Items > 5 effort get split
                            
                            from src.continuous_backlog_executor import BacklogItem
                            
                            # Create large item that should be split
                            large_item = BacklogItem(
                                id="large_refactor",
                                title="Complete Authentication System Refactor",
                                description="Massive refactor of the entire authentication system",
                                task_type=TaskType.REFACTOR,
                                impact=13,
                                effort=15,  # Much larger than threshold
                                status=TaskStatus.NEW,
                                wsjf_score=0.0,
                                created_at=datetime.now(),
                                updated_at=datetime.now(),
                                links=["https://github.com/test/repo/issues/456"],
                                acceptance_criteria=["Refactor auth system completely"],
                                security_notes="Major security implications",
                                test_notes="Comprehensive test suite required"
                            )
                            
                            executor.backlog = [large_item]
                            
                            # Run normalization which should split the item
                            executor._normalize_backlog_items()
                            
                            # Original item should be marked as DONE (split)
                            assert large_item.status == TaskStatus.DONE
                            
                            # Should have created multiple slice items
                            slices = [item for item in executor.backlog 
                                     if item.id.startswith("large_refactor_slice_")]
                            
                            assert len(slices) >= 3  # 15 effort / 5 threshold = 3 slices
                            
                            # Each slice should have reasonable effort
                            for slice_item in slices:
                                assert slice_item.effort <= executor.slice_size_threshold
                                assert slice_item.task_type == TaskType.REFACTOR
                                assert slice_item.impact == 13  # Inherits impact
                                assert "Part" in slice_item.title
                                assert slice_item.status == TaskStatus.NEW
                                
                            # Total effort should be preserved (approximately)
                            total_slice_effort = sum(slice_item.effort for slice_item in slices)
                            assert abs(total_slice_effort - 15) <= len(slices)  # Allow for rounding


if __name__ == '__main__':
    pytest.main([__file__])