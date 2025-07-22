"""
Integration tests for Performance Monitoring system

These tests verify that the performance monitoring integrates correctly
with the existing Claude Manager Service components and provides
accurate metrics in realistic scenarios.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from github_api import GitHubAPI
from orchestrator import trigger_terragon_task
from task_analyzer import find_todo_comments, analyze_open_issues
from performance_monitor import get_monitor, OperationMetrics


class TestGitHubAPIPerformanceMonitoring:
    """Test performance monitoring integration with GitHub API operations"""
    
    @pytest.fixture
    def mock_github_api(self):
        """Create a mocked GitHub API for testing"""
        with patch('github_api.Github') as mock_github_class:
            # Mock the GitHub client
            mock_client = Mock()
            mock_github_class.return_value = mock_client
            
            # Mock user for authentication test
            mock_user = Mock()
            mock_user.login = "test_user"
            mock_client.get_user.return_value = mock_user
            
            # Mock repository
            mock_repo = Mock()
            mock_repo.full_name = "test_user/test_repo"
            mock_client.get_repo.return_value = mock_repo
            
            # Mock issue creation
            mock_issue = Mock()
            mock_issue.number = 123
            mock_repo.create_issue.return_value = mock_issue
            
            # Mock existing issues check
            mock_repo.get_issues.return_value = []
            
            with patch('github_api.get_secure_config') as mock_config:
                mock_config_obj = Mock()
                mock_config_obj.get_github_token.return_value = "fake_token"
                mock_config.return_value = mock_config_obj
                
                yield GitHubAPI()
    
    def test_get_repo_performance_monitoring(self, mock_github_api):
        """Test that get_repo operations are properly monitored"""
        monitor = get_monitor()
        initial_count = len(monitor.operations)
        
        # Call the method
        repo = mock_github_api.get_repo("test_user/test_repo")
        
        # Verify the operation was monitored
        assert len(monitor.operations) > initial_count
        
        # Find the operation
        repo_operations = [
            op for op in monitor.operations 
            if op.function_name == "github_get_repository"
        ]
        
        assert len(repo_operations) > 0
        latest_op = repo_operations[-1]
        
        assert latest_op.success is True
        assert latest_op.duration > 0
        assert latest_op.module_name == "github_api"
    
    def test_create_issue_performance_monitoring(self, mock_github_api):
        """Test that create_issue operations are properly monitored"""
        monitor = get_monitor()
        initial_count = len(monitor.operations)
        
        # Call the method
        mock_github_api.create_issue(
            "test_user/test_repo",
            "Test Issue",
            "This is a test issue",
            ["bug", "test"]
        )
        
        # Verify the operation was monitored
        assert len(monitor.operations) > initial_count
        
        # Find the operation
        issue_operations = [
            op for op in monitor.operations 
            if op.function_name == "github_create_issue"
        ]
        
        assert len(issue_operations) > 0
        latest_op = issue_operations[-1]
        
        assert latest_op.success is True
        assert latest_op.duration > 0
        assert latest_op.module_name == "github_api"
    
    def test_api_call_statistics(self, mock_github_api):
        """Test that API calls are properly categorized and counted"""
        monitor = get_monitor()
        
        # Make several API calls
        for i in range(3):
            mock_github_api.get_repo(f"test_user/repo_{i}")
        
        for i in range(2):
            mock_github_api.create_issue(
                "test_user/test_repo",
                f"Issue {i}",
                f"Description {i}",
                ["test"]
            )
        
        # Check API call statistics
        api_summary = monitor.get_api_call_summary()
        
        # Should have entries for our API operations
        github_get_ops = [key for key in api_summary.keys() if "github_get_repository" in key]
        github_create_ops = [key for key in api_summary.keys() if "github_create_issue" in key]
        
        assert len(github_get_ops) > 0
        assert len(github_create_ops) > 0


class TestOrchestratorPerformanceMonitoring:
    """Test performance monitoring integration with orchestrator operations"""
    
    @pytest.fixture
    def mock_orchestrator_dependencies(self):
        """Mock dependencies for orchestrator testing"""
        with patch('orchestrator.build_prompt') as mock_build_prompt, \
             patch('orchestrator.get_template_for_labels') as mock_get_template, \
             patch('orchestrator.get_validated_config') as mock_get_config:
            
            mock_build_prompt.return_value = "Test prompt content"
            mock_get_template.return_value = "template.txt"
            mock_get_config.return_value = {
                'executor': {'terragonUsername': '@test-bot'}
            }
            
            yield {
                'build_prompt': mock_build_prompt,
                'get_template': mock_get_template,
                'get_config': mock_get_config
            }
    
    def test_terragon_task_orchestration_monitoring(self, mock_orchestrator_dependencies):
        """Test performance monitoring of Terragon task orchestration"""
        monitor = get_monitor()
        initial_count = len(monitor.operations)
        
        # Mock GitHub API
        mock_api = Mock()
        mock_api.add_comment_to_issue = Mock()
        
        # Mock issue
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = "Test Issue"
        mock_issue.body = "Test body"
        mock_issue.labels = []
        mock_issue.html_url = "https://github.com/test/test/issues/123"
        
        # Call the orchestrator function
        trigger_terragon_task(
            mock_api,
            "test_user/test_repo",
            mock_issue,
            {'executor': {'terragonUsername': '@test-bot'}}
        )
        
        # Verify monitoring captured the operation
        assert len(monitor.operations) > initial_count
        
        # Find the orchestration operation
        orchestration_ops = [
            op for op in monitor.operations 
            if op.function_name == "terragon_task_orchestration"
        ]
        
        assert len(orchestration_ops) > 0
        latest_op = orchestration_ops[-1]
        
        assert latest_op.success is True
        assert latest_op.duration > 0
        assert latest_op.memory_before is not None  # Memory tracking enabled
        assert latest_op.memory_after is not None
    
    def test_orchestrator_error_monitoring(self, mock_orchestrator_dependencies):
        """Test monitoring of orchestrator errors"""
        monitor = get_monitor()
        
        # Mock GitHub API to raise an exception
        mock_api = Mock()
        mock_api.add_comment_to_issue.side_effect = Exception("API Error")
        
        # Mock issue
        mock_issue = Mock()
        mock_issue.number = 123
        mock_issue.title = "Test Issue"
        mock_issue.body = "Test body"
        mock_issue.labels = []
        mock_issue.html_url = "https://github.com/test/test/issues/123"
        
        # Call the orchestrator function and expect it to raise
        with pytest.raises(Exception, match="API Error"):
            trigger_terragon_task(
                mock_api,
                "test_user/test_repo",
                mock_issue,
                {'executor': {'terragonUsername': '@test-bot'}}
            )
        
        # Find the failed orchestration operation
        orchestration_ops = [
            op for op in monitor.operations 
            if op.function_name == "terragon_task_orchestration"
        ]
        
        # Should have at least one failed operation
        failed_ops = [op for op in orchestration_ops if not op.success]
        assert len(failed_ops) > 0
        
        latest_failed = failed_ops[-1]
        assert latest_failed.success is False
        assert latest_failed.error_message == "API Error"


class TestTaskAnalyzerPerformanceMonitoring:
    """Test performance monitoring integration with task analyzer operations"""
    
    @pytest.fixture
    def mock_task_analyzer_dependencies(self):
        """Mock dependencies for task analyzer testing"""
        with patch('task_analyzer.get_task_tracker') as mock_get_tracker:
            mock_tracker = Mock()
            mock_tracker.has_been_processed.return_value = False
            mock_tracker.mark_as_processed = Mock()
            mock_get_tracker.return_value = mock_tracker
            
            yield mock_tracker
    
    def test_todo_scanning_performance_monitoring(self, mock_task_analyzer_dependencies):
        """Test performance monitoring of TODO comment scanning"""
        monitor = get_monitor()
        initial_count = len(monitor.operations)
        
        # Mock GitHub API
        mock_api = Mock()
        mock_api.client = Mock()
        
        # Mock search results
        mock_search_result = Mock()
        mock_search_result.path = "test_file.py"
        mock_search_result.html_url = "https://github.com/test/test/blob/main/test_file.py"
        
        mock_api.client.search_code.return_value = [mock_search_result]
        
        # Mock repository
        mock_repo = Mock()
        mock_repo.full_name = "test_user/test_repo"
        
        # Mock file content
        mock_file_content = Mock()
        mock_file_content.decoded_content = b"line 1\n# TODO: Fix this\nline 3\n"
        mock_repo.get_contents.return_value = mock_file_content
        
        mock_api.create_issue = Mock()
        
        # Call the function
        find_todo_comments(mock_api, mock_repo, "test_user/manager_repo")
        
        # Verify monitoring captured the operation
        assert len(monitor.operations) > initial_count
        
        # Find the scanning operation
        scanning_ops = [
            op for op in monitor.operations 
            if op.function_name == "scan_todo_comments"
        ]
        
        assert len(scanning_ops) > 0
        latest_op = scanning_ops[-1]
        
        assert latest_op.success is True
        assert latest_op.duration > 0
        assert latest_op.memory_before is not None  # Memory tracking enabled
    
    def test_issue_analysis_performance_monitoring(self, mock_task_analyzer_dependencies):
        """Test performance monitoring of open issue analysis"""
        monitor = get_monitor()
        initial_count = len(monitor.operations)
        
        # Mock GitHub API
        mock_api = Mock()
        mock_api.create_issue = Mock()
        
        # Mock repository
        mock_repo = Mock()
        mock_repo.full_name = "test_user/test_repo"
        
        # Mock old issue
        mock_issue = Mock()
        mock_issue.title = "Old Issue"
        mock_issue.body = "This is an old issue"
        mock_issue.number = 456
        mock_issue.html_url = "https://github.com/test/test/issues/456"
        mock_issue.created_at = time.time() - (60 * 24 * 3600)  # 60 days ago
        mock_issue.updated_at = time.time() - (40 * 24 * 3600)  # 40 days ago
        mock_issue.pull_request = None
        mock_issue.assignees = []
        
        # Mock label
        mock_label = Mock()
        mock_label.name = "bug"
        mock_issue.labels = [mock_label]
        
        mock_repo.get_issues.return_value = [mock_issue]
        
        # Call the function
        analyze_open_issues(mock_api, mock_repo, "test_user/manager_repo")
        
        # Verify monitoring captured the operation
        assert len(monitor.operations) > initial_count
        
        # Find the analysis operation
        analysis_ops = [
            op for op in monitor.operations 
            if op.function_name == "analyze_open_issues"
        ]
        
        assert len(analysis_ops) > 0
        latest_op = analysis_ops[-1]
        
        assert latest_op.success is True
        assert latest_op.duration > 0


class TestPerformanceReportGeneration:
    """Test performance report generation with real monitoring data"""
    
    def test_comprehensive_performance_report(self):
        """Test generating a comprehensive performance report with mixed operations"""
        monitor = get_monitor()
        
        # Simulate various operations with different characteristics
        base_time = time.time()
        
        # Fast successful operations
        for i in range(10):
            metrics = OperationMetrics(
                function_name="fast_operation",
                module_name="test_module",
                start_time=base_time + i,
                end_time=base_time + i + 0.01,
                duration=0.01,
                success=True,
                memory_delta=1.0
            )
            monitor.record_operation(metrics)
        
        # Slow successful operations
        for i in range(5):
            metrics = OperationMetrics(
                function_name="slow_operation",
                module_name="test_module",
                start_time=base_time + 10 + i,
                end_time=base_time + 10 + i + 1.0,
                duration=1.0,
                success=True,
                memory_delta=10.0
            )
            monitor.record_operation(metrics)
        
        # Failed operations
        for i in range(3):
            metrics = OperationMetrics(
                function_name="failing_operation",
                module_name="test_module",
                start_time=base_time + 15 + i,
                end_time=base_time + 15 + i + 0.5,
                duration=0.5,
                success=False,
                error_message="Test error",
                memory_delta=0.5
            )
            monitor.record_operation(metrics)
        
        # API operations
        for i in range(7):
            metrics = OperationMetrics(
                function_name="api_call",
                module_name="api_module",
                start_time=base_time + 18 + i,
                end_time=base_time + 18 + i + 0.3,
                duration=0.3,
                success=i < 6,  # 6 succeed, 1 fails
                memory_delta=2.0
            )
            monitor.record_operation(metrics)
        
        # Generate report
        report = monitor.get_performance_report(1)  # Last 1 hour
        
        # Verify report structure and content
        assert 'overall_stats' in report
        assert 'function_breakdown' in report
        assert 'api_call_summary' in report
        assert 'memory_stats' in report
        assert 'slowest_operations' in report
        
        # Check overall stats
        overall = report['overall_stats']
        assert overall['total_operations'] >= 25  # At least our test operations
        assert overall['successful_operations'] >= 22
        assert overall['failed_operations'] >= 3
        assert 0 <= overall['success_rate'] <= 1
        
        # Check function breakdown
        functions = report['function_breakdown']
        assert 'test_module.fast_operation' in functions
        assert 'test_module.slow_operation' in functions
        assert 'test_module.failing_operation' in functions
        
        # Verify fast operations have better performance
        fast_stats = functions['test_module.fast_operation']
        slow_stats = functions['test_module.slow_operation']
        assert fast_stats['avg_duration'] < slow_stats['avg_duration']
        
        # Check API call summary
        api_summary = report['api_call_summary']
        assert 'api_module.api_call' in api_summary
        api_stats = api_summary['api_module.api_call']
        assert api_stats['total_calls'] == 7
        assert api_stats['successful_calls'] == 6
        assert api_stats['failed_calls'] == 1
        
        # Check memory stats
        memory_stats = report['memory_stats']
        assert memory_stats is not None
        assert memory_stats['total_operations_with_memory_tracking'] >= 25
        assert 'avg_memory_delta' in memory_stats
        
        # Check slowest operations
        slowest = report['slowest_operations']
        assert len(slowest) > 0
        # Slowest should be the slow_operation
        assert slowest[0]['function'] == 'test_module.slow_operation'
        assert slowest[0]['duration'] == 1.0
    
    def test_function_specific_metrics(self):
        """Test getting specific function metrics"""
        monitor = get_monitor()
        
        # Create operations for a specific function with varying performance
        base_time = time.time()
        durations = [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.12, 0.18, 0.22, 0.08]
        
        for i, duration in enumerate(durations):
            metrics = OperationMetrics(
                function_name="target_function",
                module_name="target_module",
                start_time=base_time + i,
                end_time=base_time + i + duration,
                duration=duration,
                success=i < 8,  # 8 succeed, 2 fail
                memory_delta=duration * 10  # Memory proportional to duration
            )
            monitor.record_operation(metrics)
        
        # Get function-specific metrics
        metrics = monitor.get_function_metrics("target_function", "target_module")
        
        assert metrics is not None
        assert metrics.function_name == "target_function"
        assert metrics.module_name == "target_module"
        assert metrics.total_calls == 10
        assert metrics.successful_calls == 8
        assert metrics.failed_calls == 2
        assert metrics.success_rate == 0.8
        
        # Check duration statistics
        assert metrics.min_duration == 0.05
        assert metrics.max_duration == 0.3
        assert metrics.average_duration == sum(durations) / len(durations)
        
        # Check percentiles
        sorted_durations = sorted(durations)
        assert metrics.median_duration == (sorted_durations[4] + sorted_durations[5]) / 2
        
        # Check memory statistics
        assert metrics.avg_memory_usage is not None
        expected_avg_memory = sum(d * 10 for d in durations) / len(durations)
        assert abs(metrics.avg_memory_usage - expected_avg_memory) < 0.01


class TestPerformanceMonitoringPersistence:
    """Test persistence and data management features"""
    
    def test_metrics_save_and_load_cycle(self):
        """Test saving metrics and loading them back"""
        monitor = get_monitor()
        
        # Record some test data
        base_time = time.time()
        for i in range(5):
            metrics = OperationMetrics(
                function_name="persistent_function",
                module_name="persistent_module",
                start_time=base_time + i,
                end_time=base_time + i + 0.1,
                duration=0.1,
                success=True
            )
            monitor.record_operation(metrics)
        
        # Save metrics
        saved_file = monitor.save_metrics("test_metrics.json")
        assert saved_file is not None
        assert saved_file.exists()
        
        # Verify the saved file contains our data
        with open(saved_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'operations' in saved_data
        assert 'api_call_stats' in saved_data
        assert 'configuration' in saved_data
        
        # Check that our operations are in the saved data
        persistent_ops = [
            op for op in saved_data['operations']
            if op['function_name'] == 'persistent_function'
        ]
        assert len(persistent_ops) == 5
    
    def test_data_cleanup_functionality(self):
        """Test that old data is properly cleaned up"""
        monitor = get_monitor()
        
        # Record operations with timestamps spanning retention period
        current_time = time.time()
        old_time = current_time - (35 * 24 * 3600)  # 35 days ago (beyond retention)
        recent_time = current_time - (5 * 24 * 3600)  # 5 days ago (within retention)
        
        # Old operations (should be cleaned)
        for i in range(3):
            metrics = OperationMetrics(
                function_name="old_function",
                module_name="old_module",
                start_time=old_time + i,
                end_time=old_time + i + 0.1,
                duration=0.1,
                success=True
            )
            monitor.record_operation(metrics)
        
        # Recent operations (should be kept)
        for i in range(3):
            metrics = OperationMetrics(
                function_name="recent_function",
                module_name="recent_module",
                start_time=recent_time + i,
                end_time=recent_time + i + 0.1,
                duration=0.1,
                success=True
            )
            monitor.record_operation(metrics)
        
        # Force cleanup
        monitor._cleanup_old_data()
        
        # Check that only recent operations remain
        remaining_functions = set(op.function_name for op in monitor.operations)
        assert "old_function" not in remaining_functions
        assert "recent_function" in remaining_functions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])