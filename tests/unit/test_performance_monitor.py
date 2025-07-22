"""
Unit tests for the performance monitoring system

These tests verify the functionality of the PerformanceMonitor class,
decorators, and related utilities.
"""

import pytest
import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from performance_monitor import (
    PerformanceMonitor, 
    OperationMetrics, 
    AggregateMetrics,
    get_monitor,
    monitor_performance,
    monitor_api_call
)


class TestOperationMetrics:
    """Test the OperationMetrics dataclass"""
    
    def test_operation_metrics_creation(self):
        """Test creating operation metrics"""
        start_time = time.time()
        end_time = start_time + 1.5
        
        metrics = OperationMetrics(
            function_name="test_function",
            module_name="test_module",
            start_time=start_time,
            end_time=end_time,
            duration=1.5,
            success=True,
            memory_before=100.0,
            memory_after=110.0,
            memory_delta=10.0,
            args_count=2,
            kwargs_count=1
        )
        
        assert metrics.function_name == "test_function"
        assert metrics.module_name == "test_module"
        assert metrics.duration == 1.5
        assert metrics.success is True
        assert metrics.memory_delta == 10.0
    
    def test_operation_metrics_with_error(self):
        """Test creating operation metrics with error"""
        metrics = OperationMetrics(
            function_name="failing_function",
            module_name="test_module",
            start_time=time.time(),
            end_time=time.time() + 0.5,
            duration=0.5,
            success=False,
            error_message="Test error"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Test error"


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def monitor(self, temp_dir):
        """Create a fresh monitor instance for testing"""
        # Create a new instance bypassing singleton
        monitor = object.__new__(PerformanceMonitor)
        monitor._initialized = False
        
        # Mock the data directory
        with patch.object(PerformanceMonitor, 'data_dir', temp_dir):
            monitor.__init__()
        
        return monitor
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor._initialized is True
        assert monitor.max_operations_in_memory == 10000
        assert monitor.retention_days == 30
        assert len(monitor.operations) == 0
        assert len(monitor.function_stats) == 0
    
    def test_record_operation(self, monitor):
        """Test recording operation metrics"""
        start_time = time.time()
        metrics = OperationMetrics(
            function_name="test_func",
            module_name="test_module",
            start_time=start_time,
            end_time=start_time + 1.0,
            duration=1.0,
            success=True
        )
        
        monitor.record_operation(metrics)
        
        assert len(monitor.operations) == 1
        assert "test_module.test_func" in monitor.function_stats
        assert len(monitor.function_stats["test_module.test_func"]) == 1
    
    def test_record_api_operation(self, monitor):
        """Test recording API operation metrics"""
        start_time = time.time()
        metrics = OperationMetrics(
            function_name="api_call_function",
            module_name="github_api",
            start_time=start_time,
            end_time=start_time + 0.5,
            duration=0.5,
            success=True
        )
        
        monitor.record_operation(metrics)
        
        api_key = "github_api.api_call_function"
        assert api_key in monitor.api_call_stats
        assert monitor.api_call_stats[api_key]['success'] == 1
        assert monitor.api_call_stats[api_key]['failure'] == 0
    
    def test_record_failed_operation(self, monitor):
        """Test recording failed operation"""
        start_time = time.time()
        metrics = OperationMetrics(
            function_name="failing_func",
            module_name="test_module",
            start_time=start_time,
            end_time=start_time + 0.2,
            duration=0.2,
            success=False,
            error_message="Test error"
        )
        
        monitor.record_operation(metrics)
        
        assert len(monitor.operations) == 1
        assert monitor.operations[0].success is False
    
    def test_get_function_metrics(self, monitor):
        """Test getting aggregated function metrics"""
        # Record multiple operations for the same function
        base_time = time.time()
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for i, duration in enumerate(durations):
            metrics = OperationMetrics(
                function_name="test_func",
                module_name="test_module",
                start_time=base_time + i,
                end_time=base_time + i + duration,
                duration=duration,
                success=i < 4  # First 4 succeed, last one fails
            )
            monitor.record_operation(metrics)
        
        # Get aggregated metrics
        agg_metrics = monitor.get_function_metrics("test_func", "test_module")
        
        assert agg_metrics is not None
        assert agg_metrics.total_calls == 5
        assert agg_metrics.successful_calls == 4
        assert agg_metrics.failed_calls == 1
        assert agg_metrics.success_rate == 0.8
        assert agg_metrics.average_duration == 0.3
        assert agg_metrics.min_duration == 0.1
        assert agg_metrics.max_duration == 0.5
        assert agg_metrics.median_duration == 0.3
    
    def test_get_function_metrics_not_found(self, monitor):
        """Test getting metrics for non-existent function"""
        result = monitor.get_function_metrics("nonexistent", "module")
        assert result is None
    
    def test_get_api_call_summary(self, monitor):
        """Test getting API call summary"""
        # Record some API operations
        base_time = time.time()
        
        # Successful API calls
        for i in range(3):
            metrics = OperationMetrics(
                function_name="api_request",
                module_name="github_api",
                start_time=base_time + i,
                end_time=base_time + i + 0.1,
                duration=0.1,
                success=True
            )
            monitor.record_operation(metrics)
        
        # Failed API call
        metrics = OperationMetrics(
            function_name="api_request",
            module_name="github_api",
            start_time=base_time + 3,
            end_time=base_time + 3.2,
            duration=0.2,
            success=False
        )
        monitor.record_operation(metrics)
        
        summary = monitor.get_api_call_summary()
        api_key = "github_api.api_request"
        
        assert api_key in summary
        assert summary[api_key]['total_calls'] == 4
        assert summary[api_key]['successful_calls'] == 3
        assert summary[api_key]['failed_calls'] == 1
        assert summary[api_key]['success_rate'] == 0.75
        assert summary[api_key]['error_rate'] == 0.25
    
    def test_get_performance_report(self, monitor):
        """Test generating performance report"""
        # Record some operations
        base_time = time.time()
        
        for i in range(5):
            metrics = OperationMetrics(
                function_name=f"func_{i % 3}",
                module_name="test_module",
                start_time=base_time + i,
                end_time=base_time + i + 0.1 * (i + 1),
                duration=0.1 * (i + 1),
                success=i < 4,
                memory_delta=float(i) if i % 2 == 0 else None
            )
            monitor.record_operation(metrics)
        
        report = monitor.get_performance_report(1)
        
        assert 'overall_stats' in report
        assert 'function_breakdown' in report
        assert 'slowest_operations' in report
        
        overall = report['overall_stats']
        assert overall['total_operations'] == 5
        assert overall['successful_operations'] == 4
        assert overall['failed_operations'] == 1
        assert overall['success_rate'] == 0.8
    
    def test_get_performance_report_empty(self, monitor):
        """Test performance report with no data"""
        report = monitor.get_performance_report(24)
        assert "message" in report
        assert "No operations recorded" in report["message"]
    
    def test_percentile_calculation(self, monitor):
        """Test percentile calculation"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        assert monitor._percentile(data, 50) == 5.5  # Median
        assert monitor._percentile(data, 90) == 9.1
        assert monitor._percentile(data, 95) == 9.55
        assert monitor._percentile(data, 99) == 9.91
    
    def test_percentile_edge_cases(self, monitor):
        """Test percentile calculation edge cases"""
        # Empty data
        assert monitor._percentile([], 50) == 0.0
        
        # Single element
        assert monitor._percentile([5.0], 50) == 5.0
        assert monitor._percentile([5.0], 99) == 5.0
    
    def test_cleanup_old_data(self, monitor):
        """Test data cleanup functionality"""
        # Record operations with different timestamps
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        recent_time = time.time() - (10 * 24 * 3600)  # 10 days ago
        
        # Old operation (should be cleaned up)
        old_metrics = OperationMetrics(
            function_name="old_func",
            module_name="test_module",
            start_time=old_time,
            end_time=old_time + 1,
            duration=1.0,
            success=True
        )
        
        # Recent operation (should be kept)
        recent_metrics = OperationMetrics(
            function_name="recent_func",
            module_name="test_module",
            start_time=recent_time,
            end_time=recent_time + 1,
            duration=1.0,
            success=True
        )
        
        monitor.record_operation(old_metrics)
        monitor.record_operation(recent_metrics)
        
        # Run cleanup
        monitor._cleanup_old_data()
        
        # Check that old data is removed and recent data is kept
        assert len(monitor.operations) == 1
        assert monitor.operations[0].function_name == "recent_func"
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_metrics(self, mock_json_dump, mock_open, monitor, temp_dir):
        """Test saving metrics to file"""
        # Record some data
        metrics = OperationMetrics(
            function_name="test_func",
            module_name="test_module",
            start_time=time.time(),
            end_time=time.time() + 1,
            duration=1.0,
            success=True
        )
        monitor.record_operation(metrics)
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Save metrics
        result = monitor.save_metrics("test_metrics.json")
        
        # Verify file operations
        assert mock_open.called
        assert mock_json_dump.called
        assert result is not None


class TestMonitoringDecorators:
    """Test the monitoring decorators"""
    
    def test_monitor_performance_decorator(self):
        """Test the monitor_performance decorator"""
        
        @monitor_performance(track_memory=False, custom_name="test_operation")
        def test_function(x, y=10):
            time.sleep(0.01)  # Small delay to ensure measurable duration
            return x + y
        
        # Execute function
        result = test_function(5, y=15)
        
        # Verify function executed correctly
        assert result == 20
        
        # Verify monitoring recorded the operation
        monitor = get_monitor()
        
        # Find our operation in the recorded metrics
        found_operation = False
        for operation in monitor.operations:
            if operation.function_name == "test_operation":
                found_operation = True
                assert operation.success is True
                assert operation.duration > 0
                assert operation.args_count == 1  # 'x' argument
                assert operation.kwargs_count == 1  # 'y' keyword argument
                break
        
        assert found_operation, "Operation not found in monitoring data"
    
    def test_monitor_performance_with_error(self):
        """Test monitor_performance decorator with function that raises exception"""
        
        @monitor_performance(custom_name="failing_test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        # Execute function and expect exception
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # Verify monitoring recorded the failed operation
        monitor = get_monitor()
        
        found_operation = False
        for operation in monitor.operations:
            if operation.function_name == "failing_test_operation":
                found_operation = True
                assert operation.success is False
                assert operation.error_message == "Test error"
                assert operation.duration > 0
                break
        
        assert found_operation, "Failed operation not found in monitoring data"
    
    def test_monitor_api_call_decorator(self):
        """Test the monitor_api_call decorator"""
        
        @monitor_api_call("test_api_operation")
        def api_function():
            time.sleep(0.005)
            return {"status": "success"}
        
        # Execute function
        result = api_function()
        
        # Verify function executed correctly
        assert result["status"] == "success"
        
        # Verify monitoring recorded the operation as an API call
        monitor = get_monitor()
        
        found_operation = False
        for operation in monitor.operations:
            if operation.function_name == "test_api_operation":
                found_operation = True
                assert operation.success is True
                assert operation.duration > 0
                break
        
        assert found_operation, "API operation not found in monitoring data"
    
    @patch('psutil.Process')
    def test_monitor_performance_with_memory_tracking(self, mock_process):
        """Test monitor_performance decorator with memory tracking"""
        
        # Mock memory info
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        
        # Simulate memory usage change
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB first call
        
        @monitor_performance(track_memory=True, custom_name="memory_test_operation")
        def memory_function():
            # Simulate memory increase
            mock_process_instance.memory_info.return_value.rss = 110 * 1024 * 1024  # 110MB second call
            return "done"
        
        # Execute function
        result = memory_function()
        
        # Verify function executed correctly
        assert result == "done"
        
        # Verify memory tracking was recorded
        monitor = get_monitor()
        
        found_operation = False
        for operation in monitor.operations:
            if operation.function_name == "memory_test_operation":
                found_operation = True
                assert operation.memory_before == 100.0  # 100MB in MB
                assert operation.memory_after == 110.0   # 110MB in MB
                assert operation.memory_delta == 10.0    # 10MB difference
                break
        
        assert found_operation, "Memory-tracked operation not found in monitoring data"


class TestPerformanceAlerts:
    """Test performance alerting functionality"""
    
    @pytest.fixture
    def monitor_with_low_thresholds(self, temp_dir):
        """Create monitor with low alert thresholds for testing"""
        monitor = object.__new__(PerformanceMonitor)
        monitor._initialized = False
        
        with patch.object(PerformanceMonitor, 'data_dir', temp_dir):
            monitor.__init__()
        
        # Set low thresholds for testing
        monitor.alert_threshold_duration = 0.01  # 10ms
        monitor.alert_threshold_error_rate = 0.5  # 50%
        
        return monitor
    
    def test_duration_alert(self, monitor_with_low_thresholds):
        """Test alert for slow operation duration"""
        monitor = monitor_with_low_thresholds
        
        # Record a slow operation
        slow_metrics = OperationMetrics(
            function_name="slow_func",
            module_name="test_module",
            start_time=time.time(),
            end_time=time.time() + 0.1,  # 100ms - should trigger alert
            duration=0.1,
            success=True
        )
        
        with patch.object(monitor, '_maybe_send_alert') as mock_alert:
            monitor.record_operation(slow_metrics)
            mock_alert.assert_called_once()
            
            # Check alert message
            call_args = mock_alert.call_args
            assert "Slow operation detected" in call_args[0][1]
    
    def test_error_rate_alert(self, monitor_with_low_thresholds):
        """Test alert for high error rate"""
        monitor = monitor_with_low_thresholds
        
        # Record several failed operations
        base_time = time.time()
        for i in range(6):
            metrics = OperationMetrics(
                function_name="error_prone_func",
                module_name="test_module",
                start_time=base_time + i,
                end_time=base_time + i + 0.001,
                duration=0.001,
                success=i < 2  # Only first 2 succeed, rest fail
            )
            monitor.record_operation(metrics)
        
        # The last operation should trigger error rate alert
        with patch.object(monitor.logger, 'warning') as mock_warning:
            # We need to check if warning was called with error rate message
            warning_calls = [call for call in mock_warning.call_args_list 
                           if "High error rate detected" in str(call)]
            assert len(warning_calls) > 0


class TestMonitorSingleton:
    """Test the singleton behavior of PerformanceMonitor"""
    
    def test_singleton_behavior(self):
        """Test that get_monitor always returns the same instance"""
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2
        assert id(monitor1) == id(monitor2)
    
    def test_thread_safety(self):
        """Test thread safety of singleton creation"""
        import threading
        
        instances = []
        
        def create_monitor():
            instances.append(get_monitor())
        
        # Create multiple threads that try to get monitor instance
        threads = [threading.Thread(target=create_monitor) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance


# Integration test to verify end-to-end functionality
class TestPerformanceMonitoringIntegration:
    """Integration tests for the complete monitoring system"""
    
    def test_end_to_end_monitoring(self):
        """Test complete monitoring workflow"""
        
        # Define a test function with monitoring
        @monitor_performance(track_memory=False, custom_name="integration_test")
        def test_workflow(iterations: int):
            total = 0
            for i in range(iterations):
                total += i
                time.sleep(0.001)  # Small delay
            return total
        
        # Execute function multiple times
        results = []
        for i in range(5):
            result = test_workflow(10 + i)
            results.append(result)
        
        # Verify all executions completed
        assert len(results) == 5
        
        # Get monitoring data
        monitor = get_monitor()
        
        # Verify operations were recorded
        integration_ops = [
            op for op in monitor.operations 
            if op.function_name == "integration_test"
        ]
        
        assert len(integration_ops) == 5
        
        # Verify all operations succeeded
        for op in integration_ops:
            assert op.success is True
            assert op.duration > 0
        
        # Get aggregated metrics
        agg_metrics = monitor.get_function_metrics("integration_test")
        
        assert agg_metrics is not None
        assert agg_metrics.total_calls == 5
        assert agg_metrics.successful_calls == 5
        assert agg_metrics.failed_calls == 0
        assert agg_metrics.success_rate == 1.0
        
        # Generate performance report
        report = monitor.get_performance_report(1)
        
        assert 'overall_stats' in report
        assert report['overall_stats']['total_operations'] >= 5
        assert 'integration_test' in str(report['function_breakdown'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])