# Performance Monitoring System

The Claude Manager Service includes a comprehensive performance monitoring system that tracks execution times, API call statistics, memory usage, and provides detailed analytics for system optimization.

## Features

### Real-time Metrics Collection
- **Function Execution Times**: Track duration of every monitored function call
- **Success/Failure Rates**: Monitor operation success rates and error patterns
- **Memory Usage**: Optional memory tracking for functions with `track_memory=True`
- **API Call Statistics**: Specialized tracking for GitHub API operations
- **Automated Alerting**: Performance degradation and high error rate alerts

### Data Management
- **Configurable Retention**: Automatic cleanup of old data (default: 30 days)
- **JSON Persistence**: Save and load performance metrics to/from disk
- **In-Memory Optimization**: Configurable operation limit for memory efficiency
- **Thread-Safe Operations**: Safe for concurrent usage

### Analytics and Reporting
- **Aggregated Statistics**: Min/max/average/median/percentile calculations
- **Performance Reports**: Comprehensive reports for any time period
- **Function-Specific Metrics**: Detailed analysis for individual functions
- **Trend Analysis**: Historical performance tracking

## Usage

### Basic Monitoring

```python
from performance_monitor import monitor_performance

@monitor_performance(track_memory=True, custom_name="my_operation")
def my_function():
    # Your code here
    pass
```

### API Call Monitoring

```python
from performance_monitor import monitor_api_call

@monitor_api_call("github_create_issue")
def create_issue_function():
    # API call code
    pass
```

### Enhanced Log Performance (Backward Compatible)

```python
from performance_monitor import enhanced_log_performance

@enhanced_log_performance
def my_function():
    # This combines logging and monitoring
    pass
```

## CLI Reporting Tool

The `performance_report.py` script provides command-line access to performance data:

### Generate Performance Report
```bash
python3 performance_report.py report [hours]
```
Example:
```bash
python3 performance_report.py report 24  # Last 24 hours
python3 performance_report.py report 1   # Last 1 hour
```

### Quick Summary
```bash
python3 performance_report.py summary
```

### Function-Specific Analysis
```bash
python3 performance_report.py function github_create_issue
```

### API Call Statistics
```bash
python3 performance_report.py api-calls
```

### Save Current Metrics
```bash
python3 performance_report.py save
```

## Programmatic Access

### Get Monitor Instance
```python
from performance_monitor import get_monitor

monitor = get_monitor()
```

### Generate Performance Report
```python
# Get report for last 24 hours
report = monitor.get_performance_report(24)

print(f"Total operations: {report['overall_stats']['total_operations']}")
print(f"Success rate: {report['overall_stats']['success_rate']:.1%}")
```

### Get Function Metrics
```python
metrics = monitor.get_function_metrics("my_function", "my_module")

if metrics:
    print(f"Total calls: {metrics.total_calls}")
    print(f"Average duration: {metrics.average_duration:.3f}s")
    print(f"Success rate: {metrics.success_rate:.1%}")
```

### Get API Call Summary
```python
api_summary = monitor.get_api_call_summary()

for api_name, stats in api_summary.items():
    print(f"{api_name}: {stats['success_rate']:.1%} success rate")
```

## Configuration

The monitoring system can be configured by modifying these attributes on the monitor instance:

```python
monitor = get_monitor()

# Retention period (days)
monitor.retention_days = 7

# Maximum operations in memory
monitor.max_operations_in_memory = 5000

# Alert thresholds
monitor.alert_threshold_duration = 10.0  # seconds
monitor.alert_threshold_error_rate = 0.2  # 20%
```

## Data Structure

### OperationMetrics
- `function_name`: Name of the monitored function
- `module_name`: Module containing the function
- `start_time`/`end_time`: Execution timestamps
- `duration`: Execution time in seconds
- `success`: Whether operation completed successfully
- `error_message`: Error details if operation failed
- `memory_before`/`memory_after`/`memory_delta`: Memory usage (if tracked)
- `args_count`/`kwargs_count`: Number of function arguments

### AggregateMetrics
- Statistics aggregated over multiple operations
- Success rates, duration percentiles, error patterns
- Memory usage averages, call frequency data

## Performance Impact

The monitoring system is designed to have minimal performance overhead:

- **Decorator Overhead**: ~1-5μs per function call
- **Memory Usage**: Configurable with automatic cleanup
- **Storage**: JSON files with configurable retention
- **Background Tasks**: Non-blocking cleanup operations

## Integration with Existing Code

The monitoring system integrates seamlessly with existing Claude Manager Service components:

- **GitHub API**: All API calls are automatically monitored
- **Task Analyzer**: TODO scanning and issue analysis operations tracked
- **Orchestrator**: Task orchestration with memory usage monitoring
- **Error Handler**: Failed operations automatically recorded

## Alerting

The system provides automated alerting for:

- **Slow Operations**: Functions exceeding duration thresholds
- **High Error Rates**: Functions with elevated failure rates
- **Memory Leaks**: Significant memory usage increases

Alerts are logged with cooldown periods to prevent spam.

## File Locations

- **Module**: `src/performance_monitor.py`
- **CLI Tool**: `performance_report.py`
- **Data Storage**: `performance_data/` directory
- **Unit Tests**: `tests/unit/test_performance_monitor.py`
- **Integration Tests**: `tests/integration/test_performance_monitoring_integration.py`

## Dependencies

- `psutil`: For memory usage tracking
- `json`: For data persistence
- `threading`: For thread-safe operations
- `collections.deque`: For efficient operation storage

---

This performance monitoring system provides comprehensive insights into the Claude Manager Service's operation, enabling data-driven optimization and proactive issue identification.