# Environment Variable Configuration

This document describes all configurable environment variables for the Claude Manager Service. These variables allow you to customize behavior without modifying code.

## Performance Monitoring Configuration

### `PERF_MAX_OPERATIONS`
- **Description**: Maximum number of operations to keep in memory for performance tracking
- **Type**: Integer
- **Default**: `10000`
- **Range**: 1000 - 100000
- **Impact**: Higher values use more memory but provide more detailed performance history

### `PERF_RETENTION_DAYS`
- **Description**: Number of days to retain performance data on disk
- **Type**: Integer
- **Default**: `30`
- **Range**: 1 - 365
- **Impact**: Affects disk usage and historical analysis capabilities

### `PERF_ALERT_DURATION`
- **Description**: Duration threshold in seconds for performance alerts
- **Type**: Float
- **Default**: `30.0`
- **Range**: 0.1 - 3600.0
- **Impact**: Operations taking longer than this will trigger performance alerts

### `PERF_ALERT_ERROR_RATE`
- **Description**: Error rate threshold (0.0-1.0) for performance alerts
- **Type**: Float
- **Default**: `0.1` (10%)
- **Range**: 0.0 - 1.0
- **Impact**: Error rates above this threshold will trigger alerts

## Rate Limiting Configuration

### `RATE_LIMIT_MAX_REQUESTS`
- **Description**: Maximum number of API requests allowed per time window
- **Type**: Integer
- **Default**: `5000`
- **Range**: 100 - 50000
- **Impact**: Controls API rate limiting to prevent hitting GitHub API limits

### `RATE_LIMIT_TIME_WINDOW`
- **Description**: Time window in seconds for rate limiting
- **Type**: Float
- **Default**: `3600.0` (1 hour)
- **Range**: 60.0 - 86400.0 (1 minute - 24 hours)
- **Impact**: Larger windows allow for bursts but less fine-grained control

## Security Configuration

### `SECURITY_MAX_CONTENT_LENGTH`
- **Description**: Maximum length in characters for basic content sanitization
- **Type**: Integer
- **Default**: `50000` (~50KB)
- **Range**: 1000 - 100000
- **Impact**: Prevents DoS attacks via large content, but may truncate legitimate large content

### `SECURITY_ENHANCED_MAX_CONTENT_LENGTH`
- **Description**: Maximum length in characters for enhanced content sanitization
- **Type**: Integer
- **Default**: `60000` (~60KB)
- **Range**: 1000 - 100000
- **Impact**: Should be larger than `SECURITY_MAX_CONTENT_LENGTH` for enhanced processing

## Logging Configuration

### `LOG_LEVEL`
- **Description**: Logging level for the application
- **Type**: String
- **Default**: `INFO`
- **Allowed Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Impact**: Controls verbosity of logs

## Feature Flags

### `ENABLE_PERFORMANCE_MONITORING`
- **Description**: Enable/disable performance monitoring system
- **Type**: Boolean
- **Default**: `true`
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Impact**: Disabling saves CPU and memory but removes performance insights

### `ENABLE_RATE_LIMITING`
- **Description**: Enable/disable rate limiting for API operations
- **Type**: Boolean
- **Default**: `true`
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Impact**: Disabling may lead to API rate limit violations

### `ENABLE_ENHANCED_SECURITY`
- **Description**: Enable/disable enhanced security features
- **Type**: Boolean
- **Default**: `true`
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Impact**: Disabling reduces security but may improve performance

## Example Configuration

### Development Environment
```bash
# Performance - More lenient for development
export PERF_MAX_OPERATIONS=5000
export PERF_RETENTION_DAYS=7
export PERF_ALERT_DURATION=60.0
export PERF_ALERT_ERROR_RATE=0.2

# Rate limiting - More restrictive for testing
export RATE_LIMIT_MAX_REQUESTS=1000
export RATE_LIMIT_TIME_WINDOW=1800.0

# Security - Standard settings
export SECURITY_MAX_CONTENT_LENGTH=25000
export SECURITY_ENHANCED_MAX_CONTENT_LENGTH=30000

# Logging - Verbose for debugging
export LOG_LEVEL=DEBUG

# Features - All enabled for testing
export ENABLE_PERFORMANCE_MONITORING=true
export ENABLE_RATE_LIMITING=true
export ENABLE_ENHANCED_SECURITY=true
```

### Production Environment
```bash
# Performance - Optimized for production load
export PERF_MAX_OPERATIONS=20000
export PERF_RETENTION_DAYS=90
export PERF_ALERT_DURATION=15.0
export PERF_ALERT_ERROR_RATE=0.05

# Rate limiting - Aligned with GitHub API limits
export RATE_LIMIT_MAX_REQUESTS=4500
export RATE_LIMIT_TIME_WINDOW=3600.0

# Security - Strict settings
export SECURITY_MAX_CONTENT_LENGTH=40000
export SECURITY_ENHANCED_MAX_CONTENT_LENGTH=50000

# Logging - Standard production level
export LOG_LEVEL=INFO

# Features - All enabled for production
export ENABLE_PERFORMANCE_MONITORING=true
export ENABLE_RATE_LIMITING=true
export ENABLE_ENHANCED_SECURITY=true
```

### High-Performance Environment
```bash
# Performance - Maximum capacity
export PERF_MAX_OPERATIONS=50000
export PERF_RETENTION_DAYS=180
export PERF_ALERT_DURATION=10.0
export PERF_ALERT_ERROR_RATE=0.02

# Rate limiting - Higher limits for high-throughput
export RATE_LIMIT_MAX_REQUESTS=10000
export RATE_LIMIT_TIME_WINDOW=3600.0

# Security - Balanced settings
export SECURITY_MAX_CONTENT_LENGTH=75000
export SECURITY_ENHANCED_MAX_CONTENT_LENGTH=85000

# Logging - Minimal for performance
export LOG_LEVEL=WARNING

# Features - Selective enabling
export ENABLE_PERFORMANCE_MONITORING=true
export ENABLE_RATE_LIMITING=true
export ENABLE_ENHANCED_SECURITY=false  # Disabled for max performance
```

## Configuration Validation

The system automatically validates all environment variables on startup:

1. **Type Validation**: Ensures values can be converted to the expected type
2. **Range Validation**: Ensures values fall within acceptable ranges
3. **Dependency Validation**: Ensures related configurations are consistent
4. **Runtime Validation**: Performs additional checks for operational safety

### Error Handling

If configuration validation fails:
- The application will log detailed error messages
- Invalid values will cause startup to fail with exit code 1
- Suggestions for valid ranges will be provided in error messages

### Configuration Testing

Test your configuration using:

```bash
# Test configuration loading
python3 src/config_env.py

# Test with custom values
export PERF_MAX_OPERATIONS=15000
export LOG_LEVEL=DEBUG
python3 src/config_env.py
```

## Migration from Hardcoded Values

Previous hardcoded values have been replaced with environment variables:

| Old Hardcoded Value | New Environment Variable |
|---------------------|---------------------------|
| `max_operations_in_memory = 10000` | `PERF_MAX_OPERATIONS` |
| `retention_days = 30` | `PERF_RETENTION_DAYS` |
| `alert_threshold_duration = 30.0` | `PERF_ALERT_DURATION` |
| `alert_threshold_error_rate = 0.1` | `PERF_ALERT_ERROR_RATE` |
| `max_requests = 5000` | `RATE_LIMIT_MAX_REQUESTS` |
| `time_window = 3600.0` | `RATE_LIMIT_TIME_WINDOW` |
| `max_length = 50000` | `SECURITY_MAX_CONTENT_LENGTH` |
| `max_length = 60000` | `SECURITY_ENHANCED_MAX_CONTENT_LENGTH` |

All changes are backward compatible - if environment variables are not set, the original default values will be used.