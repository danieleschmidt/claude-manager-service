#!/usr/bin/env python3
"""
Resilience Patterns for Autonomous SDLC System

Implements advanced resilience patterns for fault tolerance and self-healing:
- Circuit Breaker Pattern
- Retry with Exponential Backoff
- Bulkhead Pattern
- Timeout Management
- Health Checks
- Graceful Degradation
"""

import asyncio
import time
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from functools import wraps

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    name: str = "default"

@dataclass
class RetryConfig:
    """Configuration for retry policy"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

@dataclass
class TimeoutConfig:
    """Configuration for timeout management"""
    operation_timeout: float = 30.0
    total_timeout: float = 300.0
    connect_timeout: float = 10.0

class CircuitBreaker:
    """Advanced circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self.logger = logging.getLogger(f"circuit_breaker.{config.name}")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self._execute(func, *args, **kwargs)
        return wrapper
    
    def _execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker logic"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.config.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
        self.failure_count = 0
        self.success_count += 1
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.config.name} opened after {self.failure_count} failures")

class RetryPolicy:
    """Advanced retry policy with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger("retry_policy")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply retry policy"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self._execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def _execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_attempts} attempts failed")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add random jitter (±25% of calculated delay)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)

class BulkheadPattern:
    """Bulkhead pattern for resource isolation"""
    
    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.logger = logging.getLogger(f"bulkhead.{name}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.semaphore.acquire()
        self.active_requests += 1
        self.total_requests += 1
        self.logger.debug(f"Acquired slot in bulkhead {self.name} ({self.active_requests} active)")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.active_requests -= 1
        if exc_type is not None:
            self.failed_requests += 1
        self.semaphore.release()
        self.logger.debug(f"Released slot in bulkhead {self.name} ({self.active_requests} active)")
    
    @property
    def utilization(self) -> float:
        """Get current utilization percentage"""
        max_concurrent = self.semaphore._initial_value
        return (self.active_requests / max_concurrent) * 100
    
    @property
    def success_rate(self) -> float:
        """Get success rate percentage"""
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.failed_requests) / self.total_requests) * 100

class TimeoutManager:
    """Advanced timeout management"""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.logger = logging.getLogger("timeout_manager")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply timeout"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.operation_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Operation timed out after {self.config.operation_timeout}s")
                raise TimeoutError(f"Operation exceeded timeout of {self.config.operation_timeout}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # For synchronous functions, we can't easily add timeout
            # This would require more complex implementation with threading
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

class HealthChecker:
    """Advanced health checking system"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.statuses: Dict[str, HealthStatus] = {}
        self.last_check_times: Dict[str, datetime] = {}
        self.logger = logging.getLogger("health_checker")
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check"""
        self.checks[name] = check_func
        self.statuses[name] = HealthStatus.HEALTHY
        self.logger.info(f"Registered health check: {name}")
    
    async def run_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await self._run_single_check(check_func)
                duration = time.time() - start_time
                
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                self.statuses[name] = status
                self.last_check_times[name] = datetime.now()
                results[name] = status
                
                self.logger.debug(f"Health check {name}: {status.value} (took {duration:.3f}s)")
                
            except Exception as e:
                self.logger.error(f"Health check {name} failed with exception: {e}")
                self.statuses[name] = HealthStatus.UNHEALTHY
                results[name] = HealthStatus.UNHEALTHY
        
        return results
    
    async def _run_single_check(self, check_func: Callable[[], bool]) -> bool:
        """Run a single health check with timeout"""
        if asyncio.iscoroutinefunction(check_func):
            return await asyncio.wait_for(check_func(), timeout=10.0)
        else:
            return check_func()
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health"""
        if not self.statuses:
            return HealthStatus.HEALTHY
        
        unhealthy_count = sum(1 for status in self.statuses.values() if status == HealthStatus.UNHEALTHY)
        total_count = len(self.statuses)
        
        if unhealthy_count == 0:
            return HealthStatus.HEALTHY
        elif unhealthy_count < total_count * 0.5:  # Less than 50% unhealthy
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

class GracefulDegradation:
    """Graceful degradation manager"""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.feature_flags: Dict[str, bool] = {}
        self.logger = logging.getLogger("graceful_degradation")
    
    def register_fallback(self, operation: str, fallback_handler: Callable):
        """Register a fallback handler for an operation"""
        self.fallback_handlers[operation] = fallback_handler
        self.logger.info(f"Registered fallback for operation: {operation}")
    
    def set_feature_flag(self, feature: str, enabled: bool):
        """Set a feature flag"""
        self.feature_flags[feature] = enabled
        self.logger.info(f"Feature flag {feature} set to {enabled}")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.feature_flags.get(feature, True)  # Default to enabled
    
    def execute_with_fallback(self, operation: str, primary_func: Callable, *args, **kwargs):
        """Execute operation with fallback"""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary operation {operation} failed: {e}")
            
            if operation in self.fallback_handlers:
                self.logger.info(f"Executing fallback for operation: {operation}")
                try:
                    return self.fallback_handlers[operation](*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback for {operation} also failed: {fallback_error}")
                    raise fallback_error
            else:
                self.logger.error(f"No fallback handler registered for operation: {operation}")
                raise e

# Custom Exceptions
class ResilienceError(Exception):
    """Base exception for resilience patterns"""
    pass

class CircuitBreakerOpenError(ResilienceError):
    """Raised when circuit breaker is open"""
    pass

class BulkheadCapacityExceededError(ResilienceError):
    """Raised when bulkhead capacity is exceeded"""
    pass

# Factory functions for easy configuration
def create_circuit_breaker(name: str, failure_threshold: int = 5, recovery_timeout: int = 60) -> CircuitBreaker:
    """Create a circuit breaker with common configuration"""
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    return CircuitBreaker(config)

def create_retry_policy(max_attempts: int = 3, base_delay: float = 1.0) -> RetryPolicy:
    """Create a retry policy with common configuration"""
    config = RetryConfig(max_attempts=max_attempts, base_delay=base_delay)
    return RetryPolicy(config)

def create_resilient_function(func: Callable, 
                             use_circuit_breaker: bool = True,
                             use_retry: bool = True,
                             use_timeout: bool = True,
                             name: str = None) -> Callable:
    """Create a resilient version of a function with all patterns applied"""
    if name is None:
        name = func.__name__
    
    resilient_func = func
    
    if use_timeout:
        timeout_config = TimeoutConfig()
        resilient_func = TimeoutManager(timeout_config)(resilient_func)
    
    if use_retry:
        retry_policy = create_retry_policy()
        resilient_func = retry_policy(resilient_func)
    
    if use_circuit_breaker:
        circuit_breaker = create_circuit_breaker(name)
        resilient_func = circuit_breaker(resilient_func)
    
    return resilient_func

# Example usage and testing functions
async def example_usage():
    """Example of how to use resilience patterns"""
    
    # Create a health checker
    health_checker = HealthChecker()
    
    # Register some health checks
    health_checker.register_check("database", lambda: True)  # Simulate healthy DB
    health_checker.register_check("api", lambda: False)     # Simulate unhealthy API
    
    # Run health checks
    health_results = await health_checker.run_checks()
    overall_health = health_checker.get_overall_health()
    
    print(f"Health results: {health_results}")
    print(f"Overall health: {overall_health}")
    
    # Create a resilient function
    @create_circuit_breaker("example_operation")
    @create_retry_policy()
    def unreliable_operation():
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("Random failure")
        return "Success"
    
    # Use bulkhead pattern
    async with BulkheadPattern("api_calls", max_concurrent=5):
        try:
            result = unreliable_operation()
            print(f"Operation result: {result}")
        except Exception as e:
            print(f"Operation failed: {e}")

if __name__ == "__main__":
    asyncio.run(example_usage())