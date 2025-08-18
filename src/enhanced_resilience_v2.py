#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - ENHANCED RESILIENCE PATTERNS V2
Advanced resilience patterns with ML-based failure prediction and adaptive recovery
"""

import asyncio
import time
import json
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import structlog
import threading
from enum import Enum

logger = structlog.get_logger("EnhancedResilienceV2")

T = TypeVar('T')

class FailurePattern(Enum):
    """Types of failure patterns for ML analysis"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit" 
    NETWORK_ERROR = "network_error"
    AUTHENTICATION = "authentication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UPSTREAM_FAILURE = "upstream_failure"
    DATA_CORRUPTION = "data_corruption"
    UNKNOWN = "unknown"

@dataclass
class FailureRecord:
    """Record of a failure for pattern analysis"""
    timestamp: datetime
    pattern: FailurePattern
    context: Dict[str, Any]
    severity: int  # 1-10
    recovery_time: Optional[float] = None
    recovery_successful: bool = False

@dataclass
class ResilienceMetrics:
    """Comprehensive resilience metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_trips: int = 0
    average_response_time: float = 0.0
    failure_patterns: Dict[str, int] = None
    recovery_success_rate: float = 0.0
    adaptive_threshold_adjustments: int = 0
    
    def __post_init__(self):
        if self.failure_patterns is None:
            self.failure_patterns = {}

class MLFailurePredictor:
    """ML-based failure prediction and pattern analysis"""
    
    def __init__(self):
        self.failure_history: List[FailureRecord] = []
        self.pattern_weights: Dict[FailurePattern, float] = {
            pattern: 1.0 for pattern in FailurePattern
        }
        self.prediction_accuracy = 0.0
        
    def record_failure(self, failure: FailureRecord):
        """Record a failure for pattern analysis"""
        self.failure_history.append(failure)
        self._update_pattern_weights()
        
        # Keep only recent failures (last 1000)
        if len(self.failure_history) > 1000:
            self.failure_history = self.failure_history[-1000:]
    
    def predict_failure_probability(self, context: Dict[str, Any]) -> float:
        """Predict probability of failure based on context"""
        if not self.failure_history:
            return 0.1  # Base probability
        
        # Analyze recent failure patterns
        recent_failures = [f for f in self.failure_history 
                          if (datetime.now(timezone.utc) - f.timestamp).total_seconds() < 3600]
        
        if not recent_failures:
            return 0.1
        
        # Calculate failure rate
        recent_count = len(recent_failures)
        failure_rate = min(recent_count / 100, 0.9)  # Cap at 90%
        
        # Adjust based on context similarity
        context_similarity = self._calculate_context_similarity(context, recent_failures)
        
        return min(failure_rate * context_similarity, 0.95)
    
    def _update_pattern_weights(self):
        """Update pattern weights based on success rates"""
        if len(self.failure_history) < 10:
            return
            
        pattern_stats = {}
        for failure in self.failure_history[-100:]:  # Recent failures
            pattern = failure.pattern
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {'total': 0, 'recovered': 0}
            pattern_stats[pattern]['total'] += 1
            if failure.recovery_successful:
                pattern_stats[pattern]['recovered'] += 1
        
        # Update weights based on recovery success rates
        for pattern, stats in pattern_stats.items():
            recovery_rate = stats['recovered'] / stats['total']
            self.pattern_weights[pattern] = 2.0 - recovery_rate  # Higher weight for harder-to-recover patterns
    
    def _calculate_context_similarity(self, context: Dict[str, Any], failures: List[FailureRecord]) -> float:
        """Calculate similarity between current context and failure contexts"""
        if not failures:
            return 1.0
        
        similarities = []
        for failure in failures:
            similarity = 0.0
            common_keys = set(context.keys()) & set(failure.context.keys())
            
            if not common_keys:
                similarities.append(0.5)  # Neutral similarity
                continue
            
            for key in common_keys:
                if context[key] == failure.context[key]:
                    similarity += 1.0 / len(common_keys)
            
            similarities.append(similarity)
        
        return statistics.mean(similarities)

class AdaptiveCircuitBreaker:
    """Circuit breaker with ML-based threshold adaptation"""
    
    def __init__(self, name: str, initial_failure_threshold: int = 5, 
                 initial_timeout: float = 60.0, predictor: Optional[MLFailurePredictor] = None):
        self.name = name
        self.failure_threshold = initial_failure_threshold
        self.timeout = initial_timeout
        self.predictor = predictor or MLFailurePredictor()
        
        # State management
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.metrics = ResilienceMetrics()
        
        # Adaptive thresholds
        self.adaptive_enabled = True
        self.threshold_adjustment_factor = 0.1
        self.min_threshold = 2
        self.max_threshold = 20
        
        self._lock = threading.Lock()
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        self.metrics.total_requests += 1
        
        # Check if circuit should be opened based on ML prediction
        if self.adaptive_enabled:
            await self._adaptive_state_check(*args, **kwargs)
        
        if self.state == "OPEN":
            await self._check_timeout()
            if self.state == "OPEN":
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success(time.time() - start_time)
            return result
        except Exception as e:
            await self._on_failure(e, *args, **kwargs)
            raise
    
    async def _adaptive_state_check(self, *args, **kwargs):
        """Adaptive state checking based on ML predictions"""
        context = {
            'function_args': str(args)[:100],  # Truncate for storage
            'current_time': datetime.now(timezone.utc).hour,
            'recent_failure_rate': self.failure_count / max(self.metrics.total_requests, 1)
        }
        
        failure_probability = self.predictor.predict_failure_probability(context)
        
        # Adjust threshold based on predicted failure probability
        if failure_probability > 0.7:  # High failure probability
            self.failure_threshold = max(self.min_threshold, 
                                       int(self.failure_threshold * (1 - self.threshold_adjustment_factor)))
            self.metrics.adaptive_threshold_adjustments += 1
        elif failure_probability < 0.3:  # Low failure probability
            self.failure_threshold = min(self.max_threshold,
                                       int(self.failure_threshold * (1 + self.threshold_adjustment_factor)))
    
    async def _on_success(self, response_time: float):
        """Handle successful call"""
        with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.successful_requests - 1) + response_time) / 
                self.metrics.successful_requests
            )
            
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= 3:  # Configurable threshold
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after recovery")
            elif self.state == "CLOSED":
                self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self, exception: Exception, *args, **kwargs):
        """Handle failed call"""
        with self._lock:
            self.metrics.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Classify failure pattern
            pattern = self._classify_failure_pattern(exception)
            self.metrics.failure_patterns[pattern.value] = (
                self.metrics.failure_patterns.get(pattern.value, 0) + 1
            )
            
            # Record failure for ML analysis
            failure_record = FailureRecord(
                timestamp=datetime.now(timezone.utc),
                pattern=pattern,
                context={
                    'exception_type': type(exception).__name__,
                    'exception_message': str(exception)[:200],
                    'function_args': str(args)[:100]
                },
                severity=self._calculate_failure_severity(exception, pattern)
            )
            self.predictor.record_failure(failure_record)
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.metrics.circuit_breaker_trips += 1
                logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def _classify_failure_pattern(self, exception: Exception) -> FailurePattern:
        """Classify the type of failure"""
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if 'timeout' in exception_str or 'timeout' in exception_type:
            return FailurePattern.TIMEOUT
        elif 'rate limit' in exception_str or '429' in exception_str:
            return FailurePattern.RATE_LIMIT
        elif 'network' in exception_str or 'connection' in exception_str:
            return FailurePattern.NETWORK_ERROR
        elif 'auth' in exception_str or '401' in exception_str or '403' in exception_str:
            return FailurePattern.AUTHENTICATION
        elif 'memory' in exception_str or 'resource' in exception_str:
            return FailurePattern.RESOURCE_EXHAUSTION
        elif '5' in exception_str and ('00' in exception_str or '02' in exception_str or '03' in exception_str):
            return FailurePattern.UPSTREAM_FAILURE
        else:
            return FailurePattern.UNKNOWN
    
    def _calculate_failure_severity(self, exception: Exception, pattern: FailurePattern) -> int:
        """Calculate severity of failure (1-10)"""
        base_severity = 5
        
        # Adjust based on pattern
        pattern_severity = {
            FailurePattern.TIMEOUT: 6,
            FailurePattern.RATE_LIMIT: 4,
            FailurePattern.NETWORK_ERROR: 7,
            FailurePattern.AUTHENTICATION: 8,
            FailurePattern.RESOURCE_EXHAUSTION: 9,
            FailurePattern.UPSTREAM_FAILURE: 6,
            FailurePattern.DATA_CORRUPTION: 10,
            FailurePattern.UNKNOWN: 5
        }
        
        return pattern_severity.get(pattern, base_severity)
    
    async def _check_timeout(self):
        """Check if circuit breaker timeout has passed"""
        if self.last_failure_time and time.time() - self.last_failure_time >= self.timeout:
            self.state = "HALF_OPEN"
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN state")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class IntelligentRetryStrategy:
    """Intelligent retry with adaptive backoff and failure pattern analysis"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, predictor: Optional[MLFailurePredictor] = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.predictor = predictor or MLFailurePredictor()
        
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with intelligent retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                return result
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    break
                
                # Calculate adaptive delay based on failure pattern
                delay = await self._calculate_adaptive_delay(e, attempt)
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        # Record failure for pattern analysis
        pattern = self._classify_failure_pattern(last_exception)
        failure_record = FailureRecord(
            timestamp=datetime.now(timezone.utc),
            pattern=pattern,
            context={'attempts': self.max_attempts, 'final_exception': str(last_exception)},
            severity=5,
            recovery_successful=False
        )
        self.predictor.record_failure(failure_record)
        
        raise last_exception
    
    async def _calculate_adaptive_delay(self, exception: Exception, attempt: int) -> float:
        """Calculate adaptive delay based on failure pattern and ML prediction"""
        pattern = self._classify_failure_pattern(exception)
        
        # Base exponential backoff
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        # Adjust based on failure pattern
        pattern_multipliers = {
            FailurePattern.RATE_LIMIT: 2.0,  # Longer delays for rate limits
            FailurePattern.NETWORK_ERROR: 1.5,
            FailurePattern.TIMEOUT: 1.2,
            FailurePattern.AUTHENTICATION: 0.5,  # Shorter delays, likely won't recover
            FailurePattern.UPSTREAM_FAILURE: 1.8,
            FailurePattern.RESOURCE_EXHAUSTION: 2.5,
        }
        
        multiplier = pattern_multipliers.get(pattern, 1.0)
        delay *= multiplier
        
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0.1, 0.3)
        delay += delay * jitter
        
        return min(delay, self.max_delay)
    
    def _classify_failure_pattern(self, exception: Exception) -> FailurePattern:
        """Classify the type of failure (same as circuit breaker)"""
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if 'timeout' in exception_str or 'timeout' in exception_type:
            return FailurePattern.TIMEOUT
        elif 'rate limit' in exception_str or '429' in exception_str:
            return FailurePattern.RATE_LIMIT
        elif 'network' in exception_str or 'connection' in exception_str:
            return FailurePattern.NETWORK_ERROR
        elif 'auth' in exception_str or '401' in exception_str or '403' in exception_str:
            return FailurePattern.AUTHENTICATION
        elif 'memory' in exception_str or 'resource' in exception_str:
            return FailurePattern.RESOURCE_EXHAUSTION
        elif '5' in exception_str and ('00' in exception_str or '02' in exception_str or '03' in exception_str):
            return FailurePattern.UPSTREAM_FAILURE
        else:
            return FailurePattern.UNKNOWN

class ResilienceOrchestrator:
    """Orchestrates all resilience patterns with ML-based coordination"""
    
    def __init__(self):
        self.predictor = MLFailurePredictor()
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.retry_strategies: Dict[str, IntelligentRetryStrategy] = {}
        self.global_metrics = ResilienceMetrics()
        
    def get_circuit_breaker(self, name: str, **kwargs) -> AdaptiveCircuitBreaker:
        """Get or create circuit breaker for a service"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = AdaptiveCircuitBreaker(
                name=name, predictor=self.predictor, **kwargs
            )
        return self.circuit_breakers[name]
    
    def get_retry_strategy(self, name: str, **kwargs) -> IntelligentRetryStrategy:
        """Get or create retry strategy for a service"""
        if name not in self.retry_strategies:
            self.retry_strategies[name] = IntelligentRetryStrategy(
                predictor=self.predictor, **kwargs
            )
        return self.retry_strategies[name]
    
    async def execute_with_resilience(self, service_name: str, func: Callable[..., T], 
                                    *args, **kwargs) -> T:
        """Execute function with full resilience patterns"""
        circuit_breaker = self.get_circuit_breaker(service_name)
        retry_strategy = self.get_retry_strategy(service_name)
        
        async def resilient_execution():
            return await retry_strategy.execute(func, *args, **kwargs)
        
        return await circuit_breaker.call(resilient_execution)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        metrics = {
            'global_metrics': asdict(self.global_metrics),
            'circuit_breakers': {},
            'failure_prediction': {
                'total_failures_analyzed': len(self.predictor.failure_history),
                'pattern_weights': {p.value: w for p, w in self.predictor.pattern_weights.items()},
                'prediction_accuracy': self.predictor.prediction_accuracy
            }
        }
        
        for name, cb in self.circuit_breakers.items():
            metrics['circuit_breakers'][name] = {
                'state': cb.state,
                'failure_count': cb.failure_count,
                'metrics': asdict(cb.metrics)
            }
        
        return metrics

# Decorator for easy resilience application
def with_resilience(service_name: str, orchestrator: Optional[ResilienceOrchestrator] = None):
    """Decorator to apply resilience patterns to any function"""
    if orchestrator is None:
        orchestrator = ResilienceOrchestrator()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            return await orchestrator.execute_with_resilience(service_name, func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global orchestrator instance
global_orchestrator = ResilienceOrchestrator()

# Example usage and testing functions
async def example_unstable_service(fail_probability: float = 0.3) -> str:
    """Example service that randomly fails"""
    import random
    if random.random() < fail_probability:
        raise Exception(f"Service failed with probability {fail_probability}")
    return f"Service succeeded at {datetime.now()}"

@with_resilience("example_service")
async def resilient_service_call() -> str:
    """Example of a service call with resilience patterns applied"""
    return await example_unstable_service(0.4)

# Test function
async def test_enhanced_resilience():
    """Test the enhanced resilience patterns"""
    logger.info("Testing Enhanced Resilience Patterns V2")
    
    orchestrator = ResilienceOrchestrator()
    
    # Test multiple calls to build ML patterns
    results = []
    for i in range(10):
        try:
            result = await orchestrator.execute_with_resilience(
                "test_service", 
                example_unstable_service,
                fail_probability=0.3
            )
            results.append(f"Success: {result}")
        except Exception as e:
            results.append(f"Failed: {e}")
        
        await asyncio.sleep(0.1)  # Brief delay between calls
    
    # Display results and metrics
    print("\nTest Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}: {result}")
    
    print("\nResilience Metrics:")
    metrics = orchestrator.get_global_metrics()
    print(json.dumps(metrics, indent=2, default=str))
    
    return True

if __name__ == "__main__":
    asyncio.run(test_enhanced_resilience())