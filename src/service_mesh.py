"""
Service Mesh Architecture and High Availability System for Claude Manager Service Generation 3

This module provides enterprise-grade high availability including:
- Service mesh architecture with sidecar pattern
- Circuit breakers for external dependencies
- Failover mechanisms and redundancy
- Graceful shutdown and startup sequences
- Health-based service discovery
- Traffic routing and load balancing
- Fault tolerance and resilience patterns
- Distributed system coordination
"""

import asyncio
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable
import logging
import signal
import weakref

from .logger import get_logger
from .performance_monitor import monitor_performance, get_monitor
from .error_handler import ServiceMeshError, CircuitBreakerError, with_enhanced_error_handling


logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class TrafficPolicy(Enum):
    """Traffic routing policies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    FAILOVER = "failover"
    CANARY = "canary"


@dataclass
class ServiceEndpoint:
    """Service endpoint definition"""
    service_id: str
    host: str
    port: int
    protocol: str = "http"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclass
class HealthCheck:
    """Health check configuration"""
    endpoint: str
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 2
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_requests: int = 3
    success_threshold: int = 2


@dataclass
class ServiceInstance:
    """Service instance with health and circuit breaker state"""
    endpoint: ServiceEndpoint
    status: ServiceStatus
    circuit_state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    active_connections: int = 0
    
    @property
    def average_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    @property
    def is_available(self) -> bool:
        return self.status == ServiceStatus.HEALTHY and self.circuit_state != CircuitState.OPEN


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_requests = 0
        self.lock = asyncio.Lock()
        
        logger.info(f"CircuitBreaker '{name}' initialized")
    
    async def call(self, func: Callable[..., Awaitable], *args, **kwargs):
        """Execute function through circuit breaker"""
        async with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN", "circuit_breaker_open")
            
            elif self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_max_requests:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' half-open limit reached", "half_open_limit")
                self.half_open_requests += 1
        
        # Execute the function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            await self._on_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self._on_failure(e)
            raise
    
    async def _on_success(self, execution_time: float):
        """Handle successful execution"""
        async with self.lock:
            self.success_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' CLOSED after recovery")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, error: Exception):
        """Handle failed execution"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures")
            
            elif self.state == CircuitState.HALF_OPEN:
                # Return to OPEN on any failure during half-open
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' returned to OPEN from HALF_OPEN")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.timeout_seconds
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'half_open_requests': self.half_open_requests,
            'config': asdict(self.config)
        }


class ServiceRegistry:
    """Service discovery and registry"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.watchers: List[Callable] = []
        self.lock = asyncio.RLock()
        
        logger.info("ServiceRegistry initialized")
    
    async def register_service(self, 
                              service_name: str, 
                              endpoint: ServiceEndpoint,
                              health_check: Optional[HealthCheck] = None,
                              circuit_breaker_config: Optional[CircuitBreakerConfig] = None) -> str:
        """Register a service instance"""
        async with self.lock:
            instance = ServiceInstance(
                endpoint=endpoint,
                status=ServiceStatus.STARTING
            )
            
            self.services[service_name].append(instance)
            
            # Set up health check
            if health_check:
                self.health_checks[endpoint.service_id] = health_check
            
            # Set up circuit breaker
            if circuit_breaker_config:
                breaker_name = f"{service_name}-{endpoint.service_id}"
                self.circuit_breakers[breaker_name] = CircuitBreaker(breaker_name, circuit_breaker_config)
            
            logger.info(f"Registered service '{service_name}' instance: {endpoint.address}")
            
            # Notify watchers
            await self._notify_watchers('register', service_name, instance)
            
            return endpoint.service_id
    
    async def deregister_service(self, service_name: str, service_id: str):
        """Deregister a service instance"""
        async with self.lock:
            if service_name in self.services:
                instances = self.services[service_name]
                for i, instance in enumerate(instances):
                    if instance.endpoint.service_id == service_id:
                        instance.status = ServiceStatus.STOPPING
                        instances.pop(i)
                        
                        # Clean up health check and circuit breaker
                        self.health_checks.pop(service_id, None)
                        breaker_name = f"{service_name}-{service_id}"
                        self.circuit_breakers.pop(breaker_name, None)
                        
                        logger.info(f"Deregistered service '{service_name}' instance: {service_id}")
                        
                        # Notify watchers
                        await self._notify_watchers('deregister', service_name, instance)
                        break
    
    async def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances for a service"""
        async with self.lock:
            if service_name not in self.services:
                return []
            
            return [
                instance for instance in self.services[service_name]
                if instance.is_available
            ]
    
    async def update_instance_status(self, service_name: str, service_id: str, status: ServiceStatus):
        """Update service instance status"""
        async with self.lock:
            if service_name in self.services:
                for instance in self.services[service_name]:
                    if instance.endpoint.service_id == service_id:
                        old_status = instance.status
                        instance.status = status
                        instance.last_health_check = datetime.now()
                        
                        if old_status != status:
                            logger.info(f"Service '{service_name}' instance {service_id} status: {old_status.value} -> {status.value}")
                            await self._notify_watchers('status_change', service_name, instance)
                        break
    
    async def record_request_result(self, service_name: str, service_id: str, success: bool, response_time: float):
        """Record request result for circuit breaker"""
        async with self.lock:
            if service_name in self.services:
                for instance in self.services[service_name]:
                    if instance.endpoint.service_id == service_id:
                        instance.response_times.append(response_time)
                        
                        if success:
                            instance.success_count += 1
                            instance.last_success = datetime.now()
                            instance.failure_count = max(0, instance.failure_count - 1)
                        else:
                            instance.failure_count += 1
                            instance.last_failure = datetime.now()
                        
                        # Update circuit breaker
                        breaker_name = f"{service_name}-{service_id}"
                        if breaker_name in self.circuit_breakers:
                            breaker = self.circuit_breakers[breaker_name]
                            if success:
                                await breaker._on_success(response_time)
                            else:
                                await breaker._on_failure(Exception("Request failed"))
                        
                        break
    
    def add_watcher(self, callback: Callable):
        """Add service registry watcher"""
        self.watchers.append(callback)
    
    async def _notify_watchers(self, event_type: str, service_name: str, instance: ServiceInstance):
        """Notify all watchers of registry events"""
        for watcher in self.watchers:
            try:
                if asyncio.iscoroutinefunction(watcher):
                    await watcher(event_type, service_name, instance)
                else:
                    watcher(event_type, service_name, instance)
            except Exception as e:
                logger.error(f"Error in registry watcher: {e}")
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service registry statistics"""
        async with self.lock:
            stats = {
                'services': {},
                'circuit_breakers': {}
            }
            
            for service_name, instances in self.services.items():
                service_stats = {
                    'total_instances': len(instances),
                    'healthy_instances': len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
                    'instances': []
                }
                
                for instance in instances:
                    instance_stats = {
                        'service_id': instance.endpoint.service_id,
                        'address': instance.endpoint.address,
                        'status': instance.status.value,
                        'circuit_state': instance.circuit_state.value,
                        'failure_count': instance.failure_count,
                        'success_count': instance.success_count,
                        'average_response_time': instance.average_response_time,
                        'active_connections': instance.active_connections
                    }
                    service_stats['instances'].append(instance_stats)
                
                stats['services'][service_name] = service_stats
            
            for name, breaker in self.circuit_breakers.items():
                stats['circuit_breakers'][name] = breaker.get_stats()
            
            return stats


class LoadBalancer:
    """Advanced load balancer with multiple algorithms"""
    
    def __init__(self, policy: TrafficPolicy = TrafficPolicy.ROUND_ROBIN):
        self.policy = policy
        self.current_index = 0
        self.request_counts = defaultdict(int)
        self.sticky_sessions: Dict[str, str] = {}  # session_id -> service_id
        
        logger.info(f"LoadBalancer initialized with {policy.value} policy")
    
    async def select_instance(self, 
                             instances: List[ServiceInstance],
                             session_id: Optional[str] = None,
                             canary_percentage: float = 0.0) -> Optional[ServiceInstance]:
        """Select instance based on load balancing policy"""
        if not instances:
            return None
        
        available_instances = [i for i in instances if i.is_available]
        if not available_instances:
            return None
        
        # Handle sticky sessions
        if session_id and session_id in self.sticky_sessions:
            target_service_id = self.sticky_sessions[session_id]
            for instance in available_instances:
                if instance.endpoint.service_id == target_service_id:
                    return instance
        
        # Handle canary deployment
        if canary_percentage > 0:
            canary_instances = [i for i in available_instances if i.endpoint.metadata.get('canary', False)]
            if canary_instances and random.random() < (canary_percentage / 100):
                selected = await self._apply_policy(canary_instances)
                if selected and session_id:
                    self.sticky_sessions[session_id] = selected.endpoint.service_id
                return selected
            
            # Filter out canary instances for regular traffic
            available_instances = [i for i in available_instances if not i.endpoint.metadata.get('canary', False)]
        
        selected = await self._apply_policy(available_instances)
        
        # Update sticky session
        if selected and session_id:
            self.sticky_sessions[session_id] = selected.endpoint.service_id
        
        return selected
    
    async def _apply_policy(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Apply load balancing policy"""
        if not instances:
            return None
        
        if self.policy == TrafficPolicy.ROUND_ROBIN:
            return self._round_robin(instances)
        elif self.policy == TrafficPolicy.LEAST_CONNECTIONS:
            return self._least_connections(instances)
        elif self.policy == TrafficPolicy.WEIGHTED:
            return self._weighted_selection(instances)
        elif self.policy == TrafficPolicy.FAILOVER:
            return self._failover_selection(instances)
        else:
            return instances[0]  # Fallback
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _weighted_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection"""
        total_weight = sum(i.endpoint.weight for i in instances)
        if total_weight == 0:
            return instances[0]
        
        target = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.endpoint.weight
            if current_weight >= target:
                return instance
        
        return instances[-1]  # Fallback
    
    def _failover_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Failover selection (primary first)"""
        # Sort by priority (lower failure count = higher priority)
        sorted_instances = sorted(instances, key=lambda x: x.failure_count)
        return sorted_instances[0]


class HealthChecker:
    """Health checking service"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.running = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("HealthChecker initialized")
    
    async def start(self):
        """Start health checking"""
        if self.running:
            return
        
        self.running = True
        
        # Start health checks for all registered services
        async with self.registry.lock:
            for service_name, instances in self.registry.services.items():
                for instance in instances:
                    service_id = instance.endpoint.service_id
                    if service_id in self.registry.health_checks:
                        self._start_health_check(service_name, instance)
        
        logger.info("HealthChecker started")
    
    async def stop(self):
        """Stop health checking"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all health check tasks
        for task in list(self.check_tasks.values()):
            task.cancel()
        
        await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        self.check_tasks.clear()
        
        logger.info("HealthChecker stopped")
    
    def _start_health_check(self, service_name: str, instance: ServiceInstance):
        """Start health check for a service instance"""
        service_id = instance.endpoint.service_id
        health_check = self.registry.health_checks.get(service_id)
        
        if health_check:
            task = asyncio.create_task(
                self._health_check_loop(service_name, instance, health_check)
            )
            self.check_tasks[service_id] = task
    
    async def _health_check_loop(self, service_name: str, instance: ServiceInstance, health_check: HealthCheck):
        """Health check loop for a service instance"""
        service_id = instance.endpoint.service_id
        consecutive_failures = 0
        consecutive_successes = 0
        
        logger.debug(f"Started health check loop for {service_name}/{service_id}")
        
        while self.running:
            try:
                # Perform health check
                success = await self._perform_health_check(instance, health_check)
                
                if success:
                    consecutive_failures = 0
                    consecutive_successes += 1
                    
                    # Mark as healthy if enough consecutive successes
                    if (instance.status != ServiceStatus.HEALTHY and 
                        consecutive_successes >= health_check.success_threshold):
                        await self.registry.update_instance_status(service_name, service_id, ServiceStatus.HEALTHY)
                
                else:
                    consecutive_successes = 0
                    consecutive_failures += 1
                    
                    # Mark as unhealthy if enough consecutive failures
                    if consecutive_failures >= health_check.failure_threshold:
                        if instance.status == ServiceStatus.HEALTHY:
                            await self.registry.update_instance_status(service_name, service_id, ServiceStatus.DEGRADED)
                        elif consecutive_failures >= health_check.failure_threshold * 2:
                            await self.registry.update_instance_status(service_name, service_id, ServiceStatus.UNHEALTHY)
                
                # Wait for next check
                await asyncio.sleep(health_check.interval_seconds)
                
            except asyncio.CancelledError:
                logger.debug(f"Health check cancelled for {service_name}/{service_id}")
                break
            except Exception as e:
                logger.error(f"Health check error for {service_name}/{service_id}: {e}")
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _perform_health_check(self, instance: ServiceInstance, health_check: HealthCheck) -> bool:
        """Perform actual health check"""
        try:
            import aiohttp
            
            url = f"{instance.endpoint.address}{health_check.endpoint}"
            timeout = aiohttp.ClientTimeout(total=health_check.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=health_check.headers) as response:
                    return response.status < 500
                    
        except Exception as e:
            logger.debug(f"Health check failed for {instance.endpoint.service_id}: {e}")
            return False


class ServiceMesh:
    """Main service mesh orchestrator"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker(self.registry)
        
        # Graceful shutdown handling
        self.shutdown_event = asyncio.Event()
        self.shutdown_timeout = 30  # seconds
        
        # Service mesh statistics
        self.request_count = 0
        self.error_count = 0
        self.average_response_time = 0.0
        
        logger.info("ServiceMesh initialized")
    
    async def start(self):
        """Start service mesh"""
        # Start health checker
        await self.health_checker.start()
        
        # Set up graceful shutdown handlers
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self._shutdown_handler)
        
        logger.info("ServiceMesh started")
    
    async def stop(self):
        """Stop service mesh"""
        self.shutdown_event.set()
        
        # Stop health checker
        await self.health_checker.stop()
        
        logger.info("ServiceMesh stopped")
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self._graceful_shutdown())
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown"""
        logger.info("Starting graceful shutdown sequence")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Wait for ongoing requests to complete
        shutdown_start = time.time()
        while time.time() - shutdown_start < self.shutdown_timeout:
            # Check if all instances are idle
            all_idle = True
            async with self.registry.lock:
                for instances in self.registry.services.values():
                    for instance in instances:
                        if instance.active_connections > 0:
                            all_idle = False
                            break
                    if not all_idle:
                        break
            
            if all_idle:
                break
            
            await asyncio.sleep(1)
        
        # Stop all components
        await self.stop()
        
        logger.info("Graceful shutdown completed")
    
    @monitor_performance(track_memory=True, custom_name="service_mesh_request")
    async def make_request(self, 
                          service_name: str,
                          method: str = "GET",
                          path: str = "/",
                          session_id: Optional[str] = None,
                          circuit_breaker: bool = True,
                          **kwargs) -> Any:
        """Make request through service mesh"""
        
        # Get healthy instances
        instances = await self.registry.get_healthy_instances(service_name)
        if not instances:
            raise ServiceMeshError(f"No healthy instances available for service '{service_name}'", "no_instances")
        
        # Select instance using load balancer
        selected_instance = await self.load_balancer.select_instance(instances, session_id)
        if not selected_instance:
            raise ServiceMeshError(f"Load balancer failed to select instance for service '{service_name}'", "load_balancer_failure")
        
        # Increment connection count
        selected_instance.active_connections += 1
        
        try:
            # Use circuit breaker if enabled
            if circuit_breaker:
                breaker_name = f"{service_name}-{selected_instance.endpoint.service_id}"
                breaker = self.registry.circuit_breakers.get(breaker_name)
                
                if breaker:
                    result = await breaker.call(self._execute_request, selected_instance, method, path, **kwargs)
                else:
                    result = await self._execute_request(selected_instance, method, path, **kwargs)
            else:
                result = await self._execute_request(selected_instance, method, path, **kwargs)
            
            return result
            
        finally:
            # Decrement connection count
            selected_instance.active_connections -= 1
    
    async def _execute_request(self, instance: ServiceInstance, method: str, path: str, **kwargs) -> Any:
        """Execute HTTP request to service instance"""
        import aiohttp
        
        start_time = time.time()
        url = f"{instance.endpoint.address}{path}"
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, **kwargs) as response:
                    result = await response.json() if response.content_type == 'application/json' else await response.text()
                    response_time = time.time() - start_time
                    
                    # Record success
                    await self.registry.record_request_result(
                        instance.endpoint.metadata.get('service_name', 'unknown'),
                        instance.endpoint.service_id,
                        True,
                        response_time
                    )
                    
                    # Update mesh statistics
                    self.request_count += 1
                    self.average_response_time = (self.average_response_time + response_time) / 2
                    
                    return result
                    
        except Exception as e:
            response_time = time.time() - start_time
            
            # Record failure
            await self.registry.record_request_result(
                instance.endpoint.metadata.get('service_name', 'unknown'),
                instance.endpoint.service_id,
                False,
                response_time
            )
            
            # Update mesh statistics
            self.request_count += 1
            self.error_count += 1
            
            raise ServiceMeshError(f"Request to {url} failed: {e}", "request_failure", e)
    
    async def register_service(self, 
                              service_name: str,
                              host: str,
                              port: int,
                              health_endpoint: str = "/health",
                              weight: float = 1.0,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register service with mesh"""
        
        service_id = str(uuid.uuid4())
        endpoint = ServiceEndpoint(
            service_id=service_id,
            host=host,
            port=port,
            weight=weight,
            metadata=metadata or {'service_name': service_name}
        )
        
        health_check = HealthCheck(
            endpoint=health_endpoint,
            interval_seconds=30,
            timeout_seconds=5
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60
        )
        
        await self.registry.register_service(
            service_name, 
            endpoint, 
            health_check, 
            circuit_breaker_config
        )
        
        return service_id
    
    async def deregister_service(self, service_name: str, service_id: str):
        """Deregister service from mesh"""
        await self.registry.deregister_service(service_name, service_id)
    
    async def get_mesh_stats(self) -> Dict[str, Any]:
        """Get comprehensive service mesh statistics"""
        registry_stats = await self.registry.get_service_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mesh_stats': {
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / max(self.request_count, 1),
                'average_response_time': self.average_response_time,
                'load_balancer_policy': self.load_balancer.policy.value
            },
            'registry_stats': registry_stats,
            'health_checker_running': self.health_checker.running
        }


# Global service mesh instance
_service_mesh: Optional[ServiceMesh] = None
_mesh_lock = asyncio.Lock()


async def get_service_mesh() -> ServiceMesh:
    """Get global service mesh instance"""
    global _service_mesh
    
    if _service_mesh is None:
        async with _mesh_lock:
            if _service_mesh is None:
                _service_mesh = ServiceMesh()
                await _service_mesh.start()
    
    return _service_mesh


# Convenience functions
@with_enhanced_error_handling("service_mesh_request")
async def call_service(service_name: str, 
                      method: str = "GET", 
                      path: str = "/",
                      session_id: Optional[str] = None,
                      **kwargs) -> Any:
    """Make service call through mesh"""
    mesh = await get_service_mesh()
    return await mesh.make_request(service_name, method, path, session_id, **kwargs)


async def register_current_service(service_name: str, 
                                  host: str = "localhost", 
                                  port: int = 8000,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
    """Register current service with mesh"""
    mesh = await get_service_mesh()
    return await mesh.register_service(service_name, host, port, metadata=metadata)


async def get_service_mesh_stats() -> Dict[str, Any]:
    """Get service mesh statistics"""
    mesh = await get_service_mesh()
    return await mesh.get_mesh_stats()