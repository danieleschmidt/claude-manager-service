#!/usr/bin/env python3
"""
Advanced Scaling and Performance Optimization System

Implements comprehensive scaling and performance optimization:
- Horizontal and vertical auto-scaling
- Intelligent load balancing
- Resource optimization and prediction
- Performance profiling and optimization
- Distributed computing capabilities
- Caching and optimization strategies
"""

import asyncio
import time
import json
import threading
import multiprocessing
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
import queue
import psutil
import math

# Optional ML imports for prediction
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ResourceType(Enum):
    """Resource types for scaling"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    WORKERS = "workers"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"

@dataclass
class ScalingPolicy:
    """Scaling policy configuration"""
    resource_type: ResourceType
    scale_up_threshold: float = 80.0    # Percentage
    scale_down_threshold: float = 30.0  # Percentage
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: int = 300        # seconds
    scale_down_cooldown: int = 600      # seconds
    evaluation_periods: int = 3

@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions"""
    cpu_utilization: float
    memory_utilization: float
    active_requests: int
    response_time: float
    error_rate: float
    throughput: float
    timestamp: datetime

@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: datetime
    resource_type: ResourceType
    direction: ScalingDirection
    from_value: int
    to_value: int
    reason: str
    success: bool

class ResourcePredictor:
    """ML-based resource usage prediction"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.cpu_history: deque = deque(maxlen=window_size)
        self.memory_history: deque = deque(maxlen=window_size)
        self.request_history: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger("resource_predictor")
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add performance metrics to prediction model"""
        self.cpu_history.append(metrics.cpu_utilization)
        self.memory_history.append(metrics.memory_utilization)
        self.request_history.append(metrics.active_requests)
    
    def predict_resource_usage(self, look_ahead_minutes: int = 15) -> Dict[str, float]:
        """Predict resource usage for the specified time ahead"""
        predictions = {}
        
        if HAS_NUMPY and len(self.cpu_history) >= 10:
            # Use simple linear regression for prediction
            predictions['cpu'] = self._predict_trend(list(self.cpu_history), look_ahead_minutes)
            predictions['memory'] = self._predict_trend(list(self.memory_history), look_ahead_minutes)
            predictions['requests'] = self._predict_trend(list(self.request_history), look_ahead_minutes)
        else:
            # Fallback to simple moving average
            if self.cpu_history:
                predictions['cpu'] = sum(self.cpu_history) / len(self.cpu_history)
                predictions['memory'] = sum(self.memory_history) / len(self.memory_history)
                predictions['requests'] = sum(self.request_history) / len(self.request_history)
        
        return predictions
    
    def _predict_trend(self, values: List[float], look_ahead: int) -> float:
        """Predict future value using linear regression"""
        if not HAS_NUMPY or len(values) < 2:
            return values[-1] if values else 0.0
        
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predict future value
            future_x = len(values) + look_ahead
            predicted = slope * future_x + intercept
            
            # Ensure prediction is within reasonable bounds
            predicted = max(0, min(100, predicted))
            
            return float(predicted)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return values[-1] if values else 0.0
    
    def get_trend_analysis(self) -> Dict[str, str]:
        """Analyze current trends"""
        analysis = {}
        
        for name, history in [
            ('cpu', self.cpu_history),
            ('memory', self.memory_history),
            ('requests', self.request_history)
        ]:
            if len(history) >= 5:
                recent = list(history)[-5:]
                older = list(history)[-10:-5] if len(history) >= 10 else recent
                
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                
                if recent_avg > older_avg * 1.1:
                    analysis[name] = "increasing"
                elif recent_avg < older_avg * 0.9:
                    analysis[name] = "decreasing"
                else:
                    analysis[name] = "stable"
            else:
                analysis[name] = "insufficient_data"
        
        return analysis

class WorkerPool:
    """Dynamic worker pool for concurrent processing"""
    
    def __init__(self, initial_workers: int = 4, max_workers: int = 16):
        self.min_workers = 1
        self.max_workers = max_workers
        self.current_workers = initial_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=initial_workers)
        self.task_queue = queue.Queue()
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger("worker_pool")
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start worker pool monitoring"""
        def monitor():
            while True:
                try:
                    self._adjust_workers()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Worker pool monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _adjust_workers(self):
        """Automatically adjust worker count based on load"""
        with self.lock:
            queue_size = self.task_queue.qsize()
            utilization = self.active_tasks / self.current_workers if self.current_workers > 0 else 0
            
            # Scale up if queue is growing or utilization is high
            if (queue_size > 5 or utilization > 0.8) and self.current_workers < self.max_workers:
                new_workers = min(self.current_workers + 2, self.max_workers)
                self._scale_workers(new_workers)
                self.logger.info(f"Scaled up workers: {self.current_workers} -> {new_workers}")
            
            # Scale down if utilization is low and queue is empty
            elif utilization < 0.3 and queue_size == 0 and self.current_workers > self.min_workers:
                new_workers = max(self.current_workers - 1, self.min_workers)
                self._scale_workers(new_workers)
                self.logger.info(f"Scaled down workers: {self.current_workers} -> {new_workers}")
    
    def _scale_workers(self, new_count: int):
        """Scale worker pool to new count"""
        if new_count != self.current_workers:
            # Shutdown old executor
            old_executor = self.executor
            
            # Create new executor
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=new_count)
            self.current_workers = new_count
            
            # Gracefully shutdown old executor
            old_executor.shutdown(wait=False)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the worker pool"""
        with self.lock:
            self.active_tasks += 1
        
        def wrapped_task():
            try:
                result = func(*args, **kwargs)
                with self.lock:
                    self.completed_tasks += 1
                return result
            except Exception as e:
                with self.lock:
                    self.failed_tasks += 1
                raise e
            finally:
                with self.lock:
                    self.active_tasks -= 1
        
        return self.executor.submit(wrapped_task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        with self.lock:
            return {
                'current_workers': self.current_workers,
                'active_tasks': self.active_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'queue_size': self.task_queue.qsize(),
                'utilization': self.active_tasks / self.current_workers if self.current_workers > 0 else 0
            }

class CacheManager:
    """Intelligent caching system with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: deque = deque()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.logger = logging.getLogger("cache_manager")
        
        # Start cleanup thread
        self._start_cleanup()
    
    def _start_cleanup(self):
        """Start background cleanup of expired entries"""
        def cleanup():
            while True:
                try:
                    self._cleanup_expired()
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        now = time.time()
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if entry['expires_at'] <= now:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry['expires_at'] <= time.time():
                    del self.cache[key]
                    try:
                        self.access_order.remove(key)
                    except ValueError:
                        pass
                    self.misses += 1
                    return None
                
                # Update access order for LRU
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)
                
                self.hits += 1
                return entry['value']
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        with self.lock:
            # Check if we need to evict
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add/update entry
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            
            # Update access order
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
            
            return True
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
                self.evictions += 1
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'utilization': (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0
            }

class LoadBalancer:
    """Intelligent load balancer for distributing requests"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.endpoints: List[Dict[str, Any]] = []
        self.current_index = 0
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        self.logger = logging.getLogger("load_balancer")
    
    def add_endpoint(self, endpoint_id: str, weight: int = 1, health_check_url: str = None):
        """Add an endpoint to the load balancer"""
        with self.lock:
            endpoint = {
                'id': endpoint_id,
                'weight': weight,
                'health_check_url': health_check_url,
                'healthy': True,
                'last_health_check': time.time(),
                'consecutive_failures': 0
            }
            self.endpoints.append(endpoint)
            self.logger.info(f"Added endpoint: {endpoint_id}")
    
    def remove_endpoint(self, endpoint_id: str):
        """Remove an endpoint from the load balancer"""
        with self.lock:
            self.endpoints = [ep for ep in self.endpoints if ep['id'] != endpoint_id]
            self.logger.info(f"Removed endpoint: {endpoint_id}")
    
    def get_endpoint(self) -> Optional[str]:
        """Get the next endpoint based on load balancing strategy"""
        with self.lock:
            healthy_endpoints = [ep for ep in self.endpoints if ep['healthy']]
            
            if not healthy_endpoints:
                return None
            
            if self.strategy == "round_robin":
                endpoint = self._round_robin(healthy_endpoints)
            elif self.strategy == "weighted_round_robin":
                endpoint = self._weighted_round_robin(healthy_endpoints)
            elif self.strategy == "least_connections":
                endpoint = self._least_connections(healthy_endpoints)
            elif self.strategy == "fastest_response":
                endpoint = self._fastest_response(healthy_endpoints)
            else:
                endpoint = healthy_endpoints[0]  # Fallback
            
            if endpoint:
                self.request_counts[endpoint['id']] += 1
                return endpoint['id']
            
            return None
    
    def _round_robin(self, endpoints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Round robin selection"""
        if not endpoints:
            return None
        
        endpoint = endpoints[self.current_index % len(endpoints)]
        self.current_index += 1
        return endpoint
    
    def _weighted_round_robin(self, endpoints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Weighted round robin selection"""
        if not endpoints:
            return None
        
        # Create weighted list
        weighted_endpoints = []
        for endpoint in endpoints:
            weighted_endpoints.extend([endpoint] * endpoint['weight'])
        
        if not weighted_endpoints:
            return endpoints[0]
        
        endpoint = weighted_endpoints[self.current_index % len(weighted_endpoints)]
        self.current_index += 1
        return endpoint
    
    def _least_connections(self, endpoints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Least connections selection"""
        if not endpoints:
            return None
        
        # Find endpoint with least connections
        min_connections = float('inf')
        selected_endpoint = None
        
        for endpoint in endpoints:
            connections = self.request_counts[endpoint['id']]
            if connections < min_connections:
                min_connections = connections
                selected_endpoint = endpoint
        
        return selected_endpoint or endpoints[0]
    
    def _fastest_response(self, endpoints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Fastest response time selection"""
        if not endpoints:
            return None
        
        fastest_time = float('inf')
        selected_endpoint = None
        
        for endpoint in endpoints:
            response_times = self.response_times[endpoint['id']]
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                if avg_time < fastest_time:
                    fastest_time = avg_time
                    selected_endpoint = endpoint
        
        return selected_endpoint or endpoints[0]
    
    def record_response_time(self, endpoint_id: str, response_time: float):
        """Record response time for an endpoint"""
        with self.lock:
            self.response_times[endpoint_id].append(response_time)
    
    def mark_endpoint_unhealthy(self, endpoint_id: str):
        """Mark an endpoint as unhealthy"""
        with self.lock:
            for endpoint in self.endpoints:
                if endpoint['id'] == endpoint_id:
                    endpoint['healthy'] = False
                    endpoint['consecutive_failures'] += 1
                    self.logger.warning(f"Marked endpoint unhealthy: {endpoint_id}")
                    break
    
    def mark_endpoint_healthy(self, endpoint_id: str):
        """Mark an endpoint as healthy"""
        with self.lock:
            for endpoint in self.endpoints:
                if endpoint['id'] == endpoint_id:
                    endpoint['healthy'] = True
                    endpoint['consecutive_failures'] = 0
                    self.logger.info(f"Marked endpoint healthy: {endpoint_id}")
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self.lock:
            stats = {
                'strategy': self.strategy,
                'total_endpoints': len(self.endpoints),
                'healthy_endpoints': len([ep for ep in self.endpoints if ep['healthy']]),
                'request_distribution': dict(self.request_counts),
                'endpoints': []
            }
            
            for endpoint in self.endpoints:
                endpoint_stats = {
                    'id': endpoint['id'],
                    'healthy': endpoint['healthy'],
                    'requests': self.request_counts[endpoint['id']],
                    'consecutive_failures': endpoint['consecutive_failures']
                }
                
                response_times = self.response_times[endpoint['id']]
                if response_times:
                    endpoint_stats['avg_response_time'] = sum(response_times) / len(response_times)
                
                stats['endpoints'].append(endpoint_stats)
            
            return stats

class AutoScaler:
    """Intelligent auto-scaling system"""
    
    def __init__(self, policies: List[ScalingPolicy] = None):
        self.policies = policies or []
        self.resource_predictor = ResourcePredictor()
        self.scaling_history: List[ScalingEvent] = []
        self.current_instances: Dict[ResourceType, int] = {}
        self.last_scale_times: Dict[ResourceType, datetime] = {}
        self.metrics_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
        self.logger = logging.getLogger("auto_scaler")
        
        # Initialize current instances
        for policy in self.policies:
            self.current_instances[policy.resource_type] = policy.min_instances
        
        # Start scaling loop
        self._start_scaling_loop()
    
    def add_policy(self, policy: ScalingPolicy):
        """Add a scaling policy"""
        with self.lock:
            self.policies.append(policy)
            if policy.resource_type not in self.current_instances:
                self.current_instances[policy.resource_type] = policy.min_instances
        self.logger.info(f"Added scaling policy for {policy.resource_type.value}")
    
    def _start_scaling_loop(self):
        """Start the auto-scaling evaluation loop"""
        def scaling_loop():
            while True:
                try:
                    self._evaluate_scaling()
                    time.sleep(60)  # Evaluate every minute
                except Exception as e:
                    self.logger.error(f"Auto-scaling loop error: {e}")
        
        scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        scaling_thread.start()
        self.logger.info("Auto-scaling loop started")
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add performance metrics for scaling decisions"""
        with self.lock:
            self.metrics_history.append(metrics)
            self.resource_predictor.add_metrics(metrics)
    
    def _evaluate_scaling(self):
        """Evaluate scaling decisions"""
        if not self.metrics_history:
            return
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
        
        for policy in self.policies:
            try:
                self._evaluate_policy(policy, recent_metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating policy for {policy.resource_type}: {e}")
    
    def _evaluate_policy(self, policy: ScalingPolicy, metrics: List[PerformanceMetrics]):
        """Evaluate a specific scaling policy"""
        if len(metrics) < policy.evaluation_periods:
            return
        
        # Get current resource utilization
        if policy.resource_type == ResourceType.CPU:
            values = [m.cpu_utilization for m in metrics[-policy.evaluation_periods:]]
        elif policy.resource_type == ResourceType.MEMORY:
            values = [m.memory_utilization for m in metrics[-policy.evaluation_periods:]]
        elif policy.resource_type == ResourceType.WORKERS:
            # Use a combination of factors for worker scaling
            cpu_values = [m.cpu_utilization for m in metrics[-policy.evaluation_periods:]]
            request_values = [m.active_requests for m in metrics[-policy.evaluation_periods:]]
            # Combine CPU and request load
            values = [(cpu + (requests / 10)) for cpu, requests in zip(cpu_values, request_values)]
        else:
            return  # Unsupported resource type
        
        avg_utilization = sum(values) / len(values)
        current_instances = self.current_instances.get(policy.resource_type, policy.min_instances)
        
        # Check cooldown periods
        now = datetime.now(timezone.utc)
        last_scale_time = self.last_scale_times.get(policy.resource_type)
        
        # Determine scaling action
        scaling_direction = ScalingDirection.STABLE
        new_instances = current_instances
        
        if avg_utilization > policy.scale_up_threshold and current_instances < policy.max_instances:
            if not last_scale_time or (now - last_scale_time).total_seconds() >= policy.scale_up_cooldown:
                scaling_direction = ScalingDirection.UP
                new_instances = min(current_instances + 1, policy.max_instances)
        
        elif avg_utilization < policy.scale_down_threshold and current_instances > policy.min_instances:
            if not last_scale_time or (now - last_scale_time).total_seconds() >= policy.scale_down_cooldown:
                scaling_direction = ScalingDirection.DOWN
                new_instances = max(current_instances - 1, policy.min_instances)
        
        # Execute scaling if needed
        if scaling_direction != ScalingDirection.STABLE:
            success = self._execute_scaling(policy.resource_type, new_instances, scaling_direction, avg_utilization)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=now,
                resource_type=policy.resource_type,
                direction=scaling_direction,
                from_value=current_instances,
                to_value=new_instances,
                reason=f"Utilization: {avg_utilization:.1f}%",
                success=success
            )
            
            with self.lock:
                self.scaling_history.append(event)
                if success:
                    self.current_instances[policy.resource_type] = new_instances
                    self.last_scale_times[policy.resource_type] = now
            
            self.logger.info(f"Scaling {policy.resource_type.value} {scaling_direction.value}: "
                           f"{current_instances} -> {new_instances} (util: {avg_utilization:.1f}%)")
    
    def _execute_scaling(self, resource_type: ResourceType, new_instances: int, 
                        direction: ScalingDirection, utilization: float) -> bool:
        """Execute the actual scaling operation"""
        try:
            # This would integrate with actual scaling mechanisms
            # For now, we'll just simulate the scaling
            
            if resource_type == ResourceType.WORKERS:
                # Scale worker processes/threads
                pass
            elif resource_type == ResourceType.CPU:
                # Scale CPU resources (e.g., in containerized environment)
                pass
            elif resource_type == ResourceType.MEMORY:
                # Scale memory resources
                pass
            
            # Simulate success
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling for {resource_type}: {e}")
            return False
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on predictions"""
        predictions = self.resource_predictor.predict_resource_usage(15)  # 15 minutes ahead
        trends = self.resource_predictor.get_trend_analysis()
        
        recommendations = {}
        
        for policy in self.policies:
            resource_name = policy.resource_type.value
            predicted_utilization = predictions.get(resource_name, 0)
            trend = trends.get(resource_name, 'stable')
            current_instances = self.current_instances.get(policy.resource_type, policy.min_instances)
            
            recommendation = {
                'current_instances': current_instances,
                'predicted_utilization': predicted_utilization,
                'trend': trend,
                'action': 'maintain'
            }
            
            if predicted_utilization > policy.scale_up_threshold and current_instances < policy.max_instances:
                recommendation['action'] = 'scale_up'
                recommendation['suggested_instances'] = min(current_instances + 1, policy.max_instances)
            elif predicted_utilization < policy.scale_down_threshold and current_instances > policy.min_instances:
                recommendation['action'] = 'scale_down'
                recommendation['suggested_instances'] = max(current_instances - 1, policy.min_instances)
            
            recommendations[resource_name] = recommendation
        
        return recommendations
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling history for specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self.lock:
            recent_events = [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'resource_type': event.resource_type.value,
                    'direction': event.direction.value,
                    'from_value': event.from_value,
                    'to_value': event.to_value,
                    'reason': event.reason,
                    'success': event.success
                }
                for event in self.scaling_history
                if event.timestamp >= cutoff_time
            ]
        
        return recent_events

class AdvancedScalingSystem:
    """Main scaling system that orchestrates all components"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.auto_scaler = AutoScaler()
        self.worker_pool = WorkerPool()
        self.cache_manager = CacheManager()
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger("advanced_scaling")
        
        # Setup default scaling policies
        self._setup_default_policies()
        
        # Start metrics collection
        self._start_metrics_collection()
    
    def _setup_default_policies(self):
        """Setup default scaling policies based on strategy"""
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            cpu_policy = ScalingPolicy(
                resource_type=ResourceType.CPU,
                scale_up_threshold=70.0,
                scale_down_threshold=20.0,
                max_instances=20,
                scale_up_cooldown=180,
                scale_down_cooldown=300
            )
            worker_policy = ScalingPolicy(
                resource_type=ResourceType.WORKERS,
                scale_up_threshold=60.0,
                scale_down_threshold=25.0,
                max_instances=32,
                scale_up_cooldown=120,
                scale_down_cooldown=240
            )
        elif self.strategy == OptimizationStrategy.CONSERVATIVE:
            cpu_policy = ScalingPolicy(
                resource_type=ResourceType.CPU,
                scale_up_threshold=90.0,
                scale_down_threshold=40.0,
                max_instances=8,
                scale_up_cooldown=600,
                scale_down_cooldown=900
            )
            worker_policy = ScalingPolicy(
                resource_type=ResourceType.WORKERS,
                scale_up_threshold=85.0,
                scale_down_threshold=35.0,
                max_instances=16,
                scale_up_cooldown=480,
                scale_down_cooldown=720
            )
        else:  # BALANCED
            cpu_policy = ScalingPolicy(
                resource_type=ResourceType.CPU,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                max_instances=12,
                scale_up_cooldown=300,
                scale_down_cooldown=600
            )
            worker_policy = ScalingPolicy(
                resource_type=ResourceType.WORKERS,
                scale_up_threshold=75.0,
                scale_down_threshold=30.0,
                max_instances=24,
                scale_up_cooldown=240,
                scale_down_cooldown=480
            )
        
        self.auto_scaler.add_policy(cpu_policy)
        self.auto_scaler.add_policy(worker_policy)
    
    def _start_metrics_collection(self):
        """Start collecting system metrics for scaling decisions"""
        def collect_metrics():
            while True:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    # Get worker pool stats for active requests simulation
                    worker_stats = self.worker_pool.get_stats()
                    active_requests = worker_stats['active_tasks']
                    
                    # Calculate response time (simulated for now)
                    response_time = 0.1 + (cpu_percent / 1000)  # Simple simulation
                    
                    # Calculate error rate (simulated)
                    error_rate = max(0, (cpu_percent - 80) / 20 * 5)  # Errors increase with high CPU
                    
                    # Calculate throughput
                    throughput = max(1, 100 - cpu_percent)  # Throughput decreases with high CPU
                    
                    metrics = PerformanceMetrics(
                        cpu_utilization=cpu_percent,
                        memory_utilization=memory_percent,
                        active_requests=active_requests,
                        response_time=response_time,
                        error_rate=error_rate,
                        throughput=throughput,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    self.auto_scaler.add_metrics(metrics)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Metrics collection error: {e}")
                    time.sleep(30)
        
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
        self.logger.info("Metrics collection started")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system scaling status"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'strategy': self.strategy.value,
            'auto_scaler': {
                'current_instances': dict(self.auto_scaler.current_instances),
                'recommendations': self.auto_scaler.get_scaling_recommendations(),
                'recent_events': len(self.auto_scaler.get_scaling_history(1))  # Last hour
            },
            'worker_pool': self.worker_pool.get_stats(),
            'cache': self.cache_manager.get_stats(),
            'load_balancer': self.load_balancer.get_stats()
        }
    
    def submit_work(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit work to the optimized worker pool"""
        return self.worker_pool.submit_task(func, *args, **kwargs)
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get result from cache"""
        return self.cache_manager.get(key)
    
    def cache_result(self, key: str, value: Any, ttl: int = None) -> bool:
        """Cache a result"""
        return self.cache_manager.set(key, value, ttl)
    
    def get_endpoint(self) -> Optional[str]:
        """Get next endpoint from load balancer"""
        return self.load_balancer.get_endpoint()
    
    def add_endpoint(self, endpoint_id: str, weight: int = 1):
        """Add endpoint to load balancer"""
        self.load_balancer.add_endpoint(endpoint_id, weight)

# Utility functions
def create_scaling_system(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> AdvancedScalingSystem:
    """Create a scaling system with specified strategy"""
    return AdvancedScalingSystem(strategy)

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scaling system
    scaling_system = create_scaling_system(OptimizationStrategy.BALANCED)
    
    # Add some load balancer endpoints
    scaling_system.add_endpoint("worker-1", weight=2)
    scaling_system.add_endpoint("worker-2", weight=1)
    scaling_system.add_endpoint("worker-3", weight=1)
    
    # Example work function
    def example_work(duration: float = 1.0):
        time.sleep(duration)
        return f"Work completed in {duration}s"
    
    # Submit some work
    futures = []
    for i in range(10):
        future = scaling_system.submit_work(example_work, 0.5)
        futures.append(future)
    
    # Wait for work to complete
    for future in futures:
        result = future.result()
        print(f"Work result: {result}")
    
    # Get system status
    status = scaling_system.get_system_status()
    print(json.dumps(status, indent=2, default=str))