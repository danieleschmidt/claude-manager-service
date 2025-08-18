#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - QUANTUM SCALING ENGINE
Advanced auto-scaling with ML-based prediction, quantum-inspired optimization,
and distributed computing capabilities
"""

import asyncio
import json
import time
import math
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
from enum import Enum
import threading
import multiprocessing
import psutil
import numpy as np
from collections import deque, defaultdict
import structlog

logger = structlog.get_logger("QuantumScalingEngine")

class ScalingStrategy(Enum):
    """Scaling strategies available"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    QUANTUM_OPTIMIZED = "quantum_optimized"
    HYBRID = "hybrid"

class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU = "cpu"
    MEMORY = "memory"
    WORKERS = "workers"
    CONNECTIONS = "connections"
    CACHE = "cache"
    BANDWIDTH = "bandwidth"

@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    worker_utilization: float
    connection_count: int
    cache_hit_rate: float
    throughput: float
    response_time: float
    error_rate: float

@dataclass
class ScalingDecision:
    """Scaling decision with reasoning"""
    timestamp: datetime
    resource_type: ResourceType
    current_value: float
    target_value: float
    scaling_factor: float
    reasoning: str
    confidence: float
    estimated_impact: Dict[str, float]

@dataclass
class QuantumState:
    """Quantum-inspired optimization state"""
    dimensions: int
    position: List[float]
    velocity: List[float]
    fitness: float
    best_position: List[float]
    best_fitness: float

class MLPredictor:
    """Machine learning-based resource prediction"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.prediction_accuracy = 0.0
        self.feature_weights: Dict[str, float] = {
            'cpu_trend': 0.3,
            'memory_trend': 0.25,
            'throughput_trend': 0.2,
            'time_of_day': 0.15,
            'day_of_week': 0.1
        }
        
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics for prediction training"""
        self.metrics_history.append(metrics)
        
    def predict_demand(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """Predict resource demand for the next horizon_minutes"""
        if len(self.metrics_history) < 10:
            return self._get_baseline_prediction()
        
        # Extract features
        features = self._extract_features()
        
        # Simple trend-based prediction (would be replaced with actual ML model)
        predictions = {}
        
        # CPU prediction
        cpu_values = [m.cpu_percent for m in list(self.metrics_history)[-20:]]
        cpu_trend = self._calculate_trend(cpu_values)
        predictions['cpu_demand'] = max(0, min(100, 
            statistics.mean(cpu_values) + cpu_trend * (horizon_minutes / 5)))
        
        # Memory prediction
        memory_values = [m.memory_percent for m in list(self.metrics_history)[-20:]]
        memory_trend = self._calculate_trend(memory_values)
        predictions['memory_demand'] = max(0, min(100,
            statistics.mean(memory_values) + memory_trend * (horizon_minutes / 5)))
        
        # Throughput prediction
        throughput_values = [m.throughput for m in list(self.metrics_history)[-20:]]
        throughput_trend = self._calculate_trend(throughput_values)
        predictions['throughput_demand'] = max(0,
            statistics.mean(throughput_values) + throughput_trend * (horizon_minutes / 5))
        
        # Response time prediction
        response_times = [m.response_time for m in list(self.metrics_history)[-20:]]
        response_trend = self._calculate_trend(response_times)
        predictions['response_time'] = max(0,
            statistics.mean(response_times) + response_trend * (horizon_minutes / 5))
        
        return predictions
    
    def _extract_features(self) -> Dict[str, float]:
        """Extract features for ML prediction"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-20:]
        
        features = {
            'cpu_mean': statistics.mean([m.cpu_percent for m in recent_metrics]),
            'memory_mean': statistics.mean([m.memory_percent for m in recent_metrics]),
            'throughput_mean': statistics.mean([m.throughput for m in recent_metrics]),
            'response_time_mean': statistics.mean([m.response_time for m in recent_metrics]),
            'error_rate_mean': statistics.mean([m.error_rate for m in recent_metrics]),
            'cpu_variance': statistics.variance([m.cpu_percent for m in recent_metrics]) if len(recent_metrics) > 1 else 0,
            'memory_variance': statistics.variance([m.memory_percent for m in recent_metrics]) if len(recent_metrics) > 1 else 0,
            'time_of_day': datetime.now().hour / 24.0,
            'day_of_week': datetime.now().weekday() / 6.0,
        }
        
        return features
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _get_baseline_prediction(self) -> Dict[str, float]:
        """Get baseline prediction when insufficient data"""
        return {
            'cpu_demand': 50.0,
            'memory_demand': 60.0,
            'throughput_demand': 1000.0,
            'response_time': 200.0
        }

class QuantumOptimizer:
    """Quantum-inspired optimization for scaling decisions"""
    
    def __init__(self, dimensions: int = 5):
        self.dimensions = dimensions
        self.population_size = 20
        self.max_iterations = 50
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0
        self.quantum_probability = 0.1
        
        # Initialize quantum states
        self.states: List[QuantumState] = []
        self._initialize_population()
        
    def _initialize_population(self):
        """Initialize quantum state population"""
        self.states = []
        for _ in range(self.population_size):
            position = [np.random.uniform(0, 1) for _ in range(self.dimensions)]
            velocity = [np.random.uniform(-0.1, 0.1) for _ in range(self.dimensions)]
            
            state = QuantumState(
                dimensions=self.dimensions,
                position=position,
                velocity=velocity,
                fitness=0.0,
                best_position=position.copy(),
                best_fitness=0.0
            )
            self.states.append(state)
    
    def optimize_scaling(self, current_metrics: ResourceMetrics, 
                        predictions: Dict[str, float],
                        constraints: Dict[str, Tuple[float, float]]) -> Dict[ResourceType, float]:
        """Optimize scaling parameters using quantum-inspired algorithm"""
        
        # Define fitness function
        def fitness_function(params: List[float]) -> float:
            # Convert normalized parameters to actual scaling values
            scaling_params = {}
            param_names = ['cpu_scale', 'memory_scale', 'worker_scale', 'cache_scale', 'connection_scale']
            
            for i, name in enumerate(param_names):
                if name in constraints:
                    min_val, max_val = constraints[name]
                    scaling_params[name] = min_val + params[i] * (max_val - min_val)
                else:
                    scaling_params[name] = params[i]
            
            # Calculate fitness based on predicted performance
            predicted_cpu = predictions.get('cpu_demand', 50) / scaling_params.get('cpu_scale', 1.0)
            predicted_memory = predictions.get('memory_demand', 60) / scaling_params.get('memory_scale', 1.0)
            predicted_response_time = predictions.get('response_time', 200) / scaling_params.get('worker_scale', 1.0)
            
            # Fitness: minimize resource usage while maintaining performance
            resource_cost = (scaling_params.get('cpu_scale', 1) + 
                           scaling_params.get('memory_scale', 1) + 
                           scaling_params.get('worker_scale', 1)) / 3.0
            
            performance_penalty = 0
            if predicted_cpu > 80:
                performance_penalty += (predicted_cpu - 80) / 20.0
            if predicted_memory > 85:
                performance_penalty += (predicted_memory - 85) / 15.0
            if predicted_response_time > 1000:
                performance_penalty += (predicted_response_time - 1000) / 1000.0
            
            return -(resource_cost + performance_penalty)
        
        # Run quantum optimization
        global_best_position = None
        global_best_fitness = float('-inf')
        
        for iteration in range(self.max_iterations):
            for state in self.states:
                # Evaluate fitness
                state.fitness = fitness_function(state.position)
                
                # Update personal best
                if state.fitness > state.best_fitness:
                    state.best_fitness = state.fitness
                    state.best_position = state.position.copy()
                
                # Update global best
                if state.fitness > global_best_fitness:
                    global_best_fitness = state.fitness
                    global_best_position = state.position.copy()
            
            # Update velocities and positions with quantum effects
            for state in self.states:
                for d in range(self.dimensions):
                    # Quantum superposition effect
                    if np.random.random() < self.quantum_probability:
                        quantum_shift = np.random.normal(0, 0.1)
                        state.position[d] += quantum_shift
                    
                    # PSO velocity update
                    r1, r2 = np.random.random(), np.random.random()
                    
                    cognitive = self.cognitive_weight * r1 * (state.best_position[d] - state.position[d])
                    social = self.social_weight * r2 * (global_best_position[d] - state.position[d])
                    
                    state.velocity[d] = (self.inertia_weight * state.velocity[d] + 
                                       cognitive + social)
                    
                    # Update position
                    state.position[d] += state.velocity[d]
                    
                    # Boundary constraints
                    state.position[d] = max(0, min(1, state.position[d]))
        
        # Convert best solution to scaling parameters
        param_names = ['cpu_scale', 'memory_scale', 'worker_scale', 'cache_scale', 'connection_scale']
        resource_types = [ResourceType.CPU, ResourceType.MEMORY, ResourceType.WORKERS, 
                         ResourceType.CACHE, ResourceType.CONNECTIONS]
        
        scaling_decisions = {}
        for i, resource_type in enumerate(resource_types):
            param_name = param_names[i]
            if param_name in constraints:
                min_val, max_val = constraints[param_name]
                scaling_decisions[resource_type] = min_val + global_best_position[i] * (max_val - min_val)
            else:
                scaling_decisions[resource_type] = 0.5 + global_best_position[i]
        
        return scaling_decisions

class AdaptiveWorkerPool:
    """Adaptive worker pool that scales based on demand"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        self.task_queue_size = 0
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.performance_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Initialize pools
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize worker pools"""
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, self.current_workers // 2))
    
    async def submit_task(self, func: Callable, *args, use_process: bool = False, **kwargs) -> Any:
        """Submit task to appropriate worker pool"""
        start_time = time.time()
        
        with self._lock:
            self.task_queue_size += 1
            self.active_tasks += 1
        
        try:
            if use_process and self.process_pool:
                future = self.process_pool.submit(func, *args, **kwargs)
            else:
                future = self.thread_pool.submit(func, *args, **kwargs)
            
            # Convert to awaitable
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            with self._lock:
                self.completed_tasks += 1
                
            execution_time = time.time() - start_time
            self.performance_history.append(execution_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_tasks += 1
            raise e
        finally:
            with self._lock:
                self.active_tasks -= 1
                self.task_queue_size = max(0, self.task_queue_size - 1)
    
    def scale_workers(self, target_workers: int):
        """Scale worker pool to target size"""
        target_workers = max(self.min_workers, min(self.max_workers, target_workers))
        
        if target_workers == self.current_workers:
            return
        
        logger.info(f"Scaling workers from {self.current_workers} to {target_workers}")
        
        # Shutdown old pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        
        # Create new pools
        self.current_workers = target_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=target_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, target_workers // 2))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker pool metrics"""
        avg_execution_time = statistics.mean(self.performance_history) if self.performance_history else 0
        
        return {
            'current_workers': self.current_workers,
            'active_tasks': self.active_tasks,
            'queue_size': self.task_queue_size,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1),
            'average_execution_time': avg_execution_time,
            'utilization': self.active_tasks / max(self.current_workers, 1)
        }

class DistributedCacheManager:
    """Distributed cache with intelligent scaling and eviction"""
    
    def __init__(self, initial_size_mb: int = 100):
        self.initial_size_mb = initial_size_mb
        self.current_size_mb = initial_size_mb
        self.max_size_mb = initial_size_mb * 10
        
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.cache_sizes: Dict[str, int] = {}
        
        self.hit_count = 0
        self.miss_count = 0
        
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, size_bytes: Optional[int] = None) -> bool:
        """Set item in cache"""
        if size_bytes is None:
            size_bytes = len(str(value))  # Rough estimate
        
        with self._lock:
            # Check if we need to evict items
            while self._get_cache_size_mb() + (size_bytes / 1024 / 1024) > self.current_size_mb:
                if not self._evict_lru_item():
                    return False  # Cache full and can't evict
            
            # Store item
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.cache_sizes[key] = size_bytes
            
            return True
    
    def _evict_lru_item(self) -> bool:
        """Evict least recently used item"""
        if not self.cache:
            return False
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove item
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.access_counts[lru_key]
        del self.cache_sizes[lru_key]
        
        return True
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB"""
        total_bytes = sum(self.cache_sizes.values())
        return total_bytes / 1024 / 1024
    
    def scale_cache(self, target_size_mb: int):
        """Scale cache to target size"""
        target_size_mb = min(self.max_size_mb, max(10, target_size_mb))
        
        with self._lock:
            if target_size_mb < self.current_size_mb:
                # Need to evict items
                while (self._get_cache_size_mb() > target_size_mb and 
                       self.cache and 
                       self._evict_lru_item()):
                    pass
            
            self.current_size_mb = target_size_mb
            
        logger.info(f"Cache scaled to {target_size_mb}MB")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        hit_rate = self.hit_count / max(self.hit_count + self.miss_count, 1)
        
        return {
            'current_size_mb': self.current_size_mb,
            'used_size_mb': self._get_cache_size_mb(),
            'utilization': self._get_cache_size_mb() / self.current_size_mb,
            'item_count': len(self.cache),
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count
        }

class QuantumScalingEngine:
    """Main quantum scaling engine orchestrating all scaling components"""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        
        # Initialize components
        self.predictor = MLPredictor()
        self.optimizer = QuantumOptimizer()
        self.worker_pool = AdaptiveWorkerPool()
        self.cache_manager = DistributedCacheManager()
        
        # Scaling state
        self.current_metrics = None
        self.scaling_history: List[ScalingDecision] = []
        self.auto_scaling_enabled = True
        
        # Performance tracking
        self.performance_baseline = {}
        self.scaling_effectiveness = deque(maxlen=50)
        
        # Monitoring
        self._running = False
        self._monitoring_task = None
    
    async def start_engine(self):
        """Start the quantum scaling engine"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Quantum Scaling Engine started")
    
    async def stop_engine(self):
        """Stop the quantum scaling engine"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Quantum Scaling Engine stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        while self._running:
            try:
                # Collect current metrics
                current_metrics = await self._collect_metrics()
                self.current_metrics = current_metrics
                
                # Add to predictor
                self.predictor.add_metrics(current_metrics)
                
                # Make scaling decisions if auto-scaling is enabled
                if self.auto_scaling_enabled:
                    await self._make_scaling_decisions(current_metrics)
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second monitoring cycle
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        # Get worker pool metrics
        worker_metrics = self.worker_pool.get_metrics()
        
        # Get cache metrics
        cache_metrics = self.cache_manager.get_metrics()
        
        # Get network metrics (simplified)
        network_stats = psutil.net_io_counters()
        
        return ResourceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory_info.percent,
            worker_utilization=worker_metrics['utilization'],
            connection_count=worker_metrics['active_tasks'],
            cache_hit_rate=cache_metrics['hit_rate'],
            throughput=worker_metrics['completed_tasks'],
            response_time=worker_metrics.get('average_execution_time', 0) * 1000,  # Convert to ms
            error_rate=1.0 - worker_metrics['success_rate']
        )
    
    async def _make_scaling_decisions(self, current_metrics: ResourceMetrics):
        """Make intelligent scaling decisions"""
        # Get predictions
        predictions = self.predictor.predict_demand(horizon_minutes=15)
        
        # Define scaling constraints
        constraints = {
            'cpu_scale': (0.5, 3.0),
            'memory_scale': (0.8, 2.5),
            'worker_scale': (0.5, 5.0),
            'cache_scale': (0.5, 10.0),
            'connection_scale': (0.5, 3.0)
        }
        
        # Get optimal scaling parameters
        if self.strategy in [ScalingStrategy.QUANTUM_OPTIMIZED, ScalingStrategy.HYBRID]:
            scaling_params = self.optimizer.optimize_scaling(current_metrics, predictions, constraints)
        else:
            scaling_params = self._simple_scaling_logic(current_metrics, predictions)
        
        # Apply scaling decisions
        await self._apply_scaling_decisions(scaling_params, current_metrics, predictions)
    
    def _simple_scaling_logic(self, metrics: ResourceMetrics, 
                            predictions: Dict[str, float]) -> Dict[ResourceType, float]:
        """Simple reactive scaling logic"""
        scaling_decisions = {}
        
        # CPU scaling
        if metrics.cpu_percent > 80:
            scaling_decisions[ResourceType.CPU] = 1.5
        elif metrics.cpu_percent < 30:
            scaling_decisions[ResourceType.CPU] = 0.8
        else:
            scaling_decisions[ResourceType.CPU] = 1.0
        
        # Worker scaling
        if metrics.worker_utilization > 0.8:
            scaling_decisions[ResourceType.WORKERS] = 1.5
        elif metrics.worker_utilization < 0.3:
            scaling_decisions[ResourceType.WORKERS] = 0.8
        else:
            scaling_decisions[ResourceType.WORKERS] = 1.0
        
        # Cache scaling
        if metrics.cache_hit_rate < 0.7:
            scaling_decisions[ResourceType.CACHE] = 1.3
        else:
            scaling_decisions[ResourceType.CACHE] = 1.0
        
        return scaling_decisions
    
    async def _apply_scaling_decisions(self, scaling_params: Dict[ResourceType, float],
                                     current_metrics: ResourceMetrics,
                                     predictions: Dict[str, float]):
        """Apply scaling decisions to system resources"""
        
        for resource_type, scale_factor in scaling_params.items():
            if abs(scale_factor - 1.0) < 0.1:  # No significant scaling needed
                continue
            
            try:
                current_value = self._get_current_resource_value(resource_type)
                target_value = current_value * scale_factor
                
                # Create scaling decision record
                decision = ScalingDecision(
                    timestamp=datetime.now(timezone.utc),
                    resource_type=resource_type,
                    current_value=current_value,
                    target_value=target_value,
                    scaling_factor=scale_factor,
                    reasoning=self._generate_scaling_reasoning(resource_type, current_metrics, predictions),
                    confidence=0.8,  # Would be calculated based on prediction accuracy
                    estimated_impact={'performance': 0.1, 'cost': 0.05}
                )
                
                # Apply scaling
                await self._scale_resource(resource_type, target_value)
                
                self.scaling_history.append(decision)
                logger.info(f"Scaled {resource_type.value} from {current_value:.1f} to {target_value:.1f}")
                
            except Exception as e:
                logger.error(f"Failed to scale {resource_type.value}: {e}")
    
    def _get_current_resource_value(self, resource_type: ResourceType) -> float:
        """Get current value for a resource type"""
        if resource_type == ResourceType.WORKERS:
            return float(self.worker_pool.current_workers)
        elif resource_type == ResourceType.CACHE:
            return self.cache_manager.current_size_mb
        elif resource_type == ResourceType.CPU:
            return psutil.cpu_count()  # Available CPUs
        elif resource_type == ResourceType.MEMORY:
            return psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
        else:
            return 1.0
    
    async def _scale_resource(self, resource_type: ResourceType, target_value: float):
        """Scale specific resource to target value"""
        if resource_type == ResourceType.WORKERS:
            self.worker_pool.scale_workers(int(target_value))
        elif resource_type == ResourceType.CACHE:
            self.cache_manager.scale_cache(int(target_value))
        # CPU and memory scaling would require container/cluster orchestration
        # This would integrate with Kubernetes, Docker Swarm, etc.
    
    def _generate_scaling_reasoning(self, resource_type: ResourceType,
                                  metrics: ResourceMetrics,
                                  predictions: Dict[str, float]) -> str:
        """Generate human-readable reasoning for scaling decision"""
        if resource_type == ResourceType.WORKERS:
            if metrics.worker_utilization > 0.8:
                return f"Worker utilization at {metrics.worker_utilization:.1%}, scaling up to handle load"
            else:
                return f"Worker utilization at {metrics.worker_utilization:.1%}, scaling down to optimize costs"
        elif resource_type == ResourceType.CACHE:
            if metrics.cache_hit_rate < 0.7:
                return f"Cache hit rate at {metrics.cache_hit_rate:.1%}, increasing cache size to improve performance"
            else:
                return f"Cache performing well at {metrics.cache_hit_rate:.1%}"
        else:
            return f"Scaling {resource_type.value} based on predictive analysis"
    
    def get_scaling_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive scaling dashboard data"""
        recent_decisions = self.scaling_history[-10:] if self.scaling_history else []
        
        # Calculate scaling effectiveness
        effectiveness = statistics.mean(self.scaling_effectiveness) if self.scaling_effectiveness else 0.5
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'strategy': self.strategy.value,
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'current_metrics': asdict(self.current_metrics) if self.current_metrics else {},
            'worker_pool': self.worker_pool.get_metrics(),
            'cache_manager': self.cache_manager.get_metrics(),
            'recent_decisions': [asdict(d) for d in recent_decisions],
            'scaling_effectiveness': effectiveness,
            'predictions': self.predictor.predict_demand() if len(self.predictor.metrics_history) > 5 else {},
            'system_status': self._get_system_health_status()
        }
    
    def _get_system_health_status(self) -> str:
        """Get overall system health status"""
        if not self.current_metrics:
            return 'unknown'
        
        # Check various health indicators
        health_score = 1.0
        
        if self.current_metrics.cpu_percent > 90:
            health_score -= 0.3
        if self.current_metrics.memory_percent > 90:
            health_score -= 0.3
        if self.current_metrics.error_rate > 0.05:
            health_score -= 0.2
        if self.current_metrics.response_time > 2000:  # 2 seconds
            health_score -= 0.2
        
        if health_score > 0.8:
            return 'excellent'
        elif health_score > 0.6:
            return 'good'
        elif health_score > 0.4:
            return 'fair'
        else:
            return 'poor'

# Example usage and testing
async def test_quantum_scaling():
    """Test the quantum scaling engine"""
    engine = QuantumScalingEngine(ScalingStrategy.HYBRID)
    
    print("Quantum Scaling Engine Test")
    print("=" * 40)
    
    # Start engine
    await engine.start_engine()
    
    # Simulate some work load
    async def mock_workload():
        """Simulate varying workload"""
        for i in range(10):
            # Submit mock tasks
            task_count = 5 + (i % 3) * 2  # Varying load
            
            for _ in range(task_count):
                await engine.worker_pool.submit_task(time.sleep, 0.1)
            
            # Test cache
            for j in range(5):
                key = f"test_key_{j}"
                if engine.cache_manager.get(key) is None:
                    engine.cache_manager.set(key, f"value_{j}", 1024)
            
            await asyncio.sleep(2)
    
    # Run workload simulation
    workload_task = asyncio.create_task(mock_workload())
    
    # Monitor for a bit
    await asyncio.sleep(20)
    
    # Get dashboard
    dashboard = engine.get_scaling_dashboard()
    
    print("\nScaling Dashboard:")
    print(json.dumps(dashboard, indent=2, default=str))
    
    # Stop engine
    await engine.stop_engine()
    workload_task.cancel()
    
    return True

if __name__ == "__main__":
    asyncio.run(test_quantum_scaling())