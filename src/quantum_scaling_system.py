#!/usr/bin/env python3
"""
QUANTUM SCALING SYSTEM - Generation 3
Advanced load balancing, auto-scaling, and distributed processing
"""

import asyncio
import json
import time
import random
import statistics
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import heapq
import threading

from src.logger import get_logger
from src.advanced_performance_engine import IntelligentCache, AsyncResourcePool, ConcurrentTaskProcessor

logger = get_logger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    QUANTUM_ADAPTIVE = "quantum_adaptive"
    HYBRID = "hybrid"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESOURCE_BASED = "resource_based"
    QUANTUM_OPTIMIZED = "quantum_optimized"


@dataclass
class Node:
    """Processing node representation"""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    health_score: float = 1.0
    last_health_check: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    is_available: bool = True
    scaling_preference: float = 1.0  # Quantum scaling factor
    
    def calculate_load_score(self) -> float:
        """Calculate composite load score"""
        connection_ratio = self.current_connections / max(self.max_connections, 1)
        resource_usage = (self.cpu_usage + self.memory_usage) / 200.0
        latency_penalty = min(self.network_latency / 1000.0, 1.0)  # Cap at 1 second
        
        load_score = (connection_ratio * 0.4 + 
                     resource_usage * 0.4 + 
                     latency_penalty * 0.2) * (2.0 - self.health_score)
        
        return max(load_score, 0.0)


@dataclass
class ScalingMetrics:
    """System scaling metrics"""
    timestamp: float
    total_requests_per_second: float
    average_response_time: float
    cpu_usage_percent: float
    memory_usage_percent: float
    active_connections: int
    queue_length: int
    error_rate: float
    throughput: float
    
    def __post_init__(self):
        self.composite_load = self._calculate_composite_load()
    
    def _calculate_composite_load(self) -> float:
        """Calculate composite system load score"""
        # Normalize metrics to 0-1 scale
        rps_load = min(self.total_requests_per_second / 1000.0, 1.0)
        response_load = min(self.average_response_time / 5000.0, 1.0)  # 5s max
        cpu_load = self.cpu_usage_percent / 100.0
        memory_load = self.memory_usage_percent / 100.0
        queue_load = min(self.queue_length / 1000.0, 1.0)
        error_load = min(self.error_rate * 10, 1.0)
        
        # Weighted composite score
        return (rps_load * 0.2 + 
                response_load * 0.2 + 
                cpu_load * 0.25 + 
                memory_load * 0.15 +
                queue_load * 0.15 +
                error_load * 0.05)


class QuantumLoadBalancer:
    """Advanced load balancer with quantum-inspired optimization"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.QUANTUM_OPTIMIZED):
        self.algorithm = algorithm
        self.nodes: List[Node] = []
        self.current_index = 0  # For round-robin
        self.connection_counts = defaultdict(int)
        self.request_history = deque(maxlen=10000)
        self.quantum_state = {}  # Quantum-inspired state tracking
        self.load_prediction_model = None
        self._lock = threading.RLock()
        
        # Initialize quantum optimization
        self._initialize_quantum_optimization()
    
    def _initialize_quantum_optimization(self):
        """Initialize quantum-inspired optimization parameters"""
        self.quantum_state = {
            "entanglement_matrix": {},  # Node relationship matrix
            "coherence_factors": {},    # Node coherence with system state
            "superposition_weights": {},  # Dynamic weight superposition
            "observation_history": deque(maxlen=1000),
            "optimization_cycles": 0
        }
    
    def add_node(self, node: Node):
        """Add a node to the load balancer"""
        with self._lock:
            self.nodes.append(node)
            self.connection_counts[node.id] = 0
            self._update_quantum_entanglement(node)
            logger.info(f"Added node {node.id} to load balancer")
    
    def remove_node(self, node_id: str):
        """Remove a node from the load balancer"""
        with self._lock:
            self.nodes = [n for n in self.nodes if n.id != node_id]
            self.connection_counts.pop(node_id, None)
            self._cleanup_quantum_state(node_id)
            logger.info(f"Removed node {node_id} from load balancer")
    
    def select_node(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[Node]:
        """Select optimal node using specified algorithm"""
        with self._lock:
            available_nodes = [n for n in self.nodes if n.is_available and n.health_score > 0.3]
            
            if not available_nodes:
                return None
            
            if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                return self._round_robin_selection(available_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
                return self._resource_based_selection(available_nodes)
            else:  # QUANTUM_OPTIMIZED
                return self._quantum_optimized_selection(available_nodes, request_context)
    
    def _round_robin_selection(self, nodes: List[Node]) -> Node:
        """Simple round-robin selection"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index = (self.current_index + 1) % len(nodes)
        return node
    
    def _weighted_round_robin_selection(self, nodes: List[Node]) -> Node:
        """Weighted round-robin based on node weights"""
        total_weight = sum(n.weight * n.health_score for n in nodes)
        if total_weight == 0:
            return nodes[0]
        
        target = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for node in nodes:
            cumulative_weight += node.weight * node.health_score
            if cumulative_weight >= target:
                return node
        
        return nodes[-1]
    
    def _least_connections_selection(self, nodes: List[Node]) -> Node:
        """Select node with least connections"""
        return min(nodes, key=lambda n: n.current_connections / max(n.weight, 0.1))
    
    def _resource_based_selection(self, nodes: List[Node]) -> Node:
        """Select node based on resource utilization"""
        return min(nodes, key=lambda n: n.calculate_load_score())
    
    def _quantum_optimized_selection(self, nodes: List[Node], request_context: Optional[Dict[str, Any]]) -> Node:
        """Quantum-inspired optimized node selection"""
        # Update quantum state with current observation
        self._update_quantum_observation(nodes, request_context)
        
        # Calculate quantum selection probabilities
        probabilities = self._calculate_quantum_probabilities(nodes, request_context)
        
        # Apply quantum superposition principle
        selection_weights = []
        for i, node in enumerate(nodes):
            # Base selection weight
            base_weight = 1.0 / max(node.calculate_load_score(), 0.01)
            
            # Quantum coherence factor
            coherence = self.quantum_state["coherence_factors"].get(node.id, 1.0)
            
            # Probability amplification
            quantum_probability = probabilities.get(node.id, 1.0 / len(nodes))
            
            # Entanglement influence from other nodes
            entanglement_factor = self._calculate_entanglement_influence(node, nodes)
            
            # Combined quantum weight
            quantum_weight = (base_weight * coherence * quantum_probability * entanglement_factor)
            selection_weights.append(quantum_weight)
        
        # Quantum measurement (probabilistic selection)
        total_weight = sum(selection_weights)
        if total_weight == 0:
            return random.choice(nodes)
        
        target = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(selection_weights):
            cumulative += weight
            if cumulative >= target:
                selected_node = nodes[i]
                self._record_quantum_selection(selected_node, request_context)
                return selected_node
        
        return nodes[-1]
    
    def _calculate_quantum_probabilities(self, nodes: List[Node], request_context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quantum selection probabilities"""
        probabilities = {}
        
        # Historical success rate influence
        for node in nodes:
            success_rate = 1.0 - (node.failed_requests / max(node.total_requests, 1))
            response_time_factor = 1.0 / max(node.average_response_time / 1000.0, 0.1)
            health_factor = node.health_score
            
            # Quantum probability calculation
            base_probability = success_rate * response_time_factor * health_factor
            
            # Context-aware adjustment
            context_adjustment = self._calculate_context_adjustment(node, request_context)
            
            probabilities[node.id] = base_probability * context_adjustment
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for node_id in probabilities:
                probabilities[node_id] /= total_prob
        
        return probabilities
    
    def _calculate_context_adjustment(self, node: Node, request_context: Optional[Dict[str, Any]]) -> float:
        """Calculate context-aware adjustment factor"""
        if not request_context:
            return 1.0
        
        adjustment = 1.0
        
        # Request type preference
        request_type = request_context.get("type", "default")
        if request_type == "cpu_intensive" and node.cpu_usage < 60:
            adjustment *= 1.2
        elif request_type == "memory_intensive" and node.memory_usage < 70:
            adjustment *= 1.2
        elif request_type == "io_bound" and node.network_latency < 50:
            adjustment *= 1.1
        
        # Geographic preference (if available)
        client_region = request_context.get("region")
        node_region = getattr(node, 'region', None)
        if client_region and node_region and client_region == node_region:
            adjustment *= 1.3
        
        return adjustment
    
    def _calculate_entanglement_influence(self, node: Node, all_nodes: List[Node]) -> float:
        """Calculate quantum entanglement influence"""
        entanglement_factor = 1.0
        
        for other_node in all_nodes:
            if other_node.id == node.id:
                continue
            
            # Get entanglement strength
            entanglement_key = f"{node.id}:{other_node.id}"
            entanglement_strength = self.quantum_state["entanglement_matrix"].get(entanglement_key, 0.0)
            
            # Influence based on other node's state
            other_load = other_node.calculate_load_score()
            if other_load > 0.8:  # High load on entangled node
                entanglement_factor *= (1.0 - entanglement_strength * 0.1)
            elif other_load < 0.3:  # Low load on entangled node
                entanglement_factor *= (1.0 + entanglement_strength * 0.1)
        
        return max(entanglement_factor, 0.5)
    
    def _update_quantum_entanglement(self, new_node: Node):
        """Update quantum entanglement matrix with new node"""
        for existing_node in self.nodes:
            if existing_node.id != new_node.id:
                # Calculate initial entanglement based on node similarity
                similarity = self._calculate_node_similarity(new_node, existing_node)
                entanglement_strength = similarity * 0.5  # Scale to 0-0.5 range
                
                key1 = f"{new_node.id}:{existing_node.id}"
                key2 = f"{existing_node.id}:{new_node.id}"
                
                self.quantum_state["entanglement_matrix"][key1] = entanglement_strength
                self.quantum_state["entanglement_matrix"][key2] = entanglement_strength
    
    def _calculate_node_similarity(self, node1: Node, node2: Node) -> float:
        """Calculate similarity between two nodes"""
        # Similarity based on capabilities and characteristics
        weight_similarity = 1.0 - abs(node1.weight - node2.weight) / max(node1.weight + node2.weight, 1.0)
        capacity_similarity = 1.0 - abs(node1.max_connections - node2.max_connections) / max(node1.max_connections + node2.max_connections, 1.0)
        
        return (weight_similarity + capacity_similarity) / 2.0
    
    def _update_quantum_observation(self, nodes: List[Node], request_context: Optional[Dict[str, Any]]):
        """Update quantum state with current system observation"""
        observation = {
            "timestamp": time.time(),
            "total_nodes": len(nodes),
            "available_nodes": len([n for n in nodes if n.is_available]),
            "average_load": statistics.mean([n.calculate_load_score() for n in nodes]),
            "system_health": statistics.mean([n.health_score for n in nodes]),
            "request_context": request_context
        }
        
        self.quantum_state["observation_history"].append(observation)
        
        # Update coherence factors based on recent performance
        self._update_coherence_factors(nodes)
        
        # Periodic quantum optimization
        self.quantum_state["optimization_cycles"] += 1
        if self.quantum_state["optimization_cycles"] % 100 == 0:
            self._optimize_quantum_parameters()
    
    def _update_coherence_factors(self, nodes: List[Node]):
        """Update quantum coherence factors for each node"""
        for node in nodes:
            # Calculate coherence based on performance stability
            recent_observations = list(self.quantum_state["observation_history"])[-10:]
            
            if len(recent_observations) < 2:
                coherence = 1.0
            else:
                # Calculate performance variance as coherence measure
                load_scores = []
                for obs in recent_observations:
                    # Find this node's load in historical observations (simplified)
                    load_scores.append(node.calculate_load_score())
                
                variance = statistics.variance(load_scores) if len(load_scores) > 1 else 0
                coherence = 1.0 / (1.0 + variance)  # Higher variance = lower coherence
            
            self.quantum_state["coherence_factors"][node.id] = coherence
    
    def _record_quantum_selection(self, selected_node: Node, request_context: Optional[Dict[str, Any]]):
        """Record quantum selection for learning"""
        selection_record = {
            "node_id": selected_node.id,
            "timestamp": time.time(),
            "load_score": selected_node.calculate_load_score(),
            "context": request_context
        }
        
        # Store in request history for pattern learning
        self.request_history.append(selection_record)
    
    def _optimize_quantum_parameters(self):
        """Periodically optimize quantum parameters"""
        logger.info("Optimizing quantum load balancing parameters")
        
        # Analyze recent performance patterns
        recent_requests = list(self.request_history)[-1000:]
        
        if len(recent_requests) < 100:
            return
        
        # Calculate node success patterns
        node_performance = defaultdict(list)
        for request in recent_requests:
            node_id = request["node_id"]
            # Simplified success metric (in real implementation, would track actual outcomes)
            success_score = 1.0 / max(request["load_score"], 0.1)
            node_performance[node_id].append(success_score)
        
        # Update quantum weights based on performance
        for node_id, scores in node_performance.items():
            if len(scores) > 10:
                avg_performance = statistics.mean(scores)
                current_coherence = self.quantum_state["coherence_factors"].get(node_id, 1.0)
                
                # Adaptive coherence adjustment
                new_coherence = (current_coherence * 0.9) + (avg_performance * 0.1)
                self.quantum_state["coherence_factors"][node_id] = new_coherence
    
    def _cleanup_quantum_state(self, node_id: str):
        """Clean up quantum state for removed node"""
        # Remove from coherence factors
        self.quantum_state["coherence_factors"].pop(node_id, None)
        
        # Remove from entanglement matrix
        keys_to_remove = []
        for key in self.quantum_state["entanglement_matrix"]:
            if node_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.quantum_state["entanglement_matrix"].pop(key, None)
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update node metrics for load balancing decisions"""
        with self._lock:
            for node in self.nodes:
                if node.id == node_id:
                    node.cpu_usage = metrics.get("cpu_usage", node.cpu_usage)
                    node.memory_usage = metrics.get("memory_usage", node.memory_usage)
                    node.network_latency = metrics.get("network_latency", node.network_latency)
                    node.current_connections = metrics.get("current_connections", node.current_connections)
                    node.health_score = metrics.get("health_score", node.health_score)
                    node.average_response_time = metrics.get("average_response_time", node.average_response_time)
                    node.is_available = metrics.get("is_available", node.is_available)
                    node.last_health_check = time.time()
                    break
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        with self._lock:
            node_stats = []
            for node in self.nodes:
                node_stats.append({
                    "id": node.id,
                    "load_score": node.calculate_load_score(),
                    "health_score": node.health_score,
                    "current_connections": node.current_connections,
                    "total_requests": node.total_requests,
                    "failed_requests": node.failed_requests,
                    "is_available": node.is_available,
                    "quantum_coherence": self.quantum_state["coherence_factors"].get(node.id, 1.0)
                })
            
            return {
                "algorithm": self.algorithm.value,
                "total_nodes": len(self.nodes),
                "available_nodes": len([n for n in self.nodes if n.is_available]),
                "total_connections": sum(n.current_connections for n in self.nodes),
                "average_load": statistics.mean([n.calculate_load_score() for n in self.nodes]) if self.nodes else 0,
                "quantum_optimization_cycles": self.quantum_state["optimization_cycles"],
                "node_stats": node_stats
            }


class AutoScaler:
    """Intelligent auto-scaling system"""
    
    def __init__(self, 
                 strategy: ScalingStrategy = ScalingStrategy.QUANTUM_ADAPTIVE,
                 min_nodes: int = 2,
                 max_nodes: int = 20,
                 scale_up_threshold: float = 0.7,
                 scale_down_threshold: float = 0.3,
                 cooldown_period: float = 300.0):
        
        self.strategy = strategy
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_actions: deque = deque(maxlen=100)
        self.last_scaling_action = 0.0
        self.prediction_model = None
        
        # Quantum-adaptive parameters
        self.quantum_scaling_state = {
            "scaling_patterns": {},
            "optimization_weights": {"reactive": 0.4, "predictive": 0.3, "adaptive": 0.3},
            "performance_feedback": deque(maxlen=500)
        }
    
    def collect_metrics(self, 
                       load_balancer_stats: Dict[str, Any], 
                       system_metrics: Dict[str, Any]) -> ScalingMetrics:
        """Collect system metrics for scaling decisions"""
        
        metrics = ScalingMetrics(
            timestamp=time.time(),
            total_requests_per_second=system_metrics.get("requests_per_second", 0.0),
            average_response_time=system_metrics.get("average_response_time", 0.0),
            cpu_usage_percent=system_metrics.get("cpu_usage", 0.0),
            memory_usage_percent=system_metrics.get("memory_usage", 0.0),
            active_connections=load_balancer_stats.get("total_connections", 0),
            queue_length=system_metrics.get("queue_length", 0),
            error_rate=system_metrics.get("error_rate", 0.0),
            throughput=system_metrics.get("throughput", 0.0)
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def should_scale(self, current_metrics: ScalingMetrics, current_node_count: int) -> Tuple[bool, str, int]:
        """Determine if scaling action is needed"""
        
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return False, "cooldown", 0
        
        if self.strategy == ScalingStrategy.REACTIVE:
            return self._reactive_scaling_decision(current_metrics, current_node_count)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return self._predictive_scaling_decision(current_metrics, current_node_count)
        elif self.strategy == ScalingStrategy.QUANTUM_ADAPTIVE:
            return self._quantum_adaptive_scaling_decision(current_metrics, current_node_count)
        else:  # HYBRID
            return self._hybrid_scaling_decision(current_metrics, current_node_count)
    
    def _reactive_scaling_decision(self, metrics: ScalingMetrics, node_count: int) -> Tuple[bool, str, int]:
        """Reactive scaling based on current load"""
        load = metrics.composite_load
        
        if load > self.scale_up_threshold and node_count < self.max_nodes:
            # Scale up
            scale_factor = math.ceil((load - self.scale_up_threshold) / 0.2)
            new_nodes = min(scale_factor, self.max_nodes - node_count)
            return True, "scale_up", new_nodes
            
        elif load < self.scale_down_threshold and node_count > self.min_nodes:
            # Scale down
            scale_factor = math.ceil((self.scale_down_threshold - load) / 0.2)
            remove_nodes = min(scale_factor, node_count - self.min_nodes)
            return True, "scale_down", remove_nodes
        
        return False, "no_action", 0
    
    def _predictive_scaling_decision(self, metrics: ScalingMetrics, node_count: int) -> Tuple[bool, str, int]:
        """Predictive scaling based on trend analysis"""
        if len(self.metrics_history) < 10:
            return self._reactive_scaling_decision(metrics, node_count)
        
        # Analyze trends
        recent_metrics = list(self.metrics_history)[-10:]
        load_trend = self._calculate_trend([m.composite_load for m in recent_metrics])
        rps_trend = self._calculate_trend([m.total_requests_per_second for m in recent_metrics])
        
        # Predict future load
        current_load = metrics.composite_load
        predicted_load = current_load + (load_trend * 5)  # Predict 5 periods ahead
        
        if predicted_load > self.scale_up_threshold and node_count < self.max_nodes:
            # Preemptive scale up
            scale_factor = math.ceil((predicted_load - self.scale_up_threshold) / 0.2)
            new_nodes = min(scale_factor, self.max_nodes - node_count)
            return True, "predictive_scale_up", new_nodes
            
        elif predicted_load < self.scale_down_threshold and node_count > self.min_nodes:
            # Preemptive scale down
            scale_factor = math.ceil((self.scale_down_threshold - predicted_load) / 0.2)
            remove_nodes = min(scale_factor, node_count - self.min_nodes)
            return True, "predictive_scale_down", remove_nodes
        
        return False, "no_action", 0
    
    def _quantum_adaptive_scaling_decision(self, metrics: ScalingMetrics, node_count: int) -> Tuple[bool, str, int]:
        """Quantum-adaptive scaling with multiple strategy fusion"""
        
        # Get decisions from multiple strategies
        reactive_decision = self._reactive_scaling_decision(metrics, node_count)
        predictive_decision = self._predictive_scaling_decision(metrics, node_count)
        
        # Quantum superposition of scaling decisions
        scaling_probabilities = self._calculate_scaling_probabilities(reactive_decision, predictive_decision, metrics)
        
        # Adaptive weight adjustment based on recent performance
        self._update_adaptive_weights()
        
        # Quantum measurement (decision collapse)
        final_decision = self._quantum_decision_collapse(scaling_probabilities, node_count)
        
        return final_decision
    
    def _hybrid_scaling_decision(self, metrics: ScalingMetrics, node_count: int) -> Tuple[bool, str, int]:
        """Hybrid scaling combining multiple strategies"""
        reactive_decision = self._reactive_scaling_decision(metrics, node_count)
        predictive_decision = self._predictive_scaling_decision(metrics, node_count)
        
        # Simple hybrid logic - reactive takes precedence for immediate needs
        if reactive_decision[0]:
            return reactive_decision
        elif predictive_decision[0]:
            return predictive_decision
        else:
            return False, "no_action", 0
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_scaling_probabilities(self, 
                                       reactive_decision: Tuple[bool, str, int],
                                       predictive_decision: Tuple[bool, str, int],
                                       metrics: ScalingMetrics) -> Dict[str, float]:
        """Calculate quantum scaling probabilities"""
        
        probabilities = {
            "scale_up": 0.0,
            "scale_down": 0.0,
            "no_action": 0.8  # Default bias toward stability
        }
        
        weights = self.quantum_scaling_state["optimization_weights"]
        
        # Reactive component
        if reactive_decision[0]:
            action = reactive_decision[1]
            if "scale_up" in action:
                probabilities["scale_up"] += weights["reactive"]
            elif "scale_down" in action:
                probabilities["scale_down"] += weights["reactive"]
        
        # Predictive component
        if predictive_decision[0]:
            action = predictive_decision[1]
            if "scale_up" in action:
                probabilities["scale_up"] += weights["predictive"]
            elif "scale_down" in action:
                probabilities["scale_down"] += weights["predictive"]
        
        # Adaptive component based on system state
        adaptive_factor = self._calculate_adaptive_factor(metrics)
        probabilities["scale_up"] += adaptive_factor * weights["adaptive"]
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for action in probabilities:
                probabilities[action] /= total_prob
        
        return probabilities
    
    def _calculate_adaptive_factor(self, metrics: ScalingMetrics) -> float:
        """Calculate adaptive scaling factor based on system learning"""
        
        # Analysis of recent performance patterns
        if len(self.metrics_history) < 5:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-5:]
        
        # Performance stability analysis
        load_variance = statistics.variance([m.composite_load for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.average_response_time for m in recent_metrics])
        error_rate_trend = self._calculate_trend([m.error_rate for m in recent_metrics])
        
        # Adaptive factor calculation
        stability_factor = 1.0 / (1.0 + load_variance)
        performance_factor = 1.0 if response_time_trend <= 0 else -0.5
        reliability_factor = 1.0 if error_rate_trend <= 0 else -0.8
        
        adaptive_factor = (stability_factor + performance_factor + reliability_factor) / 3.0
        return max(min(adaptive_factor, 1.0), -1.0)
    
    def _quantum_decision_collapse(self, probabilities: Dict[str, float], node_count: int) -> Tuple[bool, str, int]:
        """Collapse quantum superposition to final scaling decision"""
        
        # Quantum measurement simulation
        random_value = random.random()
        cumulative_prob = 0.0
        
        for action, prob in probabilities.items():
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                if action == "scale_up" and node_count < self.max_nodes:
                    # Quantum-optimized scale-up count
                    scale_count = self._calculate_quantum_scale_count(probabilities["scale_up"], "up")
                    return True, "quantum_scale_up", min(scale_count, self.max_nodes - node_count)
                    
                elif action == "scale_down" and node_count > self.min_nodes:
                    # Quantum-optimized scale-down count
                    scale_count = self._calculate_quantum_scale_count(probabilities["scale_down"], "down")
                    return True, "quantum_scale_down", min(scale_count, node_count - self.min_nodes)
                
                break
        
        return False, "no_action", 0
    
    def _calculate_quantum_scale_count(self, probability: float, direction: str) -> int:
        """Calculate optimal scaling count using quantum-inspired optimization"""
        
        # Base scaling count from probability
        base_count = math.ceil(probability * 3)  # Scale 1-3 nodes typically
        
        # Quantum optimization based on historical patterns
        if len(self.scaling_actions) > 5:
            recent_actions = list(self.scaling_actions)[-5:]
            similar_actions = [a for a in recent_actions if direction in a["action"]]
            
            if similar_actions:
                avg_effectiveness = statistics.mean([a.get("effectiveness", 0.5) for a in similar_actions])
                base_count = math.ceil(base_count * (0.5 + avg_effectiveness))
        
        return max(1, min(base_count, 3))  # Cap at 1-3 nodes per action
    
    def _update_adaptive_weights(self):
        """Update quantum adaptive weights based on performance feedback"""
        
        if len(self.quantum_scaling_state["performance_feedback"]) < 10:
            return
        
        recent_feedback = list(self.quantum_scaling_state["performance_feedback"])[-10:]
        
        # Calculate strategy effectiveness
        reactive_effectiveness = statistics.mean([f.get("reactive_score", 0.5) for f in recent_feedback])
        predictive_effectiveness = statistics.mean([f.get("predictive_score", 0.5) for f in recent_feedback])
        adaptive_effectiveness = statistics.mean([f.get("adaptive_score", 0.5) for f in recent_feedback])
        
        # Update weights with exponential moving average
        alpha = 0.1  # Learning rate
        weights = self.quantum_scaling_state["optimization_weights"]
        
        weights["reactive"] = weights["reactive"] * (1 - alpha) + reactive_effectiveness * alpha
        weights["predictive"] = weights["predictive"] * (1 - alpha) + predictive_effectiveness * alpha
        weights["adaptive"] = weights["adaptive"] * (1 - alpha) + adaptive_effectiveness * alpha
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
    
    def record_scaling_action(self, action: str, node_count: int, effectiveness: float = 0.5):
        """Record scaling action for learning"""
        
        action_record = {
            "timestamp": time.time(),
            "action": action,
            "node_count": node_count,
            "effectiveness": effectiveness,
            "metrics": asdict(self.metrics_history[-1]) if self.metrics_history else {}
        }
        
        self.scaling_actions.append(action_record)
        self.last_scaling_action = time.time()
        
        # Update quantum performance feedback
        feedback = {
            "reactive_score": effectiveness if "reactive" in action else 0.5,
            "predictive_score": effectiveness if "predictive" in action else 0.5,
            "adaptive_score": effectiveness if "adaptive" in action or "quantum" in action else 0.5
        }
        
        self.quantum_scaling_state["performance_feedback"].append(feedback)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics"""
        
        recent_actions = list(self.scaling_actions)[-10:]
        
        return {
            "strategy": self.strategy.value,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "recent_actions": len(recent_actions),
            "last_scaling_action": self.last_scaling_action,
            "metrics_history_length": len(self.metrics_history),
            "quantum_weights": self.quantum_scaling_state["optimization_weights"],
            "average_effectiveness": statistics.mean([a["effectiveness"] for a in recent_actions]) if recent_actions else 0.5
        }


# Factory functions
def create_quantum_load_balancer(algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.QUANTUM_OPTIMIZED) -> QuantumLoadBalancer:
    """Create quantum-optimized load balancer"""
    return QuantumLoadBalancer(algorithm)


def create_auto_scaler(strategy: ScalingStrategy = ScalingStrategy.QUANTUM_ADAPTIVE, **kwargs) -> AutoScaler:
    """Create intelligent auto-scaler"""
    return AutoScaler(strategy, **kwargs)


# Demo
async def quantum_scaling_demo():
    """Demonstration of quantum scaling system"""
    logger.info("Starting quantum scaling system demo")
    
    # Create quantum load balancer
    load_balancer = create_quantum_load_balancer(LoadBalancingAlgorithm.QUANTUM_OPTIMIZED)
    
    # Add test nodes
    test_nodes = [
        Node(id="node1", host="10.0.1.1", port=8080, weight=1.0, max_connections=100),
        Node(id="node2", host="10.0.1.2", port=8080, weight=1.2, max_connections=120),
        Node(id="node3", host="10.0.1.3", port=8080, weight=0.8, max_connections=80),
    ]
    
    for node in test_nodes:
        load_balancer.add_node(node)
    
    # Simulate load balancing
    logger.info("Testing quantum load balancing...")
    
    for i in range(100):
        request_context = {
            "type": "io_bound" if i % 2 == 0 else "cpu_intensive",
            "region": "us-west" if i % 3 == 0 else "us-east"
        }
        
        selected_node = load_balancer.select_node(request_context)
        if selected_node:
            # Simulate request processing
            selected_node.current_connections += 1
            selected_node.total_requests += 1
            
            # Update metrics periodically
            if i % 10 == 0:
                load_balancer.update_node_metrics(selected_node.id, {
                    "cpu_usage": random.uniform(20, 80),
                    "memory_usage": random.uniform(30, 70),
                    "network_latency": random.uniform(10, 100),
                    "health_score": random.uniform(0.7, 1.0)
                })
    
    lb_stats = load_balancer.get_load_balancer_stats()
    logger.info(f"Load balancer processed {sum(n['total_requests'] for n in lb_stats['node_stats'])} requests")
    logger.info(f"Quantum optimization cycles: {lb_stats['quantum_optimization_cycles']}")
    
    # Test auto-scaler
    logger.info("Testing quantum auto-scaler...")
    auto_scaler = create_auto_scaler(ScalingStrategy.QUANTUM_ADAPTIVE, min_nodes=2, max_nodes=10)
    
    # Simulate system metrics over time
    for i in range(20):
        system_metrics = {
            "requests_per_second": random.uniform(100, 800),
            "average_response_time": random.uniform(50, 500),
            "cpu_usage": random.uniform(20, 90),
            "memory_usage": random.uniform(30, 85),
            "queue_length": random.randint(0, 50),
            "error_rate": random.uniform(0, 0.05),
            "throughput": random.uniform(500, 2000)
        }
        
        current_metrics = auto_scaler.collect_metrics(lb_stats, system_metrics)
        current_node_count = lb_stats["available_nodes"]
        
        should_scale, action, count = auto_scaler.should_scale(current_metrics, current_node_count)
        
        if should_scale:
            logger.info(f"Scaling decision: {action} ({count} nodes)")
            
            # Simulate scaling effectiveness
            effectiveness = random.uniform(0.6, 0.9)
            auto_scaler.record_scaling_action(action, current_node_count, effectiveness)
            
            # Update node count in load balancer stats simulation
            if "scale_up" in action:
                lb_stats["available_nodes"] = min(lb_stats["available_nodes"] + count, 10)
            elif "scale_down" in action:
                lb_stats["available_nodes"] = max(lb_stats["available_nodes"] - count, 2)
        
        # Simulate time passage
        await asyncio.sleep(0.1)
    
    scaler_stats = auto_scaler.get_scaling_stats()
    logger.info(f"Auto-scaler made {scaler_stats['recent_actions']} recent scaling decisions")
    logger.info(f"Average effectiveness: {scaler_stats['average_effectiveness']:.2f}")
    
    logger.info("Quantum scaling system demo completed")


if __name__ == "__main__":
    asyncio.run(quantum_scaling_demo())