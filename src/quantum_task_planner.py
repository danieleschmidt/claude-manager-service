"""
Quantum-Inspired Task Planner for Claude Manager Service

This module implements quantum computing principles for advanced task prioritization:
- Superposition-based multi-dimensional task analysis
- Quantum entanglement for task dependency modeling
- Quantum annealing for optimal task scheduling
- Quantum interference for priority pattern detection

Quantum Principles Applied:
- Superposition: Tasks exist in multiple priority states simultaneously
- Entanglement: Task dependencies create correlated priority states
- Interference: Pattern-based priority amplification/cancellation
- Annealing: Optimal task ordering through energy minimization
"""

import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

from .logger import get_logger
from .task_prioritization import (
    classify_task_type, 
    analyze_task_complexity, 
    determine_business_impact,
    assess_urgency,
    TASK_TYPE_PRIORITIES
)

logger = get_logger(__name__)


class QuantumState(Enum):
    """Quantum states for task prioritization"""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    COHERENT = "coherent"


@dataclass
class QuantumTask:
    """Quantum representation of a task with superposition properties"""
    id: str
    content: str
    file_path: str
    line_number: int
    task_type: str
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    
    # Quantum properties
    amplitude: complex = 0.0 + 0.0j
    phase: float = 0.0
    entangled_tasks: List[str] = None
    coherence_time: float = 1.0
    
    # Priority dimensions (quantum superposition)
    priority_vector: List[float] = None
    uncertainty: float = 0.0
    
    def __post_init__(self):
        if self.entangled_tasks is None:
            self.entangled_tasks = []
        if self.priority_vector is None:
            self.priority_vector = [0.0] * 8  # 8-dimensional priority space


class QuantumTaskPlanner:
    """
    Quantum-inspired task planner implementing advanced quantum algorithms
    for optimal task prioritization and scheduling
    """
    
    def __init__(self, enable_quantum_annealing: bool = True):
        self.logger = get_logger(f"{__name__}.QuantumTaskPlanner")
        self.enable_quantum_annealing = enable_quantum_annealing
        
        # Quantum system parameters
        self.planck_constant = 6.62607015e-34
        self.temperature = 300.0  # Kelvin (room temperature)
        self.cooling_rate = 0.95
        self.min_temperature = 0.01
        
        # Priority dimensions (8D quantum space)
        self.priority_dimensions = [
            "business_impact",
            "urgency", 
            "complexity",
            "risk_factor",
            "dependency_weight",
            "resource_availability",
            "strategic_alignment",
            "innovation_potential"
        ]
        
        # Quantum entanglement patterns
        self.entanglement_patterns = {}
        
        self.logger.info("Quantum Task Planner initialized with quantum annealing")
    
    def create_quantum_task(self, task_data: Dict[str, Any]) -> QuantumTask:
        """
        Convert a classical task into a quantum task with superposition properties
        """
        task_type = task_data.get('type') or classify_task_type(
            task_data.get('content', ''), 
            task_data.get('file_path', '')
        )
        
        quantum_task = QuantumTask(
            id=task_data.get('id', f"task_{random.randint(1000, 9999)}"),
            content=task_data.get('content', ''),
            file_path=task_data.get('file_path', ''),
            line_number=task_data.get('line_number', 0),
            task_type=task_type
        )
        
        # Initialize quantum properties
        self._initialize_quantum_state(quantum_task)
        
        return quantum_task
    
    def _initialize_quantum_state(self, task: QuantumTask):
        """Initialize quantum state with superposition of priorities"""
        
        # Calculate base priority components
        business_impact = determine_business_impact(task.content, task.file_path)
        urgency = assess_urgency(task.content)
        complexity = analyze_task_complexity(task.content, task.file_path)
        
        # Quantum enhancements
        risk_factor = self._calculate_quantum_risk(task)
        dependency_weight = self._analyze_quantum_dependencies(task)
        resource_availability = self._assess_resource_quantum_state(task)
        strategic_alignment = self._calculate_strategic_quantum_alignment(task)
        innovation_potential = self._assess_quantum_innovation_potential(task)
        
        # Create 8-dimensional priority vector in superposition
        task.priority_vector = [
            business_impact,
            urgency, 
            complexity,
            risk_factor,
            dependency_weight,
            resource_availability,
            strategic_alignment,
            innovation_potential
        ]
        
        # Calculate quantum amplitude and phase
        magnitude = math.sqrt(sum(p**2 for p in task.priority_vector))
        phase = math.atan2(sum(task.priority_vector[::2]), sum(task.priority_vector[1::2]))
        
        task.amplitude = magnitude * (math.cos(phase) + 1j * math.sin(phase))
        task.phase = phase
        
        # Uncertainty principle: higher complexity = higher uncertainty
        task.uncertainty = complexity / 10.0
        
        # Coherence time based on task stability
        task.coherence_time = max(0.1, 1.0 - task.uncertainty)
        
        self.logger.debug(f"Initialized quantum task {task.id} with amplitude {task.amplitude:.3f}")
    
    def _calculate_quantum_risk(self, task: QuantumTask) -> float:
        """Calculate quantum risk factor using uncertainty principle"""
        
        risk_indicators = {
            'security': 9.0,
            'production': 8.0,
            'database': 7.0,
            'migration': 7.0,
            'external': 6.0,
            'integration': 6.0,
            'breaking': 8.0,
            'deprecated': 5.0
        }
        
        content_lower = task.content.lower()
        file_lower = task.file_path.lower()
        
        base_risk = 3.0
        risk_amplification = 0.0
        
        # Content-based risk
        for indicator, weight in risk_indicators.items():
            if indicator in content_lower:
                risk_amplification += weight
        
        # File-based risk
        critical_paths = ['auth', 'security', 'payment', 'api', 'database']
        for path in critical_paths:
            if path in file_lower:
                risk_amplification += 2.0
                break
        
        # Quantum interference pattern for risk
        interference = math.sin(len(task.content) * 0.1) * 0.5
        
        total_risk = base_risk + min(risk_amplification, 6.0) + interference
        
        return min(max(total_risk, 1.0), 10.0)
    
    def _analyze_quantum_dependencies(self, task: QuantumTask) -> float:
        """Analyze task dependencies using quantum entanglement principles"""
        
        dependency_keywords = {
            'depends': 3.0,
            'requires': 3.0,
            'after': 2.0,
            'before': 2.0,
            'blocks': 4.0,
            'blocked': 4.0,
            'prerequisite': 3.5,
            'related': 1.5
        }
        
        content_lower = task.content.lower()
        dependency_weight = 1.0
        
        for keyword, weight in dependency_keywords.items():
            if keyword in content_lower:
                dependency_weight += weight
        
        # File co-location suggests dependency entanglement
        if 'test' in task.file_path.lower():
            dependency_weight += 1.0
        
        # Apply quantum entanglement decay
        entanglement_decay = math.exp(-dependency_weight / 5.0)
        quantum_dependency = dependency_weight * (1 + entanglement_decay)
        
        return min(max(quantum_dependency, 1.0), 10.0)
    
    def _assess_resource_quantum_state(self, task: QuantumTask) -> float:
        """Assess resource availability using quantum superposition"""
        
        # Base resource availability (simulated)
        base_availability = 7.0
        
        # Task complexity affects resource requirements
        complexity_factor = analyze_task_complexity(task.content, task.file_path)
        resource_demand = complexity_factor / 10.0
        
        # Quantum superposition of resource states
        available_state = base_availability * (1 - resource_demand * 0.3)
        constrained_state = base_availability * (1 - resource_demand * 0.8)
        
        # Probabilistic resource state collapse
        probability_available = 0.7  # 70% chance of good resource availability
        
        quantum_availability = (
            probability_available * available_state + 
            (1 - probability_available) * constrained_state
        )
        
        return min(max(quantum_availability, 1.0), 10.0)
    
    def _calculate_strategic_quantum_alignment(self, task: QuantumTask) -> float:
        """Calculate strategic alignment using quantum coherence"""
        
        strategic_keywords = {
            'optimization': 7.0,
            'performance': 6.0,
            'scalability': 8.0,
            'security': 9.0,
            'user experience': 7.0,
            'automation': 6.0,
            'ai': 8.0,
            'machine learning': 8.0,
            'innovation': 7.0,
            'modernization': 6.0
        }
        
        content_lower = task.content.lower()
        alignment_score = 4.0  # Base alignment
        
        for keyword, weight in strategic_keywords.items():
            if keyword in content_lower:
                alignment_score += weight * 0.3
        
        # Quantum coherence bonus for aligned tasks
        coherence_bonus = math.cos(alignment_score * 0.5) * 0.5
        
        quantum_alignment = alignment_score + coherence_bonus
        
        return min(max(quantum_alignment, 1.0), 10.0)
    
    def _assess_quantum_innovation_potential(self, task: QuantumTask) -> float:
        """Assess innovation potential using quantum tunneling principles"""
        
        innovation_indicators = {
            'new': 3.0,
            'novel': 4.0,
            'innovative': 5.0,
            'creative': 3.0,
            'experimental': 4.0,
            'prototype': 3.5,
            'research': 4.0,
            'explore': 3.0,
            'discover': 3.5,
            'breakthrough': 5.0
        }
        
        content_lower = task.content.lower()
        innovation_score = 2.0  # Base innovation potential
        
        for indicator, weight in innovation_indicators.items():
            if indicator in content_lower:
                innovation_score += weight * 0.5
        
        # Quantum tunneling effect for breakthrough potential
        tunneling_probability = math.exp(-innovation_score / 3.0)
        breakthrough_bonus = tunneling_probability * 2.0
        
        quantum_innovation = innovation_score + breakthrough_bonus
        
        return min(max(quantum_innovation, 1.0), 10.0)
    
    def create_quantum_entanglement(self, task1: QuantumTask, task2: QuantumTask) -> float:
        """Create quantum entanglement between related tasks"""
        
        # Check for entanglement conditions
        entanglement_strength = 0.0
        
        # Same file entanglement
        if task1.file_path == task2.file_path:
            entanglement_strength += 0.5
        
        # Same task type entanglement  
        if task1.task_type == task2.task_type:
            entanglement_strength += 0.3
        
        # Content similarity entanglement
        content_similarity = self._calculate_content_similarity(task1.content, task2.content)
        entanglement_strength += content_similarity * 0.4
        
        # Create bidirectional entanglement
        if entanglement_strength > 0.3:
            task1.entangled_tasks.append(task2.id)
            task2.entangled_tasks.append(task1.id)
            task1.quantum_state = QuantumState.ENTANGLED
            task2.quantum_state = QuantumState.ENTANGLED
            
            # Store entanglement pattern
            entanglement_key = f"{task1.id}_{task2.id}"
            self.entanglement_patterns[entanglement_key] = entanglement_strength
            
            self.logger.debug(f"Created quantum entanglement between {task1.id} and {task2.id} "
                            f"with strength {entanglement_strength:.3f}")
        
        return entanglement_strength
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using quantum interference"""
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Classical Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Quantum interference enhancement
        interference = math.cos(intersection * 0.5) * 0.1
        
        return min(jaccard + interference, 1.0)
    
    def quantum_annealing_schedule(self, tasks: List[QuantumTask], max_iterations: int = 1000) -> List[QuantumTask]:
        """
        Apply quantum annealing to find optimal task prioritization
        """
        if not self.enable_quantum_annealing:
            return self._classical_priority_sort(tasks)
        
        self.logger.info(f"Starting quantum annealing for {len(tasks)} tasks")
        
        # Initialize quantum system
        current_order = tasks.copy()
        current_energy = self._calculate_system_energy(current_order)
        best_order = current_order.copy()
        best_energy = current_energy
        
        temperature = self.temperature
        
        for iteration in range(max_iterations):
            # Generate quantum tunneling move
            new_order = self._quantum_tunneling_move(current_order)
            new_energy = self._calculate_system_energy(new_order)
            
            # Energy difference
            delta_energy = new_energy - current_energy
            
            # Quantum acceptance probability
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_order = new_order
                current_energy = new_energy
                
                # Update best solution
                if current_energy < best_energy:
                    best_order = current_order.copy()
                    best_energy = current_energy
            
            # Cooling schedule
            temperature *= self.cooling_rate
            temperature = max(temperature, self.min_temperature)
            
            if iteration % 100 == 0:
                self.logger.debug(f"Annealing iteration {iteration}: energy={current_energy:.3f}, temp={temperature:.3f}")
        
        self.logger.info(f"Quantum annealing complete. Final energy: {best_energy:.3f}")
        
        # Collapse quantum states
        for task in best_order:
            task.quantum_state = QuantumState.COLLAPSED
        
        return best_order
    
    def _calculate_system_energy(self, task_order: List[QuantumTask]) -> float:
        """Calculate total system energy for quantum annealing"""
        
        total_energy = 0.0
        
        for i, task in enumerate(task_order):
            # Position energy (earlier positions have lower energy for high priority)
            priority_score = self._calculate_quantum_priority(task)
            position_energy = i * (10.0 - priority_score)  # Higher priority = lower energy when early
            
            # Entanglement energy
            entanglement_energy = 0.0
            for j, other_task in enumerate(task_order):
                if other_task.id in task.entangled_tasks:
                    # Entangled tasks prefer to be close together
                    distance = abs(i - j)
                    entanglement_key = f"{task.id}_{other_task.id}"
                    strength = self.entanglement_patterns.get(entanglement_key, 0.0)
                    entanglement_energy += distance * strength * 5.0
            
            # Dependency violation energy
            dependency_energy = self._calculate_dependency_violations(task, i, task_order)
            
            total_energy += position_energy + entanglement_energy + dependency_energy
        
        return total_energy
    
    def _quantum_tunneling_move(self, current_order: List[QuantumTask]) -> List[QuantumTask]:
        """Generate quantum tunneling move for annealing"""
        
        new_order = current_order.copy()
        
        # Random quantum tunneling moves
        move_type = random.choice(['swap', 'insert', 'reverse'])
        
        if move_type == 'swap' and len(new_order) >= 2:
            i, j = random.sample(range(len(new_order)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
        elif move_type == 'insert' and len(new_order) >= 2:
            i = random.randint(0, len(new_order) - 1)
            j = random.randint(0, len(new_order) - 1)
            task = new_order.pop(i)
            new_order.insert(j, task)
            
        elif move_type == 'reverse' and len(new_order) >= 3:
            i = random.randint(0, len(new_order) - 3)
            j = random.randint(i + 2, len(new_order))
            new_order[i:j] = reversed(new_order[i:j])
        
        return new_order
    
    def _calculate_dependency_violations(self, task: QuantumTask, position: int, task_order: List[QuantumTask]) -> float:
        """Calculate energy penalty for dependency violations"""
        
        violation_energy = 0.0
        
        # Simple dependency check based on content keywords
        content_lower = task.content.lower()
        
        if 'after' in content_lower or 'depends' in content_lower:
            # This task should come after its dependencies
            # For simplification, assume earlier tasks are dependencies
            if position < len(task_order) * 0.3:  # Task is too early
                violation_energy += 10.0
        
        if 'before' in content_lower or 'blocks' in content_lower:
            # This task should come before others
            # For simplification, assume later tasks depend on this
            if position > len(task_order) * 0.7:  # Task is too late
                violation_energy += 10.0
        
        return violation_energy
    
    def _calculate_quantum_priority(self, task: QuantumTask) -> float:
        """Calculate quantum priority using superposition collapse"""
        
        # Quantum superposition collapse
        priority_probabilities = [abs(p)**2 for p in task.priority_vector]
        total_probability = sum(priority_probabilities)
        
        if total_probability > 0:
            normalized_probabilities = [p/total_probability for p in priority_probabilities]
        else:
            normalized_probabilities = [1.0/len(task.priority_vector)] * len(task.priority_vector)
        
        # Expected value from quantum measurement
        quantum_priority = sum(
            prob * value for prob, value in zip(normalized_probabilities, task.priority_vector)
        )
        
        # Quantum enhancement based on task type
        type_multiplier = TASK_TYPE_PRIORITIES.get(task.task_type, 3.0)
        
        # Final quantum priority with uncertainty
        uncertainty_factor = 1.0 + task.uncertainty * 0.1
        final_priority = quantum_priority * (type_multiplier / 10.0) * uncertainty_factor
        
        return min(max(final_priority, 1.0), 10.0)
    
    def _classical_priority_sort(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Fallback classical priority sorting"""
        
        return sorted(tasks, key=lambda t: self._calculate_quantum_priority(t), reverse=True)
    
    def prioritize_tasks(self, task_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main method to prioritize tasks using quantum-inspired algorithms
        """
        self.logger.info(f"Quantum prioritization of {len(task_data_list)} tasks")
        
        # Convert to quantum tasks
        quantum_tasks = [self.create_quantum_task(task_data) for task_data in task_data_list]
        
        # Create quantum entanglements
        for i in range(len(quantum_tasks)):
            for j in range(i + 1, len(quantum_tasks)):
                self.create_quantum_entanglement(quantum_tasks[i], quantum_tasks[j])
        
        # Apply quantum annealing
        optimized_tasks = self.quantum_annealing_schedule(quantum_tasks)
        
        # Convert back to classical format with quantum insights
        prioritized_results = []
        for rank, task in enumerate(optimized_tasks):
            quantum_priority = self._calculate_quantum_priority(task)
            
            result = {
                'id': task.id,
                'content': task.content,
                'file_path': task.file_path,
                'line_number': task.line_number,
                'type': task.task_type,
                'task_type': task.task_type,
                'priority_score': quantum_priority,
                'quantum_rank': rank + 1,
                'quantum_state': task.quantum_state.value,
                'entangled_tasks': task.entangled_tasks,
                'uncertainty': task.uncertainty,
                'priority_reason': self._generate_quantum_priority_reason(task, quantum_priority),
                'quantum_insights': {
                    'amplitude': abs(task.amplitude),
                    'phase': task.phase,
                    'coherence_time': task.coherence_time,
                    'priority_dimensions': dict(zip(self.priority_dimensions, task.priority_vector))
                }
            }
            
            prioritized_results.append(result)
        
        self.logger.info(f"Quantum prioritization complete. Highest priority: {prioritized_results[0]['priority_score']:.3f}")
        
        return prioritized_results
    
    def _generate_quantum_priority_reason(self, task: QuantumTask, priority_score: float) -> str:
        """Generate quantum-enhanced priority reasoning"""
        
        reasons = []
        
        # Quantum state information
        if task.quantum_state == QuantumState.ENTANGLED:
            reasons.append("quantum entangled with related tasks")
        
        # Task type reasoning
        if task.task_type == 'security':
            reasons.append("critical security quantum state")
        elif task.task_type == 'bug':
            reasons.append("high-priority bug quantum correction")
        elif task.task_type == 'performance':
            reasons.append("performance optimization quantum enhancement")
        else:
            reasons.append(f"{task.task_type} quantum task")
        
        # Quantum insights
        max_dimension = max(range(len(task.priority_vector)), key=lambda i: task.priority_vector[i])
        dominant_dimension = self.priority_dimensions[max_dimension]
        reasons.append(f"dominated by {dominant_dimension}")
        
        # Uncertainty principle
        if task.uncertainty > 0.5:
            reasons.append("high quantum uncertainty")
        
        # Priority level with quantum enhancement
        if priority_score >= 8.0:
            priority_level = "Quantum Critical"
        elif priority_score >= 6.0:
            priority_level = "Quantum High"
        elif priority_score >= 4.0:
            priority_level = "Quantum Medium"
        else:
            priority_level = "Quantum Low"
        
        return f"{priority_level}: {', '.join(reasons)}"
    
    def generate_quantum_insights_report(self, prioritized_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quantum insights report"""
        
        total_tasks = len(prioritized_tasks)
        entangled_tasks = sum(1 for task in prioritized_tasks if task.get('quantum_state') == 'entangled')
        
        # Priority distribution
        priority_distribution = {
            'critical': sum(1 for task in prioritized_tasks if task['priority_score'] >= 8.0),
            'high': sum(1 for task in prioritized_tasks if 6.0 <= task['priority_score'] < 8.0),
            'medium': sum(1 for task in prioritized_tasks if 4.0 <= task['priority_score'] < 6.0),
            'low': sum(1 for task in prioritized_tasks if task['priority_score'] < 4.0)
        }
        
        # Quantum metrics
        avg_uncertainty = sum(task['uncertainty'] for task in prioritized_tasks) / total_tasks
        
        # Entanglement network stats
        entanglement_density = len(self.entanglement_patterns) / (total_tasks * (total_tasks - 1) / 2) if total_tasks > 1 else 0.0
        
        report = {
            'quantum_metrics': {
                'total_tasks': total_tasks,
                'entangled_tasks': entangled_tasks,
                'entanglement_density': entanglement_density,
                'average_uncertainty': avg_uncertainty,
                'quantum_annealing_enabled': self.enable_quantum_annealing
            },
            'priority_distribution': priority_distribution,
            'top_priorities': [
                {
                    'id': task['id'],
                    'priority_score': task['priority_score'],
                    'quantum_state': task['quantum_state'],
                    'content_preview': task['content'][:100] + '...' if len(task['content']) > 100 else task['content']
                }
                for task in prioritized_tasks[:5]
            ],
            'entanglement_patterns': dict(list(self.entanglement_patterns.items())[:10]),  # Top 10 entanglements
            'dimensional_analysis': self._analyze_priority_dimensions(prioritized_tasks),
            'generation_timestamp': datetime.now().isoformat(),
            'quantum_coherence_time': sum(
                task.get('quantum_insights', {}).get('coherence_time', 0) 
                for task in prioritized_tasks
            ) / total_tasks if total_tasks > 0 else 0.0
        }
        
        return report
    
    def _analyze_priority_dimensions(self, prioritized_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant priority dimensions across all tasks"""
        
        dimension_totals = {dim: 0.0 for dim in self.priority_dimensions}
        
        for task in prioritized_tasks:
            quantum_insights = task.get('quantum_insights', {})
            priority_dims = quantum_insights.get('priority_dimensions', {})
            
            for dim, value in priority_dims.items():
                if dim in dimension_totals:
                    dimension_totals[dim] += value
        
        # Normalize by number of tasks
        total_tasks = len(prioritized_tasks)
        if total_tasks > 0:
            dimension_averages = {dim: total / total_tasks for dim, total in dimension_totals.items()}
        else:
            dimension_averages = dimension_totals
        
        return dimension_averages


# Factory function for easy integration
def create_quantum_task_planner(enable_annealing: bool = True) -> QuantumTaskPlanner:
    """Factory function to create quantum task planner"""
    return QuantumTaskPlanner(enable_quantum_annealing=enable_annealing)


# Example usage and testing
if __name__ == "__main__":
    # Test quantum task planner
    planner = create_quantum_task_planner(enable_annealing=True)
    
    test_tasks = [
        {
            'id': 'quantum_task_1',
            'content': 'TODO: Fix critical SQL injection vulnerability in authentication system',
            'file_path': 'src/auth/security.py',
            'line_number': 42,
            'type': 'security'
        },
        {
            'id': 'quantum_task_2',
            'content': 'TODO: Optimize database queries for better performance',
            'file_path': 'src/database/queries.py',
            'line_number': 15,
            'type': 'performance'
        },
        {
            'id': 'quantum_task_3',
            'content': 'TODO: Add comprehensive unit tests for authentication module',
            'file_path': 'tests/test_auth.py',
            'line_number': 1,
            'type': 'testing'
        },
        {
            'id': 'quantum_task_4',
            'content': 'TODO: Update API documentation with new endpoints',
            'file_path': 'docs/api.md',
            'line_number': 25,
            'type': 'documentation'
        }
    ]
    
    # Run quantum prioritization
    quantum_results = planner.prioritize_tasks(test_tasks)
    
    print("\n=== QUANTUM TASK PRIORITIZATION RESULTS ===")
    for i, task in enumerate(quantum_results[:3]):
        print(f"\nRank {i+1}: {task['id']}")
        print(f"  Priority Score: {task['priority_score']:.3f}")
        print(f"  Quantum State: {task['quantum_state']}")
        print(f"  Reason: {task['priority_reason']}")
        print(f"  Entangled Tasks: {len(task['entangled_tasks'])}")
        print(f"  Uncertainty: {task['uncertainty']:.3f}")
    
    # Generate quantum insights report
    insights = planner.generate_quantum_insights_report(quantum_results)
    print(f"\n=== QUANTUM INSIGHTS ===")
    print(f"Total Tasks: {insights['quantum_metrics']['total_tasks']}")
    print(f"Entangled Tasks: {insights['quantum_metrics']['entangled_tasks']}")
    print(f"Entanglement Density: {insights['quantum_metrics']['entanglement_density']:.3f}")
    print(f"Average Uncertainty: {insights['quantum_metrics']['average_uncertainty']:.3f}")
    print(f"Quantum Coherence Time: {insights['quantum_coherence_time']:.3f}")