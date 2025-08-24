#!/usr/bin/env python3
"""
NEXT-GENERATION AUTONOMOUS EXECUTOR

Hybrid AI system combining multiple execution strategies with
self-learning capabilities and predictive optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import uuid
import random
import math


class ExecutorStrategy(Enum):
    """Different execution strategies for task handling"""
    CONSERVATIVE = "conservative"  # Safe, proven approaches
    AGGRESSIVE = "aggressive"      # Fast, experimental approaches  
    BALANCED = "balanced"          # Middle ground
    ADAPTIVE = "adaptive"          # Learning-based approach
    HYBRID = "hybrid"              # Multiple strategies combined


class LearningSignal(Enum):
    """Signals for reinforcement learning"""
    SUCCESS = 1.0
    PARTIAL_SUCCESS = 0.5
    FAILURE = -1.0
    TIMEOUT = -0.5
    ERROR = -0.8


class NextGenExecutor:
    """Advanced autonomous executor with self-learning capabilities"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logger()
        
        # Execution strategies and learning
        self.strategies = {
            ExecutorStrategy.CONSERVATIVE: ConservativeStrategy(),
            ExecutorStrategy.AGGRESSIVE: AggressiveStrategy(), 
            ExecutorStrategy.BALANCED: BalancedStrategy(),
            ExecutorStrategy.ADAPTIVE: AdaptiveStrategy(),
            ExecutorStrategy.HYBRID: HybridStrategy()
        }
        
        # Performance tracking for learning
        self.strategy_performance: Dict[ExecutorStrategy, List[float]] = {
            strategy: [] for strategy in ExecutorStrategy
        }
        
        # Current state
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.completed_executions: List[Dict[str, Any]] = []
        self.learning_model = SimpleLearningModel()
        
        # Configuration
        self.config = {
            "max_concurrent_executions": 3,
            "learning_rate": 0.1,
            "exploration_rate": 0.2,
            "strategy_switch_threshold": 0.3
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("NextGenExecutor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def execute_with_learning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with continuous learning and adaptation"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting execution {execution_id}: {task.get('title', 'Unknown')}")
        
        try:
            # Select optimal strategy using learning model
            selected_strategy = await self._select_strategy(task)
            
            # Execute with selected strategy
            result = await self._execute_with_strategy(execution_id, task, selected_strategy)
            
            # Learn from result
            await self._learn_from_execution(selected_strategy, result)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["strategy_used"] = selected_strategy.value
            
            self.completed_executions.append(result)
            
            self.logger.info(f"Execution {execution_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Execution {execution_id} failed: {e}")
            
            # Learn from failure
            failure_result = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            if "selected_strategy" in locals():
                await self._learn_from_execution(selected_strategy, failure_result)
            
            return failure_result
    
    async def _select_strategy(self, task: Dict[str, Any]) -> ExecutorStrategy:
        """Select optimal strategy using learning model and exploration"""
        # Extract task features for decision making
        task_features = self._extract_task_features(task)
        
        # Exploration vs exploitation
        if random.random() < self.config["exploration_rate"]:
            # Exploration: try random strategy
            selected = random.choice(list(ExecutorStrategy))
            self.logger.debug(f"Exploring with strategy: {selected.value}")
            return selected
        
        # Exploitation: use learning model to select best strategy
        strategy_scores = {}
        
        for strategy in ExecutorStrategy:
            # Get historical performance
            performance_history = self.strategy_performance[strategy]
            
            if len(performance_history) == 0:
                # No history, give neutral score
                base_score = 0.5
            else:
                # Recent performance weighted more heavily
                recent_performance = performance_history[-10:]  # Last 10 executions
                base_score = sum(recent_performance) / len(recent_performance)
            
            # Adjust score based on task characteristics
            adjusted_score = self.learning_model.predict_strategy_success(
                strategy, task_features, base_score
            )
            
            strategy_scores[strategy] = adjusted_score
        
        # Select strategy with highest predicted success
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        
        self.logger.info(f"Selected strategy: {best_strategy.value} (score: {strategy_scores[best_strategy]:.3f})")
        return best_strategy
    
    def _extract_task_features(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from task for ML model"""
        features = {}
        
        # Basic features
        features["title_length"] = len(task.get("title", ""))
        features["description_length"] = len(task.get("description", ""))
        features["priority"] = task.get("priority", 3)
        features["complexity"] = task.get("complexity_score", 5.0)
        
        # Time-based features  
        created_at = task.get("created_at")
        if created_at:
            try:
                created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                age_hours = (datetime.now().astimezone() - created_time).total_seconds() / 3600
                features["age_hours"] = min(age_hours, 168)  # Cap at 1 week
            except:
                features["age_hours"] = 0
        else:
            features["age_hours"] = 0
        
        # Repository features
        repo = task.get("repository", "")
        features["repo_complexity"] = hash(repo) % 10  # Simple repo complexity proxy
        
        # Label-based features
        labels = task.get("labels", [])
        features["has_bug_label"] = 1.0 if any("bug" in label.lower() for label in labels) else 0.0
        features["has_enhancement_label"] = 1.0 if any("enhancement" in label.lower() for label in labels) else 0.0
        features["has_documentation_label"] = 1.0 if any("doc" in label.lower() for label in labels) else 0.0
        
        return features
    
    async def _execute_with_strategy(self, execution_id: str, task: Dict[str, Any], strategy: ExecutorStrategy) -> Dict[str, Any]:
        """Execute task using specified strategy"""
        self.active_executions[execution_id] = {
            "task": task,
            "strategy": strategy,
            "start_time": time.time()
        }
        
        try:
            # Get strategy executor
            executor = self.strategies[strategy]
            
            # Execute with strategy
            result = await executor.execute(task)
            
            # Clean up active execution
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            return result
            
        except Exception as e:
            # Clean up on error
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            raise
    
    async def _learn_from_execution(self, strategy: ExecutorStrategy, result: Dict[str, Any]) -> None:
        """Learn from execution result to improve future decisions"""
        # Convert result to learning signal
        if result.get("success", False):
            if result.get("execution_time", 0) < 60:  # Fast success
                signal = LearningSignal.SUCCESS.value
            else:  # Slow success
                signal = LearningSignal.PARTIAL_SUCCESS.value
        else:
            error = result.get("error", "")
            if "timeout" in error.lower():
                signal = LearningSignal.TIMEOUT.value
            else:
                signal = LearningSignal.FAILURE.value
        
        # Update strategy performance
        self.strategy_performance[strategy].append(signal)
        
        # Keep only recent performance (last 100 executions)
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
        
        # Update learning model
        self.learning_model.update(strategy, signal)
        
        self.logger.debug(f"Learned from execution: {strategy.value} -> {signal:.2f}")
    
    async def execute_batch_with_optimization(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tasks with dynamic optimization"""
        self.logger.info(f"Starting batch execution of {len(tasks)} tasks")
        
        results = []
        
        # Group tasks by characteristics for optimal batching
        task_groups = self._group_tasks_for_batching(tasks)
        
        for group_name, group_tasks in task_groups.items():
            self.logger.info(f"Processing task group '{group_name}' ({len(group_tasks)} tasks)")
            
            # Execute group with appropriate concurrency
            max_concurrent = min(self.config["max_concurrent_executions"], len(group_tasks))
            
            # Process tasks in batches
            for i in range(0, len(group_tasks), max_concurrent):
                batch = group_tasks[i:i + max_concurrent]
                
                # Execute batch concurrently
                execution_tasks = [
                    self.execute_with_learning(task) for task in batch
                ]
                
                batch_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Batch task failed: {result}")
                        results.append({
                            "success": False,
                            "error": str(result),
                            "task_index": i + j
                        })
                    else:
                        results.append(result)
                
                # Adaptive delay between batches based on performance
                if i + max_concurrent < len(group_tasks):
                    delay = self._calculate_adaptive_delay(batch_results)
                    if delay > 0:
                        await asyncio.sleep(delay)
        
        # Analyze batch performance and adjust parameters
        await self._analyze_batch_performance(results)
        
        self.logger.info(f"Batch execution completed: {len(results)} results")
        return results
    
    def _group_tasks_for_batching(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tasks by characteristics for optimal batch processing"""
        groups = {
            "high_priority": [],
            "bug_fixes": [],
            "enhancements": [],
            "documentation": [],
            "maintenance": []
        }
        
        for task in tasks:
            priority = task.get("priority", 3)
            labels = task.get("labels", [])
            
            if priority >= 4:
                groups["high_priority"].append(task)
            elif any("bug" in label.lower() for label in labels):
                groups["bug_fixes"].append(task)
            elif any("enhancement" in label.lower() for label in labels):
                groups["enhancements"].append(task)
            elif any("doc" in label.lower() for label in labels):
                groups["documentation"].append(task)
            else:
                groups["maintenance"].append(task)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _calculate_adaptive_delay(self, batch_results: List[Any]) -> float:
        """Calculate adaptive delay between batches based on performance"""
        if not batch_results:
            return 1.0
        
        # Count successful vs failed executions
        successful = sum(1 for r in batch_results 
                        if isinstance(r, dict) and r.get("success", False))
        success_rate = successful / len(batch_results)
        
        # More delay if low success rate (system might be overloaded)
        if success_rate < 0.5:
            return 5.0  # 5 second delay
        elif success_rate < 0.8:
            return 2.0  # 2 second delay
        else:
            return 0.5  # Minimal delay
    
    async def _analyze_batch_performance(self, results: List[Dict[str, Any]]) -> None:
        """Analyze batch performance and adjust system parameters"""
        if not results:
            return
        
        successful = sum(1 for r in results if r.get("success", False))
        success_rate = successful / len(results)
        
        avg_execution_time = sum(r.get("execution_time", 0) for r in results) / len(results)
        
        self.logger.info(f"Batch performance: {success_rate:.2%} success rate, {avg_execution_time:.2f}s avg time")
        
        # Adjust parameters based on performance
        if success_rate < 0.7:
            # Low success rate - reduce concurrency
            self.config["max_concurrent_executions"] = max(1, self.config["max_concurrent_executions"] - 1)
            self.logger.info("Reduced concurrency due to low success rate")
        elif success_rate > 0.9 and avg_execution_time < 30:
            # High success rate and fast execution - increase concurrency
            self.config["max_concurrent_executions"] = min(10, self.config["max_concurrent_executions"] + 1)
            self.logger.info("Increased concurrency due to high performance")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_executions = len(self.completed_executions)
        
        if total_executions == 0:
            return {"message": "No executions completed yet"}
        
        successful = sum(1 for e in self.completed_executions if e.get("success", False))
        success_rate = successful / total_executions
        
        avg_execution_time = sum(e.get("execution_time", 0) for e in self.completed_executions) / total_executions
        
        # Strategy performance
        strategy_summary = {}
        for strategy, performance in self.strategy_performance.items():
            if performance:
                avg_performance = sum(performance) / len(performance)
                strategy_summary[strategy.value] = {
                    "average_score": avg_performance,
                    "executions": len(performance),
                    "recent_trend": sum(performance[-5:]) / min(len(performance), 5)
                }
        
        return {
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "active_executions": len(self.active_executions),
            "strategy_performance": strategy_summary,
            "current_config": self.config
        }


class SimpleLearningModel:
    """Simple learning model for strategy selection"""
    
    def __init__(self):
        self.strategy_weights = {strategy: 1.0 for strategy in ExecutorStrategy}
        self.feature_importance = {
            "complexity": 0.3,
            "priority": 0.2,
            "age_hours": 0.1,
            "has_bug_label": 0.2,
            "has_enhancement_label": 0.1,
            "description_length": 0.1
        }
    
    def predict_strategy_success(self, strategy: ExecutorStrategy, features: Dict[str, float], base_score: float) -> float:
        """Predict success probability for strategy given task features"""
        # Simple linear combination
        feature_adjustment = 0.0
        
        # Strategy-specific feature preferences
        if strategy == ExecutorStrategy.CONSERVATIVE:
            # Conservative prefers lower complexity, documented tasks
            feature_adjustment -= features.get("complexity", 5.0) * 0.05
            feature_adjustment += features.get("has_documentation_label", 0.0) * 0.1
        elif strategy == ExecutorStrategy.AGGRESSIVE:
            # Aggressive prefers high priority, recent tasks
            feature_adjustment += features.get("priority", 3.0) * 0.05
            feature_adjustment -= features.get("age_hours", 0.0) * 0.001
        elif strategy == ExecutorStrategy.BALANCED:
            # Balanced has no strong preferences
            pass
        elif strategy == ExecutorStrategy.ADAPTIVE:
            # Adaptive prefers tasks with clear patterns
            feature_adjustment += (features.get("has_bug_label", 0.0) + 
                                 features.get("has_enhancement_label", 0.0)) * 0.1
        
        # Apply strategy weight
        weight_adjustment = (self.strategy_weights[strategy] - 1.0) * 0.1
        
        # Combine all factors
        final_score = base_score + feature_adjustment + weight_adjustment
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, final_score))
    
    def update(self, strategy: ExecutorStrategy, signal: float) -> None:
        """Update model based on execution result"""
        learning_rate = 0.1
        
        # Update strategy weight based on signal
        self.strategy_weights[strategy] += learning_rate * signal
        
        # Keep weights in reasonable range
        self.strategy_weights[strategy] = max(0.1, min(2.0, self.strategy_weights[strategy]))


# Strategy implementations

class ExecutionStrategy:
    """Base class for execution strategies"""
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task and return result"""
        raise NotImplementedError


class ConservativeStrategy(ExecutionStrategy):
    """Conservative execution strategy - safe and thorough"""
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with conservative approach"""
        # Simulate conservative execution (longer but more reliable)
        execution_time = random.uniform(30, 90)  # Longer execution
        success_probability = 0.85  # High success rate
        
        await asyncio.sleep(min(execution_time / 30, 3))  # Simulate work
        
        success = random.random() < success_probability
        
        return {
            "success": success,
            "output": f"Conservative execution {'completed' if success else 'failed'}",
            "details": {
                "approach": "thorough_analysis",
                "risk_assessment": "low",
                "validation_steps": 5
            }
        }


class AggressiveStrategy(ExecutionStrategy):
    """Aggressive execution strategy - fast but risky"""
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with aggressive approach"""
        # Simulate aggressive execution (faster but less reliable)
        execution_time = random.uniform(10, 30)  # Shorter execution
        success_probability = 0.65  # Lower success rate
        
        await asyncio.sleep(min(execution_time / 30, 1))  # Simulate work
        
        success = random.random() < success_probability
        
        return {
            "success": success,
            "output": f"Aggressive execution {'completed' if success else 'failed'}",
            "details": {
                "approach": "rapid_implementation",
                "risk_assessment": "high",
                "validation_steps": 2
            }
        }


class BalancedStrategy(ExecutionStrategy):
    """Balanced execution strategy - middle ground"""
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with balanced approach"""
        # Simulate balanced execution
        execution_time = random.uniform(20, 60)  # Medium execution time
        success_probability = 0.75  # Medium success rate
        
        await asyncio.sleep(min(execution_time / 30, 2))  # Simulate work
        
        success = random.random() < success_probability
        
        return {
            "success": success,
            "output": f"Balanced execution {'completed' if success else 'failed'}",
            "details": {
                "approach": "measured_implementation",
                "risk_assessment": "medium",
                "validation_steps": 3
            }
        }


class AdaptiveStrategy(ExecutionStrategy):
    """Adaptive execution strategy - learns from context"""
    
    def __init__(self):
        self.execution_history = []
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with adaptive approach based on task characteristics"""
        # Adapt based on task complexity
        complexity = task.get("complexity_score", 5.0)
        
        if complexity < 3.0:
            # Simple task - use aggressive approach
            execution_time = random.uniform(10, 25)
            success_probability = 0.8
        elif complexity > 7.0:
            # Complex task - use conservative approach
            execution_time = random.uniform(45, 90)
            success_probability = 0.85
        else:
            # Medium complexity - balanced approach
            execution_time = random.uniform(25, 50)
            success_probability = 0.75
        
        await asyncio.sleep(min(execution_time / 30, 3))  # Simulate work
        
        success = random.random() < success_probability
        
        # Learn from this execution
        self.execution_history.append({
            "complexity": complexity,
            "success": success,
            "execution_time": execution_time
        })
        
        # Keep only recent history
        self.execution_history = self.execution_history[-50:]
        
        return {
            "success": success,
            "output": f"Adaptive execution {'completed' if success else 'failed'}",
            "details": {
                "approach": "context_aware",
                "risk_assessment": "adaptive",
                "complexity_adjustment": complexity,
                "learning_samples": len(self.execution_history)
            }
        }


class HybridStrategy(ExecutionStrategy):
    """Hybrid strategy - combines multiple approaches"""
    
    def __init__(self):
        self.sub_strategies = {
            "conservative": ConservativeStrategy(),
            "aggressive": AggressiveStrategy(),
            "balanced": BalancedStrategy()
        }
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using hybrid approach with multiple sub-strategies"""
        # Select sub-strategy based on task characteristics
        priority = task.get("priority", 3)
        complexity = task.get("complexity_score", 5.0)
        labels = task.get("labels", [])
        
        if priority >= 4 and any("critical" in label.lower() for label in labels):
            # Critical tasks - use conservative
            selected_strategy = "conservative"
        elif complexity < 3.0:
            # Simple tasks - use aggressive
            selected_strategy = "aggressive"
        else:
            # Default - use balanced
            selected_strategy = "balanced"
        
        # Execute with selected sub-strategy
        sub_executor = self.sub_strategies[selected_strategy]
        result = await sub_executor.execute(task)
        
        # Add hybrid-specific information
        result["details"]["hybrid_strategy"] = selected_strategy
        result["details"]["hybrid_reasoning"] = f"Selected {selected_strategy} based on priority={priority}, complexity={complexity:.1f}"
        
        return result


# CLI Interface
async def main():
    """Main entry point for next-gen executor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Next-Generation Autonomous Executor")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--tasks", type=int, default=5, help="Number of test tasks to generate")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for execution")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize executor
    executor = NextGenExecutor(args.config)
    
    # Generate test tasks
    test_tasks = []
    for i in range(args.tasks):
        task = {
            "id": str(uuid.uuid4()),
            "title": f"Test Task {i+1}",
            "description": f"This is test task number {i+1} for demonstrating the next-gen executor",
            "priority": random.randint(1, 5),
            "complexity_score": random.uniform(1.0, 10.0),
            "labels": random.choice([
                ["bug", "high-priority"],
                ["enhancement", "feature"],
                ["documentation"],
                ["maintenance", "cleanup"],
                ["critical", "security"]
            ]),
            "repository": f"test-repo-{i % 3}",
            "created_at": datetime.now().isoformat()
        }
        test_tasks.append(task)
    
    print(f"\n🚀 Starting Next-Generation Autonomous Executor")
    print(f"📋 Generated {len(test_tasks)} test tasks")
    print(f"📦 Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Execute tasks in batches
    start_time = time.time()
    
    if args.batch_size == 1:
        # Execute tasks individually
        results = []
        for i, task in enumerate(test_tasks, 1):
            print(f"\n[{i}/{len(test_tasks)}] Executing: {task['title']}")
            result = await executor.execute_with_learning(task)
            results.append(result)
            
            if result.get("success"):
                print(f"✅ Success in {result.get('execution_time', 0):.2f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    else:
        # Execute in batch
        results = await executor.execute_batch_with_optimization(test_tasks)
    
    total_time = time.time() - start_time
    
    # Display results summary
    print("\n" + "=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r.get("success", False))
    success_rate = (successful / len(results)) * 100 if results else 0
    
    print(f"📈 Success Rate: {success_rate:.1f}% ({successful}/{len(results)})")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"🏃 Average per Task: {total_time / len(test_tasks):.2f}s")
    
    # Show performance summary
    performance = executor.get_performance_summary()
    
    print("\n🧠 LEARNING PERFORMANCE:")
    for strategy, metrics in performance["strategy_performance"].items():
        print(f"  {strategy}: {metrics['average_score']:.3f} avg score ({metrics['executions']} executions)")
    
    print(f"\n⚙️  Current Config:")
    for key, value in performance["current_config"].items():
        print(f"  {key}: {value}")
    
    print("\n✨ Next-Generation Executor completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())