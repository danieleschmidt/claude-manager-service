#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - INTELLIGENT TASK DISCOVERY SYSTEM
AI-powered task discovery with adaptive learning and predictive analysis
"""

import asyncio
import json
import re
import ast
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import structlog
from github import Github, GithubException

from .core_system import Task, CoreLogger


@dataclass
class IntelligentTask(Task):
    """Enhanced task with AI insights"""
    complexity_score: float = 0.0
    estimated_effort: int = 0  # in minutes
    confidence: float = 0.0
    suggested_approach: str = ""
    related_files: List[str] = None
    dependencies: List[str] = None
    business_impact: str = "low"
    technical_debt_score: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        if self.related_files is None:
            self.related_files = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CodeMetrics:
    """Code quality and complexity metrics"""
    cyclomatic_complexity: int = 0
    lines_of_code: int = 0
    test_coverage: float = 0.0
    technical_debt_ratio: float = 0.0
    maintainability_index: float = 0.0
    code_duplication: float = 0.0


class IntelligentTaskDiscovery:
    """AI-powered task discovery with advanced analysis"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.logger = structlog.get_logger("IntelligentTaskDiscovery")
        self.github_client = Github(github_token) if github_token else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Pattern libraries for intelligent recognition
        self.bug_patterns = [
            r"(?i)(bug|error|exception|crash|fail|broken)",
            r"(?i)(memory leak|race condition|deadlock)",
            r"(?i)(null pointer|segfault|assertion failed)",
            r"(?i)(timeout|performance issue|slow query)"
        ]
        
        self.security_patterns = [
            r"(?i)(sql injection|xss|csrf|authentication)",
            r"(?i)(privilege escalation|buffer overflow)",
            r"(?i)(hardcoded password|api key|secret)",
            r"(?i)(insecure|vulnerability|security)"
        ]
        
        self.performance_patterns = [
            r"(?i)(slow|performance|optimization|bottleneck)",
            r"(?i)(memory usage|cpu intensive|disk io)",
            r"(?i)(cache|index|query optimization)",
            r"(?i)(scaling|load balancing|throughput)"
        ]
        
    async def discover_intelligent_tasks(self, repo_path: str = ".") -> List[IntelligentTask]:
        """Main entry point for intelligent task discovery"""
        self.logger.info("Starting intelligent task discovery", repo_path=repo_path)
        
        tasks = []
        
        # Parallel discovery processes
        discovery_tasks = [
            self._discover_code_quality_issues(repo_path),
            self._discover_architecture_opportunities(repo_path),
            self._discover_performance_bottlenecks(repo_path),
            self._discover_security_vulnerabilities(repo_path),
            self._discover_test_gaps(repo_path),
            self._discover_documentation_needs(repo_path),
            self._discover_dependency_issues(repo_path),
            self._discover_refactoring_opportunities(repo_path)
        ]
        
        if self.github_client:
            discovery_tasks.append(self._discover_github_insights(repo_path))
        
        # Execute all discovery tasks concurrently
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error("Discovery task failed", error=str(result))
            elif isinstance(result, list):
                tasks.extend(result)
        
        # Apply intelligent prioritization
        prioritized_tasks = await self._prioritize_tasks_with_ai(tasks)
        
        # Remove duplicates and merge similar tasks
        deduplicated_tasks = self._deduplicate_tasks(prioritized_tasks)
        
        self.logger.info("Discovery complete", 
                        total_tasks=len(deduplicated_tasks),
                        high_priority=len([t for t in deduplicated_tasks if t.priority >= 8]))
        
        return deduplicated_tasks
    
    async def _discover_code_quality_issues(self, repo_path: str) -> List[IntelligentTask]:
        """Discover code quality issues using static analysis"""
        tasks = []
        
        try:
            for py_file in Path(repo_path).glob("**/*.py"):
                if py_file.is_file() and not self._is_ignored_file(py_file):
                    metrics = await self._analyze_code_metrics(py_file)
                    
                    # High complexity functions
                    if metrics.cyclomatic_complexity > 10:
                        task = IntelligentTask(
                            id=f"complexity_{py_file.stem}_{int(time.time())}",
                            title=f"Reduce complexity in {py_file.name}",
                            description=f"Function has complexity score of {metrics.cyclomatic_complexity}. Consider breaking into smaller functions.",
                            priority=7,
                            task_type="code_quality",
                            file_path=str(py_file),
                            complexity_score=metrics.cyclomatic_complexity,
                            estimated_effort=30 + (metrics.cyclomatic_complexity * 5),
                            confidence=0.85,
                            suggested_approach="Extract methods, reduce nesting, use early returns",
                            business_impact="medium",
                            technical_debt_score=metrics.cyclomatic_complexity / 20.0
                        )
                        tasks.append(task)
                    
                    # Low maintainability
                    if metrics.maintainability_index < 50:
                        task = IntelligentTask(
                            id=f"maintainability_{py_file.stem}_{int(time.time())}",
                            title=f"Improve maintainability of {py_file.name}",
                            description=f"Maintainability index: {metrics.maintainability_index:.1f}. Needs refactoring.",
                            priority=6,
                            task_type="refactoring",
                            file_path=str(py_file),
                            complexity_score=100 - metrics.maintainability_index,
                            estimated_effort=45,
                            confidence=0.80,
                            suggested_approach="Add documentation, reduce complexity, improve naming",
                            business_impact="medium",
                            technical_debt_score=(100 - metrics.maintainability_index) / 100.0
                        )
                        tasks.append(task)
                    
        except Exception as e:
            self.logger.error("Error in code quality discovery", error=str(e))
            
        return tasks
    
    async def _discover_architecture_opportunities(self, repo_path: str) -> List[IntelligentTask]:
        """Discover architectural improvement opportunities"""
        tasks = []
        
        try:
            # Analyze import dependencies
            dependency_graph = await self._build_dependency_graph(repo_path)
            
            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(dependency_graph)
            for cycle in circular_deps:
                task = IntelligentTask(
                    id=f"circular_dep_{len(tasks)}_{int(time.time())}",
                    title=f"Resolve circular dependency",
                    description=f"Circular dependency detected: {' -> '.join(cycle)}",
                    priority=8,
                    task_type="architecture",
                    complexity_score=len(cycle) * 2.0,
                    estimated_effort=60 + (len(cycle) * 15),
                    confidence=0.90,
                    suggested_approach="Extract common interface, use dependency injection",
                    related_files=cycle,
                    business_impact="high",
                    technical_debt_score=0.8
                )
                tasks.append(task)
            
            # Detect large modules that should be split
            large_modules = await self._detect_large_modules(repo_path)
            for module_path, metrics in large_modules:
                if metrics.lines_of_code > 500:
                    task = IntelligentTask(
                        id=f"large_module_{Path(module_path).stem}_{int(time.time())}",
                        title=f"Split large module {Path(module_path).name}",
                        description=f"Module has {metrics.lines_of_code} lines. Consider splitting into smaller modules.",
                        priority=6,
                        task_type="architecture",
                        file_path=module_path,
                        complexity_score=metrics.lines_of_code / 100.0,
                        estimated_effort=90 + (metrics.lines_of_code // 50),
                        confidence=0.75,
                        suggested_approach="Extract related functionality into separate modules",
                        business_impact="medium",
                        technical_debt_score=metrics.lines_of_code / 1000.0
                    )
                    tasks.append(task)
                    
        except Exception as e:
            self.logger.error("Error in architecture discovery", error=str(e))
            
        return tasks
    
    async def _discover_performance_bottlenecks(self, repo_path: str) -> List[IntelligentTask]:
        """Discover potential performance bottlenecks"""
        tasks = []
        
        try:
            for py_file in Path(repo_path).glob("**/*.py"):
                if py_file.is_file() and not self._is_ignored_file(py_file):
                    bottlenecks = await self._analyze_performance_patterns(py_file)
                    
                    for bottleneck in bottlenecks:
                        task = IntelligentTask(
                            id=f"performance_{py_file.stem}_{bottleneck['line']}_{int(time.time())}",
                            title=f"Optimize performance in {py_file.name}",
                            description=f"Line {bottleneck['line']}: {bottleneck['description']}",
                            priority=bottleneck['severity'],
                            task_type="performance",
                            file_path=str(py_file),
                            line_number=bottleneck['line'],
                            complexity_score=bottleneck['complexity'],
                            estimated_effort=bottleneck['effort'],
                            confidence=bottleneck['confidence'],
                            suggested_approach=bottleneck['solution'],
                            business_impact="high" if bottleneck['severity'] >= 8 else "medium",
                            technical_debt_score=bottleneck['severity'] / 10.0
                        )
                        tasks.append(task)
                        
        except Exception as e:
            self.logger.error("Error in performance discovery", error=str(e))
            
        return tasks
    
    async def _discover_security_vulnerabilities(self, repo_path: str) -> List[IntelligentTask]:
        """Discover potential security vulnerabilities"""
        tasks = []
        
        try:
            for py_file in Path(repo_path).glob("**/*.py"):
                if py_file.is_file() and not self._is_ignored_file(py_file):
                    vulnerabilities = await self._analyze_security_patterns(py_file)
                    
                    for vuln in vulnerabilities:
                        task = IntelligentTask(
                            id=f"security_{py_file.stem}_{vuln['line']}_{int(time.time())}",
                            title=f"Fix security issue in {py_file.name}",
                            description=f"Line {vuln['line']}: {vuln['description']}",
                            priority=9,  # Security is always high priority
                            task_type="security",
                            file_path=str(py_file),
                            line_number=vuln['line'],
                            complexity_score=vuln['complexity'],
                            estimated_effort=vuln['effort'],
                            confidence=vuln['confidence'],
                            suggested_approach=vuln['solution'],
                            business_impact="critical",
                            technical_debt_score=0.9
                        )
                        tasks.append(task)
                        
        except Exception as e:
            self.logger.error("Error in security discovery", error=str(e))
            
        return tasks
    
    async def _discover_test_gaps(self, repo_path: str) -> List[IntelligentTask]:
        """Discover gaps in test coverage"""
        tasks = []
        
        try:
            coverage_data = await self._analyze_test_coverage(repo_path)
            
            for file_path, coverage in coverage_data.items():
                if coverage < 70:  # Below 70% coverage
                    task = IntelligentTask(
                        id=f"test_coverage_{Path(file_path).stem}_{int(time.time())}",
                        title=f"Improve test coverage for {Path(file_path).name}",
                        description=f"Current coverage: {coverage:.1f}%. Needs more comprehensive tests.",
                        priority=7,
                        task_type="testing",
                        file_path=file_path,
                        complexity_score=(100 - coverage) / 10.0,
                        estimated_effort=int((100 - coverage) * 2),
                        confidence=0.85,
                        suggested_approach="Add unit tests for uncovered branches and edge cases",
                        business_impact="high",
                        technical_debt_score=(100 - coverage) / 100.0
                    )
                    tasks.append(task)
                    
        except Exception as e:
            self.logger.error("Error in test gap discovery", error=str(e))
            
        return tasks
    
    async def _discover_documentation_needs(self, repo_path: str) -> List[IntelligentTask]:
        """Discover documentation gaps and needs"""
        tasks = []
        
        try:
            # Check for missing docstrings
            undocumented_functions = await self._find_undocumented_functions(repo_path)
            
            for func_info in undocumented_functions:
                task = IntelligentTask(
                    id=f"docstring_{func_info['file_stem']}_{func_info['line']}_{int(time.time())}",
                    title=f"Add docstring to {func_info['name']}",
                    description=f"Function '{func_info['name']}' in {func_info['file']} lacks documentation",
                    priority=5,
                    task_type="documentation",
                    file_path=func_info['file'],
                    line_number=func_info['line'],
                    complexity_score=2.0,
                    estimated_effort=15,
                    confidence=0.95,
                    suggested_approach="Add comprehensive docstring with parameters, returns, and examples",
                    business_impact="medium",
                    technical_debt_score=0.3
                )
                tasks.append(task)
                
            # Check for missing README sections
            readme_tasks = await self._analyze_readme_completeness(repo_path)
            tasks.extend(readme_tasks)
            
        except Exception as e:
            self.logger.error("Error in documentation discovery", error=str(e))
            
        return tasks
    
    async def _discover_dependency_issues(self, repo_path: str) -> List[IntelligentTask]:
        """Discover dependency-related issues"""
        tasks = []
        
        try:
            # Check for outdated dependencies
            outdated_deps = await self._check_outdated_dependencies(repo_path)
            
            for dep_name, current_version, latest_version in outdated_deps:
                severity = self._calculate_dependency_severity(current_version, latest_version)
                
                task = IntelligentTask(
                    id=f"dep_update_{dep_name}_{int(time.time())}",
                    title=f"Update {dep_name} dependency",
                    description=f"Update from {current_version} to {latest_version}",
                    priority=severity,
                    task_type="dependencies",
                    complexity_score=3.0 if severity >= 8 else 1.0,
                    estimated_effort=30 if severity >= 8 else 15,
                    confidence=0.80,
                    suggested_approach="Update dependency and run tests to ensure compatibility",
                    business_impact="medium" if severity >= 8 else "low",
                    technical_debt_score=severity / 10.0
                )
                tasks.append(task)
                
            # Check for unused dependencies
            unused_deps = await self._find_unused_dependencies(repo_path)
            
            for dep_name in unused_deps:
                task = IntelligentTask(
                    id=f"dep_remove_{dep_name}_{int(time.time())}",
                    title=f"Remove unused dependency {dep_name}",
                    description=f"Dependency {dep_name} appears to be unused",
                    priority=4,
                    task_type="dependencies",
                    complexity_score=1.0,
                    estimated_effort=10,
                    confidence=0.70,
                    suggested_approach="Verify dependency is unused and remove from requirements",
                    business_impact="low",
                    technical_debt_score=0.2
                )
                tasks.append(task)
                
        except Exception as e:
            self.logger.error("Error in dependency discovery", error=str(e))
            
        return tasks
    
    async def _discover_refactoring_opportunities(self, repo_path: str) -> List[IntelligentTask]:
        """Discover code refactoring opportunities"""
        tasks = []
        
        try:
            # Detect code duplication
            duplications = await self._detect_code_duplication(repo_path)
            
            for duplication in duplications:
                task = IntelligentTask(
                    id=f"duplication_{len(tasks)}_{int(time.time())}",
                    title=f"Eliminate code duplication",
                    description=f"Duplicate code found in {len(duplication['files'])} files",
                    priority=6,
                    task_type="refactoring",
                    complexity_score=len(duplication['files']) * 2.0,
                    estimated_effort=45 + (len(duplication['files']) * 15),
                    confidence=0.75,
                    suggested_approach="Extract common functionality into shared utility",
                    related_files=duplication['files'],
                    business_impact="medium",
                    technical_debt_score=0.6
                )
                tasks.append(task)
                
            # Detect long parameter lists
            long_param_functions = await self._detect_long_parameter_lists(repo_path)
            
            for func_info in long_param_functions:
                task = IntelligentTask(
                    id=f"long_params_{func_info['file_stem']}_{func_info['line']}_{int(time.time())}",
                    title=f"Refactor function with long parameter list",
                    description=f"Function '{func_info['name']}' has {func_info['param_count']} parameters",
                    priority=5,
                    task_type="refactoring",
                    file_path=func_info['file'],
                    line_number=func_info['line'],
                    complexity_score=func_info['param_count'] / 3.0,
                    estimated_effort=30,
                    confidence=0.80,
                    suggested_approach="Use parameter object or builder pattern",
                    business_impact="medium",
                    technical_debt_score=func_info['param_count'] / 15.0
                )
                tasks.append(task)
                
        except Exception as e:
            self.logger.error("Error in refactoring discovery", error=str(e))
            
        return tasks
    
    async def _discover_github_insights(self, repo_path: str) -> List[IntelligentTask]:
        """Discover insights from GitHub data"""
        tasks = []
        
        if not self.github_client:
            return tasks
            
        try:
            # This would typically analyze GitHub issues, PRs, and activity
            # For now, returning empty list as GitHub analysis requires repo access
            pass
        except Exception as e:
            self.logger.error("Error in GitHub discovery", error=str(e))
            
        return tasks
    
    # Helper methods for analysis
    
    async def _analyze_code_metrics(self, file_path: Path) -> CodeMetrics:
        """Analyze code metrics for a Python file"""
        metrics = CodeMetrics()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST for complexity analysis
            tree = ast.parse(content)
            
            # Count lines of code (excluding comments and blank lines)
            lines = content.split('\n')
            metrics.lines_of_code = len([line for line in lines 
                                       if line.strip() and not line.strip().startswith('#')])
            
            # Calculate cyclomatic complexity (simplified)
            complexity_visitor = ComplexityVisitor()
            complexity_visitor.visit(tree)
            metrics.cyclomatic_complexity = complexity_visitor.complexity
            
            # Estimate maintainability index (simplified formula)
            metrics.maintainability_index = max(0, 
                171 - 5.2 * complexity_visitor.complexity - 0.23 * metrics.lines_of_code)
            
        except Exception as e:
            self.logger.error("Error analyzing code metrics", file=str(file_path), error=str(e))
            
        return metrics
    
    def _is_ignored_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis"""
        ignored_patterns = [
            "__pycache__", ".git", ".pytest_cache", "venv", "env",
            "node_modules", ".tox", "build", "dist"
        ]
        
        for pattern in ignored_patterns:
            if pattern in str(file_path):
                return True
                
        return False
    
    async def _build_dependency_graph(self, repo_path: str) -> Dict[str, Set[str]]:
        """Build module dependency graph"""
        graph = {}
        
        try:
            for py_file in Path(repo_path).glob("**/*.py"):
                if py_file.is_file() and not self._is_ignored_file(py_file):
                    imports = await self._extract_imports(py_file)
                    graph[str(py_file)] = imports
        except Exception as e:
            self.logger.error("Error building dependency graph", error=str(e))
            
        return graph
    
    async def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract import statements from a Python file"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        
        except Exception as e:
            self.logger.error("Error extracting imports", file=str(file_path), error=str(e))
            
        return imports
    
    def _detect_circular_dependencies(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect circular dependencies in the graph"""
        # Simplified cycle detection - would use proper graph algorithms in production
        return []
    
    async def _prioritize_tasks_with_ai(self, tasks: List[IntelligentTask]) -> List[IntelligentTask]:
        """Use AI-based prioritization considering multiple factors"""
        
        for task in tasks:
            # Calculate composite priority score
            priority_factors = {
                'complexity': task.complexity_score * 0.2,
                'business_impact': self._impact_to_score(task.business_impact) * 0.3,
                'technical_debt': task.technical_debt_score * 0.25,
                'confidence': task.confidence * 0.15,
                'effort': (100 - min(task.estimated_effort, 100)) / 100 * 0.1
            }
            
            # Calculate weighted priority (1-10 scale)
            weighted_priority = sum(priority_factors.values()) * 10
            task.priority = max(1, min(10, int(weighted_priority)))
        
        # Sort by priority (highest first)
        return sorted(tasks, key=lambda x: x.priority, reverse=True)
    
    def _impact_to_score(self, impact: str) -> float:
        """Convert business impact to numeric score"""
        impact_map = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        return impact_map.get(impact, 0.4)
    
    def _deduplicate_tasks(self, tasks: List[IntelligentTask]) -> List[IntelligentTask]:
        """Remove duplicate and similar tasks"""
        # Simplified deduplication - would use more sophisticated similarity in production
        seen_titles = set()
        deduplicated = []
        
        for task in tasks:
            if task.title not in seen_titles:
                seen_titles.add(task.title)
                deduplicated.append(task)
                
        return deduplicated
    
    # Placeholder methods for specific analysis features
    
    async def _detect_large_modules(self, repo_path: str) -> List[Tuple[str, CodeMetrics]]:
        return []
    
    async def _analyze_performance_patterns(self, file_path: Path) -> List[Dict]:
        return []
    
    async def _analyze_security_patterns(self, file_path: Path) -> List[Dict]:
        return []
    
    async def _analyze_test_coverage(self, repo_path: str) -> Dict[str, float]:
        return {}
    
    async def _find_undocumented_functions(self, repo_path: str) -> List[Dict]:
        return []
    
    async def _analyze_readme_completeness(self, repo_path: str) -> List[IntelligentTask]:
        return []
    
    async def _check_outdated_dependencies(self, repo_path: str) -> List[Tuple[str, str, str]]:
        return []
    
    def _calculate_dependency_severity(self, current: str, latest: str) -> int:
        return 5
    
    async def _find_unused_dependencies(self, repo_path: str) -> List[str]:
        return []
    
    async def _detect_code_duplication(self, repo_path: str) -> List[Dict]:
        return []
    
    async def _detect_long_parameter_lists(self, repo_path: str) -> List[Dict]:
        return []


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity"""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Try(self, node):
        self.complexity += len(node.handlers)
        self.generic_visit(node)
        
    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)


# Example usage
async def main():
    """Example usage of intelligent task discovery"""
    discovery = IntelligentTaskDiscovery()
    tasks = await discovery.discover_intelligent_tasks(".")
    
    print(f"Discovered {len(tasks)} intelligent tasks:")
    for task in tasks[:10]:  # Show top 10
        print(f"- {task.title} (Priority: {task.priority}, Effort: {task.estimated_effort}min)")


if __name__ == "__main__":
    asyncio.run(main())