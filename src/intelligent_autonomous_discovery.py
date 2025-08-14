#!/usr/bin/env python3
"""
Intelligent Autonomous Task Discovery System - Generation 1
Advanced AI-powered task identification and prioritization system
"""

import asyncio
import json
import logging
import time
import os
import re
import ast
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NICE_TO_HAVE = "nice_to_have"

class TaskCategory(Enum):
    """Task categories"""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"

class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"      # <1 hour
    SIMPLE = "simple"        # 1-4 hours
    MODERATE = "moderate"    # 1-2 days
    COMPLEX = "complex"      # 3-5 days
    EPIC = "epic"           # >1 week

@dataclass
class TaskMetrics:
    """Metrics for task analysis"""
    lines_of_code_affected: int = 0
    files_affected: int = 0
    estimated_hours: float = 0.0
    risk_score: float = 0.0
    impact_score: float = 0.0
    technical_debt_score: float = 0.0

@dataclass
class DiscoveredTask:
    """Represents a discovered task"""
    id: str
    title: str
    description: str
    category: TaskCategory
    priority: TaskPriority
    complexity: TaskComplexity
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_context: Optional[str] = None
    suggested_solution: Optional[str] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class IntelligentTaskDiscovery:
    """
    Advanced task discovery system using pattern recognition and AI heuristics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.discovered_tasks: List[DiscoveredTask] = []
        self.analyzers = {
            'todo_analyzer': self._analyze_todo_comments,
            'bug_analyzer': self._analyze_potential_bugs,
            'performance_analyzer': self._analyze_performance_issues,
            'security_analyzer': self._analyze_security_issues,
            'refactor_analyzer': self._analyze_refactoring_opportunities,
            'test_analyzer': self._analyze_testing_gaps,
            'documentation_analyzer': self._analyze_documentation_gaps,
            'dependency_analyzer': self._analyze_dependency_issues,
            'code_smell_analyzer': self._analyze_code_smells,
            'architecture_analyzer': self._analyze_architecture_issues
        }
        
        # Pattern recognition rules
        self.patterns = self._initialize_patterns()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'source_paths': ['src/', 'lib/', 'app/'],
            'test_paths': ['tests/', 'test/', 'spec/'],
            'doc_paths': ['docs/', 'documentation/', 'README.md'],
            'ignore_patterns': [
                '__pycache__',
                '.git',
                '.pytest_cache',
                'node_modules',
                '.venv',
                'venv'
            ],
            'analysis_depth': 'deep',  # shallow, medium, deep
            'min_confidence_threshold': 0.6,
            'max_tasks_per_category': 10,
            'enable_ai_suggestions': True,
            'priority_weights': {
                'security': 1.0,
                'bug': 0.9,
                'performance': 0.7,
                'refactor': 0.5,
                'documentation': 0.3
            }
        }
    
    def _initialize_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize pattern recognition rules"""
        return {
            'todo_patterns': [
                {
                    'pattern': r'TODO[\s\:]+(.+)',
                    'category': TaskCategory.MAINTENANCE,
                    'priority': TaskPriority.MEDIUM,
                    'complexity': TaskComplexity.SIMPLE
                },
                {
                    'pattern': r'FIXME[\s\:]+(.+)',
                    'category': TaskCategory.BUG_FIX,
                    'priority': TaskPriority.HIGH,
                    'complexity': TaskComplexity.MODERATE
                },
                {
                    'pattern': r'HACK[\s\:]+(.+)',
                    'category': TaskCategory.REFACTOR,
                    'priority': TaskPriority.MEDIUM,
                    'complexity': TaskComplexity.MODERATE
                },
                {
                    'pattern': r'XXX[\s\:]+(.+)',
                    'category': TaskCategory.BUG_FIX,
                    'priority': TaskPriority.HIGH,
                    'complexity': TaskComplexity.SIMPLE
                }
            ],
            'bug_patterns': [
                {
                    'pattern': r'except\s*:\s*pass',
                    'description': 'Silent exception handling - potential bug masking',
                    'category': TaskCategory.BUG_FIX,
                    'priority': TaskPriority.HIGH,
                    'complexity': TaskComplexity.SIMPLE
                },
                {
                    'pattern': r'print\s*\([\'"]debug',
                    'description': 'Debug print statements left in code',
                    'category': TaskCategory.MAINTENANCE,
                    'priority': TaskPriority.LOW,
                    'complexity': TaskComplexity.TRIVIAL
                },
                {
                    'pattern': r'time\.sleep\([0-9]+\)',
                    'description': 'Hard-coded sleep statements - potential timing issues',
                    'category': TaskCategory.REFACTOR,
                    'priority': TaskPriority.MEDIUM,
                    'complexity': TaskComplexity.SIMPLE
                }
            ],
            'security_patterns': [
                {
                    'pattern': r'password\s*=\s*[\'"][^\'"]+[\'"]',
                    'description': 'Hard-coded password detected',
                    'category': TaskCategory.SECURITY,
                    'priority': TaskPriority.CRITICAL,
                    'complexity': TaskComplexity.SIMPLE
                },
                {
                    'pattern': r'eval\s*\(',
                    'description': 'Use of eval() - security risk',
                    'category': TaskCategory.SECURITY,
                    'priority': TaskPriority.CRITICAL,
                    'complexity': TaskComplexity.MODERATE
                },
                {
                    'pattern': r'shell\s*=\s*True',
                    'description': 'Shell injection vulnerability',
                    'category': TaskCategory.SECURITY,
                    'priority': TaskPriority.HIGH,
                    'complexity': TaskComplexity.MODERATE
                }
            ],
            'performance_patterns': [
                {
                    'pattern': r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(',
                    'description': 'Inefficient iteration over range(len())',
                    'category': TaskCategory.PERFORMANCE,
                    'priority': TaskPriority.MEDIUM,
                    'complexity': TaskComplexity.SIMPLE
                },
                {
                    'pattern': r'\.append\s*\([^)]+\)\s*[\r\n]\s*for',
                    'description': 'Potential list comprehension opportunity',
                    'category': TaskCategory.PERFORMANCE,
                    'priority': TaskPriority.LOW,
                    'complexity': TaskComplexity.SIMPLE
                }
            ]
        }
    
    async def discover_all_tasks(self, repo_path: str = ".") -> List[DiscoveredTask]:
        """Discover all tasks in the repository"""
        logger.info(f"Starting intelligent task discovery for {repo_path}")
        
        self.discovered_tasks = []
        
        # Run all analyzers
        for analyzer_name, analyzer_func in self.analyzers.items():
            try:
                logger.info(f"Running {analyzer_name}")
                tasks = await analyzer_func(repo_path)
                self.discovered_tasks.extend(tasks)
                logger.info(f"Found {len(tasks)} tasks from {analyzer_name}")
                
            except Exception as e:
                logger.error(f"Error in {analyzer_name}: {str(e)}")
        
        # Post-process tasks
        self._deduplicate_tasks()
        self._prioritize_tasks()
        self._estimate_complexity()
        
        # Filter by confidence threshold
        confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        filtered_tasks = [
            task for task in self.discovered_tasks 
            if task.confidence >= confidence_threshold
        ]
        
        logger.info(f"Discovery complete: {len(filtered_tasks)} high-confidence tasks found")
        
        return filtered_tasks
    
    async def _analyze_todo_comments(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze TODO, FIXME, and similar comments"""
        tasks = []
        
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern_info in self.patterns['todo_patterns']:
                            match = re.search(pattern_info['pattern'], line, re.IGNORECASE)
                            if match:
                                description = match.group(1).strip() if match.groups() else line.strip()
                                
                                task = DiscoveredTask(
                                    id=f"todo_{file_path.stem}_{line_num}",
                                    title=f"Address {pattern_info['pattern'].split('[')[0]} in {file_path.name}",
                                    description=description,
                                    category=pattern_info['category'],
                                    priority=pattern_info['priority'],
                                    complexity=pattern_info['complexity'],
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    code_context=self._get_code_context(lines, line_num),
                                    confidence=0.9,
                                    tags=['todo', 'comment']
                                )
                                
                                tasks.append(task)
                                
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path}: {str(e)}")
        
        return tasks
    
    async def _analyze_potential_bugs(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze code for potential bugs using pattern recognition"""
        tasks = []
        
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern_info in self.patterns['bug_patterns']:
                            if re.search(pattern_info['pattern'], line):
                                task = DiscoveredTask(
                                    id=f"bug_{file_path.stem}_{line_num}",
                                    title=f"Fix potential bug: {pattern_info['description']}",
                                    description=f"Found pattern '{pattern_info['pattern']}' which may indicate: {pattern_info['description']}",
                                    category=pattern_info['category'],
                                    priority=pattern_info['priority'],
                                    complexity=pattern_info['complexity'],
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    code_context=self._get_code_context(lines, line_num),
                                    confidence=0.7,
                                    tags=['bug', 'pattern-match']
                                )
                                
                                tasks.append(task)
                                
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for bugs: {str(e)}")
        
        return tasks
    
    async def _analyze_performance_issues(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze code for performance optimization opportunities"""
        tasks = []
        
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()
                    
                    # Check for performance anti-patterns
                    for line_num, line in enumerate(lines, 1):
                        for pattern_info in self.patterns['performance_patterns']:
                            if re.search(pattern_info['pattern'], line):
                                task = DiscoveredTask(
                                    id=f"perf_{file_path.stem}_{line_num}",
                                    title=f"Optimize performance: {pattern_info['description']}",
                                    description=f"Performance improvement opportunity: {pattern_info['description']}",
                                    category=pattern_info['category'],
                                    priority=pattern_info['priority'],
                                    complexity=pattern_info['complexity'],
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    code_context=self._get_code_context(lines, line_num),
                                    confidence=0.6,
                                    tags=['performance', 'optimization']
                                )
                                
                                tasks.append(task)
                    
                    # Analyze function complexity for performance issues
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                complexity = self._calculate_function_complexity(node)
                                if complexity > 15:  # High complexity threshold
                                    task = DiscoveredTask(
                                        id=f"complex_{file_path.stem}_{node.name}",
                                        title=f"Refactor complex function: {node.name}",
                                        description=f"Function '{node.name}' has high complexity ({complexity}), consider refactoring",
                                        category=TaskCategory.REFACTOR,
                                        priority=TaskPriority.MEDIUM,
                                        complexity=TaskComplexity.MODERATE,
                                        file_path=str(file_path),
                                        line_number=node.lineno,
                                        confidence=0.8,
                                        tags=['complexity', 'refactor']
                                    )
                                    
                                    tasks.append(task)
                    except:
                        pass  # Skip AST parsing errors
                                
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for performance: {str(e)}")
        
        return tasks
    
    async def _analyze_security_issues(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze code for security vulnerabilities"""
        tasks = []
        
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern_info in self.patterns['security_patterns']:
                            if re.search(pattern_info['pattern'], line):
                                task = DiscoveredTask(
                                    id=f"sec_{file_path.stem}_{line_num}",
                                    title=f"Security issue: {pattern_info['description']}",
                                    description=f"Security vulnerability detected: {pattern_info['description']}",
                                    category=pattern_info['category'],
                                    priority=pattern_info['priority'],
                                    complexity=pattern_info['complexity'],
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    code_context=self._get_code_context([l.rstrip() for l in lines], line_num),
                                    confidence=0.85,
                                    tags=['security', 'vulnerability']
                                )
                                
                                tasks.append(task)
                                
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for security: {str(e)}")
        
        return tasks
    
    async def _analyze_refactoring_opportunities(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze code for refactoring opportunities"""
        tasks = []
        
        # Find duplicate code blocks
        code_blocks = {}
        
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Look for long functions (>50 lines)
                    try:
                        tree = ast.parse(''.join(lines))
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                if hasattr(node, 'end_lineno'):
                                    func_length = node.end_lineno - node.lineno
                                    if func_length > 50:
                                        task = DiscoveredTask(
                                            id=f"long_func_{file_path.stem}_{node.name}",
                                            title=f"Refactor long function: {node.name}",
                                            description=f"Function '{node.name}' is {func_length} lines long, consider breaking it down",
                                            category=TaskCategory.REFACTOR,
                                            priority=TaskPriority.MEDIUM,
                                            complexity=TaskComplexity.MODERATE,
                                            file_path=str(file_path),
                                            line_number=node.lineno,
                                            confidence=0.7,
                                            tags=['refactor', 'long-function']
                                        )
                                        
                                        tasks.append(task)
                    except:
                        pass
                                
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for refactoring: {str(e)}")
        
        return tasks
    
    async def _analyze_testing_gaps(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze for missing or inadequate tests"""
        tasks = []
        
        # Find source files without corresponding tests
        source_files = set()
        test_files = set()
        
        # Collect source files
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if full_path.exists():
                for file_path in full_path.rglob("*.py"):
                    if self._should_ignore_file(file_path):
                        continue
                    source_files.add(file_path.stem)
        
        # Collect test files
        for test_path in self.config['test_paths']:
            full_path = Path(repo_path) / test_path
            if full_path.exists():
                for file_path in full_path.rglob("test_*.py"):
                    test_files.add(file_path.stem.replace('test_', ''))
                for file_path in full_path.rglob("*_test.py"):
                    test_files.add(file_path.stem.replace('_test', ''))
        
        # Find untested modules
        untested = source_files - test_files
        
        for module_name in untested:
            if module_name not in ['__init__']:
                task = DiscoveredTask(
                    id=f"test_{module_name}",
                    title=f"Add tests for module: {module_name}",
                    description=f"Module '{module_name}' appears to lack test coverage",
                    category=TaskCategory.TESTING,
                    priority=TaskPriority.MEDIUM,
                    complexity=TaskComplexity.MODERATE,
                    confidence=0.6,
                    tags=['testing', 'coverage']
                )
                
                tasks.append(task)
        
        return tasks
    
    async def _analyze_documentation_gaps(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze for missing or inadequate documentation"""
        tasks = []
        
        # Check for missing README
        readme_path = Path(repo_path) / "README.md"
        if not readme_path.exists():
            task = DiscoveredTask(
                id="missing_readme",
                title="Create README.md file",
                description="Repository is missing a README.md file",
                category=TaskCategory.DOCUMENTATION,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.SIMPLE,
                confidence=1.0,
                tags=['documentation', 'readme']
            )
            tasks.append(task)
        
        # Check for functions without docstrings
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            if not ast.get_docstring(node):
                                task = DiscoveredTask(
                                    id=f"doc_{file_path.stem}_{node.name}",
                                    title=f"Add docstring to {type(node).__name__.lower()}: {node.name}",
                                    description=f"{type(node).__name__} '{node.name}' is missing documentation",
                                    category=TaskCategory.DOCUMENTATION,
                                    priority=TaskPriority.LOW,
                                    complexity=TaskComplexity.TRIVIAL,
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    confidence=0.8,
                                    tags=['documentation', 'docstring']
                                )
                                
                                tasks.append(task)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for documentation: {str(e)}")
        
        return tasks
    
    async def _analyze_dependency_issues(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze dependency-related issues"""
        tasks = []
        
        requirements_path = Path(repo_path) / "requirements.txt"
        if requirements_path.exists():
            try:
                with open(requirements_path, 'r') as f:
                    requirements = f.readlines()
                
                # Check for unpinned dependencies
                for line_num, line in enumerate(requirements, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' not in line and '>=' not in line and '<=' not in line:
                            task = DiscoveredTask(
                                id=f"dep_pin_{line_num}",
                                title=f"Pin dependency version: {line}",
                                description=f"Dependency '{line}' should have a pinned version for reproducible builds",
                                category=TaskCategory.MAINTENANCE,
                                priority=TaskPriority.MEDIUM,
                                complexity=TaskComplexity.TRIVIAL,
                                file_path=str(requirements_path),
                                line_number=line_num,
                                confidence=0.7,
                                tags=['dependencies', 'version-pinning']
                            )
                            
                            tasks.append(task)
                            
            except Exception as e:
                logger.warning(f"Error analyzing requirements.txt: {str(e)}")
        
        return tasks
    
    async def _analyze_code_smells(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze for code smells and anti-patterns"""
        tasks = []
        
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Check for god classes (>500 lines)
                    if len(lines) > 500:
                        task = DiscoveredTask(
                            id=f"god_class_{file_path.stem}",
                            title=f"Refactor large class/module: {file_path.name}",
                            description=f"File '{file_path.name}' has {len(lines)} lines and may be too large",
                            category=TaskCategory.REFACTOR,
                            priority=TaskPriority.MEDIUM,
                            complexity=TaskComplexity.COMPLEX,
                            file_path=str(file_path),
                            confidence=0.6,
                            tags=['refactor', 'god-class']
                        )
                        
                        tasks.append(task)
                    
                    # Check for dead code (functions with no calls)
                    content = ''.join(lines)
                    try:
                        tree = ast.parse(content)
                        defined_functions = set()
                        called_functions = set()
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                defined_functions.add(node.name)
                            elif isinstance(node, ast.Call):
                                if hasattr(node.func, 'id'):
                                    called_functions.add(node.func.id)
                        
                        # Find potentially unused functions
                        unused_functions = defined_functions - called_functions
                        for func_name in unused_functions:
                            if not func_name.startswith('_'):  # Skip private functions
                                task = DiscoveredTask(
                                    id=f"unused_{file_path.stem}_{func_name}",
                                    title=f"Review potentially unused function: {func_name}",
                                    description=f"Function '{func_name}' may be unused - consider removal or testing",
                                    category=TaskCategory.MAINTENANCE,
                                    priority=TaskPriority.LOW,
                                    complexity=TaskComplexity.SIMPLE,
                                    file_path=str(file_path),
                                    confidence=0.5,  # Lower confidence for dead code detection
                                    tags=['dead-code', 'cleanup']
                                )
                                
                                tasks.append(task)
                    except:
                        pass
                                
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for code smells: {str(e)}")
        
        return tasks
    
    async def _analyze_architecture_issues(self, repo_path: str) -> List[DiscoveredTask]:
        """Analyze for architectural issues and improvements"""
        tasks = []
        
        # Check for missing configuration management
        config_files = ['config.py', 'settings.py', 'config.json', '.env']
        config_found = any((Path(repo_path) / cf).exists() for cf in config_files)
        
        if not config_found:
            task = DiscoveredTask(
                id="missing_config",
                title="Implement configuration management",
                description="No configuration files found - consider adding proper config management",
                category=TaskCategory.FEATURE,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.MODERATE,
                confidence=0.7,
                tags=['architecture', 'configuration']
            )
            tasks.append(task)
        
        # Check for missing error handling patterns
        for source_path in self.config['source_paths']:
            full_path = Path(repo_path) / source_path
            if not full_path.exists():
                continue
                
            for file_path in full_path.rglob("*.py"):
                if self._should_ignore_file(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Count try/except blocks vs functions
                    tree = ast.parse(content)
                    function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                    try_count = len([n for n in ast.walk(tree) if isinstance(n, ast.Try)])
                    
                    if function_count > 5 and try_count == 0:
                        task = DiscoveredTask(
                            id=f"error_handling_{file_path.stem}",
                            title=f"Add error handling to {file_path.name}",
                            description=f"File has {function_count} functions but no error handling",
                            category=TaskCategory.REFACTOR,
                            priority=TaskPriority.MEDIUM,
                            complexity=TaskComplexity.MODERATE,
                            file_path=str(file_path),
                            confidence=0.6,
                            tags=['error-handling', 'robustness']
                        )
                        tasks.append(task)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path} for architecture: {str(e)}")
        
        return tasks
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        for pattern in self.config['ignore_patterns']:
            if pattern in str(file_path):
                return True
        return False
    
    def _get_code_context(self, lines: List[str], line_number: int, context_size: int = 3) -> str:
        """Get code context around a specific line"""
        start = max(0, line_number - context_size - 1)
        end = min(len(lines), line_number + context_size)
        
        context_lines = []
        for i in range(start, end):
            marker = " -> " if i == line_number - 1 else "    "
            context_lines.append(f"{i+1:3d}{marker}{lines[i].rstrip()}")
        
        return "\n".join(context_lines)
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, (ast.BoolOp, ast.Compare)):
                complexity += 1
        
        return complexity
    
    def _deduplicate_tasks(self):
        """Remove duplicate tasks based on similarity"""
        seen_signatures = set()
        unique_tasks = []
        
        for task in self.discovered_tasks:
            # Create signature based on file, line, and category
            signature = f"{task.file_path}:{task.line_number}:{task.category.value}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_tasks.append(task)
        
        self.discovered_tasks = unique_tasks
    
    def _prioritize_tasks(self):
        """Adjust task priorities based on configured weights"""
        priority_weights = self.config.get('priority_weights', {})
        
        for task in self.discovered_tasks:
            # Apply category-based priority adjustments
            if task.category.value in priority_weights:
                weight = priority_weights[task.category.value]
                
                # Boost priority for high-weight categories
                if weight >= 0.8 and task.priority != TaskPriority.CRITICAL:
                    if task.priority == TaskPriority.MEDIUM:
                        task.priority = TaskPriority.HIGH
                    elif task.priority == TaskPriority.LOW:
                        task.priority = TaskPriority.MEDIUM
    
    def _estimate_complexity(self):
        """Estimate task complexity based on various factors"""
        for task in self.discovered_tasks:
            # Complexity is already set during analysis, but we can refine it
            if task.category == TaskCategory.SECURITY:
                # Security tasks tend to be more complex
                if task.complexity == TaskComplexity.SIMPLE:
                    task.complexity = TaskComplexity.MODERATE
            
            elif task.category == TaskCategory.REFACTOR:
                # Refactoring can be complex depending on scope
                if task.metrics.lines_of_code_affected > 100:
                    task.complexity = TaskComplexity.COMPLEX
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> List[DiscoveredTask]:
        """Get tasks filtered by priority"""
        return [task for task in self.discovered_tasks if task.priority == priority]
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[DiscoveredTask]:
        """Get tasks filtered by category"""
        return [task for task in self.discovered_tasks if task.category == category]
    
    def get_critical_tasks(self) -> List[DiscoveredTask]:
        """Get all critical priority tasks"""
        return self.get_tasks_by_priority(TaskPriority.CRITICAL)
    
    def export_tasks_json(self, file_path: str):
        """Export discovered tasks to JSON file"""
        tasks_data = []
        
        for task in self.discovered_tasks:
            task_dict = {
                'id': task.id,
                'title': task.title,
                'description': task.description,
                'category': task.category.value,
                'priority': task.priority.value,
                'complexity': task.complexity.value,
                'file_path': task.file_path,
                'line_number': task.line_number,
                'confidence': task.confidence,
                'tags': task.tags,
                'created_at': task.created_at.isoformat()
            }
            tasks_data.append(task_dict)
        
        with open(file_path, 'w') as f:
            json.dump(tasks_data, f, indent=2)
        
        logger.info(f"Exported {len(tasks_data)} tasks to {file_path}")

# Example usage
async def main():
    """Example usage of intelligent task discovery"""
    
    print("🔍 Starting Intelligent Autonomous Task Discovery")
    
    # Initialize discovery system
    discovery = IntelligentTaskDiscovery()
    
    # Discover all tasks
    tasks = await discovery.discover_all_tasks(".")
    
    # Display results
    print(f"\n📋 Discovered {len(tasks)} high-confidence tasks")
    
    # Group by priority
    for priority in TaskPriority:
        priority_tasks = discovery.get_tasks_by_priority(priority)
        if priority_tasks:
            print(f"\n{priority.value.upper()} Priority ({len(priority_tasks)} tasks):")
            for task in priority_tasks[:5]:  # Show first 5
                print(f"  • {task.title}")
                if task.file_path:
                    print(f"    📁 {task.file_path}:{task.line_number or 'N/A'}")
                print(f"    📊 {task.category.value} | {task.complexity.value} | confidence: {task.confidence:.1f}")
                print()
    
    # Export to file
    discovery.export_tasks_json("discovered_tasks.json")
    
    print("✅ Task discovery completed!")

if __name__ == "__main__":
    asyncio.run(main())