#!/usr/bin/env python3
"""
Terragon Value Discovery Engine

Advanced value discovery system that continuously analyzes codebases to identify
high-value work items using multiple scoring algorithms (WSJF, ICE, Technical Debt).
"""

import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import aiofiles
from enum import Enum
import hashlib


class TaskType(Enum):
    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    FEATURE = "feature"
    TECHNICAL_DEBT = "technical_debt"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValueItem:
    """Represents a discovered value item (TODO, bug, enhancement, etc.)"""
    id: str
    title: str
    description: str
    task_type: TaskType
    file_path: str
    line_number: int
    priority: Priority
    
    # WSJF Components
    business_value: float
    time_criticality: float
    risk_reduction: float
    job_size: float
    wsjf_score: float
    
    # ICE Components  
    impact: float
    confidence: float
    ease: float
    ice_score: float
    
    # Technical Debt Score
    technical_debt_score: float
    
    # Metadata
    discovered_date: datetime
    last_updated: datetime
    source: str
    tags: List[str]
    estimated_effort_hours: float
    
    # Context
    surrounding_code: str
    dependencies: List[str]
    affected_components: List[str]


class ValueDiscoveryEngine:
    """Main engine for discovering and scoring value items"""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path(".")
        self.metrics_file = Path(".terragon/value-metrics.json")
        self.discovered_items: List[ValueItem] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "value_discovery": {
                "code_analysis": {
                    "patterns": {
                        "todo_patterns": ["TODO:", "FIXME:", "HACK:", "XXX:", "BUG:"],
                        "debt_patterns": ["@deprecated", "# DEBT:", "# TECHNICAL_DEBT"],
                        "security_patterns": ["# SECURITY:", "# VULN:", "password", "secret", "token"]
                    }
                },
                "scoring": {
                    "algorithm": "WSJF"
                }
            }
        }
    
    async def discover_value_items(self) -> List[ValueItem]:
        """Main method to discover all value items in the repository"""
        print("🔍 Starting comprehensive value discovery...")
        
        tasks = [
            self._discover_todo_items(),
            self._discover_failing_tests(),
            self._discover_security_vulnerabilities(),
            self._discover_performance_issues(),
            self._discover_technical_debt(),
            self._discover_documentation_gaps(),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results and deduplicate
        all_items = []
        for result in results:
            all_items.extend(result)
        
        self.discovered_items = self._deduplicate_items(all_items)
        
        # Score all items
        for item in self.discovered_items:
            self._calculate_scores(item)
        
        # Sort by WSJF score (highest first)
        self.discovered_items.sort(key=lambda x: x.wsjf_score, reverse=True)
        
        print(f"✅ Discovered {len(self.discovered_items)} value items")
        return self.discovered_items
    
    async def _discover_todo_items(self) -> List[ValueItem]:
        """Discover TODO/FIXME comments in code"""
        items = []
        patterns = self.config["value_discovery"]["code_analysis"]["patterns"]["todo_patterns"]
        
        # Compile regex patterns
        combined_pattern = "|".join(re.escape(p) for p in patterns)
        regex = re.compile(rf'({combined_pattern})\s*(.+)', re.IGNORECASE)
        
        # Scan Python files
        python_files = list(self.repo_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_exclude_file(file_path):
                continue
                
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        match = regex.search(line)
                        if match:
                            todo_type = match.group(1).upper()
                            description = match.group(2).strip()
                            
                            # Get surrounding context
                            context_start = max(0, line_num - 3)
                            context_end = min(len(lines), line_num + 3)
                            context = '\n'.join(lines[context_start:context_end])
                            
                            item = ValueItem(
                                id=self._generate_id(file_path, line_num, description),
                                title=f"{todo_type}: {description[:50]}...",
                                description=description,
                                task_type=self._classify_task_type(description, todo_type),
                                file_path=str(file_path),
                                line_number=line_num,
                                priority=Priority.MEDIUM,  # Will be recalculated
                                business_value=0,
                                time_criticality=0,
                                risk_reduction=0,
                                job_size=0,
                                wsjf_score=0,
                                impact=0,
                                confidence=0,
                                ease=0,
                                ice_score=0,
                                technical_debt_score=0,
                                discovered_date=datetime.now(),
                                last_updated=datetime.now(),
                                source="code_analysis",
                                tags=[todo_type.lower()],
                                estimated_effort_hours=self._estimate_effort(description),
                                surrounding_code=context,
                                dependencies=[],
                                affected_components=self._identify_components(file_path)
                            )
                            items.append(item)
                            
            except Exception as e:
                print(f"Warning: Could not scan {file_path}: {e}")
        
        print(f"📝 Found {len(items)} TODO/FIXME items")
        return items
    
    async def _discover_failing_tests(self) -> List[ValueItem]:
        """Discover failing tests and test gaps"""
        items = []
        
        # This would integrate with pytest to find failing tests
        # For now, we'll scan for test files and identify potential issues
        test_files = list(self.repo_root.rglob("test_*.py")) + list(self.repo_root.rglob("*_test.py"))
        
        for test_file in test_files:
            try:
                async with aiofiles.open(test_file, 'r') as f:
                    content = await f.read()
                    
                    # Look for @pytest.mark.skip or @pytest.mark.xfail
                    skip_pattern = re.compile(r'@pytest\.mark\.(skip|xfail).*\n.*def (test_\w+)', re.MULTILINE)
                    
                    for match in skip_pattern.finditer(content):
                        mark_type = match.group(1)
                        test_name = match.group(2)
                        
                        line_num = content[:match.start()].count('\n') + 1
                        
                        item = ValueItem(
                            id=self._generate_id(test_file, line_num, test_name),
                            title=f"Fix {mark_type}ped test: {test_name}",
                            description=f"Test {test_name} is currently {mark_type}ped and needs attention",
                            task_type=TaskType.TESTING,
                            file_path=str(test_file),
                            line_number=line_num,
                            priority=Priority.HIGH,
                            business_value=0,
                            time_criticality=0,
                            risk_reduction=0,
                            job_size=0,
                            wsjf_score=0,
                            impact=0,
                            confidence=0,
                            ease=0,
                            ice_score=0,
                            technical_debt_score=0,
                            discovered_date=datetime.now(),
                            last_updated=datetime.now(),
                            source="test_analysis",
                            tags=["testing", mark_type],
                            estimated_effort_hours=2.0,
                            surrounding_code="",
                            dependencies=[],
                            affected_components=["testing"]
                        )
                        items.append(item)
                        
            except Exception as e:
                print(f"Warning: Could not analyze test file {test_file}: {e}")
        
        print(f"🧪 Found {len(items)} test-related items")
        return items
    
    async def _discover_security_vulnerabilities(self) -> List[ValueItem]:
        """Discover potential security vulnerabilities"""
        items = []
        
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'subprocess\.call\([^)]*shell=True', "Shell injection vulnerability"),
            (r'eval\s*\(', "Eval() usage - potential code injection"),
            (r'exec\s*\(', "Exec() usage - potential code injection"),
            (r'pickle\.loads\s*\(', "Unsafe pickle deserialization"),
        ]
        
        python_files = list(self.repo_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_exclude_file(file_path):
                continue
                
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    lines = content.split('\n')
                    
                    for pattern, description in security_patterns:
                        regex = re.compile(pattern, re.IGNORECASE)
                        
                        for line_num, line in enumerate(lines, 1):
                            if regex.search(line):
                                item = ValueItem(
                                    id=self._generate_id(file_path, line_num, description),
                                    title=f"Security: {description}",
                                    description=f"{description} at line {line_num}",
                                    task_type=TaskType.SECURITY,
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    priority=Priority.CRITICAL,
                                    business_value=0,
                                    time_criticality=0,
                                    risk_reduction=0,
                                    job_size=0,
                                    wsjf_score=0,
                                    impact=0,
                                    confidence=0,
                                    ease=0,
                                    ice_score=0,
                                    technical_debt_score=0,
                                    discovered_date=datetime.now(),
                                    last_updated=datetime.now(),
                                    source="security_analysis",
                                    tags=["security", "vulnerability"],
                                    estimated_effort_hours=4.0,
                                    surrounding_code=line.strip(),
                                    dependencies=[],
                                    affected_components=["security"]
                                )
                                items.append(item)
                                
            except Exception as e:
                print(f"Warning: Could not scan {file_path} for security issues: {e}")
        
        print(f"🔒 Found {len(items)} potential security issues")
        return items
    
    async def _discover_performance_issues(self) -> List[ValueItem]:
        """Discover potential performance issues"""
        items = []
        
        performance_patterns = [
            (r'time\.sleep\s*\(\s*[0-9]+', "Synchronous sleep in code"),
            (r'for\s+\w+\s+in\s+.*\.keys\(\):', "Inefficient dictionary iteration"),
            (r'len\s*\(\s*list\s*\(', "Inefficient list length check"),
            (r'\.join\s*\(\s*\[.*for.*in.*\]', "List comprehension in join"),
        ]
        
        python_files = list(self.repo_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_exclude_file(file_path):
                continue
                
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    lines = content.split('\n')
                    
                    for pattern, description in performance_patterns:
                        regex = re.compile(pattern, re.IGNORECASE)
                        
                        for line_num, line in enumerate(lines, 1):
                            if regex.search(line):
                                item = ValueItem(
                                    id=self._generate_id(file_path, line_num, description),
                                    title=f"Performance: {description}",
                                    description=f"{description} at line {line_num}",
                                    task_type=TaskType.PERFORMANCE,
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    priority=Priority.MEDIUM,
                                    business_value=0,
                                    time_criticality=0,
                                    risk_reduction=0,
                                    job_size=0,
                                    wsjf_score=0,
                                    impact=0,
                                    confidence=0,
                                    ease=0,
                                    ice_score=0,
                                    technical_debt_score=0,
                                    discovered_date=datetime.now(),
                                    last_updated=datetime.now(),
                                    source="performance_analysis",
                                    tags=["performance"],
                                    estimated_effort_hours=3.0,
                                    surrounding_code=line.strip(),
                                    dependencies=[],
                                    affected_components=["performance"]
                                )
                                items.append(item)
                                
            except Exception as e:
                print(f"Warning: Could not scan {file_path} for performance issues: {e}")
        
        print(f"⚡ Found {len(items)} potential performance issues")
        return items
    
    async def _discover_technical_debt(self) -> List[ValueItem]:
        """Discover technical debt indicators"""
        items = []
        
        debt_patterns = [
            (r'#\s*DEBT:', "Technical debt marker"),
            (r'#\s*HACK:', "Code hack marker"),
            (r'#\s*FIXME:', "Code fix needed"),
            (r'@deprecated', "Deprecated code"),
            (r'raise\s+NotImplementedError', "Unimplemented functionality"),
        ]
        
        python_files = list(self.repo_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_exclude_file(file_path):
                continue
                
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    lines = content.split('\n')
                    
                    # Also check for long functions (> 50 lines)
                    current_function = None
                    function_start_line = 0
                    
                    for line_num, line in enumerate(lines, 1):
                        # Check for function definitions
                        func_match = re.match(r'\s*def\s+(\w+)', line)
                        if func_match:
                            # Check if previous function was too long
                            if current_function and (line_num - function_start_line) > 50:
                                item = ValueItem(
                                    id=self._generate_id(file_path, function_start_line, f"Long function {current_function}"),
                                    title=f"Refactor long function: {current_function}",
                                    description=f"Function {current_function} is {line_num - function_start_line} lines long and should be refactored",
                                    task_type=TaskType.REFACTORING,
                                    file_path=str(file_path),
                                    line_number=function_start_line,
                                    priority=Priority.LOW,
                                    business_value=0,
                                    time_criticality=0,
                                    risk_reduction=0,
                                    job_size=0,
                                    wsjf_score=0,
                                    impact=0,
                                    confidence=0,
                                    ease=0,
                                    ice_score=0,
                                    technical_debt_score=0,
                                    discovered_date=datetime.now(),
                                    last_updated=datetime.now(),
                                    source="debt_analysis",
                                    tags=["technical_debt", "refactoring"],
                                    estimated_effort_hours=6.0,
                                    surrounding_code="",
                                    dependencies=[],
                                    affected_components=["maintainability"]
                                )
                                items.append(item)
                            
                            current_function = func_match.group(1)
                            function_start_line = line_num
                        
                        # Check for debt patterns
                        for pattern, description in debt_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                item = ValueItem(
                                    id=self._generate_id(file_path, line_num, description),
                                    title=f"Technical Debt: {description}",
                                    description=f"{description} at line {line_num}",
                                    task_type=TaskType.TECHNICAL_DEBT,
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    priority=Priority.LOW,
                                    business_value=0,
                                    time_criticality=0,
                                    risk_reduction=0,
                                    job_size=0,
                                    wsjf_score=0,
                                    impact=0,
                                    confidence=0,
                                    ease=0,
                                    ice_score=0,
                                    technical_debt_score=0,
                                    discovered_date=datetime.now(),
                                    last_updated=datetime.now(),
                                    source="debt_analysis",
                                    tags=["technical_debt"],
                                    estimated_effort_hours=2.0,
                                    surrounding_code=line.strip(),
                                    dependencies=[],
                                    affected_components=["maintainability"]
                                )
                                items.append(item)
                                
            except Exception as e:
                print(f"Warning: Could not scan {file_path} for technical debt: {e}")
        
        print(f"🔧 Found {len(items)} technical debt items")
        return items
    
    async def _discover_documentation_gaps(self) -> List[ValueItem]:
        """Discover documentation gaps"""
        items = []
        
        python_files = list(self.repo_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_exclude_file(file_path):
                continue
                
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        # Check for functions without docstrings
                        if re.match(r'\s*def\s+\w+', line):
                            # Look ahead for docstring
                            has_docstring = False
                            for i in range(line_num, min(line_num + 5, len(lines))):
                                if '"""' in lines[i] or "'''" in lines[i]:
                                    has_docstring = True
                                    break
                            
                            if not has_docstring:
                                func_name_match = re.search(r'def\s+(\w+)', line)
                                if func_name_match:
                                    func_name = func_name_match.group(1)
                                    # Skip private methods and tests
                                    if not func_name.startswith('_') and not func_name.startswith('test_'):
                                        item = ValueItem(
                                            id=self._generate_id(file_path, line_num, f"Missing docstring {func_name}"),
                                            title=f"Add docstring to {func_name}",
                                            description=f"Function {func_name} is missing a docstring",
                                            task_type=TaskType.DOCUMENTATION,
                                            file_path=str(file_path),
                                            line_number=line_num,
                                            priority=Priority.LOW,
                                            business_value=0,
                                            time_criticality=0,
                                            risk_reduction=0,
                                            job_size=0,
                                            wsjf_score=0,
                                            impact=0,
                                            confidence=0,
                                            ease=0,
                                            ice_score=0,
                                            technical_debt_score=0,
                                            discovered_date=datetime.now(),
                                            last_updated=datetime.now(),
                                            source="documentation_analysis",
                                            tags=["documentation"],
                                            estimated_effort_hours=0.5,
                                            surrounding_code=line.strip(),
                                            dependencies=[],
                                            affected_components=["documentation"]
                                        )
                                        items.append(item)
                                        
            except Exception as e:
                print(f"Warning: Could not analyze {file_path} for documentation: {e}")
        
        print(f"📚 Found {len(items)} documentation gaps")
        return items
    
    def _calculate_scores(self, item: ValueItem) -> None:
        """Calculate WSJF, ICE, and Technical Debt scores for an item"""
        
        # WSJF Scoring
        item.business_value = self._calculate_business_value(item)
        item.time_criticality = self._calculate_time_criticality(item)
        item.risk_reduction = self._calculate_risk_reduction(item)
        item.job_size = self._calculate_job_size(item)
        
        # Prevent division by zero
        if item.job_size > 0:
            item.wsjf_score = (item.business_value + item.time_criticality + item.risk_reduction) / item.job_size
        else:
            item.wsjf_score = 0
        
        # ICE Scoring
        item.impact = item.business_value  # Reuse business value as impact
        item.confidence = self._calculate_confidence(item)
        item.ease = 5.0 - item.job_size  # Inverse of job size
        item.ice_score = (item.impact * item.confidence * item.ease) / 125  # Normalize to 0-1
        
        # Technical Debt Score
        item.technical_debt_score = self._calculate_technical_debt(item)
        
        # Update priority based on WSJF score
        if item.wsjf_score >= 8.0:
            item.priority = Priority.CRITICAL
        elif item.wsjf_score >= 6.0:
            item.priority = Priority.HIGH
        elif item.wsjf_score >= 3.0:
            item.priority = Priority.MEDIUM
        else:
            item.priority = Priority.LOW
    
    def _calculate_business_value(self, item: ValueItem) -> float:
        """Calculate business value score (1-5)"""
        base_score = {
            TaskType.SECURITY: 5.0,
            TaskType.BUG: 4.0,
            TaskType.PERFORMANCE: 3.0,
            TaskType.TESTING: 3.0,
            TaskType.DOCUMENTATION: 2.0,
            TaskType.REFACTORING: 2.0,
            TaskType.FEATURE: 3.0,
            TaskType.TECHNICAL_DEBT: 1.0,
        }.get(item.task_type, 2.0)
        
        # Boost for critical files
        critical_files = ["orchestrator", "security", "task_analyzer", "github_api"]
        if any(cf in item.file_path for cf in critical_files):
            base_score += 1.0
        
        return min(5.0, base_score)
    
    def _calculate_time_criticality(self, item: ValueItem) -> float:
        """Calculate time criticality score (1-5)"""
        urgency_keywords = {
            "critical": 5.0,
            "urgent": 4.0,
            "blocking": 5.0,
            "security": 4.0,
            "vulnerability": 5.0,
            "bug": 3.0,
            "performance": 2.0,
            "slow": 2.0,
        }
        
        description_lower = item.description.lower()
        max_score = 1.0
        
        for keyword, score in urgency_keywords.items():
            if keyword in description_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_risk_reduction(self, item: ValueItem) -> float:
        """Calculate risk reduction score (1-5)"""
        risk_keywords = {
            "security": 5.0,
            "vulnerability": 5.0,
            "injection": 5.0,
            "bug": 4.0,
            "crash": 4.0,
            "error": 3.0,
            "performance": 2.0,
            "maintainability": 2.0,
        }
        
        description_lower = item.description.lower()
        max_score = 1.0
        
        for keyword, score in risk_keywords.items():
            if keyword in description_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_job_size(self, item: ValueItem) -> float:
        """Calculate job size score (1-5)"""
        if item.estimated_effort_hours <= 1:
            return 1.0
        elif item.estimated_effort_hours <= 4:
            return 2.0
        elif item.estimated_effort_hours <= 16:
            return 3.0
        elif item.estimated_effort_hours <= 40:
            return 4.0
        else:
            return 5.0
    
    def _calculate_confidence(self, item: ValueItem) -> float:
        """Calculate confidence score (1-5)"""
        # Higher confidence for automated discovery
        if item.source == "security_analysis":
            return 4.0
        elif item.source == "test_analysis":
            return 4.5
        elif item.source == "code_analysis":
            return 3.5
        else:
            return 3.0
    
    def _calculate_technical_debt(self, item: ValueItem) -> float:
        """Calculate technical debt score"""
        if item.task_type == TaskType.TECHNICAL_DEBT:
            return item.estimated_effort_hours * 0.5
        elif item.task_type == TaskType.REFACTORING:
            return item.estimated_effort_hours * 0.3
        else:
            return 0.0
    
    def _classify_task_type(self, description: str, todo_type: str) -> TaskType:
        """Classify the task type based on description and TODO type"""
        desc_lower = description.lower()
        
        # Security indicators
        security_keywords = ["security", "auth", "token", "password", "vulnerability", "injection"]
        if any(kw in desc_lower for kw in security_keywords):
            return TaskType.SECURITY
        
        # Performance indicators
        perf_keywords = ["performance", "slow", "optimize", "cache", "timeout"]
        if any(kw in desc_lower for kw in perf_keywords):
            return TaskType.PERFORMANCE
        
        # Testing indicators
        test_keywords = ["test", "coverage", "mock", "pytest"]
        if any(kw in desc_lower for kw in test_keywords):
            return TaskType.TESTING
        
        # Documentation indicators
        doc_keywords = ["doc", "comment", "readme", "guide"]
        if any(kw in desc_lower for kw in doc_keywords):
            return TaskType.DOCUMENTATION
        
        # Bug indicators
        bug_keywords = ["bug", "fix", "error", "crash", "issue"]
        if any(kw in desc_lower for kw in bug_keywords) or todo_type == "FIXME":
            return TaskType.BUG
        
        # Refactoring indicators
        refactor_keywords = ["refactor", "cleanup", "simplify", "duplicate"]
        if any(kw in desc_lower for kw in refactor_keywords):
            return TaskType.REFACTORING
        
        # Default to technical debt for TODO items
        return TaskType.TECHNICAL_DEBT
    
    def _estimate_effort(self, description: str) -> float:
        """Estimate effort in hours based on description"""
        desc_lower = description.lower()
        
        # High effort indicators
        high_effort = ["refactor", "rewrite", "implement", "design", "architecture"]
        if any(kw in desc_lower for kw in high_effort):
            return 8.0
        
        # Medium effort indicators  
        medium_effort = ["fix", "update", "improve", "optimize"]
        if any(kw in desc_lower for kw in medium_effort):
            return 4.0
        
        # Low effort indicators
        low_effort = ["add", "remove", "change", "comment", "doc"]
        if any(kw in desc_lower for kw in low_effort):
            return 1.0
        
        # Default medium effort
        return 2.0
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis"""
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            "venv",
            "env",
            ".env"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in exclude_patterns)
    
    def _identify_components(self, file_path: Path) -> List[str]:
        """Identify which components are affected by this file"""
        components = []
        path_str = str(file_path)
        
        if "test" in path_str:
            components.append("testing")
        if "security" in path_str:
            components.append("security")
        if "performance" in path_str:
            components.append("performance")
        if "github" in path_str:
            components.append("github_integration")
        if "orchestrator" in path_str:
            components.append("orchestration")
        if "web" in path_str:
            components.append("web_interface")
        
        return components if components else ["core"]
    
    def _generate_id(self, file_path: Path, line_number: int, description: str) -> str:
        """Generate unique ID for value item"""
        content = f"{file_path}:{line_number}:{description}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _deduplicate_items(self, items: List[ValueItem]) -> List[ValueItem]:
        """Remove duplicate items based on ID"""
        seen_ids = set()
        unique_items = []
        
        for item in items:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_items.append(item)
        
        return unique_items
    
    async def save_results(self, output_path: str = ".terragon/discovered_backlog.json") -> None:
        """Save discovered items to JSON file"""
        output_data = {
            "metadata": {
                "discovery_date": datetime.now().isoformat(),
                "total_items": len(self.discovered_items),
                "config_version": "1.0.0",
                "repository": self.config.get("repository_info", {}).get("name", "unknown")
            },
            "summary": {
                "by_priority": {
                    priority.value: len([item for item in self.discovered_items if item.priority == priority])
                    for priority in Priority
                },
                "by_type": {
                    task_type.value: len([item for item in self.discovered_items if item.task_type == task_type])
                    for task_type in TaskType
                },
                "total_estimated_hours": sum(item.estimated_effort_hours for item in self.discovered_items)
            },
            "items": [
                {
                    **asdict(item),
                    "discovered_date": item.discovered_date.isoformat(),
                    "last_updated": item.last_updated.isoformat(),
                    "task_type": item.task_type.value,
                    "priority": item.priority.value
                }
                for item in self.discovered_items
            ]
        }
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(output_data, indent=2))
        
        print(f"💾 Saved {len(self.discovered_items)} items to {output_path}")


async def main():
    """Main entry point for value discovery"""
    engine = ValueDiscoveryEngine()
    
    # Discover all value items
    items = await engine.discover_value_items()
    
    # Save results
    await engine.save_results()
    
    # Print summary
    print("\n📊 Discovery Summary:")
    print(f"Total items discovered: {len(items)}")
    
    # Summary by priority
    for priority in Priority:
        count = len([item for item in items if item.priority == priority])
        print(f"  {priority.value.capitalize()}: {count}")
    
    # Top 10 highest value items
    print("\n🏆 Top 10 Highest Value Items (by WSJF score):")
    for i, item in enumerate(items[:10], 1):
        print(f"  {i}. [{item.priority.value.upper()}] {item.title} (WSJF: {item.wsjf_score:.2f})")


if __name__ == "__main__":
    asyncio.run(main())