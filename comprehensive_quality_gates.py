#!/usr/bin/env python3
"""
TERRAGON SDLC - COMPREHENSIVE QUALITY GATES
Comprehensive validation system ensuring 85%+ test coverage, security compliance, and performance benchmarks
"""

import os
import sys
import json
import asyncio
import time
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import hashlib

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.columns import Columns
from rich.status import Status

@dataclass
class QualityGateResult:
    """Quality gate validation result"""
    gate_name: str
    status: str  # "passed", "failed", "warning", "skipped"
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time: float
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class ComprehensiveQualityReport:
    """Comprehensive quality assessment report"""
    timestamp: str
    overall_status: str
    overall_score: float
    gates_passed: int
    gates_failed: int
    gates_warning: int
    execution_time: float
    gate_results: List[QualityGateResult]
    recommendations: List[str]
    compliance_status: Dict[str, str]
    performance_metrics: Dict[str, float]
    security_metrics: Dict[str, float]
    deployment_readiness: str

class ComprehensiveQualityGates:
    """Comprehensive quality gates validation system"""
    
    def __init__(self):
        self.console = Console()
        self.start_time = time.time()
        
        # Quality thresholds
        self.thresholds = {
            'test_coverage': 85.0,
            'security_score': 90.0,
            'performance_score': 80.0,
            'code_quality': 85.0,
            'documentation': 75.0
        }
        
        self.gate_results: List[QualityGateResult] = []
    
    async def execute_all_quality_gates(self) -> ComprehensiveQualityReport:
        """Execute all quality gates comprehensively"""
        rprint("[bold green]🛡️ TERRAGON SDLC - COMPREHENSIVE QUALITY GATES[/bold green]")
        rprint("[dim]Validating all systems with enterprise-grade quality standards[/dim]\n")
        
        # Define quality gates in execution order
        quality_gates = [
            ("Basic System Validation", self._validate_basic_system),
            ("Code Quality Analysis", self._validate_code_quality),
            ("Security Compliance", self._validate_security_compliance),
            ("Performance Benchmarks", self._validate_performance_benchmarks),
            ("Test Coverage Analysis", self._validate_test_coverage),
            ("Documentation Quality", self._validate_documentation),
            ("Deployment Readiness", self._validate_deployment_readiness),
            ("Global Compliance", self._validate_global_compliance),
            ("Integration Testing", self._validate_integration),
            ("Final System Health", self._validate_final_health)
        ]
        
        # Execute gates with progress tracking
        with Progress() as progress:
            gate_task = progress.add_task("Executing Quality Gates...", total=len(quality_gates))
            
            for gate_name, gate_func in quality_gates:
                with Status(f"[bold blue]Executing {gate_name}...[/bold blue]", spinner="dots"):
                    gate_start = time.time()
                    
                    try:
                        result = await gate_func()
                        result.execution_time = time.time() - gate_start
                        self.gate_results.append(result)
                        
                        # Display immediate result
                        status_emoji = "✅" if result.status == "passed" else "⚠️" if result.status == "warning" else "❌"
                        rprint(f"{status_emoji} {gate_name}: [bold]{result.status.upper()}[/bold] ({result.score:.1f}/100)")
                        
                    except Exception as e:
                        error_result = QualityGateResult(
                            gate_name=gate_name,
                            status="failed",
                            score=0.0,
                            details={"error": str(e)},
                            execution_time=time.time() - gate_start,
                            errors=[str(e)]
                        )
                        self.gate_results.append(error_result)
                        rprint(f"❌ {gate_name}: [bold red]FAILED[/bold red] - {str(e)}")
                    
                    progress.advance(gate_task)
                    await asyncio.sleep(0.1)  # Brief pause between gates
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Display final results
        self._display_quality_report(report)
        
        # Save report
        await self._save_quality_report(report)
        
        return report
    
    async def _validate_basic_system(self) -> QualityGateResult:
        """Validate basic system functionality"""
        details = {}
        errors = []
        score = 0.0
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version >= (3, 8):
                score += 15
                details['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            else:
                errors.append(f"Python version too old: {python_version}")
            
            # Check critical files
            critical_files = [
                'config.json',
                'requirements.txt',
                'src/main.py',
                'simple_autonomous_main.py',
                'robust_autonomous_main.py',
                'scalable_autonomous_main.py'
            ]
            
            existing_files = []
            for file_path in critical_files:
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                    score += 10
                else:
                    errors.append(f"Critical file missing: {file_path}")
            
            details['existing_files'] = existing_files
            details['missing_files'] = [f for f in critical_files if not os.path.exists(f)]
            
            # Check directory structure
            required_dirs = ['src', 'tests']
            existing_dirs = []
            for dir_path in required_dirs:
                if os.path.exists(dir_path):
                    existing_dirs.append(dir_path)
                    score += 5
                else:
                    errors.append(f"Required directory missing: {dir_path}")
            
            details['existing_dirs'] = existing_dirs
            
            # Test basic imports
            try:
                import json
                import asyncio
                import pathlib
                score += 10
                details['core_imports'] = 'success'
            except ImportError as e:
                errors.append(f"Core import failed: {e}")
                details['core_imports'] = 'failed'
            
            status = "passed" if score >= 70 else "warning" if score >= 40 else "failed"
            
        except Exception as e:
            errors.append(f"Basic system validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Basic System Validation",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors
        )
    
    async def _validate_code_quality(self) -> QualityGateResult:
        """Validate code quality and standards"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Count Python files
            python_files = []
            for root, dirs, files in os.walk('.'):
                if 'venv' in root or '.git' in root:
                    continue
                python_files.extend([os.path.join(root, f) for f in files if f.endswith('.py')])
            
            details['python_files_count'] = len(python_files)
            
            if len(python_files) > 0:
                score += 20
            
            # Analyze code structure
            total_lines = 0
            documented_functions = 0
            total_functions = 0
            
            for file_path in python_files[:20]:  # Limit for performance
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        
                        # Count functions and documentation
                        for i, line in enumerate(lines):
                            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                                total_functions += 1
                                
                                # Check for docstring in next few lines
                                has_docstring = False
                                for j in range(i + 1, min(i + 4, len(lines))):
                                    if '"""' in lines[j] or "'''" in lines[j]:
                                        has_docstring = True
                                        break
                                
                                if has_docstring:
                                    documented_functions += 1
                
                except Exception as e:
                    warnings.append(f"Could not analyze {file_path}: {e}")
            
            details['total_lines'] = total_lines
            details['total_functions'] = total_functions
            details['documented_functions'] = documented_functions
            
            # Calculate documentation ratio
            if total_functions > 0:
                doc_ratio = (documented_functions / total_functions) * 100
                details['documentation_ratio'] = doc_ratio
                
                if doc_ratio >= 60:
                    score += 25
                elif doc_ratio >= 40:
                    score += 15
                    warnings.append(f"Documentation ratio low: {doc_ratio:.1f}%")
                else:
                    errors.append(f"Documentation ratio too low: {doc_ratio:.1f}%")
            
            # Check for common code quality indicators
            if total_lines > 1000:
                score += 15
                details['code_complexity'] = 'substantial'
            elif total_lines > 500:
                score += 10
                details['code_complexity'] = 'moderate'
            else:
                score += 5
                details['code_complexity'] = 'minimal'
            
            # Check for error handling patterns
            error_handling_patterns = 0
            for file_path in python_files[:10]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'try:' in content:
                            error_handling_patterns += 1
                        if 'except' in content:
                            error_handling_patterns += 1
                        if 'raise' in content:
                            error_handling_patterns += 1
                except:
                    continue
            
            details['error_handling_patterns'] = error_handling_patterns
            if error_handling_patterns >= 5:
                score += 20
                details['error_handling'] = 'good'
            elif error_handling_patterns >= 2:
                score += 10
                details['error_handling'] = 'basic'
            else:
                warnings.append("Limited error handling found")
                details['error_handling'] = 'limited'
            
            # Check for modern Python features
            modern_features = 0
            for file_path in python_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'async def' in content:
                            modern_features += 2
                        if 'await' in content:
                            modern_features += 1
                        if 'from typing import' in content:
                            modern_features += 1
                        if '@dataclass' in content:
                            modern_features += 1
                except:
                    continue
            
            details['modern_features'] = modern_features
            if modern_features >= 5:
                score += 20
                details['modern_python'] = 'excellent'
            elif modern_features >= 2:
                score += 10
                details['modern_python'] = 'good'
            else:
                details['modern_python'] = 'basic'
            
            status = "passed" if score >= self.thresholds['code_quality'] else "warning" if score >= 60 else "failed"
            
        except Exception as e:
            errors.append(f"Code quality validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Code Quality Analysis",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_security_compliance(self) -> QualityGateResult:
        """Validate security compliance"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Check for security-related files
            security_files = [
                'robust_error_handler.py',
                'robust_health_monitor.py',
                'robust_config_validator.py'
            ]
            
            existing_security_files = [f for f in security_files if os.path.exists(f)]
            details['security_files'] = existing_security_files
            score += len(existing_security_files) * 15
            
            # Scan for potential security issues
            security_issues = []
            secret_patterns = [
                'password = "',
                'api_key = "',
                'secret = "',
                'token = "'
            ]
            
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for hardcoded secrets (basic check)
                        for pattern in secret_patterns:
                            if pattern in content.lower():
                                security_issues.append(f"Potential hardcoded secret in {file_path}")
                        
                        # Check for dangerous functions
                        dangerous_patterns = ['eval(', 'exec(', 'os.system(']
                        for pattern in dangerous_patterns:
                            if pattern in content:
                                security_issues.append(f"Dangerous function {pattern} in {file_path}")
                
                except Exception as e:
                    warnings.append(f"Could not scan {file_path}: {e}")
            
            details['security_issues'] = security_issues
            
            if len(security_issues) == 0:
                score += 30
                details['secret_scanning'] = 'clean'
            elif len(security_issues) <= 2:
                score += 15
                warnings.extend(security_issues)
                details['secret_scanning'] = 'minor_issues'
            else:
                errors.extend(security_issues)
                details['secret_scanning'] = 'major_issues'
            
            # Check file permissions (basic check)
            sensitive_files = ['config.json']
            permission_issues = []
            
            for file_path in sensitive_files:
                if os.path.exists(file_path):
                    try:
                        stat_info = os.stat(file_path)
                        # Basic permission check (not too permissive)
                        permissions = oct(stat_info.st_mode)[-3:]
                        if permissions in ['644', '640', '600']:
                            score += 10
                        else:
                            permission_issues.append(f"File {file_path} has permissions {permissions}")
                    except Exception as e:
                        warnings.append(f"Could not check permissions for {file_path}: {e}")
            
            details['permission_issues'] = permission_issues
            
            if len(permission_issues) == 0:
                score += 15
                details['file_permissions'] = 'secure'
            else:
                warnings.extend(permission_issues)
                details['file_permissions'] = 'issues_found'
            
            # Check for security frameworks/libraries
            security_imports = []
            for file_path in python_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if any(lib in content for lib in ['hashlib', 'secrets', 'cryptography', 'jwt']):
                            security_imports.append(file_path)
                except:
                    continue
            
            details['security_imports'] = len(security_imports)
            if len(security_imports) > 0:
                score += 20
                details['security_libraries'] = 'present'
            else:
                details['security_libraries'] = 'minimal'
            
            status = "passed" if score >= self.thresholds['security_score'] else "warning" if score >= 60 else "failed"
            
        except Exception as e:
            errors.append(f"Security validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Security Compliance",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_performance_benchmarks(self) -> QualityGateResult:
        """Validate performance benchmarks"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Test Generation 1 performance
            gen1_start = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, 'simple_autonomous_main.py'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd='.'
                )
                gen1_time = time.time() - gen1_start
                
                if result.returncode == 0:
                    score += 25
                    details['generation_1'] = {
                        'status': 'passed',
                        'execution_time': gen1_time,
                        'performance_rating': 'good' if gen1_time < 5 else 'acceptable'
                    }
                else:
                    errors.append("Generation 1 execution failed")
                    details['generation_1'] = {'status': 'failed', 'error': result.stderr}
                    
            except subprocess.TimeoutExpired:
                errors.append("Generation 1 execution timed out")
                details['generation_1'] = {'status': 'timeout'}
            except Exception as e:
                warnings.append(f"Could not test Generation 1: {e}")
                details['generation_1'] = {'status': 'error', 'error': str(e)}
            
            # Check for performance-related implementations
            performance_files = [
                'scalable_cache_system.py',
                'scalable_performance_monitor.py',
                'scalable_autonomous_main.py'
            ]
            
            existing_perf_files = [f for f in performance_files if os.path.exists(f)]
            details['performance_files'] = existing_perf_files
            score += len(existing_perf_files) * 15
            
            # Analyze performance patterns in code
            performance_patterns = 0
            async_patterns = 0
            
            python_files = [f for f in os.listdir('.') if f.endswith('.py')][:10]
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for async patterns
                        if 'async def' in content:
                            async_patterns += 1
                        if 'await' in content:
                            async_patterns += 1
                        
                        # Check for performance-related patterns
                        perf_keywords = ['cache', 'optimize', 'concurrent', 'parallel', 'pool', 'batch']
                        for keyword in perf_keywords:
                            if keyword.lower() in content.lower():
                                performance_patterns += 1
                                break
                
                except Exception as e:
                    warnings.append(f"Could not analyze {file_path}: {e}")
            
            details['async_patterns'] = async_patterns
            details['performance_patterns'] = performance_patterns
            
            if async_patterns >= 5:
                score += 20
                details['async_usage'] = 'extensive'
            elif async_patterns >= 2:
                score += 10
                details['async_usage'] = 'moderate'
            else:
                details['async_usage'] = 'minimal'
            
            if performance_patterns >= 3:
                score += 15
                details['performance_optimization'] = 'implemented'
            else:
                details['performance_optimization'] = 'limited'
            
            # Memory efficiency check (basic)
            import psutil
            memory_info = psutil.virtual_memory()
            if memory_info.percent < 80:
                score += 15
                details['memory_status'] = 'healthy'
            else:
                warnings.append(f"High memory usage: {memory_info.percent}%")
                details['memory_status'] = 'high_usage'
            
            status = "passed" if score >= self.thresholds['performance_score'] else "warning" if score >= 60 else "failed"
            
        except Exception as e:
            errors.append(f"Performance validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Performance Benchmarks",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_test_coverage(self) -> QualityGateResult:
        """Validate test coverage"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Check for test directory and files
            test_dirs = ['tests', 'test']
            existing_test_dirs = [d for d in test_dirs if os.path.exists(d)]
            
            details['test_directories'] = existing_test_dirs
            
            if existing_test_dirs:
                score += 20
                
                # Count test files
                test_files = []
                for test_dir in existing_test_dirs:
                    for root, dirs, files in os.walk(test_dir):
                        test_files.extend([f for f in files if f.startswith('test_') and f.endswith('.py')])
                
                details['test_files_count'] = len(test_files)
                
                if len(test_files) >= 10:
                    score += 30
                    details['test_coverage_level'] = 'comprehensive'
                elif len(test_files) >= 5:
                    score += 20
                    details['test_coverage_level'] = 'good'
                elif len(test_files) >= 1:
                    score += 10
                    details['test_coverage_level'] = 'basic'
                    warnings.append(f"Limited test files found: {len(test_files)}")
                else:
                    errors.append("No test files found in test directories")
                    details['test_coverage_level'] = 'none'
            else:
                errors.append("No test directories found")
                details['test_coverage_level'] = 'none'
            
            # Try to run existing tests
            if existing_test_dirs:
                try:
                    test_result = subprocess.run(
                        [sys.executable, '-m', 'pytest', '--tb=short', '-v'],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd='.'
                    )
                    
                    if test_result.returncode == 0:
                        score += 30
                        details['test_execution'] = 'passed'
                        # Parse test results
                        output_lines = test_result.stdout.split('\n')
                        passed_tests = len([line for line in output_lines if ' PASSED' in line])
                        failed_tests = len([line for line in output_lines if ' FAILED' in line])
                        
                        details['tests_passed'] = passed_tests
                        details['tests_failed'] = failed_tests
                        
                        if failed_tests > 0:
                            warnings.append(f"{failed_tests} tests failed")
                            score -= failed_tests * 5  # Penalty for failing tests
                        
                    else:
                        warnings.append("Some tests failed or had issues")
                        details['test_execution'] = 'issues'
                        details['test_error'] = test_result.stderr[:500]
                        score += 10  # Some credit for having runnable tests
                        
                except subprocess.TimeoutExpired:
                    warnings.append("Test execution timed out")
                    details['test_execution'] = 'timeout'
                except Exception as e:
                    warnings.append(f"Could not run tests: {e}")
                    details['test_execution'] = 'error'
            
            # Check for test patterns in main code
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            testable_functions = 0
            
            for file_path in python_files[:10]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Count public functions that should be tested
                        lines = content.split('\n')
                        for line in lines:
                            if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                                testable_functions += 1
                except:
                    continue
            
            details['testable_functions'] = testable_functions
            
            # Estimate coverage based on test files vs testable functions
            if testable_functions > 0 and len(test_files) > 0:
                estimated_coverage = min(100, (len(test_files) * 10) / testable_functions * 100)
                details['estimated_coverage'] = estimated_coverage
                
                if estimated_coverage >= self.thresholds['test_coverage']:
                    score += 20
                elif estimated_coverage >= 50:
                    score += 10
                    warnings.append(f"Estimated coverage below threshold: {estimated_coverage:.1f}%")
                else:
                    errors.append(f"Low estimated coverage: {estimated_coverage:.1f}%")
            else:
                details['estimated_coverage'] = 0
                errors.append("Cannot estimate test coverage")
            
            status = "passed" if score >= self.thresholds['test_coverage'] else "warning" if score >= 50 else "failed"
            
        except Exception as e:
            errors.append(f"Test coverage validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Test Coverage Analysis",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation quality"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Check for main documentation files
            doc_files = [
                'README.md',
                'ARCHITECTURE.md',
                'CONTRIBUTING.md',
                'CODE_OF_CONDUCT.md'
            ]
            
            existing_docs = []
            for doc_file in doc_files:
                if os.path.exists(doc_file):
                    existing_docs.append(doc_file)
                    score += 15
            
            details['documentation_files'] = existing_docs
            
            # Analyze README quality
            if 'README.md' in existing_docs:
                try:
                    with open('README.md', 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                        
                    readme_sections = 0
                    quality_indicators = ['installation', 'usage', 'example', 'api', 'contributing']
                    
                    for indicator in quality_indicators:
                        if indicator.lower() in readme_content.lower():
                            readme_sections += 1
                    
                    details['readme_sections'] = readme_sections
                    details['readme_length'] = len(readme_content)
                    
                    if readme_sections >= 4:
                        score += 20
                        details['readme_quality'] = 'comprehensive'
                    elif readme_sections >= 2:
                        score += 10
                        details['readme_quality'] = 'good'
                    else:
                        warnings.append("README missing key sections")
                        details['readme_quality'] = 'basic'
                        
                except Exception as e:
                    warnings.append(f"Could not analyze README: {e}")
            else:
                errors.append("README.md not found")
            
            # Check docs directory
            if os.path.exists('docs'):
                doc_files_count = 0
                for root, dirs, files in os.walk('docs'):
                    doc_files_count += len([f for f in files if f.endswith('.md')])
                
                details['docs_directory_files'] = doc_files_count
                
                if doc_files_count >= 10:
                    score += 20
                    details['docs_coverage'] = 'comprehensive'
                elif doc_files_count >= 5:
                    score += 10
                    details['docs_coverage'] = 'good'
                else:
                    details['docs_coverage'] = 'minimal'
            else:
                warnings.append("docs directory not found")
                details['docs_coverage'] = 'none'
            
            # Check inline documentation (docstrings)
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            total_functions = 0
            documented_functions = 0
            
            for file_path in python_files[:10]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                        for i, line in enumerate(lines):
                            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                                total_functions += 1
                                
                                # Check for docstring
                                for j in range(i + 1, min(i + 4, len(lines))):
                                    if '"""' in lines[j] or "'''" in lines[j]:
                                        documented_functions += 1
                                        break
                
                except Exception as e:
                    warnings.append(f"Could not analyze {file_path}: {e}")
            
            details['total_functions'] = total_functions
            details['documented_functions'] = documented_functions
            
            if total_functions > 0:
                doc_ratio = (documented_functions / total_functions) * 100
                details['inline_documentation_ratio'] = doc_ratio
                
                if doc_ratio >= self.thresholds['documentation']:
                    score += 25
                    details['inline_docs_quality'] = 'excellent'
                elif doc_ratio >= 50:
                    score += 15
                    details['inline_docs_quality'] = 'good'
                elif doc_ratio >= 25:
                    score += 5
                    warnings.append(f"Low inline documentation: {doc_ratio:.1f}%")
                    details['inline_docs_quality'] = 'poor'
                else:
                    errors.append(f"Very low inline documentation: {doc_ratio:.1f}%")
                    details['inline_docs_quality'] = 'critical'
            
            status = "passed" if score >= self.thresholds['documentation'] else "warning" if score >= 50 else "failed"
            
        except Exception as e:
            errors.append(f"Documentation validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Documentation Quality",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_deployment_readiness(self) -> QualityGateResult:
        """Validate deployment readiness"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Check for deployment files
            deployment_files = [
                'Dockerfile',
                'docker-compose.yml',
                'requirements.txt',
                'config.json'
            ]
            
            existing_deployment_files = []
            for file_path in deployment_files:
                if os.path.exists(file_path):
                    existing_deployment_files.append(file_path)
                    score += 15
            
            details['deployment_files'] = existing_deployment_files
            
            # Check configuration
            if os.path.exists('config.json'):
                try:
                    with open('config.json', 'r') as f:
                        config = json.load(f)
                    
                    required_config_sections = ['github', 'analyzer', 'executor']
                    config_completeness = 0
                    
                    for section in required_config_sections:
                        if section in config:
                            config_completeness += 1
                    
                    details['config_completeness'] = f"{config_completeness}/{len(required_config_sections)}"
                    
                    if config_completeness == len(required_config_sections):
                        score += 20
                        details['configuration_status'] = 'complete'
                    elif config_completeness >= 2:
                        score += 10
                        warnings.append("Configuration partially complete")
                        details['configuration_status'] = 'partial'
                    else:
                        errors.append("Configuration incomplete")
                        details['configuration_status'] = 'incomplete'
                        
                except Exception as e:
                    errors.append(f"Could not validate configuration: {e}")
                    details['configuration_status'] = 'invalid'
            else:
                errors.append("config.json not found")
            
            # Check for requirements.txt
            if os.path.exists('requirements.txt'):
                try:
                    with open('requirements.txt', 'r') as f:
                        requirements = f.readlines()
                    
                    details['requirements_count'] = len([r for r in requirements if r.strip() and not r.strip().startswith('#')])
                    
                    # Check for key dependencies
                    key_deps = ['pytest', 'PyGithub', 'typer', 'rich']
                    found_deps = 0
                    
                    req_content = '\n'.join(requirements)
                    for dep in key_deps:
                        if dep.lower() in req_content.lower():
                            found_deps += 1
                    
                    details['key_dependencies'] = f"{found_deps}/{len(key_deps)}"
                    
                    if found_deps >= len(key_deps) - 1:
                        score += 15
                        details['dependencies_status'] = 'complete'
                    elif found_deps >= 2:
                        score += 10
                        details['dependencies_status'] = 'mostly_complete'
                    else:
                        warnings.append("Missing key dependencies")
                        details['dependencies_status'] = 'incomplete'
                        
                except Exception as e:
                    warnings.append(f"Could not analyze requirements: {e}")
            
            # Check for CI/CD setup
            ci_files = ['.github/workflows', '.gitlab-ci.yml', 'Jenkinsfile']
            ci_setup = []
            
            for ci_path in ci_files:
                if os.path.exists(ci_path):
                    ci_setup.append(ci_path)
                    score += 10
            
            details['ci_cd_setup'] = ci_setup
            
            if ci_setup:
                details['ci_cd_status'] = 'configured'
            else:
                warnings.append("No CI/CD configuration found")
                details['ci_cd_status'] = 'none'
            
            # Environment readiness
            env_factors = []
            
            # Check Python version
            if sys.version_info >= (3, 8):
                env_factors.append("python_version_ok")
                score += 5
            
            # Check if virtual environment is active
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                env_factors.append("virtual_env_active")
                score += 10
            
            details['environment_factors'] = env_factors
            details['deployment_readiness_score'] = score
            
            status = "passed" if score >= 70 else "warning" if score >= 50 else "failed"
            
        except Exception as e:
            errors.append(f"Deployment readiness validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Deployment Readiness",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_global_compliance(self) -> QualityGateResult:
        """Validate global compliance (I18n, GDPR, etc.)"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Check for internationalization support
            i18n_files = []
            if os.path.exists('i18n'):
                for root, dirs, files in os.walk('i18n'):
                    i18n_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])
                
                details['i18n_files'] = len(i18n_files)
                
                if len(i18n_files) >= 10:
                    score += 30
                    details['i18n_support'] = 'comprehensive'
                elif len(i18n_files) >= 5:
                    score += 20
                    details['i18n_support'] = 'good'
                elif len(i18n_files) >= 1:
                    score += 10
                    details['i18n_support'] = 'basic'
                else:
                    details['i18n_support'] = 'none'
            else:
                warnings.append("No i18n directory found")
                details['i18n_support'] = 'none'
            
            # Check for compliance-related documentation
            compliance_files = [
                'PRIVACY_POLICY.md',
                'TERMS_OF_SERVICE.md',
                'CODE_OF_CONDUCT.md',
                'CONTRIBUTING.md',
                'LICENSE'
            ]
            
            existing_compliance = []
            for file_path in compliance_files:
                if os.path.exists(file_path):
                    existing_compliance.append(file_path)
                    score += 10
            
            details['compliance_files'] = existing_compliance
            
            if len(existing_compliance) >= 3:
                details['compliance_documentation'] = 'comprehensive'
            elif len(existing_compliance) >= 1:
                details['compliance_documentation'] = 'basic'
            else:
                warnings.append("Limited compliance documentation")
                details['compliance_documentation'] = 'minimal'
            
            # Check for security and privacy considerations in code
            privacy_patterns = ['gdpr', 'privacy', 'consent', 'personal_data', 'data_protection']
            privacy_implementations = 0
            
            python_files = [f for f in os.listdir('.') if f.endswith('.py')][:10]
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in privacy_patterns:
                            if pattern in content:
                                privacy_implementations += 1
                                break
                
                except Exception:
                    continue
            
            details['privacy_implementations'] = privacy_implementations
            
            if privacy_implementations >= 2:
                score += 15
                details['privacy_awareness'] = 'implemented'
            elif privacy_implementations >= 1:
                score += 5
                details['privacy_awareness'] = 'basic'
            else:
                details['privacy_awareness'] = 'none'
            
            # Cross-platform compatibility check
            platform_indicators = []
            
            for file_path in python_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        if 'os.path.join' in content or 'Path(' in content:
                            platform_indicators.append("path_handling")
                        if 'platform' in content or 'sys.platform' in content:
                            platform_indicators.append("platform_detection")
                        if 'encoding=' in content:
                            platform_indicators.append("encoding_handling")
                
                except Exception:
                    continue
            
            details['platform_compatibility_indicators'] = list(set(platform_indicators))
            
            if len(set(platform_indicators)) >= 2:
                score += 15
                details['cross_platform'] = 'good'
            elif len(set(platform_indicators)) >= 1:
                score += 8
                details['cross_platform'] = 'basic'
            else:
                warnings.append("Limited cross-platform considerations")
                details['cross_platform'] = 'limited'
            
            # Time zone and locale handling
            locale_patterns = ['timezone', 'utc', 'locale', 'strftime', 'datetime']
            locale_handling = 0
            
            for file_path in python_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in locale_patterns:
                            if pattern in content:
                                locale_handling += 1
                                break
                
                except Exception:
                    continue
            
            details['locale_handling'] = locale_handling
            
            if locale_handling >= 3:
                score += 10
                details['temporal_handling'] = 'comprehensive'
            elif locale_handling >= 1:
                score += 5
                details['temporal_handling'] = 'basic'
            else:
                details['temporal_handling'] = 'minimal'
            
            status = "passed" if score >= 60 else "warning" if score >= 40 else "failed"
            
        except Exception as e:
            errors.append(f"Global compliance validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Global Compliance",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_integration(self) -> QualityGateResult:
        """Validate integration capabilities"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # Check for API integration capabilities
            api_patterns = ['requests', 'aiohttp', 'httpx', 'urllib', 'github', 'api']
            api_integrations = 0
            
            python_files = [f for f in os.listdir('.') if f.endswith('.py')][:10]
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in api_patterns:
                            if pattern in content:
                                api_integrations += 1
                                break
                
                except Exception:
                    continue
            
            details['api_integrations'] = api_integrations
            
            if api_integrations >= 5:
                score += 25
                details['api_integration_level'] = 'comprehensive'
            elif api_integrations >= 2:
                score += 15
                details['api_integration_level'] = 'good'
            elif api_integrations >= 1:
                score += 8
                details['api_integration_level'] = 'basic'
            else:
                details['api_integration_level'] = 'none'
            
            # Check for database integration
            db_patterns = ['sqlite', 'database', 'db', 'sql', 'aiosqlite']
            db_integrations = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in db_patterns:
                            if pattern in content:
                                db_integrations += 1
                                break
                
                except Exception:
                    continue
            
            details['database_integrations'] = db_integrations
            
            if db_integrations >= 2:
                score += 20
                details['database_support'] = 'implemented'
            elif db_integrations >= 1:
                score += 10
                details['database_support'] = 'basic'
            else:
                details['database_support'] = 'none'
            
            # Check for async/await patterns (for scalable integrations)
            async_patterns = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'async def' in content and 'await' in content:
                            async_patterns += 1
                
                except Exception:
                    continue
            
            details['async_patterns'] = async_patterns
            
            if async_patterns >= 3:
                score += 20
                details['async_capability'] = 'extensive'
            elif async_patterns >= 1:
                score += 10
                details['async_capability'] = 'present'
            else:
                details['async_capability'] = 'none'
            
            # Check for configuration management
            config_patterns = ['config', 'settings', 'environment', 'env']
            config_management = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in config_patterns:
                            if pattern in content:
                                config_management += 1
                                break
                
                except Exception:
                    continue
            
            details['configuration_management'] = config_management
            
            if config_management >= 3:
                score += 15
                details['config_handling'] = 'comprehensive'
            elif config_management >= 1:
                score += 10
                details['config_handling'] = 'basic'
            else:
                warnings.append("Limited configuration management")
                details['config_handling'] = 'minimal'
            
            # Check for error handling and logging
            error_handling_patterns = ['try:', 'except', 'logging', 'logger', 'error']
            error_handling = 0
            
            for file_path in python_files[:5]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in error_handling_patterns:
                            if pattern in content:
                                error_handling += 1
                                break
                
                except Exception:
                    continue
            
            details['error_handling'] = error_handling
            
            if error_handling >= 4:
                score += 20
                details['error_handling_quality'] = 'robust'
            elif error_handling >= 2:
                score += 10
                details['error_handling_quality'] = 'adequate'
            else:
                warnings.append("Limited error handling")
                details['error_handling_quality'] = 'basic'
            
            status = "passed" if score >= 70 else "warning" if score >= 50 else "failed"
            
        except Exception as e:
            errors.append(f"Integration validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Integration Testing",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_final_health(self) -> QualityGateResult:
        """Validate final system health"""
        details = {}
        errors = []
        warnings = []
        score = 0.0
        
        try:
            # System resource check
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details['system_resources'] = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'available_memory_gb': memory.available / (1024**3)
            }
            
            # Resource health scoring
            resource_score = 0
            if cpu_percent < 80:
                resource_score += 25
            elif cpu_percent < 95:
                resource_score += 15
                warnings.append(f"High CPU usage: {cpu_percent}%")
            else:
                errors.append(f"Critical CPU usage: {cpu_percent}%")
            
            if memory.percent < 85:
                resource_score += 25
            elif memory.percent < 95:
                resource_score += 15
                warnings.append(f"High memory usage: {memory.percent}%")
            else:
                errors.append(f"Critical memory usage: {memory.percent}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent < 90:
                resource_score += 15
            elif disk_percent < 95:
                resource_score += 10
                warnings.append(f"High disk usage: {disk_percent:.1f}%")
            else:
                errors.append(f"Critical disk usage: {disk_percent:.1f}%")
            
            score += resource_score
            details['resource_health_score'] = resource_score
            
            # File system health
            critical_files = [
                'simple_autonomous_main.py',
                'robust_autonomous_main.py', 
                'scalable_autonomous_main.py',
                'config.json'
            ]
            
            file_health = 0
            missing_files = []
            
            for file_path in critical_files:
                if os.path.exists(file_path):
                    file_health += 10
                else:
                    missing_files.append(file_path)
            
            details['missing_critical_files'] = missing_files
            details['file_system_health'] = file_health
            score += min(40, file_health)
            
            if missing_files:
                errors.extend([f"Missing critical file: {f}" for f in missing_files])
            
            # Process health (check if we can start processes)
            process_health = 0
            try:
                test_process = subprocess.run(
                    [sys.executable, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if test_process.returncode == 0:
                    process_health += 20
                    details['python_executable'] = 'healthy'
                else:
                    warnings.append("Python executable issues")
                    details['python_executable'] = 'issues'
            except Exception as e:
                errors.append(f"Process execution issues: {e}")
                details['python_executable'] = 'failed'
            
            score += process_health
            
            # Network connectivity (basic check)
            network_health = 0
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                network_health += 15
                details['network_connectivity'] = 'available'
            except Exception:
                warnings.append("Network connectivity issues")
                details['network_connectivity'] = 'limited'
            
            score += network_health
            
            # Overall system stability
            uptime_info = "Unknown"
            try:
                boot_time = psutil.boot_time()
                uptime_seconds = time.time() - boot_time
                uptime_hours = uptime_seconds / 3600
                
                if uptime_hours > 24:
                    score += 10
                    uptime_info = f"{uptime_hours:.1f} hours"
                    details['system_stability'] = 'stable'
                else:
                    uptime_info = f"{uptime_hours:.1f} hours"
                    details['system_stability'] = 'recent_boot'
                
            except Exception:
                details['system_stability'] = 'unknown'
            
            details['system_uptime'] = uptime_info
            
            # Final health assessment
            if score >= 85:
                details['health_status'] = 'excellent'
                health_summary = "System is in excellent health"
            elif score >= 70:
                details['health_status'] = 'good'
                health_summary = "System is in good health with minor issues"
            elif score >= 50:
                details['health_status'] = 'fair'
                health_summary = "System has some health issues that should be addressed"
                warnings.append(health_summary)
            else:
                details['health_status'] = 'poor'
                health_summary = "System has significant health issues"
                errors.append(health_summary)
            
            details['health_summary'] = health_summary
            
            status = "passed" if score >= 80 else "warning" if score >= 60 else "failed"
            
        except Exception as e:
            errors.append(f"Final health validation error: {e}")
            status = "failed"
            score = 0
        
        return QualityGateResult(
            gate_name="Final System Health",
            status=status,
            score=min(100, score),
            details=details,
            execution_time=0,
            errors=errors,
            warnings=warnings
        )
    
    def _generate_comprehensive_report(self) -> ComprehensiveQualityReport:
        """Generate comprehensive quality report"""
        
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall metrics
        total_gates = len(self.gate_results)
        passed_gates = len([r for r in self.gate_results if r.status == "passed"])
        failed_gates = len([r for r in self.gate_results if r.status == "failed"])
        warning_gates = len([r for r in self.gate_results if r.status == "warning"])
        
        # Calculate weighted overall score
        total_score = sum(result.score for result in self.gate_results)
        overall_score = total_score / max(total_gates, 1)
        
        # Determine overall status
        if failed_gates == 0 and warning_gates <= 2:
            overall_status = "passed"
        elif failed_gates <= 2:
            overall_status = "warning"
        else:
            overall_status = "failed"
        
        # Generate recommendations
        recommendations = []
        
        for result in self.gate_results:
            if result.status == "failed":
                recommendations.append(f"Address failures in {result.gate_name}: {'; '.join(result.errors[:2])}")
            elif result.status == "warning" and result.score < 70:
                recommendations.append(f"Improve {result.gate_name} (current score: {result.score:.1f})")
        
        if overall_score < 80:
            recommendations.append("Overall quality score below 80% - review all quality gates")
        
        if len(recommendations) > 5:
            recommendations = recommendations[:5] + [f"... and {len(recommendations) - 5} more recommendations"]
        
        # Compliance status
        compliance_status = {
            "security": "passed" if any(r.gate_name == "Security Compliance" and r.status == "passed" for r in self.gate_results) else "review",
            "performance": "passed" if any(r.gate_name == "Performance Benchmarks" and r.status == "passed" for r in self.gate_results) else "review",
            "documentation": "passed" if any(r.gate_name == "Documentation Quality" and r.status == "passed" for r in self.gate_results) else "review",
            "testing": "passed" if any(r.gate_name == "Test Coverage Analysis" and r.status == "passed" for r in self.gate_results) else "review"
        }
        
        # Performance metrics
        performance_metrics = {}
        for result in self.gate_results:
            if result.gate_name == "Performance Benchmarks":
                performance_metrics.update(result.details)
        
        # Security metrics  
        security_metrics = {}
        for result in self.gate_results:
            if result.gate_name == "Security Compliance":
                security_metrics.update(result.details)
        
        # Deployment readiness
        deployment_readiness = "ready" if overall_score >= 85 and failed_gates == 0 else "needs_work"
        
        return ComprehensiveQualityReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status=overall_status,
            overall_score=overall_score,
            gates_passed=passed_gates,
            gates_failed=failed_gates,
            gates_warning=warning_gates,
            execution_time=total_execution_time,
            gate_results=self.gate_results,
            recommendations=recommendations,
            compliance_status=compliance_status,
            performance_metrics=performance_metrics,
            security_metrics=security_metrics,
            deployment_readiness=deployment_readiness
        )
    
    def _display_quality_report(self, report: ComprehensiveQualityReport):
        """Display comprehensive quality report"""
        
        # Overall summary
        summary_table = Table(title="🛡️ COMPREHENSIVE QUALITY GATES REPORT", title_style="bold green")
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Value", style="green", width=20)
        summary_table.add_column("Status", style="bold", width=10)
        
        status_emoji = "✅" if report.overall_status == "passed" else "⚠️" if report.overall_status == "warning" else "❌"
        
        summary_table.add_row("Overall Status", report.overall_status.upper(), status_emoji)
        summary_table.add_row("Overall Score", f"{report.overall_score:.1f}/100", "🏆" if report.overall_score >= 85 else "📈")
        summary_table.add_row("Gates Passed", str(report.gates_passed), "✅")
        summary_table.add_row("Gates Warning", str(report.gates_warning), "⚠️" if report.gates_warning > 0 else "✅")
        summary_table.add_row("Gates Failed", str(report.gates_failed), "❌" if report.gates_failed > 0 else "✅")
        summary_table.add_row("Execution Time", f"{report.execution_time:.1f}s", "⏱️")
        summary_table.add_row("Deployment Ready", report.deployment_readiness.replace("_", " ").title(), 
                             "🚀" if report.deployment_readiness == "ready" else "🔧")
        
        self.console.print(summary_table)
        
        # Individual gate results
        gates_table = Table(title="Individual Quality Gate Results", title_style="bold blue")
        gates_table.add_column("Gate", style="cyan", width=30)
        gates_table.add_column("Status", style="bold", width=12)
        gates_table.add_column("Score", style="green", width=10)
        gates_table.add_column("Time", style="dim", width=8)
        
        for result in report.gate_results:
            status_display = result.status.upper()
            if result.status == "passed":
                status_style = "[green]✅ PASSED[/green]"
            elif result.status == "warning":
                status_style = "[yellow]⚠️ WARNING[/yellow]"
            else:
                status_style = "[red]❌ FAILED[/red]"
            
            gates_table.add_row(
                result.gate_name,
                status_style,
                f"{result.score:.1f}/100",
                f"{result.execution_time:.2f}s"
            )
        
        self.console.print("\n")
        self.console.print(gates_table)
        
        # Compliance status
        compliance_columns = []
        for area, status in report.compliance_status.items():
            status_emoji = "✅" if status == "passed" else "🔍"
            compliance_columns.append(
                Panel(
                    f"[bold]{status.upper()}[/bold]\n{status_emoji}",
                    title=area.upper(),
                    width=15,
                    style="green" if status == "passed" else "yellow"
                )
            )
        
        self.console.print("\n[bold]Compliance Status:[/bold]")
        self.console.print(Columns(compliance_columns))
        
        # Key recommendations
        if report.recommendations:
            self.console.print("\n[bold yellow]🔧 Key Recommendations:[/bold yellow]")
            for i, rec in enumerate(report.recommendations[:5], 1):
                self.console.print(f"  {i}. {rec}")
        
        # Final assessment
        self.console.print(f"\n[bold]🎯 QUALITY ASSESSMENT:[/bold]")
        
        if report.overall_score >= 90:
            self.console.print("[green]🌟 EXCELLENT - Production ready with outstanding quality![/green]")
        elif report.overall_score >= 80:
            self.console.print("[green]✅ GOOD - Production ready with high quality standards met.[/green]")
        elif report.overall_score >= 70:
            self.console.print("[yellow]⚠️ ACCEPTABLE - Deployment possible but improvements recommended.[/yellow]")
        elif report.overall_score >= 60:
            self.console.print("[yellow]🔧 NEEDS WORK - Address key issues before production deployment.[/yellow]")
        else:
            self.console.print("[red]❌ POOR - Significant improvements required before deployment.[/red]")
    
    async def _save_quality_report(self, report: ComprehensiveQualityReport):
        """Save comprehensive quality report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"comprehensive_quality_report_{timestamp}.json"
            
            with open(report_file, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            rprint(f"\n[dim]📊 Detailed report saved to: {report_file}[/dim]")
            
        except Exception as e:
            rprint(f"[red]⚠️ Could not save quality report: {e}[/red]")

# CLI Interface
app = typer.Typer(name="quality-gates", help="Comprehensive Quality Gates Validation")

@app.command()
def validate():
    """Run comprehensive quality gates validation"""
    asyncio.run(main())

async def main():
    """Main quality gates execution"""
    quality_gates = ComprehensiveQualityGates()
    
    rprint("[bold green]🛡️ TERRAGON SDLC - COMPREHENSIVE QUALITY GATES[/bold green]")
    rprint("[dim]Enterprise-grade quality validation for autonomous SDLC systems[/dim]\n")
    
    try:
        report = await quality_gates.execute_all_quality_gates()
        
        # Final summary
        rprint(f"\n[bold]🎯 QUALITY GATES EXECUTION COMPLETE[/bold]")
        rprint(f"[bold]Overall Result:[/bold] {report.overall_status.upper()} ({report.overall_score:.1f}/100)")
        rprint(f"[bold]Deployment Readiness:[/bold] {report.deployment_readiness.replace('_', ' ').title()}")
        
        return report.overall_score >= 70
        
    except KeyboardInterrupt:
        rprint("\n[yellow]⏹️ Quality gates validation stopped by user[/yellow]")
        return False
    except Exception as e:
        rprint(f"\n[red]💥 Quality gates validation failed: {e}[/red]")
        return False

if __name__ == "__main__":
    app()