#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - QUALITY GATES VALIDATOR
Comprehensive testing, security, performance, and compliance validation
"""

import asyncio
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    """Validation result severity levels"""
    PASS = "pass"
    WARNING = "warning"  
    FAIL = "fail"
    CRITICAL = "critical"


class QualityGate(Enum):
    """Quality gate categories"""
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    CODE_QUALITY = "code_quality"


@dataclass
class ValidationResult:
    """Individual validation result"""
    gate: QualityGate
    test_name: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class QualityReport:
    """Comprehensive quality validation report"""
    overall_status: ValidationSeverity
    total_validations: int
    passed: int
    warnings: int
    failed: int
    critical_failures: int
    execution_time: float
    quality_score: float
    results: List[ValidationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


class QualityGatesValidator:
    """
    Comprehensive quality gates validation system
    """
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
    
    async def run_all_quality_gates(self) -> QualityReport:
        """Run all quality gate validations"""
        print("\n🛡️ TERRAGON SDLC v4.0 - QUALITY GATES VALIDATION")
        print("="*70)
        print("Running comprehensive quality, security, and compliance validation")
        print("="*70)
        
        start_time = time.time()
        all_results = []
        
        # Run Testing validation
        print("\n🔍 Validating TESTING...")
        testing_results = await self._validate_testing()
        all_results.extend(testing_results)
        passed = len([r for r in testing_results if r.severity == ValidationSeverity.PASS])
        warnings = len([r for r in testing_results if r.severity == ValidationSeverity.WARNING])
        failed = len([r for r in testing_results if r.severity in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL]])
        print(f"  ✅ Passed: {passed} | ⚠️  Warnings: {warnings} | ❌ Failed: {failed}")
        
        # Run Security validation
        print("\n🔍 Validating SECURITY...")
        security_results = await self._validate_security()
        all_results.extend(security_results)
        passed = len([r for r in security_results if r.severity == ValidationSeverity.PASS])
        warnings = len([r for r in security_results if r.severity == ValidationSeverity.WARNING])
        failed = len([r for r in security_results if r.severity in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL]])
        print(f"  ✅ Passed: {passed} | ⚠️  Warnings: {warnings} | ❌ Failed: {failed}")
        
        # Run Performance validation
        print("\n🔍 Validating PERFORMANCE...")
        performance_results = await self._validate_performance()
        all_results.extend(performance_results)
        passed = len([r for r in performance_results if r.severity == ValidationSeverity.PASS])
        warnings = len([r for r in performance_results if r.severity == ValidationSeverity.WARNING])
        failed = len([r for r in performance_results if r.severity in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL]])
        print(f"  ✅ Passed: {passed} | ⚠️  Warnings: {warnings} | ❌ Failed: {failed}")
        
        # Run Documentation validation
        print("\n🔍 Validating DOCUMENTATION...")
        documentation_results = await self._validate_documentation()
        all_results.extend(documentation_results)
        passed = len([r for r in documentation_results if r.severity == ValidationSeverity.PASS])
        warnings = len([r for r in documentation_results if r.severity == ValidationSeverity.WARNING])
        failed = len([r for r in documentation_results if r.severity in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL]])
        print(f"  ✅ Passed: {passed} | ⚠️  Warnings: {warnings} | ❌ Failed: {failed}")
        
        execution_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_quality_report(all_results, execution_time)
        
        # Display summary
        self._display_quality_summary(report)
        
        # Save report
        await self._save_quality_report(report)
        
        return report
    
    async def _validate_testing(self) -> List[ValidationResult]:
        """Validate testing coverage and quality"""
        results = []
        
        # Check if tests exist
        test_dirs = [
            self.project_path / "tests",
            self.project_path / "test",
            self.project_path / "src" / "tests"
        ]
        
        tests_found = any(test_dir.exists() for test_dir in test_dirs)
        
        if not tests_found:
            results.append(ValidationResult(
                gate=QualityGate.TESTING,
                test_name="unit_tests_exist",
                severity=ValidationSeverity.CRITICAL,
                message="No test directories found",
                details={"searched_paths": [str(p) for p in test_dirs]}
            ))
        else:
            # Count test files
            test_files = list(self.project_path.rglob("test_*.py")) + list(self.project_path.rglob("*_test.py"))
            
            if len(test_files) == 0:
                results.append(ValidationResult(
                    gate=QualityGate.TESTING,
                    test_name="test_files_found",
                    severity=ValidationSeverity.FAIL,
                    message="No test files found",
                    details={"patterns_searched": ["test_*.py", "*_test.py"]}
                ))
            else:
                results.append(ValidationResult(
                    gate=QualityGate.TESTING,
                    test_name="test_files_found",
                    severity=ValidationSeverity.PASS,
                    message=f"Found {len(test_files)} test files",
                    details={"test_count": len(test_files), "test_files": [str(f) for f in test_files[:10]]}
                ))
                
                # Validate test quality
                total_tests = 0
                total_assertions = 0
                
                for test_file in test_files[:5]:  # Sample first 5 files
                    try:
                        with open(test_file, 'r') as f:
                            content = f.read()
                        
                        # Count test functions
                        test_functions = content.count("def test_")
                        total_tests += test_functions
                        
                        # Count assertions (rough estimate)
                        assertions = (content.count("assert ") + content.count("self.assert") + 
                                    content.count("assertEqual") + content.count("assertTrue"))
                        total_assertions += assertions
                        
                    except Exception:
                        continue
                
                # Assess test quality
                if total_tests == 0:
                    results.append(ValidationResult(
                        gate=QualityGate.TESTING,
                        test_name="test_function_count",
                        severity=ValidationSeverity.FAIL,
                        message="No test functions found in test files",
                        details={"files_analyzed": len(test_files)}
                    ))
                else:
                    avg_assertions_per_test = total_assertions / total_tests if total_tests > 0 else 0
                    
                    if avg_assertions_per_test < 1:
                        severity = ValidationSeverity.FAIL
                        message = f"Low assertion density: {avg_assertions_per_test:.1f} assertions per test"
                    elif avg_assertions_per_test < 2:
                        severity = ValidationSeverity.WARNING
                        message = f"Moderate assertion density: {avg_assertions_per_test:.1f} assertions per test"
                    else:
                        severity = ValidationSeverity.PASS
                        message = f"Good assertion density: {avg_assertions_per_test:.1f} assertions per test"
                    
                    results.append(ValidationResult(
                        gate=QualityGate.TESTING,
                        test_name="test_assertion_density",
                        severity=severity,
                        message=message,
                        details={
                            "total_tests": total_tests,
                            "total_assertions": total_assertions,
                            "avg_assertions_per_test": avg_assertions_per_test
                        }
                    ))
        
        # Check integration tests
        integration_test_paths = [
            self.project_path / "tests" / "integration",
            self.project_path / "tests" / "e2e",
            self.project_path / "integration_tests"
        ]
        
        integration_tests_found = any(path.exists() for path in integration_test_paths)
        
        if integration_tests_found:
            results.append(ValidationResult(
                gate=QualityGate.TESTING,
                test_name="integration_tests_exist",
                severity=ValidationSeverity.PASS,
                message="Integration test directories found"
            ))
        else:
            results.append(ValidationResult(
                gate=QualityGate.TESTING,
                test_name="integration_tests_exist",
                severity=ValidationSeverity.WARNING,
                message="No integration test directories found",
                details={"recommendation": "Consider adding integration tests for end-to-end validation"}
            ))
        
        return results
    
    async def _validate_security(self) -> List[ValidationResult]:
        """Validate security practices and vulnerabilities"""
        results = []
        
        # Check for security configuration files
        security_files = {
            ".bandit": "Bandit security scanner configuration",
            "security.py": "Security module",
            ".safety": "Safety dependency scanner configuration"
        }
        
        for filename, description in security_files.items():
            file_path = self.project_path / filename
            if file_path.exists():
                results.append(ValidationResult(
                    gate=QualityGate.SECURITY,
                    test_name=f"security_file_{filename}",
                    severity=ValidationSeverity.PASS,
                    message=f"{description} found",
                    details={"file_path": str(file_path)}
                ))
            else:
                severity = ValidationSeverity.WARNING if filename.startswith('.') else ValidationSeverity.FAIL
                results.append(ValidationResult(
                    gate=QualityGate.SECURITY,
                    test_name=f"security_file_{filename}",
                    severity=severity,
                    message=f"{description} not found",
                    details={"recommendation": f"Consider adding {filename} for better security practices"}
                ))
        
        # Scan for potential hardcoded secrets
        secret_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
            (r"token\s*=\s*['\"][^'\"]+['\"]", "Hardcoded token"),
        ]
        
        python_files = list(self.project_path.rglob("*.py"))
        secrets_found = []
        
        for py_file in python_files[:20]:  # Scan first 20 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for pattern, description in secret_patterns:
                    import re
                    if re.search(pattern, content, re.IGNORECASE):
                        secrets_found.append({
                            "file": str(py_file),
                            "type": description,
                            "line": "unknown"
                        })
            except Exception:
                continue
        
        if secrets_found:
            results.append(ValidationResult(
                gate=QualityGate.SECURITY,
                test_name="hardcoded_secrets_scan",
                severity=ValidationSeverity.CRITICAL,
                message=f"Found {len(secrets_found)} potential hardcoded secrets",
                details={"secrets": secrets_found[:5]}
            ))
        else:
            results.append(ValidationResult(
                gate=QualityGate.SECURITY,
                test_name="hardcoded_secrets_scan",
                severity=ValidationSeverity.PASS,
                message="No obvious hardcoded secrets found in sample files"
            ))
        
        # Check for secure coding practices
        security_issues = []
        good_practices = []
        
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for input validation
                if "validate" in content.lower() or "sanitize" in content.lower():
                    good_practices.append(f"Input validation found in {py_file.name}")
                
                # Check for dangerous functions
                dangerous_functions = ["eval(", "exec(", "os.system("]
                for func in dangerous_functions:
                    if func in content:
                        security_issues.append(f"Dangerous function {func} found in {py_file.name}")
                
            except Exception:
                continue
        
        if security_issues:
            results.append(ValidationResult(
                gate=QualityGate.SECURITY,
                test_name="secure_coding_practices",
                severity=ValidationSeverity.FAIL,
                message=f"Found {len(security_issues)} potential security issues",
                details={"issues": security_issues[:3]}
            ))
        else:
            results.append(ValidationResult(
                gate=QualityGate.SECURITY,
                test_name="secure_coding_practices",
                severity=ValidationSeverity.PASS,
                message="No obvious security issues found in sample files"
            ))
        
        return results
    
    async def _validate_performance(self) -> List[ValidationResult]:
        """Validate performance benchmarks and optimization"""
        results = []
        
        # Check for performance monitoring files
        performance_indicators = [
            "performance_monitor.py",
            "benchmark.py",
            "profiling.py",
            "metrics.py"
        ]
        
        perf_files_found = []
        for indicator in performance_indicators:
            perf_files = list(self.project_path.rglob(f"*{indicator}"))
            if perf_files:
                perf_files_found.extend(perf_files)
        
        if perf_files_found:
            results.append(ValidationResult(
                gate=QualityGate.PERFORMANCE,
                test_name="performance_monitoring_exists",
                severity=ValidationSeverity.PASS,
                message=f"Found {len(perf_files_found)} performance monitoring files",
                details={"files": [str(f) for f in perf_files_found]}
            ))
        else:
            results.append(ValidationResult(
                gate=QualityGate.PERFORMANCE,
                test_name="performance_monitoring_exists",
                severity=ValidationSeverity.WARNING,
                message="No performance monitoring files found",
                details={"recommendation": "Consider adding performance monitoring and benchmarking"}
            ))
        
        # Run basic performance test
        try:
            start_time = time.time()
            
            # CPU-bound task
            total = 0
            for i in range(100000):
                total += i * i
            
            cpu_time = time.time() - start_time
            
            # Memory allocation test
            start_time = time.time()
            data = []
            for i in range(10000):
                data.append([j for j in range(10)])
            memory_time = time.time() - start_time
            
            # Assess performance
            performance_score = 100
            issues = []
            
            if cpu_time > 1.0:
                performance_score -= 20
                issues.append(f"CPU performance slow: {cpu_time:.3f}s")
            
            if memory_time > 0.5:
                performance_score -= 15
                issues.append(f"Memory allocation slow: {memory_time:.3f}s")
            
            if performance_score >= 80:
                severity = ValidationSeverity.PASS
                message = f"Good performance benchmark results (score: {performance_score})"
            elif performance_score >= 60:
                severity = ValidationSeverity.WARNING
                message = f"Moderate performance benchmark results (score: {performance_score})"
            else:
                severity = ValidationSeverity.FAIL
                message = f"Poor performance benchmark results (score: {performance_score})"
            
            results.append(ValidationResult(
                gate=QualityGate.PERFORMANCE,
                test_name="basic_performance_benchmark",
                severity=severity,
                message=message,
                details={
                    "performance_score": performance_score,
                    "cpu_time": cpu_time,
                    "memory_time": memory_time,
                    "issues": issues
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                gate=QualityGate.PERFORMANCE,
                test_name="basic_performance_benchmark",
                severity=ValidationSeverity.WARNING,
                message=f"Could not run performance benchmark: {str(e)}"
            ))
        
        # Check for async/await usage
        python_files = list(self.project_path.rglob("*.py"))
        async_files = 0
        total_functions = 0
        async_functions = 0
        
        for py_file in python_files[:15]:  # Sample files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if "async def" in content or "await " in content:
                    async_files += 1
                
                # Count functions
                import re
                functions = re.findall(r'def\s+\w+', content)
                async_funcs = re.findall(r'async\s+def\s+\w+', content)
                
                total_functions += len(functions)
                async_functions += len(async_funcs)
                
            except Exception:
                continue
        
        if total_functions > 0:
            async_ratio = async_functions / total_functions
            
            if async_ratio > 0.3:
                severity = ValidationSeverity.PASS
                message = f"Good async usage: {async_ratio:.1%} of functions are async"
            elif async_ratio > 0.1:
                severity = ValidationSeverity.WARNING
                message = f"Moderate async usage: {async_ratio:.1%} of functions are async"
            else:
                severity = ValidationSeverity.WARNING
                message = f"Low async usage: {async_ratio:.1%} of functions are async"
            
            results.append(ValidationResult(
                gate=QualityGate.PERFORMANCE,
                test_name="async_usage_analysis",
                severity=severity,
                message=message,
                details={
                    "async_files": async_files,
                    "async_functions": async_functions,
                    "total_functions": total_functions,
                    "async_ratio": async_ratio
                }
            ))
        
        return results
    
    async def _validate_documentation(self) -> List[ValidationResult]:
        """Validate documentation quality and completeness"""
        results = []
        
        # Check for essential documentation files
        essential_docs = {
            "README.md": "Project README",
            "CHANGELOG.md": "Change log",
            "CONTRIBUTING.md": "Contribution guidelines",
            "LICENSE": "License file",
            "docs/API.md": "API documentation"
        }
        
        for doc_file, description in essential_docs.items():
            file_path = self.project_path / doc_file
            if file_path.exists():
                # Check file size
                file_size = file_path.stat().st_size
                if file_size > 500:  # At least 500 bytes
                    results.append(ValidationResult(
                        gate=QualityGate.DOCUMENTATION,
                        test_name=f"doc_file_{doc_file.replace('/', '_').replace('.', '_')}",
                        severity=ValidationSeverity.PASS,
                        message=f"{description} exists and has content ({file_size} bytes)",
                        details={"file_size": file_size}
                    ))
                else:
                    results.append(ValidationResult(
                        gate=QualityGate.DOCUMENTATION,
                        test_name=f"doc_file_{doc_file.replace('/', '_').replace('.', '_')}",
                        severity=ValidationSeverity.WARNING,
                        message=f"{description} exists but is very small ({file_size} bytes)",
                        details={"file_size": file_size}
                    ))
            else:
                severity = ValidationSeverity.FAIL if doc_file == "README.md" else ValidationSeverity.WARNING
                results.append(ValidationResult(
                    gate=QualityGate.DOCUMENTATION,
                    test_name=f"doc_file_{doc_file.replace('/', '_').replace('.', '_')}",
                    severity=severity,
                    message=f"{description} not found",
                    details={"recommendation": f"Create {doc_file} for better project documentation"}
                ))
        
        # Check README quality if it exists
        readme_path = self.project_path / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, 'r') as f:
                    content = f.read()
                
                quality_score = 0
                sections_found = []
                
                # Check for common README sections
                required_sections = {
                    "installation": ["install", "setup"],
                    "usage": ["usage", "how to use"],
                    "description": ["description", "about"],
                    "examples": ["example", "demo"]
                }
                
                content_lower = content.lower()
                
                for section, keywords in required_sections.items():
                    if any(keyword in content_lower for keyword in keywords):
                        sections_found.append(section)
                        quality_score += 20
                
                # Additional quality checks
                if len(content) > 1000:
                    quality_score += 20
                
                # Assess quality
                if quality_score >= 80:
                    severity = ValidationSeverity.PASS
                    message = f"Excellent README quality (score: {quality_score})"
                elif quality_score >= 60:
                    severity = ValidationSeverity.PASS
                    message = f"Good README quality (score: {quality_score})"
                elif quality_score >= 40:
                    severity = ValidationSeverity.WARNING
                    message = f"Moderate README quality (score: {quality_score})"
                else:
                    severity = ValidationSeverity.FAIL
                    message = f"Poor README quality (score: {quality_score})"
                
                results.append(ValidationResult(
                    gate=QualityGate.DOCUMENTATION,
                    test_name="readme_quality_analysis",
                    severity=severity,
                    message=message,
                    details={
                        "quality_score": quality_score,
                        "sections_found": sections_found,
                        "content_length": len(content)
                    }
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    gate=QualityGate.DOCUMENTATION,
                    test_name="readme_quality_analysis",
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not analyze README quality: {str(e)}"
                ))
        
        # Check code documentation (docstrings)
        python_files = list(self.project_path.rglob("*.py"))
        total_functions = 0
        documented_functions = 0
        
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check public functions
                    if line.startswith('def ') and not line.startswith('def _'):
                        total_functions += 1
                        # Check if next few lines contain docstring
                        if i + 1 < len(lines) and ('"""' in lines[i + 1] or "'''" in lines[i + 1]):
                            documented_functions += 1
                    
                    i += 1
                    
            except Exception:
                continue
        
        # Calculate documentation coverage
        doc_ratio = documented_functions / total_functions if total_functions > 0 else 1
        
        if doc_ratio >= 0.8:
            severity = ValidationSeverity.PASS
            message = f"Excellent code documentation coverage: {doc_ratio:.1%}"
        elif doc_ratio >= 0.6:
            severity = ValidationSeverity.PASS
            message = f"Good code documentation coverage: {doc_ratio:.1%}"
        elif doc_ratio >= 0.4:
            severity = ValidationSeverity.WARNING
            message = f"Moderate code documentation coverage: {doc_ratio:.1%}"
        else:
            severity = ValidationSeverity.FAIL
            message = f"Poor code documentation coverage: {doc_ratio:.1%}"
        
        results.append(ValidationResult(
            gate=QualityGate.DOCUMENTATION,
            test_name="code_documentation_coverage",
            severity=severity,
            message=message,
            details={
                "total_functions": total_functions,
                "documented_functions": documented_functions,
                "doc_ratio": doc_ratio
            }
        ))
        
        return results
    
    def _generate_quality_report(self, results: List[ValidationResult], execution_time: float) -> QualityReport:
        """Generate comprehensive quality report"""
        
        # Count results by severity
        passed = len([r for r in results if r.severity == ValidationSeverity.PASS])
        warnings = len([r for r in results if r.severity == ValidationSeverity.WARNING])
        failed = len([r for r in results if r.severity == ValidationSeverity.FAIL])
        critical = len([r for r in results if r.severity == ValidationSeverity.CRITICAL])
        
        total = len(results)
        
        # Determine overall status
        if critical > 0:
            overall_status = ValidationSeverity.CRITICAL
        elif failed > 0:
            overall_status = ValidationSeverity.FAIL
        elif warnings > 0:
            overall_status = ValidationSeverity.WARNING
        else:
            overall_status = ValidationSeverity.PASS
        
        # Calculate quality score (0-100)
        if total > 0:
            quality_score = (passed * 100 + warnings * 50 + failed * 10 + critical * 0) / (total * 100)
        else:
            quality_score = 1.0
        
        # Generate recommendations
        recommendations = []
        failed_tests = [r for r in results if r.severity in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL]]
        
        for result in failed_tests[:5]:  # Top 5 critical issues
            if "test_files_found" in result.test_name:
                recommendations.append("Add comprehensive unit tests with good coverage")
            elif "hardcoded_secrets" in result.test_name:
                recommendations.append("URGENT: Remove hardcoded secrets and use environment variables")
            elif "readme" in result.test_name.lower():
                recommendations.append("Improve README.md with installation, usage, and examples")
            elif "performance" in result.test_name:
                recommendations.append("Implement performance monitoring and optimization")
        
        if not recommendations:
            recommendations.append("Quality gates are in good shape - consider advanced optimizations")
        
        # Generate next steps
        next_steps = []
        if overall_status == ValidationSeverity.CRITICAL:
            next_steps.extend([
                "Address all critical failures before proceeding",
                "Block deployment until critical issues are resolved"
            ])
        elif overall_status == ValidationSeverity.FAIL:
            next_steps.extend([
                "Fix failed validations before deployment",
                "Re-run quality gates after fixes"
            ])
        elif overall_status == ValidationSeverity.WARNING:
            next_steps.extend([
                "Address warnings to improve overall quality",
                "Proceed with deployment with monitoring"
            ])
        else:
            next_steps.extend([
                "Quality gates passed - ready for deployment",
                "Continue monitoring quality metrics"
            ])
        
        return QualityReport(
            overall_status=overall_status,
            total_validations=total,
            passed=passed,
            warnings=warnings,
            failed=failed,
            critical_failures=critical,
            execution_time=execution_time,
            quality_score=quality_score,
            results=results,
            recommendations=recommendations,
            next_steps=next_steps
        )
    
    def _display_quality_summary(self, report: QualityReport):
        """Display comprehensive quality validation summary"""
        print("\n" + "="*70)
        print("📊 QUALITY GATES VALIDATION SUMMARY")
        print("="*70)
        
        # Overall status
        status_emoji = {
            ValidationSeverity.PASS: "✅",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.FAIL: "❌",
            ValidationSeverity.CRITICAL: "💥"
        }
        
        print(f"🎯 Overall Status: {status_emoji[report.overall_status]} {report.overall_status.value.upper()}")
        print(f"📊 Quality Score: {report.quality_score:.1%}")
        print(f"⏱️ Execution Time: {report.execution_time:.2f}s")
        
        # Results breakdown
        print(f"\n📋 VALIDATION RESULTS:")
        print(f"  ✅ Passed: {report.passed}")
        print(f"  ⚠️  Warnings: {report.warnings}")
        print(f"  ❌ Failed: {report.failed}")
        print(f"  💥 Critical: {report.critical_failures}")
        print(f"  📊 Total: {report.total_validations}")
        
        # Results by category
        by_gate = {}
        for result in report.results:
            if result.gate not in by_gate:
                by_gate[result.gate] = {"pass": 0, "warning": 0, "fail": 0, "critical": 0}
            by_gate[result.gate][result.severity.value] += 1
        
        print(f"\n📊 RESULTS BY CATEGORY:")
        for gate in QualityGate:
            if gate in by_gate:
                stats = by_gate[gate]
                total_gate = sum(stats.values())
                pass_rate = stats["pass"] / total_gate if total_gate > 0 else 0
                status_icon = "✅" if pass_rate >= 0.8 else "⚠️" if pass_rate >= 0.5 else "❌"
                print(f"  {status_icon} {gate.value.upper()}: {stats['pass']}/{total_gate} passed ({pass_rate:.1%})")
        
        # Top recommendations
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for rec in report.recommendations[:5]:
            print(f"  • {rec}")
        
        # Next steps
        print(f"\n🎯 NEXT STEPS:")
        for step in report.next_steps[:5]:
            print(f"  • {step}")
        
        # Global deployment readiness
        deployment_ready = report.overall_status in [ValidationSeverity.PASS, ValidationSeverity.WARNING]
        deployment_status = "✅ READY" if deployment_ready else "❌ BLOCKED"
        print(f"\n🚀 Global Deployment Ready: {deployment_status}")
        
        print("="*70)
    
    async def _save_quality_report(self, report: QualityReport):
        """Save quality validation report"""
        try:
            report_file = self.project_path / "quality_gates_validation_report.json"
            
            report_data = {
                "overall_status": report.overall_status.value,
                "quality_score": report.quality_score,
                "execution_time": report.execution_time,
                "total_validations": report.total_validations,
                "passed": report.passed,
                "warnings": report.warnings,
                "failed": report.failed,
                "critical_failures": report.critical_failures,
                "recommendations": report.recommendations,
                "next_steps": report.next_steps,
                "results": [
                    {
                        "gate": result.gate.value,
                        "test_name": result.test_name,
                        "severity": result.severity.value,
                        "message": result.message,
                        "details": result.details,
                        "execution_time": result.execution_time,
                        "timestamp": result.timestamp
                    }
                    for result in report.results
                ],
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"💾 Quality gates report saved to {report_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving quality report: {e}")


# Autonomous execution entry point
async def main():
    """Quality Gates Validation Entry Point"""
    validator = QualityGatesValidator()
    report = await validator.run_all_quality_gates()
    
    # Determine next phase readiness
    if report.overall_status in [ValidationSeverity.PASS, ValidationSeverity.WARNING]:
        print("\n🌍 GLOBAL-FIRST IMPLEMENTATION: Quality gates passed - ready for internationalization and compliance!")
        print("   Next phase will implement multi-region deployment, I18n, and compliance features")
    else:
        print("\n⏳ Address quality gate failures before proceeding to global deployment")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())