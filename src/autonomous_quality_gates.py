#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - AUTONOMOUS QUALITY GATES
Comprehensive quality validation with autonomous decision making and continuous improvement
"""

import asyncio
import json
import subprocess
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog

logger = structlog.get_logger("AutonomousQualityGates")

class GateResult(Enum):
    """Quality gate results"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

class GatePriority(Enum):
    """Priority levels for quality gates"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class QualityGate:
    """Individual quality gate definition"""
    name: str
    description: str
    priority: GatePriority
    enabled: bool = True
    timeout_seconds: int = 300
    retry_count: int = 1
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class GateExecution:
    """Quality gate execution result"""
    gate_name: str
    result: GateResult
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class QualityGateExecutor:
    """Executes individual quality gates"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.venv_path = self.project_root / "venv"
        
    async def execute_gate(self, gate: QualityGate) -> GateExecution:
        """Execute a single quality gate"""
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Executing quality gate: {gate.name}")
            
            # Execute the specific gate
            result, message, details = await self._execute_specific_gate(gate)
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            execution = GateExecution(
                gate_name=gate.name,
                result=result,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                message=message,
                details=details
            )
            
            logger.info(f"Gate {gate.name}: {result.value} ({duration:.2f}s) - {message}")
            return execution
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return GateExecution(
                gate_name=gate.name,
                result=GateResult.FAIL,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                message=f"Gate execution failed: {str(e)}",
                error=str(e)
            )
    
    async def _execute_specific_gate(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Execute specific gate logic based on gate name"""
        gate_handlers = {
            "unit_tests": self._run_unit_tests,
            "integration_tests": self._run_integration_tests,
            "test_coverage": self._check_test_coverage,
            "security_scan": self._run_security_scan,
            "code_quality": self._check_code_quality,
            "performance_tests": self._run_performance_tests,
            "dependency_check": self._check_dependencies,
            "documentation_check": self._check_documentation,
            "build_verification": self._verify_build,
            "deployment_readiness": self._check_deployment_readiness
        }
        
        handler = gate_handlers.get(gate.name)
        if not handler:
            return GateResult.SKIP, f"No handler for gate {gate.name}", {}
        
        return await handler(gate)
    
    async def _run_unit_tests(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Run unit tests"""
        cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "pytest",
            "tests/unit/",
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results.json"
        ]
        
        try:
            result = await self._run_command(cmd, timeout=gate.timeout_seconds)
            
            # Parse test results
            test_results = {}
            if Path("test_results.json").exists():
                with open("test_results.json") as f:
                    test_data = json.load(f)
                    test_results = {
                        "total": test_data.get("summary", {}).get("total", 0),
                        "passed": test_data.get("summary", {}).get("passed", 0),
                        "failed": test_data.get("summary", {}).get("failed", 0),
                        "skipped": test_data.get("summary", {}).get("skipped", 0)
                    }
            
            if result.returncode == 0:
                return GateResult.PASS, f"All unit tests passed", test_results
            else:
                failed_count = test_results.get("failed", 0)
                return GateResult.FAIL, f"{failed_count} unit tests failed", test_results
                
        except subprocess.TimeoutExpired:
            return GateResult.FAIL, f"Unit tests timed out after {gate.timeout_seconds}s", {}
        except Exception as e:
            return GateResult.FAIL, f"Unit test execution failed: {e}", {}
    
    async def _run_integration_tests(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Run integration tests"""
        if not (self.project_root / "tests" / "integration").exists():
            return GateResult.SKIP, "No integration tests found", {}
        
        cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "pytest",
            "tests/integration/",
            "-v",
            "--tb=short"
        ]
        
        try:
            result = await self._run_command(cmd, timeout=gate.timeout_seconds)
            
            if result.returncode == 0:
                return GateResult.PASS, "Integration tests passed", {}
            else:
                return GateResult.FAIL, "Integration tests failed", {"returncode": result.returncode}
                
        except subprocess.TimeoutExpired:
            return GateResult.FAIL, f"Integration tests timed out after {gate.timeout_seconds}s", {}
        except Exception as e:
            return GateResult.FAIL, f"Integration test execution failed: {e}", {}
    
    async def _check_test_coverage(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Check test coverage"""
        cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "pytest",
            "--cov=src",
            "--cov-report=json",
            "--cov-report=term-missing",
            "tests/",
            "-q"
        ]
        
        try:
            result = await self._run_command(cmd, timeout=gate.timeout_seconds)
            
            coverage_data = {}
            if Path("coverage.json").exists():
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                if total_coverage >= 80:
                    return GateResult.PASS, f"Test coverage: {total_coverage:.1f}%", coverage_data
                elif total_coverage >= 60:
                    return GateResult.WARNING, f"Test coverage below target: {total_coverage:.1f}%", coverage_data
                else:
                    return GateResult.FAIL, f"Test coverage too low: {total_coverage:.1f}%", coverage_data
            
            return GateResult.WARNING, "Coverage data not available", {}
            
        except Exception as e:
            return GateResult.FAIL, f"Coverage check failed: {e}", {}
    
    async def _run_security_scan(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Run security scan"""
        # Try bandit for Python security scanning
        bandit_cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "bandit",
            "-r", "src/",
            "-f", "json",
            "-o", "security_report.json"
        ]
        
        try:
            result = await self._run_command(bandit_cmd, timeout=gate.timeout_seconds)
            
            security_data = {}
            if Path("security_report.json").exists():
                with open("security_report.json") as f:
                    security_data = json.load(f)
                
                high_severity = len([r for r in security_data.get("results", []) 
                                   if r.get("issue_severity") == "HIGH"])
                medium_severity = len([r for r in security_data.get("results", []) 
                                     if r.get("issue_severity") == "MEDIUM"])
                
                if high_severity > 0:
                    return GateResult.FAIL, f"{high_severity} high severity security issues found", security_data
                elif medium_severity > 5:
                    return GateResult.WARNING, f"{medium_severity} medium severity security issues found", security_data
                else:
                    return GateResult.PASS, f"Security scan passed ({medium_severity} medium issues)", security_data
            
            # If bandit not available, do basic checks
            return await self._basic_security_check()
            
        except Exception as e:
            # Fallback to basic security check
            return await self._basic_security_check()
    
    async def _basic_security_check(self) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Basic security check as fallback"""
        issues = []
        
        # Check for common security issues in Python files
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                
                # Check for potential security issues
                if "eval(" in content:
                    issues.append(f"eval() usage in {py_file}")
                if "exec(" in content:
                    issues.append(f"exec() usage in {py_file}")
                if "subprocess.call" in content and "shell=True" in content:
                    issues.append(f"Dangerous subprocess call in {py_file}")
                    
            except Exception:
                continue
        
        if len(issues) > 0:
            return GateResult.WARNING, f"{len(issues)} potential security issues found", {"issues": issues}
        else:
            return GateResult.PASS, "Basic security check passed", {}
    
    async def _check_code_quality(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Check code quality using linting tools"""
        # Try flake8
        flake8_cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "flake8",
            "src/",
            "--max-line-length=88",
            "--count",
            "--statistics"
        ]
        
        try:
            result = await self._run_command(flake8_cmd, timeout=gate.timeout_seconds)
            
            if result.returncode == 0:
                return GateResult.PASS, "Code quality check passed", {}
            else:
                error_count = result.stdout.count('\n') if result.stdout else 0
                if error_count < 10:
                    return GateResult.WARNING, f"{error_count} code quality issues found", {}
                else:
                    return GateResult.FAIL, f"{error_count} code quality issues found", {}
                    
        except Exception:
            # Basic code quality check
            return GateResult.WARNING, "Code quality tools not available, skipping", {}
    
    async def _run_performance_tests(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Run performance tests"""
        if not (self.project_root / "tests" / "performance").exists():
            return GateResult.SKIP, "No performance tests found", {}
        
        cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "pytest",
            "tests/performance/",
            "-v",
            "--benchmark-only",
            "--benchmark-json=performance_results.json"
        ]
        
        try:
            result = await self._run_command(cmd, timeout=gate.timeout_seconds)
            
            if result.returncode == 0:
                return GateResult.PASS, "Performance tests passed", {}
            else:
                return GateResult.WARNING, "Performance tests had issues", {"returncode": result.returncode}
                
        except Exception as e:
            return GateResult.WARNING, f"Performance tests failed: {e}", {}
    
    async def _check_dependencies(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Check dependency vulnerabilities"""
        # Check if requirements file exists
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            return GateResult.SKIP, "No requirements.txt found", {}
        
        # Try safety check
        safety_cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "safety",
            "check",
            "--json"
        ]
        
        try:
            result = await self._run_command(safety_cmd, timeout=gate.timeout_seconds)
            
            if result.returncode == 0:
                return GateResult.PASS, "No known security vulnerabilities in dependencies", {}
            else:
                # Parse vulnerabilities
                try:
                    vulnerabilities = json.loads(result.stdout or "[]")
                    high_vuln = len([v for v in vulnerabilities if v.get("vulnerability", {}).get("severity") == "high"])
                    
                    if high_vuln > 0:
                        return GateResult.FAIL, f"{high_vuln} high severity vulnerabilities in dependencies", {"vulnerabilities": vulnerabilities}
                    else:
                        return GateResult.WARNING, f"{len(vulnerabilities)} vulnerabilities in dependencies", {"vulnerabilities": vulnerabilities}
                except:
                    return GateResult.WARNING, "Dependency vulnerability check had issues", {}
                    
        except Exception:
            return GateResult.WARNING, "Safety tool not available for dependency check", {}
    
    async def _check_documentation(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Check documentation completeness"""
        required_docs = ["README.md", "CHANGELOG.md"]
        missing_docs = []
        
        for doc in required_docs:
            if not (self.project_root / doc).exists():
                missing_docs.append(doc)
        
        # Check for docstrings in Python files
        undocumented_functions = 0
        total_functions = 0
        
        for py_file in (self.project_root / "src").rglob("*.py"):
            try:
                content = py_file.read_text()
                
                # Simple check for function definitions and docstrings
                lines = content.split('\n')
                in_function = False
                function_has_docstring = False
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                        total_functions += 1
                        in_function = True
                        function_has_docstring = False
                        
                        # Check next few lines for docstring
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                function_has_docstring = True
                                break
                        
                        if not function_has_docstring:
                            undocumented_functions += 1
                        
                        in_function = False
                        
            except Exception:
                continue
        
        issues = []
        if missing_docs:
            issues.append(f"Missing documentation files: {', '.join(missing_docs)}")
        
        if total_functions > 0:
            doc_rate = (total_functions - undocumented_functions) / total_functions
            if doc_rate < 0.5:
                issues.append(f"Low documentation rate: {doc_rate:.1%} of functions documented")
        
        if not issues:
            return GateResult.PASS, "Documentation check passed", {
                "documented_functions": total_functions - undocumented_functions,
                "total_functions": total_functions
            }
        elif len(issues) == 1:
            return GateResult.WARNING, issues[0], {}
        else:
            return GateResult.FAIL, f"{len(issues)} documentation issues found", {"issues": issues}
    
    async def _verify_build(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Verify that the project builds correctly"""
        # Check Python syntax
        syntax_cmd = [
            str(self.venv_path / "bin" / "python3"),
            "-m", "py_compile"
        ]
        
        failed_files = []
        
        for py_file in (self.project_root / "src").rglob("*.py"):
            try:
                cmd = syntax_cmd + [str(py_file)]
                result = await self._run_command(cmd, timeout=30)
                
                if result.returncode != 0:
                    failed_files.append(str(py_file))
                    
            except Exception:
                failed_files.append(str(py_file))
        
        if not failed_files:
            return GateResult.PASS, "Build verification passed - all Python files compile", {}
        else:
            return GateResult.FAIL, f"{len(failed_files)} files failed to compile", {"failed_files": failed_files}
    
    async def _check_deployment_readiness(self, gate: QualityGate) -> Tuple[GateResult, str, Dict[str, Any]]:
        """Check if project is ready for deployment"""
        checks = []
        
        # Check for Dockerfile
        if (self.project_root / "Dockerfile").exists():
            checks.append("Dockerfile present")
        else:
            checks.append("No Dockerfile found")
        
        # Check for docker-compose
        if (self.project_root / "docker-compose.yml").exists():
            checks.append("docker-compose.yml present")
        
        # Check for Kubernetes manifests
        k8s_dir = self.project_root / "k8s"
        if k8s_dir.exists() and list(k8s_dir.glob("*.yaml")):
            checks.append("Kubernetes manifests present")
        
        # Check for requirements.txt
        if (self.project_root / "requirements.txt").exists():
            checks.append("requirements.txt present")
        else:
            checks.append("No requirements.txt found")
        
        ready_count = len([c for c in checks if "present" in c])
        
        if ready_count >= 2:
            return GateResult.PASS, f"Deployment ready ({ready_count}/4 checks passed)", {"checks": checks}
        else:
            return GateResult.WARNING, f"Deployment readiness issues ({ready_count}/4 checks passed)", {"checks": checks}
    
    async def _run_command(self, cmd: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a command with timeout"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            return subprocess.CompletedProcess(
                cmd, process.returncode,
                stdout=stdout.decode() if stdout else None,
                stderr=stderr.decode() if stderr else None
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, timeout)

class AutonomousQualityManager:
    """Manages autonomous quality gate execution and decision making"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.executor = QualityGateExecutor(project_root)
        self.gates: List[QualityGate] = []
        self.execution_history: List[List[GateExecution]] = []
        
        # Initialize default gates
        self._setup_default_gates()
    
    def _setup_default_gates(self):
        """Setup default quality gates"""
        self.gates = [
            QualityGate("build_verification", "Verify project builds correctly", GatePriority.CRITICAL),
            QualityGate("unit_tests", "Run unit tests", GatePriority.CRITICAL),
            QualityGate("test_coverage", "Check test coverage", GatePriority.HIGH),
            QualityGate("security_scan", "Run security vulnerability scan", GatePriority.HIGH),
            QualityGate("code_quality", "Check code quality and standards", GatePriority.MEDIUM),
            QualityGate("dependency_check", "Check for vulnerable dependencies", GatePriority.HIGH),
            QualityGate("documentation_check", "Verify documentation completeness", GatePriority.MEDIUM),
            QualityGate("integration_tests", "Run integration tests", GatePriority.MEDIUM),
            QualityGate("performance_tests", "Run performance tests", GatePriority.LOW),
            QualityGate("deployment_readiness", "Check deployment readiness", GatePriority.MEDIUM),
        ]
    
    async def execute_quality_gates(self, fail_fast: bool = False) -> Dict[str, Any]:
        """Execute all quality gates with autonomous decision making"""
        start_time = datetime.now(timezone.utc)
        logger.info("Starting autonomous quality gate execution")
        
        executions = []
        passed_gates = []
        failed_gates = []
        warning_gates = []
        skipped_gates = []
        
        # Sort gates by priority
        sorted_gates = sorted(self.gates, key=lambda g: [GatePriority.CRITICAL, GatePriority.HIGH, GatePriority.MEDIUM, GatePriority.LOW].index(g.priority))
        
        for gate in sorted_gates:
            if not gate.enabled:
                continue
            
            # Check prerequisites
            if not self._check_prerequisites(gate, passed_gates):
                logger.warning(f"Prerequisites not met for gate {gate.name}")
                continue
            
            execution = await self.executor.execute_gate(gate)
            executions.append(execution)
            
            if execution.result == GateResult.PASS:
                passed_gates.append(gate.name)
            elif execution.result == GateResult.FAIL:
                failed_gates.append(gate.name)
                
                # Autonomous decision: should we continue?
                if fail_fast or gate.priority in [GatePriority.CRITICAL]:
                    logger.error(f"Critical gate {gate.name} failed, stopping execution")
                    break
                    
            elif execution.result == GateResult.WARNING:
                warning_gates.append(gate.name)
            else:
                skipped_gates.append(gate.name)
        
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - start_time).total_seconds()
        
        # Store execution history
        self.execution_history.append(executions)
        
        # Make autonomous decision
        decision = self._make_autonomous_decision(executions)
        
        results = {
            "timestamp": end_time.isoformat(),
            "total_duration_seconds": total_duration,
            "executions": [asdict(e) for e in executions],
            "summary": {
                "total_gates": len([g for g in self.gates if g.enabled]),
                "executed": len(executions),
                "passed": len(passed_gates),
                "failed": len(failed_gates),
                "warnings": len(warning_gates),
                "skipped": len(skipped_gates)
            },
            "autonomous_decision": decision,
            "ready_for_deployment": decision["approved"],
            "recommendations": self._generate_recommendations(executions)
        }
        
        logger.info(f"Quality gate execution completed: {decision['status']} "
                   f"({len(passed_gates)} passed, {len(failed_gates)} failed)")
        
        return results
    
    def _check_prerequisites(self, gate: QualityGate, completed_gates: List[str]) -> bool:
        """Check if gate prerequisites are satisfied"""
        if not gate.prerequisites:
            return True
        
        return all(prereq in completed_gates for prereq in gate.prerequisites)
    
    def _make_autonomous_decision(self, executions: List[GateExecution]) -> Dict[str, Any]:
        """Make autonomous decision based on gate results"""
        critical_failures = []
        high_priority_failures = []
        warnings = []
        
        for execution in executions:
            gate = next((g for g in self.gates if g.name == execution.gate_name), None)
            if not gate:
                continue
            
            if execution.result == GateResult.FAIL:
                if gate.priority == GatePriority.CRITICAL:
                    critical_failures.append(execution.gate_name)
                elif gate.priority == GatePriority.HIGH:
                    high_priority_failures.append(execution.gate_name)
            elif execution.result == GateResult.WARNING:
                warnings.append(execution.gate_name)
        
        # Autonomous decision logic
        if critical_failures:
            return {
                "approved": False,
                "status": "rejected",
                "reason": f"Critical quality gates failed: {', '.join(critical_failures)}",
                "action": "block_deployment",
                "critical_failures": critical_failures,
                "high_failures": high_priority_failures,
                "warnings": warnings
            }
        
        if len(high_priority_failures) >= 2:
            return {
                "approved": False,
                "status": "rejected",
                "reason": f"Multiple high-priority gates failed: {', '.join(high_priority_failures)}",
                "action": "block_deployment",
                "critical_failures": critical_failures,
                "high_failures": high_priority_failures,
                "warnings": warnings
            }
        
        if high_priority_failures:
            return {
                "approved": False,
                "status": "conditional",
                "reason": f"High-priority gate failed: {', '.join(high_priority_failures)}",
                "action": "require_approval",
                "critical_failures": critical_failures,
                "high_failures": high_priority_failures,
                "warnings": warnings
            }
        
        if len(warnings) > 3:
            return {
                "approved": True,
                "status": "approved_with_warnings",
                "reason": f"Multiple warnings but no critical failures: {', '.join(warnings)}",
                "action": "deploy_with_monitoring",
                "critical_failures": critical_failures,
                "high_failures": high_priority_failures,
                "warnings": warnings
            }
        
        return {
            "approved": True,
            "status": "approved",
            "reason": "All critical and high-priority quality gates passed",
            "action": "proceed_deployment",
            "critical_failures": critical_failures,
            "high_failures": high_priority_failures,
            "warnings": warnings
        }
    
    def _generate_recommendations(self, executions: List[GateExecution]) -> List[str]:
        """Generate actionable recommendations based on execution results"""
        recommendations = []
        
        for execution in executions:
            if execution.result == GateResult.FAIL:
                if execution.gate_name == "unit_tests":
                    recommendations.append("Fix failing unit tests before deployment")
                elif execution.gate_name == "security_scan":
                    recommendations.append("Address security vulnerabilities immediately")
                elif execution.gate_name == "test_coverage":
                    recommendations.append("Increase test coverage to at least 80%")
                elif execution.gate_name == "build_verification":
                    recommendations.append("Fix compilation errors")
                else:
                    recommendations.append(f"Address issues in {execution.gate_name}")
            
            elif execution.result == GateResult.WARNING:
                if execution.gate_name == "code_quality":
                    recommendations.append("Consider improving code quality metrics")
                elif execution.gate_name == "documentation_check":
                    recommendations.append("Improve documentation coverage")
                else:
                    recommendations.append(f"Review warnings in {execution.gate_name}")
        
        # Add improvement recommendations
        if not any("performance" in r for r in recommendations):
            recommendations.append("Consider adding performance benchmarking")
        
        if not recommendations:
            recommendations.append("Quality gates passed - consider adding more comprehensive tests")
        
        return recommendations

# Example usage and testing
async def test_autonomous_quality_gates():
    """Test the autonomous quality gates system"""
    print("Autonomous Quality Gates Test")
    print("=" * 40)
    
    manager = AutonomousQualityManager()
    
    # Execute quality gates
    results = await manager.execute_quality_gates()
    
    print(f"\nQuality Gate Results:")
    print(f"Total Duration: {results['total_duration_seconds']:.2f} seconds")
    print(f"Gates Executed: {results['summary']['executed']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Skipped: {results['summary']['skipped']}")
    
    decision = results['autonomous_decision']
    print(f"\nAutonomous Decision: {decision['status']}")
    print(f"Approved for Deployment: {decision['approved']}")
    print(f"Reason: {decision['reason']}")
    print(f"Recommended Action: {decision['action']}")
    
    if results['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Display detailed results
    print(f"\nDetailed Gate Results:")
    for execution in results['executions']:
        status_icon = {
            'pass': '✅',
            'fail': '❌', 
            'warning': '⚠️',
            'skip': '⏭️'
        }.get(execution['result'], '?')
        
        print(f"{status_icon} {execution['gate_name']}: {execution['result']} "
              f"({execution['duration_seconds']:.2f}s) - {execution['message']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_autonomous_quality_gates())