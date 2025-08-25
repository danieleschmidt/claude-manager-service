#!/usr/bin/env python3
"""
COMPREHENSIVE QUALITY GATES - FINAL VALIDATION
Complete quality assurance framework for the autonomous SDLC system
"""

import asyncio
import json
import time
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import concurrent.futures
import threading

# Simple logger since imports are failing
class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def warning(self, msg): print(f"[WARNING] {msg}")

logger = SimpleLogger()


@dataclass
class QualityGateResult:
    """Quality gate validation result"""
    gate_name: str
    status: str  # passed, failed, warning, skipped
    score: float
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: str


class ComprehensiveQualityGates:
    """Comprehensive quality assurance system"""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        self.total_score = 0.0
        self.gates_passed = 0
        self.gates_failed = 0
        
    async def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results"""
        logger.info("Starting comprehensive quality gates validation")
        
        # Define all quality gates
        quality_gates = [
            ("Code Structure Validation", self._validate_code_structure),
            ("Dependency Analysis", self._validate_dependencies),
            ("Security Scan", self._security_validation),
            ("Performance Benchmarks", self._performance_validation),
            ("Documentation Check", self._documentation_validation),
            ("Configuration Validation", self._configuration_validation),
            ("Integration Points", self._integration_validation),
            ("Error Handling Coverage", self._error_handling_validation),
            ("Monitoring Systems", self._monitoring_validation),
            ("Scalability Assessment", self._scalability_validation)
        ]
        
        # Run gates concurrently
        tasks = []
        for gate_name, gate_func in quality_gates:
            task = self._run_quality_gate(gate_name, gate_func)
            tasks.append(task)
        
        # Wait for all gates to complete
        gate_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in gate_results:
            if isinstance(result, Exception):
                self.results.append(QualityGateResult(
                    gate_name="Unknown Gate",
                    status="failed",
                    score=0.0,
                    message=f"Exception: {str(result)}",
                    details={},
                    execution_time=0.0,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))
            else:
                self.results.append(result)
        
        return self._generate_final_report()
    
    async def _run_quality_gate(self, gate_name: str, gate_func) -> QualityGateResult:
        """Run individual quality gate"""
        start_time = time.time()
        
        try:
            logger.info(f"Running quality gate: {gate_name}")
            
            if asyncio.iscoroutinefunction(gate_func):
                result = await gate_func()
            else:
                result = gate_func()
            
            execution_time = time.time() - start_time
            
            if isinstance(result, QualityGateResult):
                result.execution_time = execution_time
                return result
            else:
                # Convert simple result to QualityGateResult
                status = "passed" if result.get("success", False) else "failed"
                return QualityGateResult(
                    gate_name=gate_name,
                    status=status,
                    score=result.get("score", 0.0),
                    message=result.get("message", ""),
                    details=result.get("details", {}),
                    execution_time=execution_time,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quality gate {gate_name} failed with exception: {e}")
            
            return QualityGateResult(
                gate_name=gate_name,
                status="failed",
                score=0.0,
                message=f"Exception: {str(e)}",
                details={"exception_type": type(e).__name__},
                execution_time=execution_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    def _validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization"""
        repo_path = Path("/root/repo")
        
        # Check essential directories
        required_dirs = ["src", "tests", "docs", "scripts"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not (repo_path / dir_name).exists():
                missing_dirs.append(dir_name)
        
        # Count Python files
        src_files = list((repo_path / "src").glob("*.py")) if (repo_path / "src").exists() else []
        test_files = list((repo_path / "tests").glob("**/*.py")) if (repo_path / "tests").exists() else []
        
        # Check for key files
        key_files = ["README.md", "requirements.txt", "config.json"]
        missing_files = []
        
        for file_name in key_files:
            if not (repo_path / file_name).exists():
                missing_files.append(file_name)
        
        # Calculate score
        structure_score = 100.0
        structure_score -= len(missing_dirs) * 15  # -15 per missing directory
        structure_score -= len(missing_files) * 10  # -10 per missing file
        structure_score = max(structure_score, 0.0)
        
        success = structure_score >= 70.0
        
        return {
            "success": success,
            "score": structure_score,
            "message": f"Code structure validation {'passed' if success else 'failed'}",
            "details": {
                "src_files_count": len(src_files),
                "test_files_count": len(test_files),
                "missing_directories": missing_dirs,
                "missing_files": missing_files,
                "structure_score": structure_score
            }
        }
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies and package management"""
        repo_path = Path("/root/repo")
        
        # Check requirements.txt
        requirements_file = repo_path / "requirements.txt"
        dependencies = []
        
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                return {
                    "success": False,
                    "score": 0.0,
                    "message": f"Failed to read requirements.txt: {e}",
                    "details": {}
                }
        
        # Basic dependency validation
        essential_deps = ["PyGithub", "pytest", "aiohttp", "pydantic", "typer"]
        missing_deps = []
        
        for dep in essential_deps:
            if not any(dep.lower() in d.lower() for d in dependencies):
                missing_deps.append(dep)
        
        # Check for security vulnerabilities (basic check)
        vulnerable_patterns = ["==", "<=", "<"]  # Pinned versions can be vulnerable
        potentially_vulnerable = []
        
        for dep in dependencies:
            if any(pattern in dep for pattern in vulnerable_patterns):
                potentially_vulnerable.append(dep)
        
        # Calculate score
        dep_score = 100.0
        dep_score -= len(missing_deps) * 10
        dep_score -= len(potentially_vulnerable) * 2
        dep_score = max(dep_score, 0.0)
        
        success = dep_score >= 70.0 and len(missing_deps) == 0
        
        return {
            "success": success,
            "score": dep_score,
            "message": f"Dependency validation {'passed' if success else 'failed'}",
            "details": {
                "total_dependencies": len(dependencies),
                "missing_essential": missing_deps,
                "potentially_vulnerable": potentially_vulnerable,
                "dependency_score": dep_score
            }
        }
    
    def _security_validation(self) -> Dict[str, Any]:
        """Security validation and vulnerability assessment"""
        repo_path = Path("/root/repo")
        security_issues = []
        security_score = 100.0
        
        # Check for hardcoded secrets (basic patterns)
        secret_patterns = [
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]"
        ]
        
        try:
            # Check Python files for potential secrets
            python_files = list((repo_path / "src").glob("**/*.py")) if (repo_path / "src").exists() else []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in secret_patterns:
                            import re
                            if re.search(pattern, content, re.IGNORECASE):
                                security_issues.append(f"Potential hardcoded secret in {py_file.name}")
                                security_score -= 15
                except Exception:
                    continue
        
        except Exception as e:
            security_issues.append(f"Failed to scan for secrets: {e}")
            security_score -= 20
        
        # Check for secure configuration
        config_file = repo_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                    # Check for insecure configurations
                    if any(key in str(config).lower() for key in ["password", "secret", "key"]):
                        if not any(key in str(config).lower() for key in ["env", "environment", "var"]):
                            security_issues.append("Potential credentials in config file")
                            security_score -= 10
                            
            except Exception as e:
                security_issues.append(f"Failed to validate config security: {e}")
                security_score -= 5
        
        # Check for security-related files
        security_files = ["SECURITY.md", ".gitignore"]
        missing_security_files = []
        
        for sec_file in security_files:
            if not (repo_path / sec_file).exists():
                missing_security_files.append(sec_file)
                security_score -= 5
        
        security_score = max(security_score, 0.0)
        success = security_score >= 80.0 and len(security_issues) <= 2
        
        return {
            "success": success,
            "score": security_score,
            "message": f"Security validation {'passed' if success else 'failed'} with {len(security_issues)} issues",
            "details": {
                "security_issues": security_issues,
                "missing_security_files": missing_security_files,
                "security_score": security_score,
                "files_scanned": len(python_files) if 'python_files' in locals() else 0
            }
        }
    
    def _performance_validation(self) -> Dict[str, Any]:
        """Performance benchmarks and optimization validation"""
        
        # Performance metrics to validate
        performance_score = 100.0
        performance_issues = []
        
        # Check for performance-related files
        repo_path = Path("/root/repo")
        perf_indicators = {
            "caching": ["cache", "redis", "memcache"],
            "async": ["async", "await", "asyncio"],
            "optimization": ["optimize", "performance", "benchmark"],
            "monitoring": ["monitor", "metric", "prometheus"]
        }
        
        detected_optimizations = {}
        
        try:
            src_path = repo_path / "src"
            if src_path.exists():
                python_files = list(src_path.glob("**/*.py"))
                
                for category, keywords in perf_indicators.items():
                    detected_optimizations[category] = 0
                    
                    for py_file in python_files:
                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                for keyword in keywords:
                                    if keyword in content:
                                        detected_optimizations[category] += 1
                                        break
                        except Exception:
                            continue
                
                # Score based on detected optimizations
                total_files = len(python_files)
                if total_files > 0:
                    for category, count in detected_optimizations.items():
                        if count == 0:
                            performance_issues.append(f"No {category} optimizations detected")
                            performance_score -= 10
        
        except Exception as e:
            performance_issues.append(f"Failed to analyze performance patterns: {e}")
            performance_score -= 20
        
        # Check for performance configuration
        config_files = ["performance.json", "monitoring_config.json"]
        perf_config_found = any((repo_path / cf).exists() for cf in config_files)
        
        if perf_config_found:
            performance_score += 10
        else:
            performance_issues.append("No performance configuration files found")
            performance_score -= 5
        
        performance_score = max(performance_score, 0.0)
        success = performance_score >= 70.0 and len(performance_issues) <= 3
        
        return {
            "success": success,
            "score": performance_score,
            "message": f"Performance validation {'passed' if success else 'failed'}",
            "details": {
                "detected_optimizations": detected_optimizations,
                "performance_issues": performance_issues,
                "performance_score": performance_score,
                "config_files_present": perf_config_found
            }
        }
    
    def _documentation_validation(self) -> Dict[str, Any]:
        """Documentation completeness and quality validation"""
        repo_path = Path("/root/repo")
        doc_score = 100.0
        doc_issues = []
        
        # Check for essential documentation files
        essential_docs = ["README.md", "ARCHITECTURE.md"]
        optional_docs = ["CONTRIBUTING.md", "LICENSE", "CHANGELOG.md", "SECURITY.md"]
        
        missing_essential = []
        missing_optional = []
        
        for doc in essential_docs:
            if not (repo_path / doc).exists():
                missing_essential.append(doc)
                doc_score -= 20
        
        for doc in optional_docs:
            if not (repo_path / doc).exists():
                missing_optional.append(doc)
                doc_score -= 5
        
        # Check README quality
        readme_path = repo_path / "README.md"
        readme_quality = 0
        
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read().lower()
                    
                    # Check for essential README sections
                    readme_sections = ["installation", "usage", "configuration", "development"]
                    for section in readme_sections:
                        if section in readme_content:
                            readme_quality += 25
                    
                    # Bonus for comprehensive README
                    if len(readme_content) > 5000:  # Substantial README
                        readme_quality += 10
                        
            except Exception as e:
                doc_issues.append(f"Failed to analyze README: {e}")
                doc_score -= 10
        
        # Check docs directory
        docs_dir = repo_path / "docs"
        docs_count = 0
        
        if docs_dir.exists():
            docs_count = len(list(docs_dir.glob("**/*.md")))
            if docs_count > 5:
                doc_score += 10
            elif docs_count > 0:
                doc_score += 5
        else:
            doc_issues.append("No docs directory found")
            doc_score -= 10
        
        doc_score = max(doc_score + (readme_quality * 0.3), 0.0)
        success = doc_score >= 80.0 and len(missing_essential) == 0
        
        return {
            "success": success,
            "score": doc_score,
            "message": f"Documentation validation {'passed' if success else 'failed'}",
            "details": {
                "missing_essential_docs": missing_essential,
                "missing_optional_docs": missing_optional,
                "readme_quality_score": readme_quality,
                "docs_directory_files": docs_count,
                "documentation_score": doc_score,
                "issues": doc_issues
            }
        }
    
    def _configuration_validation(self) -> Dict[str, Any]:
        """Configuration management validation"""
        repo_path = Path("/root/repo")
        config_score = 100.0
        config_issues = []
        
        # Check for configuration files
        config_files = ["config.json", "pyproject.toml", "pytest.ini"]
        found_configs = []
        
        for config_file in config_files:
            if (repo_path / config_file).exists():
                found_configs.append(config_file)
        
        if not found_configs:
            config_issues.append("No configuration files found")
            config_score -= 30
        
        # Validate main config.json
        main_config = repo_path / "config.json"
        if main_config.exists():
            try:
                with open(main_config, 'r') as f:
                    config_data = json.load(f)
                    
                    # Check for essential config sections
                    essential_sections = ["github", "analyzer", "executor"]
                    missing_sections = []
                    
                    for section in essential_sections:
                        if section not in config_data:
                            missing_sections.append(section)
                            config_score -= 15
                    
                    if missing_sections:
                        config_issues.append(f"Missing config sections: {missing_sections}")
                        
            except Exception as e:
                config_issues.append(f"Invalid JSON in config.json: {e}")
                config_score -= 25
        
        # Check environment variable handling
        env_file = repo_path / ".env"
        env_example = repo_path / ".env.example"
        
        if env_example.exists():
            config_score += 10
        elif env_file.exists():
            config_issues.append("Found .env file but no .env.example for documentation")
            config_score -= 5
        
        config_score = max(config_score, 0.0)
        success = config_score >= 70.0 and len(config_issues) <= 2
        
        return {
            "success": success,
            "score": config_score,
            "message": f"Configuration validation {'passed' if success else 'failed'}",
            "details": {
                "found_config_files": found_configs,
                "config_issues": config_issues,
                "config_score": config_score,
                "has_env_example": (repo_path / ".env.example").exists()
            }
        }
    
    def _integration_validation(self) -> Dict[str, Any]:
        """Integration points and API validation"""
        repo_path = Path("/root/repo")
        integration_score = 100.0
        integration_issues = []
        
        # Check for integration-related files
        integration_indicators = [
            "github_api.py",
            "orchestrator.py", 
            "async_",
            "webhook",
            "api"
        ]
        
        found_integrations = []
        
        try:
            src_path = repo_path / "src"
            if src_path.exists():
                for py_file in src_path.glob("**/*.py"):
                    file_name = py_file.name.lower()
                    
                    for indicator in integration_indicators:
                        if indicator in file_name:
                            found_integrations.append(py_file.name)
                            break
        
        except Exception as e:
            integration_issues.append(f"Failed to analyze integration files: {e}")
            integration_score -= 20
        
        if len(found_integrations) < 3:
            integration_issues.append("Limited integration components found")
            integration_score -= 15
        
        # Check for API documentation
        api_docs = ["API.md", "docs/API.md", "api_documentation.md"]
        has_api_docs = any((repo_path / doc).exists() for doc in api_docs)
        
        if not has_api_docs:
            integration_issues.append("No API documentation found")
            integration_score -= 10
        
        # Check for webhook/event handling
        webhook_files = []
        if src_path.exists():
            for py_file in src_path.glob("**/*.py"):
                if "webhook" in py_file.name.lower() or "event" in py_file.name.lower():
                    webhook_files.append(py_file.name)
        
        if not webhook_files:
            integration_issues.append("No webhook/event handling found")
            integration_score -= 10
        
        integration_score = max(integration_score, 0.0)
        success = integration_score >= 70.0 and len(found_integrations) >= 2
        
        return {
            "success": success,
            "score": integration_score,
            "message": f"Integration validation {'passed' if success else 'failed'}",
            "details": {
                "found_integration_files": found_integrations,
                "webhook_files": webhook_files,
                "has_api_documentation": has_api_docs,
                "integration_issues": integration_issues,
                "integration_score": integration_score
            }
        }
    
    def _error_handling_validation(self) -> Dict[str, Any]:
        """Error handling and resilience validation"""
        repo_path = Path("/root/repo")
        error_score = 100.0
        error_issues = []
        
        # Check for error handling patterns
        error_patterns = [
            "try:",
            "except",
            "raise",
            "logging.error",
            "logger.error",
            "finally:",
            "ErrorHandler",
            "exception"
        ]
        
        error_handling_stats = {pattern: 0 for pattern in error_patterns}
        
        try:
            src_path = repo_path / "src"
            if src_path.exists():
                python_files = list(src_path.glob("**/*.py"))
                
                for py_file in python_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            for pattern in error_patterns:
                                count = content.lower().count(pattern.lower())
                                error_handling_stats[pattern] += count
                    except Exception:
                        continue
                
                # Evaluate error handling coverage
                total_files = len(python_files)
                if total_files > 0:
                    try_except_ratio = (error_handling_stats["try:"] + error_handling_stats["except"]) / total_files
                    logging_ratio = (error_handling_stats["logging.error"] + error_handling_stats["logger.error"]) / total_files
                    
                    if try_except_ratio < 1.0:  # Less than 1 try-except per file
                        error_issues.append("Insufficient exception handling")
                        error_score -= 20
                    
                    if logging_ratio < 0.5:  # Less than 0.5 error logs per file
                        error_issues.append("Insufficient error logging")
                        error_score -= 15
        
        except Exception as e:
            error_issues.append(f"Failed to analyze error handling: {e}")
            error_score -= 25
        
        # Check for dedicated error handling modules
        error_modules = ["error_handler.py", "exceptions.py", "resilience.py"]
        found_error_modules = []
        
        for module in error_modules:
            if (repo_path / "src" / module).exists():
                found_error_modules.append(module)
        
        if not found_error_modules:
            error_issues.append("No dedicated error handling modules found")
            error_score -= 10
        
        error_score = max(error_score, 0.0)
        success = error_score >= 70.0 and len(error_issues) <= 2
        
        return {
            "success": success,
            "score": error_score,
            "message": f"Error handling validation {'passed' if success else 'failed'}",
            "details": {
                "error_handling_patterns": error_handling_stats,
                "error_modules_found": found_error_modules,
                "error_issues": error_issues,
                "error_handling_score": error_score
            }
        }
    
    def _monitoring_validation(self) -> Dict[str, Any]:
        """Monitoring and observability validation"""
        repo_path = Path("/root/repo")
        monitoring_score = 100.0
        monitoring_issues = []
        
        # Check for monitoring-related files
        monitoring_files = [
            "monitoring",
            "prometheus",
            "grafana",
            "alerts",
            "health_check",
            "metrics"
        ]
        
        found_monitoring = []
        
        try:
            for root_item in repo_path.iterdir():
                if root_item.is_file() or root_item.is_dir():
                    name_lower = root_item.name.lower()
                    for mon_term in monitoring_files:
                        if mon_term in name_lower:
                            found_monitoring.append(root_item.name)
                            break
        
        except Exception as e:
            monitoring_issues.append(f"Failed to scan for monitoring files: {e}")
            monitoring_score -= 20
        
        # Check source code for monitoring patterns
        monitoring_patterns = [
            "prometheus",
            "metric",
            "alert",
            "health",
            "monitor",
            "telemetry",
            "observability"
        ]
        
        monitoring_code_stats = {pattern: 0 for pattern in monitoring_patterns}
        
        try:
            src_path = repo_path / "src"
            if src_path.exists():
                for py_file in src_path.glob("**/*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            for pattern in monitoring_patterns:
                                if pattern in content:
                                    monitoring_code_stats[pattern] += 1
                    except Exception:
                        continue
        
        except Exception as e:
            monitoring_issues.append(f"Failed to analyze monitoring code: {e}")
            monitoring_score -= 15
        
        # Evaluate monitoring coverage
        monitoring_coverage = sum(1 for count in monitoring_code_stats.values() if count > 0)
        
        if monitoring_coverage < 3:
            monitoring_issues.append("Limited monitoring coverage in code")
            monitoring_score -= 20
        
        if not found_monitoring:
            monitoring_issues.append("No monitoring configuration files found")
            monitoring_score -= 15
        
        # Check for health endpoints
        health_indicators = ["health", "/health", "healthz", "status"]
        has_health_check = any(
            indicator in str(monitoring_code_stats) or 
            indicator in str(found_monitoring)
            for indicator in health_indicators
        )
        
        if not has_health_check:
            monitoring_issues.append("No health check endpoints found")
            monitoring_score -= 10
        
        monitoring_score = max(monitoring_score, 0.0)
        success = monitoring_score >= 70.0 and monitoring_coverage >= 3
        
        return {
            "success": success,
            "score": monitoring_score,
            "message": f"Monitoring validation {'passed' if success else 'failed'}",
            "details": {
                "found_monitoring_files": found_monitoring,
                "monitoring_code_patterns": monitoring_code_stats,
                "monitoring_coverage": monitoring_coverage,
                "has_health_check": has_health_check,
                "monitoring_issues": monitoring_issues,
                "monitoring_score": monitoring_score
            }
        }
    
    def _scalability_validation(self) -> Dict[str, Any]:
        """Scalability and performance validation"""
        repo_path = Path("/root/repo")
        scalability_score = 100.0
        scalability_issues = []
        
        # Check for scalability patterns
        scalability_patterns = [
            "async",
            "await", 
            "asyncio",
            "concurrent",
            "threading",
            "multiprocessing",
            "pool",
            "cache",
            "scale",
            "load_balancing",
            "auto_scal"
        ]
        
        scalability_stats = {pattern: 0 for pattern in scalability_patterns}
        
        try:
            src_path = repo_path / "src"
            if src_path.exists():
                python_files = list(src_path.glob("**/*.py"))
                
                for py_file in python_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            for pattern in scalability_patterns:
                                count = content.count(pattern)
                                scalability_stats[pattern] += count
                    except Exception:
                        continue
                
                # Evaluate scalability implementation
                async_usage = scalability_stats["async"] + scalability_stats["await"] + scalability_stats["asyncio"]
                concurrency_usage = scalability_stats["concurrent"] + scalability_stats["threading"] + scalability_stats["multiprocessing"]
                caching_usage = scalability_stats["cache"] + scalability_stats["pool"]
                
                if async_usage < 5:
                    scalability_issues.append("Limited async/await usage")
                    scalability_score -= 15
                
                if concurrency_usage < 3:
                    scalability_issues.append("Limited concurrency implementation")
                    scalability_score -= 15
                
                if caching_usage < 2:
                    scalability_issues.append("Limited caching implementation")
                    scalability_score -= 10
        
        except Exception as e:
            scalability_issues.append(f"Failed to analyze scalability patterns: {e}")
            scalability_score -= 20
        
        # Check for scalability configuration
        scalability_configs = [
            "docker-compose.yml",
            "k8s/",
            "deployment.yaml",
            "scaling_config.json"
        ]
        
        found_scalability_configs = []
        for config in scalability_configs:
            config_path = repo_path / config
            if config_path.exists():
                found_scalability_configs.append(config)
        
        if not found_scalability_configs:
            scalability_issues.append("No scalability deployment configurations found")
            scalability_score -= 10
        
        # Check for performance optimization files
        perf_files = [
            "performance",
            "optimization",
            "benchmark",
            "profiling"
        ]
        
        found_perf_files = []
        try:
            for item in repo_path.glob("**/*"):
                if item.is_file():
                    name_lower = item.name.lower()
                    for perf_term in perf_files:
                        if perf_term in name_lower:
                            found_perf_files.append(str(item.relative_to(repo_path)))
                            break
        except Exception as e:
            scalability_issues.append(f"Failed to scan for performance files: {e}")
        
        scalability_score = max(scalability_score, 0.0)
        success = scalability_score >= 70.0 and len(scalability_issues) <= 3
        
        return {
            "success": success,
            "score": scalability_score,
            "message": f"Scalability validation {'passed' if success else 'failed'}",
            "details": {
                "scalability_patterns": scalability_stats,
                "found_scalability_configs": found_scalability_configs,
                "found_performance_files": found_perf_files,
                "scalability_issues": scalability_issues,
                "scalability_score": scalability_score
            }
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final quality report"""
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall statistics
        total_gates = len(self.results)
        passed_gates = len([r for r in self.results if r.status == "passed"])
        failed_gates = len([r for r in self.results if r.status == "failed"])
        warning_gates = len([r for r in self.results if r.status == "warning"])
        
        # Calculate overall score
        if total_gates > 0:
            total_score = sum(r.score for r in self.results) / total_gates
            pass_rate = passed_gates / total_gates * 100
        else:
            total_score = 0.0
            pass_rate = 0.0
        
        # Determine overall status
        if pass_rate >= 90:
            overall_status = "EXCELLENT"
        elif pass_rate >= 80:
            overall_status = "GOOD"
        elif pass_rate >= 70:
            overall_status = "ACCEPTABLE"
        elif pass_rate >= 50:
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            overall_status = "POOR"
        
        # Generate recommendations
        recommendations = []
        
        for result in self.results:
            if result.status == "failed":
                recommendations.append(f"Fix {result.gate_name}: {result.message}")
            elif result.status == "warning":
                recommendations.append(f"Review {result.gate_name}: {result.message}")
        
        # Critical issues
        critical_issues = [
            r for r in self.results 
            if r.status == "failed" and r.score < 50
        ]
        
        report = {
            "overall_status": overall_status,
            "total_score": round(total_score, 2),
            "pass_rate": round(pass_rate, 1),
            "execution_time": round(total_execution_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": {
                "total_gates": total_gates,
                "passed": passed_gates,
                "failed": failed_gates,
                "warnings": warning_gates,
                "critical_issues": len(critical_issues)
            },
            "gate_results": [asdict(result) for result in self.results],
            "recommendations": recommendations,
            "critical_issues": [
                {
                    "gate": issue.gate_name,
                    "message": issue.message,
                    "score": issue.score
                }
                for issue in critical_issues
            ],
            "summary": {
                "quality_assessment": overall_status,
                "ready_for_production": pass_rate >= 80 and len(critical_issues) == 0,
                "major_concerns": len(critical_issues),
                "improvement_areas": len([r for r in self.results if r.score < 70])
            }
        }
        
        return report


async def main():
    """Run comprehensive quality gates"""
    quality_gates = ComprehensiveQualityGates()
    
    print("🚀 TERRAGON SDLC v4.0 - COMPREHENSIVE QUALITY GATES")
    print("=" * 60)
    
    # Run all quality gates
    final_report = await quality_gates.run_all_quality_gates()
    
    # Display results
    print(f"\n📊 QUALITY ASSESSMENT: {final_report['overall_status']}")
    print(f"🎯 Overall Score: {final_report['total_score']}/100")
    print(f"✅ Pass Rate: {final_report['pass_rate']}%")
    print(f"⏱️  Execution Time: {final_report['execution_time']}s")
    
    print(f"\n📈 STATISTICS:")
    stats = final_report['statistics']
    print(f"  Total Gates: {stats['total_gates']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Warnings: {stats['warnings']}")
    print(f"  Critical Issues: {stats['critical_issues']}")
    
    # Show critical issues
    if final_report['critical_issues']:
        print(f"\n🚨 CRITICAL ISSUES:")
        for issue in final_report['critical_issues']:
            print(f"  - {issue['gate']}: {issue['message']} (Score: {issue['score']})")
    
    # Show recommendations
    if final_report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in final_report['recommendations'][:5]:  # Show top 5
            print(f"  - {rec}")
    
    # Production readiness
    summary = final_report['summary']
    print(f"\n🏭 PRODUCTION READINESS:")
    print(f"  Ready for Production: {'✅ YES' if summary['ready_for_production'] else '❌ NO'}")
    print(f"  Major Concerns: {summary['major_concerns']}")
    print(f"  Areas Needing Improvement: {summary['improvement_areas']}")
    
    # Save detailed report
    report_file = f"/root/repo/quality_gates_final_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved: {report_file}")
    
    return final_report


if __name__ == "__main__":
    asyncio.run(main())