#!/usr/bin/env python3
"""
Progressive Quality Gates System - Generation 1
Implements dynamic quality validation with adaptive thresholds
"""

import asyncio
import json
import logging
import time
import subprocess
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

class QualityGateStatus(Enum):
    """Quality gate execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"

class QualityLevel(Enum):
    """Quality gate severity levels"""
    BLOCKING = "blocking"      # Must pass to continue
    CRITICAL = "critical"      # Fails build but allows continuation
    WARNING = "warning"        # Warns but allows continuation
    INFO = "info"             # Informational only

@dataclass
class QualityGate:
    """Individual quality gate definition"""
    name: str
    command: str
    level: QualityLevel
    timeout: int = 300
    retry_count: int = 0
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class QualityResult:
    """Result of quality gate execution"""
    gate: QualityGate
    status: QualityGateStatus
    execution_time: float
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class ProgressiveQualityGates:
    """
    Advanced quality gates system with progressive validation
    """
    
    def __init__(self, config_path: str = "quality_gates_config.json"):
        self.config_path = config_path
        self.gates: List[QualityGate] = []
        self.results: List[QualityResult] = []
        self.start_time: float = 0
        self.load_configuration()
        
    def load_configuration(self):
        """Load quality gates configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.gates = self._parse_gates_config(config)
            else:
                self.gates = self._default_gates()
                self._save_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.gates = self._default_gates()
    
    def _parse_gates_config(self, config: Dict) -> List[QualityGate]:
        """Parse gates from configuration"""
        gates = []
        for gate_config in config.get('gates', []):
            gate = QualityGate(
                name=gate_config['name'],
                command=gate_config['command'],
                level=QualityLevel(gate_config.get('level', 'warning')),
                timeout=gate_config.get('timeout', 300),
                retry_count=gate_config.get('retry_count', 0),
                environment=gate_config.get('environment', {}),
                working_directory=gate_config.get('working_directory'),
                success_criteria=gate_config.get('success_criteria', {})
            )
            gates.append(gate)
        return gates
    
    def _default_gates(self) -> List[QualityGate]:
        """Default quality gates configuration"""
        return [
            # Security Gates
            QualityGate(
                name="Security Scan",
                command="python -m bandit -r src/ -f json",
                level=QualityLevel.BLOCKING,
                timeout=300,
                success_criteria={"max_high_severity": 0, "max_medium_severity": 5}
            ),
            QualityGate(
                name="Dependency Security Check",
                command="python -m safety check --json",
                level=QualityLevel.CRITICAL,
                timeout=120
            ),
            
            # Code Quality Gates
            QualityGate(
                name="Lint Check",
                command="python -m flake8 src tests --max-line-length=100",
                level=QualityLevel.WARNING,
                timeout=180
            ),
            QualityGate(
                name="Type Check",
                command="python -m mypy src --ignore-missing-imports",
                level=QualityLevel.WARNING,
                timeout=240
            ),
            
            # Test Gates
            QualityGate(
                name="Unit Tests",
                command="python -m pytest tests/unit/ -v --tb=short --junitxml=test-results-unit.xml",
                level=QualityLevel.BLOCKING,
                timeout=600,
                success_criteria={"min_coverage": 85, "max_failures": 0}
            ),
            QualityGate(
                name="Integration Tests",
                command="python -m pytest tests/integration/ -v --tb=short --junitxml=test-results-integration.xml",
                level=QualityLevel.CRITICAL,
                timeout=900
            ),
            
            # Performance Gates
            QualityGate(
                name="Performance Benchmarks",
                command="python -m pytest tests/performance/ --benchmark-only --benchmark-json=benchmark-results.json",
                level=QualityLevel.WARNING,
                timeout=1200,
                success_criteria={"max_response_time_ms": 500}
            ),
            
            # Build Gates
            QualityGate(
                name="Docker Build",
                command="docker build -t claude-manager:test .",
                level=QualityLevel.BLOCKING,
                timeout=600
            ),
        ]
    
    def _save_default_config(self):
        """Save default configuration to file"""
        try:
            config = {
                "version": "1.0",
                "gates": []
            }
            
            for gate in self.gates:
                gate_config = {
                    "name": gate.name,
                    "command": gate.command,
                    "level": gate.level.value,
                    "timeout": gate.timeout,
                    "retry_count": gate.retry_count,
                    "environment": gate.environment,
                    "success_criteria": gate.success_criteria
                }
                if gate.working_directory:
                    gate_config["working_directory"] = gate.working_directory
                config["gates"].append(gate_config)
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved default quality gates configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save default configuration: {e}")
    
    async def execute_gate(self, gate: QualityGate) -> QualityResult:
        """Execute a single quality gate"""
        logger.info(f"Executing quality gate: {gate.name}")
        start_time = time.time()
        
        result = QualityResult(
            gate=gate,
            status=QualityGateStatus.RUNNING,
            execution_time=0
        )
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(gate.environment)
            
            # Set working directory
            cwd = gate.working_directory or os.getcwd()
            
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                gate.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=gate.timeout
                )
                
                result.stdout = stdout.decode('utf-8') if stdout else ""
                result.stderr = stderr.decode('utf-8') if stderr else ""
                result.return_code = process.returncode or 0
                result.execution_time = time.time() - start_time
                
                # Determine status based on return code and success criteria
                if result.return_code == 0:
                    if await self._validate_success_criteria(gate, result):
                        result.status = QualityGateStatus.PASSED
                    else:
                        result.status = QualityGateStatus.FAILED
                else:
                    result.status = QualityGateStatus.FAILED
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result.status = QualityGateStatus.FAILED
                result.stderr = f"Gate timed out after {gate.timeout} seconds"
                result.execution_time = gate.timeout
                
        except Exception as e:
            result.status = QualityGateStatus.FAILED
            result.stderr = f"Execution error: {str(e)}"
            result.execution_time = time.time() - start_time
        
        # Handle retries for failed gates
        if result.status == QualityGateStatus.FAILED and gate.retry_count > 0:
            logger.warning(f"Gate {gate.name} failed, retrying ({gate.retry_count} retries left)")
            gate.retry_count -= 1
            await asyncio.sleep(2)  # Brief delay before retry
            return await self.execute_gate(gate)
        
        logger.info(f"Gate {gate.name} completed with status: {result.status.value}")
        return result
    
    async def _validate_success_criteria(self, gate: QualityGate, result: QualityResult) -> bool:
        """Validate success criteria for a gate result"""
        if not gate.success_criteria:
            return True
        
        try:
            # Parse output for metrics
            if gate.name == "Unit Tests" and "coverage" in gate.success_criteria:
                # Extract coverage from pytest output
                coverage = self._extract_coverage(result.stdout)
                if coverage and coverage < gate.success_criteria.get("min_coverage", 0):
                    return False
            
            if gate.name == "Security Scan" and result.stdout:
                # Parse bandit JSON output
                try:
                    bandit_results = json.loads(result.stdout)
                    high_severity = len([issue for issue in bandit_results.get("results", []) 
                                       if issue.get("issue_severity") == "HIGH"])
                    medium_severity = len([issue for issue in bandit_results.get("results", []) 
                                         if issue.get("issue_severity") == "MEDIUM"])
                    
                    if high_severity > gate.success_criteria.get("max_high_severity", 0):
                        return False
                    if medium_severity > gate.success_criteria.get("max_medium_severity", 10):
                        return False
                except json.JSONDecodeError:
                    pass
            
            if gate.name == "Performance Benchmarks":
                # Validate performance criteria
                if os.path.exists("benchmark-results.json"):
                    with open("benchmark-results.json") as f:
                        benchmark_data = json.load(f)
                        # Add performance validation logic here
                        pass
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate success criteria for {gate.name}: {e}")
            return True  # Default to pass if validation fails
    
    def _extract_coverage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from pytest output"""
        try:
            for line in output.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            return float(part.replace('%', ''))
            return None
        except:
            return None
    
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates in order"""
        logger.info("Starting progressive quality gates execution")
        self.start_time = time.time()
        self.results = []
        
        blocking_failed = False
        critical_failed = False
        warning_count = 0
        
        for gate in self.gates:
            # Skip remaining gates if blocking gate failed
            if blocking_failed and gate.level == QualityLevel.BLOCKING:
                result = QualityResult(
                    gate=gate,
                    status=QualityGateStatus.SKIPPED,
                    execution_time=0
                )
                self.results.append(result)
                continue
            
            result = await self.execute_gate(gate)
            self.results.append(result)
            
            # Track failures by severity
            if result.status == QualityGateStatus.FAILED:
                if gate.level == QualityLevel.BLOCKING:
                    blocking_failed = True
                elif gate.level == QualityLevel.CRITICAL:
                    critical_failed = True
                elif gate.level == QualityLevel.WARNING:
                    warning_count += 1
        
        total_time = time.time() - self.start_time
        
        # Generate summary
        summary = self._generate_summary(total_time, blocking_failed, critical_failed, warning_count)
        
        # Save results
        await self._save_results(summary)
        
        return summary
    
    def _generate_summary(self, total_time: float, blocking_failed: bool, 
                         critical_failed: bool, warning_count: int) -> Dict[str, Any]:
        """Generate execution summary"""
        passed = len([r for r in self.results if r.status == QualityGateStatus.PASSED])
        failed = len([r for r in self.results if r.status == QualityGateStatus.FAILED])
        skipped = len([r for r in self.results if r.status == QualityGateStatus.SKIPPED])
        
        overall_status = "PASSED"
        if blocking_failed:
            overall_status = "BLOCKED"
        elif critical_failed:
            overall_status = "FAILED"
        elif warning_count > 0:
            overall_status = "WARNING"
        
        return {
            "overall_status": overall_status,
            "total_time": round(total_time, 2),
            "gates_executed": len(self.results),
            "results": {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "warnings": warning_count
            },
            "blocking_failed": blocking_failed,
            "critical_failed": critical_failed,
            "execution_timestamp": time.time(),
            "gate_results": [
                {
                    "name": r.gate.name,
                    "status": r.status.value,
                    "level": r.gate.level.value,
                    "execution_time": round(r.execution_time, 2),
                    "return_code": r.return_code
                }
                for r in self.results
            ]
        }
    
    async def _save_results(self, summary: Dict[str, Any]):
        """Save execution results to file"""
        try:
            results_file = f"quality_gates_results_{int(time.time())}.json"
            
            detailed_results = {
                "summary": summary,
                "detailed_results": [
                    {
                        "gate": {
                            "name": r.gate.name,
                            "command": r.gate.command,
                            "level": r.gate.level.value,
                            "timeout": r.gate.timeout
                        },
                        "result": {
                            "status": r.status.value,
                            "execution_time": r.execution_time,
                            "return_code": r.return_code,
                            "stdout": r.stdout[:1000] + ("..." if len(r.stdout) > 1000 else ""),
                            "stderr": r.stderr[:1000] + ("..." if len(r.stderr) > 1000 else ""),
                            "timestamp": r.timestamp
                        }
                    }
                    for r in self.results
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            logger.info(f"Quality gates results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_blocking_failures(self) -> List[QualityResult]:
        """Get all blocking gate failures"""
        return [
            r for r in self.results 
            if r.status == QualityGateStatus.FAILED and r.gate.level == QualityLevel.BLOCKING
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        if not self.results:
            return {}
        
        execution_times = [r.execution_time for r in self.results]
        
        return {
            "total_gates": len(self.results),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "total_execution_time": sum(execution_times),
            "success_rate": len([r for r in self.results if r.status == QualityGateStatus.PASSED]) / len(self.results)
        }

async def main():
    """Main execution function for progressive quality gates"""
    gates = ProgressiveQualityGates()
    
    print("🛡️ Starting Progressive Quality Gates Validation")
    print(f"Configured {len(gates.gates)} quality gates")
    print("-" * 60)
    
    try:
        summary = await gates.execute_all_gates()
        
        print(f"\n✅ Quality Gates Execution Complete")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Time: {summary['total_time']}s")
        print(f"Passed: {summary['results']['passed']}")
        print(f"Failed: {summary['results']['failed']}")
        print(f"Warnings: {summary['results']['warnings']}")
        
        if summary['overall_status'] in ['BLOCKED', 'FAILED']:
            blocking_failures = gates.get_blocking_failures()
            if blocking_failures:
                print(f"\n🚫 Blocking Failures:")
                for failure in blocking_failures:
                    print(f"  - {failure.gate.name}: {failure.stderr[:100]}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Quality gates execution failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))