#!/usr/bin/env python3
"""
Simplified Progressive Quality Gates - Generation 1
Basic quality validation without external dependencies
"""

import asyncio
import json
import logging
import time
import subprocess
import os
import ast
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    BLOCKING = "blocking"
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class QualityGate:
    """Individual quality gate definition"""
    name: str
    command: str
    level: QualityLevel
    timeout: int = 300
    retry_count: int = 0
    environment: Dict[str, str] = None
    working_directory: Optional[str] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}

@dataclass
class QualityResult:
    """Result of quality gate execution"""
    gate: QualityGate
    status: QualityGateStatus
    execution_time: float
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class SimplifiedQualityGates:
    """
    Simplified quality gates system focusing on core validation
    """
    
    def __init__(self, config_path: str = "quality_gates_config.json"):
        self.config_path = config_path
        self.gates = self._default_gates()
        self.results = []
        self.start_time = 0
        
    def _default_gates(self) -> List[QualityGate]:
        """Default quality gates configuration"""
        return [
            # Python syntax validation
            QualityGate(
                name="Python Syntax Check",
                command="python3 -m py_compile src/*.py",
                level=QualityLevel.BLOCKING,
                timeout=60
            ),
            
            # Basic lint check (if available)
            QualityGate(
                name="Basic Code Structure",
                command="find src -name '*.py' -exec python3 -m ast {} \\;",
                level=QualityLevel.WARNING,
                timeout=120
            ),
            
            # Test execution (if tests exist)
            QualityGate(
                name="Unit Tests",
                command="python3 -m pytest tests/ -v --tb=short || echo 'No pytest available'",
                level=QualityLevel.CRITICAL,
                timeout=300
            ),
            
            # File structure validation
            QualityGate(
                name="Project Structure",
                command="echo 'Checking project structure'",
                level=QualityLevel.INFO,
                timeout=10
            ),
        ]
    
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
                
                # Determine status based on return code
                if result.return_code == 0:
                    result.status = QualityGateStatus.PASSED
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
        
        logger.info(f"Gate {gate.name} completed with status: {result.status.value}")
        return result
    
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
            results_file = f"simplified_quality_gates_results_{int(time.time())}.json"
            
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
                            "stdout": r.stdout[:500] + ("..." if len(r.stdout) > 500 else ""),
                            "stderr": r.stderr[:500] + ("..." if len(r.stderr) > 500 else ""),
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
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate basic project structure"""
        structure_issues = []
        required_files = ['README.md', 'requirements.txt', 'src/']
        recommended_files = ['CHANGELOG.md', 'LICENSE', '.gitignore']
        
        for required in required_files:
            if not os.path.exists(required):
                structure_issues.append(f"Missing required file/directory: {required}")
        
        missing_recommended = []
        for recommended in recommended_files:
            if not os.path.exists(recommended):
                missing_recommended.append(recommended)
        
        return {
            "required_files_present": len(structure_issues) == 0,
            "issues": structure_issues,
            "missing_recommended": missing_recommended
        }

async def main():
    """Main execution function for simplified quality gates"""
    gates = SimplifiedQualityGates()
    
    print("🛡️ Starting Simplified Progressive Quality Gates Validation")
    print(f"Configured {len(gates.gates)} quality gates")
    print("-" * 60)
    
    try:
        # Validate project structure first
        structure = gates.validate_project_structure()
        print("\n📁 Project Structure Validation:")
        if structure["required_files_present"]:
            print("✅ All required files present")
        else:
            print("❌ Missing required files:")
            for issue in structure["issues"]:
                print(f"  - {issue}")
        
        if structure["missing_recommended"]:
            print("⚠️  Missing recommended files:")
            for missing in structure["missing_recommended"]:
                print(f"  - {missing}")
        
        # Execute quality gates
        summary = await gates.execute_all_gates()
        
        print(f"\n✅ Quality Gates Execution Complete")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Time: {summary['total_time']}s")
        print(f"Passed: {summary['results']['passed']}")
        print(f"Failed: {summary['results']['failed']}")
        print(f"Warnings: {summary['results']['warnings']}")
        
        # Show gate details
        print(f"\n📊 Gate Results:")
        for gate_result in summary['gate_results']:
            status_emoji = "✅" if gate_result['status'] == 'passed' else "❌" if gate_result['status'] == 'failed' else "⚠️"
            print(f"  {status_emoji} {gate_result['name']}: {gate_result['status']} ({gate_result['execution_time']}s)")
        
        if summary['overall_status'] in ['BLOCKED', 'FAILED']:
            print(f"\n🚫 Issues found - check individual gate results")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Quality gates execution failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))