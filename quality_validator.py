#!/usr/bin/env python3
"""
Claude Manager Service - Quality Gates Validator

Comprehensive quality validation including tests, security scans, performance benchmarks,
and production readiness checks.
"""

import asyncio
import json
import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

@dataclass
class QualityGateResult:
    """Quality gate validation result"""
    gate_name: str
    status: str  # 'pass', 'fail', 'warning'
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    duration: float
    timestamp: str
    
@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    status: str
    gates: List[QualityGateResult]
    recommendations: List[str]
    timestamp: str
    metadata: Dict[str, Any]

class QualityGateValidator:
    """Comprehensive quality gate validation system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = self._setup_logger()
        
        # Quality gate configuration
        self.quality_gates = {
            'code_structure': self._validate_code_structure,
            'functionality': self._validate_functionality,
            'security': self._validate_security,
            'performance': self._validate_performance,
            'documentation': self._validate_documentation,
            'deployment_readiness': self._validate_deployment_readiness
        }
        
        # Quality thresholds
        self.thresholds = {
            'minimum_score': 85.0,
            'critical_gates': ['functionality', 'security'],
            'warning_threshold': 70.0
        }
    
    def _setup_logger(self):
        """Setup logger for quality validation"""
        logger = logging.getLogger("quality-validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def run_quality_gates(self) -> QualityReport:
        """Run all quality gates and generate comprehensive report"""
        self.logger.info("Starting comprehensive quality gate validation")
        
        gate_results = []
        total_score = 0
        critical_failures = []
        
        for gate_name, gate_func in self.quality_gates.items():
            start_time = time.time()
            
            try:
                self.logger.info(f"Running quality gate: {gate_name}")
                result = await gate_func()
                
                duration = time.time() - start_time
                
                gate_result = QualityGateResult(
                    gate_name=gate_name,
                    status=result['status'],
                    score=result['score'],
                    message=result['message'],
                    details=result.get('details', {}),
                    duration=duration,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                
                gate_results.append(gate_result)
                total_score += result['score']
                
                if gate_name in self.thresholds['critical_gates'] and result['status'] == 'fail':
                    critical_failures.append(gate_name)
                
                self.logger.info(
                    f"Quality gate {gate_name} completed: {result['status']} "
                    f"(Score: {result['score']:.1f}/100)"
                )
                
            except Exception as e:
                duration = time.time() - start_time
                
                gate_result = QualityGateResult(
                    gate_name=gate_name,
                    status='fail',
                    score=0.0,
                    message=f"Gate execution failed: {str(e)}",
                    details={'error': str(e)},
                    duration=duration,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                
                gate_results.append(gate_result)
                critical_failures.append(gate_name)
                
                self.logger.error(f"Quality gate {gate_name} failed: {e}")
        
        # Calculate overall score and status
        average_score = total_score / len(self.quality_gates) if self.quality_gates else 0
        
        if critical_failures:
            overall_status = 'fail'
        elif average_score >= self.thresholds['minimum_score']:
            overall_status = 'pass'
        elif average_score >= self.thresholds['warning_threshold']:
            overall_status = 'warning'
        else:
            overall_status = 'fail'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results, average_score)
        
        report = QualityReport(
            overall_score=average_score,
            status=overall_status,
            gates=gate_results,
            recommendations=recommendations,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            metadata={
                'project_root': str(self.project_root),
                'critical_failures': critical_failures,
                'total_gates': len(self.quality_gates),
                'passed_gates': len([g for g in gate_results if g.status == 'pass']),
                'failed_gates': len([g for g in gate_results if g.status == 'fail'])
            }
        )
        
        self.logger.info(
            f"Quality validation completed: {overall_status} "
            f"(Overall score: {average_score:.1f}/100)"
        )
        
        return report
    
    async def _validate_code_structure(self) -> Dict[str, Any]:
        """Validate project code structure and organization"""
        score = 0
        details = {}
        issues = []
        
        # Check for essential files
        essential_files = [
            'README.md',
            'requirements.txt', 
            'config.json',
            '.gitignore'
        ]
        
        missing_files = []
        for file in essential_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
            else:
                score += 15  # 15 points per essential file
        
        if missing_files:
            issues.append(f"Missing essential files: {', '.join(missing_files)}")
        
        details['missing_files'] = missing_files
        
        # Check source code organization
        src_dir = self.project_root / 'src'
        if src_dir.exists():
            score += 20
            python_files = list(src_dir.glob('**/*.py'))
            details['python_files_count'] = len(python_files)
            
            # Check for __init__.py files
            init_files = list(src_dir.glob('**/__init__.py'))
            if init_files:
                score += 10
            else:
                issues.append("Missing __init__.py files in src directory")
                
            details['init_files_count'] = len(init_files)
        else:
            issues.append("No src directory found")
        
        # Check for tests directory
        tests_dir = self.project_root / 'tests'
        if tests_dir.exists():
            score += 20
            test_files = list(tests_dir.glob('**/*.py'))
            details['test_files_count'] = len(test_files)
        else:
            issues.append("No tests directory found")
            details['test_files_count'] = 0
        
        # Check for documentation
        docs_dir = self.project_root / 'docs'
        if docs_dir.exists():
            score += 15
            doc_files = list(docs_dir.glob('**/*.md'))
            details['doc_files_count'] = len(doc_files)
        else:
            details['doc_files_count'] = 0
        
        details['issues'] = issues
        
        # Cap score at 100
        score = min(score, 100)
        
        status = 'pass' if score >= 80 else ('warning' if score >= 60 else 'fail')
        message = f"Code structure score: {score}/100"
        if issues:
            message += f" - Issues: {len(issues)}"
        
        return {
            'status': status,
            'score': score,
            'message': message,
            'details': details
        }
    
    async def _validate_functionality(self) -> Dict[str, Any]:
        """Validate core functionality through execution tests"""
        score = 0
        details = {}
        issues = []
        
        # Test basic CLI functionality
        cli_tests = [
            ('simple_main.py', ['status']),
            ('robust_main.py', ['status']),
            ('scalable_main.py', ['status'])
        ]
        
        successful_tests = 0
        for script, args in cli_tests:
            script_path = self.project_root / script
            if script_path.exists():
                try:
                    cmd = [sys.executable, str(script_path)] + args
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=30,
                        cwd=self.project_root
                    )
                    
                    if result.returncode == 0:
                        successful_tests += 1
                        score += 30  # 30 points per working CLI
                    else:
                        issues.append(f"{script} failed: {result.stderr[:100]}")
                        
                except subprocess.TimeoutExpired:
                    issues.append(f"{script} timed out")
                except Exception as e:
                    issues.append(f"{script} error: {str(e)}")
            else:
                issues.append(f"Missing script: {script}")
        
        details['cli_tests'] = {
            'total': len(cli_tests),
            'successful': successful_tests,
            'failed': len(cli_tests) - successful_tests
        }
        
        # Test configuration loading
        config_file = self.project_root / 'config.json'
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                # Validate config structure
                required_sections = ['github', 'analyzer', 'executor']
                if all(section in config for section in required_sections):
                    score += 10
                else:
                    issues.append("Invalid configuration structure")
                    
            except json.JSONDecodeError:
                issues.append("Invalid JSON in configuration file")
            except Exception as e:
                issues.append(f"Configuration error: {str(e)}")
        
        details['issues'] = issues
        
        # Cap score at 100
        score = min(score, 100)
        
        status = 'pass' if score >= 70 else ('warning' if score >= 50 else 'fail')
        message = f"Functionality score: {score}/100"
        if issues:
            message += f" - Issues: {len(issues)}"
        
        return {
            'status': status,
            'score': score,
            'message': message,
            'details': details
        }
    
    async def _validate_security(self) -> Dict[str, Any]:
        """Validate security measures and practices"""
        score = 0
        details = {}
        issues = []
        warnings = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'(?i)(password|passwd|pwd)\s*=\s*["\'][a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{8,}["\']',
            r'(?i)(api[_-]?key|apikey)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'(?i)(secret|auth[_-]?token)\s*=\s*["\'][a-zA-Z0-9]{16,}["\']',
            r'(?i)github[_-]?token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']'
        ]
        
        python_files = list(self.project_root.glob('**/*.py'))
        security_issues = []
        
        for file_path in python_files:
            # Skip test files and documentation - they may contain example secrets
            if any(part in str(file_path).lower() for part in ['test', 'tests', 'docs', 'example']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    import re
                    if re.search(pattern, content):
                        security_issues.append(f"Potential hardcoded secret in {file_path.name}")
                        
            except Exception:
                continue  # Skip files that can't be read
        
        if not security_issues:
            score += 40  # 40 points for no hardcoded secrets
        else:
            issues.extend(security_issues)
        
        # Check file permissions (basic check)
        config_file = self.project_root / 'config.json'
        if config_file.exists():
            try:
                stat = config_file.stat()
                # Check if file is world-readable (basic security check)
                if stat.st_mode & 0o044:
                    warnings.append("Configuration file is world-readable")
                else:
                    score += 20
            except Exception:
                pass
        
        # Check for .env file handling
        env_file = self.project_root / '.env'
        gitignore_file = self.project_root / '.gitignore'
        
        if env_file.exists():
            warnings.append(".env file found - ensure it's not committed")
            
        if gitignore_file.exists():
            try:
                with open(gitignore_file) as f:
                    gitignore_content = f.read()
                
                if '.env' in gitignore_content or '*.env' in gitignore_content:
                    score += 20  # Good practice
                else:
                    warnings.append(".env patterns not found in .gitignore")
                    
            except Exception:
                pass
        
        # Check for security documentation
        security_files = [
            'SECURITY.md',
            'docs/SECURITY.md',
            'security.md'
        ]
        
        security_doc_found = any((self.project_root / f).exists() for f in security_files)
        if security_doc_found:
            score += 20
        else:
            warnings.append("No security documentation found")
        
        details['security_issues'] = security_issues
        details['warnings'] = warnings
        details['python_files_scanned'] = len(python_files)
        
        # Cap score at 100
        score = min(score, 100)
        
        # Security is critical - higher threshold
        status = 'pass' if score >= 80 else ('warning' if score >= 60 else 'fail')
        message = f"Security score: {score}/100"
        if security_issues:
            message += f" - Critical issues: {len(security_issues)}"
        if warnings:
            message += f" - Warnings: {len(warnings)}"
        
        return {
            'status': status,
            'score': score,
            'message': message,
            'details': details
        }
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics through benchmarks"""
        score = 0
        details = {}
        issues = []
        
        # Performance benchmark tests
        benchmark_results = {}
        
        # Test scalable implementation performance
        scalable_script = self.project_root / 'scalable_main.py'
        if scalable_script.exists():
            try:
                # Run benchmark
                start_time = time.time()
                result = subprocess.run(
                    [sys.executable, str(scalable_script), 'benchmark'],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.project_root
                )
                
                benchmark_duration = time.time() - start_time
                
                if result.returncode == 0:
                    score += 40  # 40 points for successful benchmark
                    
                    # Parse performance score from output
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if 'Performance Score:' in line:
                            try:
                                perf_score_str = line.split(':')[1].strip()
                                perf_score = float(perf_score_str.split('s')[0])
                                
                                # Score based on performance (lower is better)
                                if perf_score < 0.05:  # Under 50ms per operation
                                    score += 30
                                elif perf_score < 0.1:  # Under 100ms per operation  
                                    score += 20
                                else:
                                    score += 10
                                    
                                benchmark_results['performance_score'] = perf_score
                                break
                            except (ValueError, IndexError):
                                pass
                    
                    benchmark_results['duration'] = benchmark_duration
                    benchmark_results['output'] = result.stdout[:500]  # First 500 chars
                else:
                    issues.append(f"Benchmark failed: {result.stderr[:100]}")
                    
            except subprocess.TimeoutExpired:
                issues.append("Benchmark timed out (>60s)")
            except Exception as e:
                issues.append(f"Benchmark error: {str(e)}")
        else:
            issues.append("Scalable implementation not found")
        
        # Check for performance monitoring features
        monitoring_features = 0
        perf_indicators = [
            'performance_monitor',
            'MetricsCollector', 
            'IntelligentCache',
            'ConcurrentProcessor',
            'AutoScaler'
        ]
        
        python_files = list(self.project_root.glob('**/*.py'))
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for indicator in perf_indicators:
                    if indicator in content:
                        monitoring_features += 1
                        break  # Count once per file
                        
            except Exception:
                continue
        
        if monitoring_features >= 3:
            score += 20
        elif monitoring_features >= 1:
            score += 10
        
        details['benchmark_results'] = benchmark_results
        details['monitoring_features_found'] = monitoring_features
        details['issues'] = issues
        
        # Cap score at 100
        score = min(score, 100)
        
        status = 'pass' if score >= 70 else ('warning' if score >= 50 else 'fail')
        message = f"Performance score: {score}/100"
        if issues:
            message += f" - Issues: {len(issues)}"
        
        return {
            'status': status,
            'score': score,
            'message': message,
            'details': details
        }
    
    async def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation quality and completeness"""
        score = 0
        details = {}
        issues = []
        
        # Check README.md
        readme_file = self.project_root / 'README.md'
        if readme_file.exists():
            try:
                with open(readme_file) as f:
                    readme_content = f.read()
                
                # Check README sections
                required_sections = [
                    'description',
                    'installation', 
                    'usage',
                    'configuration'
                ]
                
                found_sections = []
                for section in required_sections:
                    if section.lower() in readme_content.lower():
                        found_sections.append(section)
                
                score += len(found_sections) * 15  # 15 points per section
                
                details['readme_length'] = len(readme_content)
                details['readme_sections'] = found_sections
                
                if len(readme_content) < 500:
                    issues.append("README is too brief (< 500 characters)")
                    
            except Exception as e:
                issues.append(f"Error reading README: {str(e)}")
        else:
            issues.append("README.md not found")
        
        # Check for additional documentation
        doc_files = list(self.project_root.glob('**/*.md'))
        doc_files.extend(list(self.project_root.glob('docs/**/*')))
        
        details['total_doc_files'] = len(doc_files)
        
        if len(doc_files) >= 5:
            score += 20
        elif len(doc_files) >= 3:
            score += 10
        
        # Check for code comments
        python_files = list(self.project_root.glob('**/*.py'))
        total_lines = 0
        comment_lines = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line in lines:
                    total_lines += 1
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        comment_lines += 1
                        
            except Exception:
                continue
        
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            details['comment_ratio'] = comment_ratio
            
            if comment_ratio >= 0.2:  # 20% comments
                score += 20
            elif comment_ratio >= 0.1:  # 10% comments
                score += 10
            else:
                issues.append(f"Low comment ratio: {comment_ratio:.1%}")
        
        details['issues'] = issues
        details['total_code_lines'] = total_lines
        details['comment_lines'] = comment_lines
        
        # Cap score at 100
        score = min(score, 100)
        
        status = 'pass' if score >= 70 else ('warning' if score >= 50 else 'fail')
        message = f"Documentation score: {score}/100"
        if issues:
            message += f" - Issues: {len(issues)}"
        
        return {
            'status': status,
            'score': score,
            'message': message,
            'details': details
        }
    
    async def _validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate production deployment readiness"""
        score = 0
        details = {}
        issues = []
        
        # Check for deployment files
        deployment_files = {
            'Dockerfile': 20,
            'docker-compose.yml': 15,
            'requirements.txt': 15,
            'Makefile': 10,
            'k8s/deployment.yaml': 15,
            '.github/workflows': 10
        }
        
        found_files = []
        for file_path, points in deployment_files.items():
            if (self.project_root / file_path).exists():
                found_files.append(file_path)
                score += points
        
        details['deployment_files_found'] = found_files
        
        # Check configuration management
        config_files = ['config.json', 'config/production.json']
        config_found = any((self.project_root / f).exists() for f in config_files)
        
        if config_found:
            score += 15
        else:
            issues.append("No configuration files found")
        
        # Check for environment variable documentation
        env_docs = ['ENVIRONMENT_VARIABLES.md', '.env.example']
        env_doc_found = any((self.project_root / f).exists() for f in env_docs)
        
        if env_doc_found:
            score += 10
        else:
            issues.append("No environment variable documentation")
        
        # Check for health check endpoints
        health_check_indicators = ['health', 'healthz', '/status', 'health_check']
        health_check_found = False
        
        python_files = list(self.project_root.glob('**/*.py'))
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for indicator in health_check_indicators:
                    if indicator in content.lower():
                        health_check_found = True
                        break
                        
                if health_check_found:
                    break
                    
            except Exception:
                continue
        
        if health_check_found:
            score += 15
        else:
            issues.append("No health check endpoints found")
        
        details['health_check_found'] = health_check_found
        details['issues'] = issues
        
        # Cap score at 100
        score = min(score, 100)
        
        status = 'pass' if score >= 75 else ('warning' if score >= 60 else 'fail')
        message = f"Deployment readiness score: {score}/100"
        if issues:
            message += f" - Issues: {len(issues)}"
        
        return {
            'status': status,
            'score': score,
            'message': message,
            'details': details
        }
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult], 
                                overall_score: float) -> List[str]:
        """Generate actionable recommendations based on gate results"""
        recommendations = []
        
        # Overall score recommendations
        if overall_score < 70:
            recommendations.append(
                "🚨 CRITICAL: Overall quality score is below acceptable threshold. "
                "Address failing quality gates immediately."
            )
        elif overall_score < 85:
            recommendations.append(
                "⚠️ WARNING: Overall quality score needs improvement. "
                "Focus on low-scoring quality gates."
            )
        
        # Gate-specific recommendations
        for gate in gate_results:
            if gate.status == 'fail':
                if gate.gate_name == 'functionality':
                    recommendations.append(
                        "🔧 FUNCTIONALITY: Fix core functionality issues before deployment. "
                        "Ensure all CLI commands work correctly."
                    )
                elif gate.gate_name == 'security':
                    recommendations.append(
                        "🔒 SECURITY: Address security vulnerabilities immediately. "
                        "Remove hardcoded secrets and improve file permissions."
                    )
                elif gate.gate_name == 'performance':
                    recommendations.append(
                        "⚡ PERFORMANCE: Optimize performance bottlenecks. "
                        "Consider implementing caching and concurrent processing."
                    )
                elif gate.gate_name == 'documentation':
                    recommendations.append(
                        "📝 DOCUMENTATION: Improve documentation quality. "
                        "Add comprehensive README and code comments."
                    )
                elif gate.gate_name == 'deployment_readiness':
                    recommendations.append(
                        "🚀 DEPLOYMENT: Prepare deployment infrastructure. "
                        "Add Dockerfile, health checks, and configuration management."
                    )
            
            elif gate.status == 'warning' and gate.score < 80:
                recommendations.append(
                    f"💡 IMPROVEMENT: {gate.gate_name.title()} can be enhanced. "
                    f"Current score: {gate.score:.1f}/100"
                )
        
        # Success recommendations
        if overall_score >= 90:
            recommendations.append(
                "🎉 EXCELLENT: High quality score achieved! "
                "Consider documenting best practices for future projects."
            )
        elif overall_score >= 85:
            recommendations.append(
                "✅ GOOD: Quality standards met. "
                "Focus on minor improvements for excellence."
            )
        
        return recommendations

# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Main entry point for quality validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Claude Manager Service - Quality Gates Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Gates:
  • Code Structure: Project organization and file structure
  • Functionality: Core feature validation through testing
  • Security: Security best practices and vulnerability scanning  
  • Performance: Performance benchmarks and optimization
  • Documentation: Documentation quality and completeness
  • Deployment: Production deployment readiness

Examples:
  python3 quality_validator.py --detailed       # Detailed quality report
  python3 quality_validator.py --output report.json  # Save report to file
  python3 quality_validator.py --gate security  # Run specific gate only
        """)
    
    parser.add_argument('--output', '-o',
                       help='Output file for quality report (JSON format)')
    
    parser.add_argument('--detailed', '-d',
                       action='store_true',
                       help='Show detailed gate information')
    
    parser.add_argument('--gate', '-g',
                       help='Run specific quality gate only')
    
    parser.add_argument('--threshold', '-t',
                       type=float,
                       default=85.0,
                       help='Minimum quality score threshold')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = QualityGateValidator()
    if args.threshold != 85.0:
        validator.thresholds['minimum_score'] = args.threshold
    
    try:
        # Run quality gates
        if args.gate:
            # Run specific gate
            if args.gate not in validator.quality_gates:
                print(f"❌ Unknown quality gate: {args.gate}")
                print(f"Available gates: {', '.join(validator.quality_gates.keys())}")
                sys.exit(1)
            
            print(f"🔍 Running quality gate: {args.gate}")
            gate_func = validator.quality_gates[args.gate]
            result = await gate_func()
            
            status_emoji = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}.get(result['status'], '❓')
            print(f"{status_emoji} {args.gate.title()}: {result['message']}")
            
            if args.detailed:
                print(f"\nDetails:")
                for key, value in result.get('details', {}).items():
                    print(f"  {key}: {value}")
        else:
            # Run all quality gates
            print("🛡️ Running comprehensive quality validation...")
            report = await validator.run_quality_gates()
            
            # Display results
            status_emoji = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}.get(report.status, '❓')
            print(f"\n{status_emoji} Overall Quality: {report.status.upper()}")
            print(f"📊 Score: {report.overall_score:.1f}/100")
            
            print(f"\n📋 Quality Gate Results:")
            for gate in report.gates:
                gate_emoji = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}.get(gate.status, '❓')
                print(f"  {gate_emoji} {gate.gate_name.replace('_', ' ').title()}: {gate.score:.1f}/100")
                
                if args.detailed:
                    print(f"    Message: {gate.message}")
                    print(f"    Duration: {gate.duration:.2f}s")
                    if gate.details:
                        for key, value in gate.details.items():
                            if isinstance(value, list) and len(value) > 3:
                                print(f"    {key}: {value[:3]}... ({len(value)} total)")
                            else:
                                print(f"    {key}: {value}")
                    print()
            
            print(f"\n💡 Recommendations:")
            for i, recommendation in enumerate(report.recommendations, 1):
                print(f"  {i}. {recommendation}")
            
            # Save report if requested
            if args.output:
                report_dict = asdict(report)
                with open(args.output, 'w') as f:
                    json.dump(report_dict, f, indent=2, default=str)
                print(f"\n📄 Quality report saved to {args.output}")
            
            # Exit with appropriate code
            if report.status == 'fail':
                print(f"\n❌ Quality validation failed (Score: {report.overall_score:.1f} < {args.threshold})")
                sys.exit(1)
            elif report.status == 'warning':
                print(f"\n⚠️ Quality validation passed with warnings")
            else:
                print(f"\n✅ Quality validation passed successfully!")
                
    except Exception as e:
        print(f"❌ Quality validation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏸️ Quality validation cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)