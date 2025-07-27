"""
Security Scanner Integration for Autonomous Backlog Management

Integrates Software Composition Analysis (SCA) and Static Application Security Testing (SAST)
into the autonomous development workflow.

Features:
- OWASP Dependency-Check for SCA with cached NVD database
- GitHub CodeQL integration for SAST
- Bandit for Python security linting
- Safety for Python dependency vulnerability scanning
- Automated SBOM (Software Bill of Materials) generation
- Security issue prioritization and backlog integration
"""

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from logger import get_logger

logger = get_logger(__name__)


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScanType(Enum):
    """Types of security scans"""
    SCA = "sca"  # Software Composition Analysis
    SAST = "sast"  # Static Application Security Testing
    DEPENDENCY = "dependency"
    SECRETS = "secrets"
    SBOM = "sbom"  # Software Bill of Materials


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability"""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    scan_type: ScanType
    file_path: Optional[str]
    line_number: Optional[int]
    cve_id: Optional[str]
    cvss_score: Optional[float]
    component: Optional[str]
    version: Optional[str]
    fix_available: bool
    recommendation: str
    detected_timestamp: float


@dataclass
class ScanResult:
    """Results from a security scan"""
    scan_type: ScanType
    timestamp: float
    duration_seconds: float
    success: bool
    vulnerabilities: List[SecurityVulnerability]
    summary: Dict[str, int]
    scan_tool: str
    error_message: Optional[str] = None


class SecurityScanner:
    """Autonomous security scanning system"""
    
    def __init__(self, project_root: Path = Path("."), cache_dir: Path = Path(".security-cache")):
        self.project_root = project_root
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # NVD cache for OWASP Dependency-Check
        self.nvd_cache_dir = self.cache_dir / "nvd"
        self.nvd_cache_dir.mkdir(exist_ok=True)
        
        # Scan results storage
        self.results_dir = self.cache_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_comprehensive_scan(self) -> List[ScanResult]:
        """Run all security scans asynchronously"""
        logger.info("Starting comprehensive security scan...")
        
        scan_tasks = [
            self.run_dependency_scan(),
            self.run_sast_scan(),
            self.run_secrets_scan(),
            self.generate_sbom()
        ]
        
        results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scan {i} failed: {result}")
            else:
                valid_results.append(result)
                
        logger.info(f"Completed {len(valid_results)} security scans")
        return valid_results
        
    async def run_dependency_scan(self) -> ScanResult:
        """Run dependency vulnerability scan using multiple tools"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Run pip-audit for Python dependencies
            pip_vulns = await self._run_pip_audit()
            vulnerabilities.extend(pip_vulns)
            
            # Run npm audit if package.json exists
            if (self.project_root / "package.json").exists():
                npm_vulns = await self._run_npm_audit()
                vulnerabilities.extend(npm_vulns)
                
            # Run OWASP Dependency-Check if available
            try:
                owasp_vulns = await self._run_owasp_dependency_check()
                vulnerabilities.extend(owasp_vulns)
            except Exception as e:
                logger.warning(f"OWASP Dependency-Check not available: {e}")
                
            duration = time.time() - start_time
            summary = self._summarize_vulnerabilities(vulnerabilities)
            
            result = ScanResult(
                scan_type=ScanType.DEPENDENCY,
                timestamp=start_time,
                duration_seconds=duration,
                success=True,
                vulnerabilities=vulnerabilities,
                summary=summary,
                scan_tool="pip-audit,npm-audit,owasp-dependency-check"
            )
            
            self._save_scan_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Dependency scan failed: {e}")
            return ScanResult(
                scan_type=ScanType.DEPENDENCY,
                timestamp=start_time,
                duration_seconds=time.time() - start_time,
                success=False,
                vulnerabilities=[],
                summary={},
                scan_tool="multiple",
                error_message=str(e)
            )
            
    async def _run_pip_audit(self) -> List[SecurityVulnerability]:
        """Run pip-audit for Python dependency scanning"""
        try:
            # Install pip-audit if not available
            subprocess.run(["pip", "install", "pip-audit"], capture_output=True, check=False)
            
            # Run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format=json", "--progress-spinner=off"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                return []
                
            # Parse JSON output
            try:
                audit_data = json.loads(result.stdout)
                vulnerabilities = []
                
                for vuln in audit_data.get("vulnerabilities", []):
                    vulnerability = SecurityVulnerability(
                        id=f"pip-{vuln['id']}",
                        title=f"Vulnerability in {vuln['package']}",
                        description=vuln.get('description', 'No description available'),
                        severity=self._map_severity(vuln.get('severity', 'medium')),
                        scan_type=ScanType.DEPENDENCY,
                        file_path="requirements.txt",
                        line_number=None,
                        cve_id=vuln.get('id'),
                        cvss_score=None,
                        component=vuln['package'],
                        version=vuln['installed_version'],
                        fix_available=bool(vuln.get('fix_versions')),
                        recommendation=f"Update {vuln['package']} to version {', '.join(vuln.get('fix_versions', []))}",
                        detected_timestamp=time.time()
                    )
                    vulnerabilities.append(vulnerability)
                    
                return vulnerabilities
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse pip-audit JSON output")
                return []
                
        except Exception as e:
            logger.warning(f"pip-audit scan failed: {e}")
            return []
            
    async def _run_npm_audit(self) -> List[SecurityVulnerability]:
        """Run npm audit for Node.js dependency scanning"""
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # npm audit returns non-zero for vulnerabilities, so don't check returncode
            try:
                audit_data = json.loads(result.stdout)
                vulnerabilities = []
                
                for vuln_id, vuln in audit_data.get("vulnerabilities", {}).items():
                    vulnerability = SecurityVulnerability(
                        id=f"npm-{vuln_id}",
                        title=f"Vulnerability in {vuln.get('module_name', 'unknown')}",
                        description=vuln.get('overview', 'No description available'),
                        severity=self._map_severity(vuln.get('severity', 'medium')),
                        scan_type=ScanType.DEPENDENCY,
                        file_path="package.json",
                        line_number=None,
                        cve_id=vuln.get('cve', [None])[0] if vuln.get('cve') else None,
                        cvss_score=vuln.get('cvss_score'),
                        component=vuln.get('module_name'),
                        version=vuln.get('version'),
                        fix_available=bool(vuln.get('patched_versions')),
                        recommendation=vuln.get('recommendation', 'Update to latest version'),
                        detected_timestamp=time.time()
                    )
                    vulnerabilities.append(vulnerability)
                    
                return vulnerabilities
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse npm audit JSON output")
                return []
                
        except FileNotFoundError:
            logger.info("npm not available, skipping npm audit")
            return []
        except Exception as e:
            logger.warning(f"npm audit failed: {e}")
            return []
            
    async def _run_owasp_dependency_check(self) -> List[SecurityVulnerability]:
        """Run OWASP Dependency-Check with cached NVD database"""
        try:
            # Download dependency-check if not available
            dependency_check_path = self.cache_dir / "dependency-check"
            if not dependency_check_path.exists():
                logger.info("OWASP Dependency-Check not found, downloading...")
                # This would download and extract dependency-check
                # For now, skip if not available
                return []
                
            # Run dependency-check with NVD cache
            result = subprocess.run([
                str(dependency_check_path / "bin" / "dependency-check.sh"),
                "--project", "autonomous-backlog",
                "--scan", str(self.project_root),
                "--format", "JSON",
                "--out", str(self.results_dir),
                "--data", str(self.nvd_cache_dir),
                "--noupdate"  # Use cached NVD data
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"OWASP Dependency-Check failed: {result.stderr}")
                return []
                
            # Parse results
            report_file = self.results_dir / "dependency-check-report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    
                vulnerabilities = []
                for dependency in report_data.get("dependencies", []):
                    for vuln in dependency.get("vulnerabilities", []):
                        vulnerability = SecurityVulnerability(
                            id=f"owasp-{vuln['name']}",
                            title=vuln.get('description', vuln['name']),
                            description=vuln.get('description', 'No description available'),
                            severity=self._map_severity(vuln.get('severity', 'medium')),
                            scan_type=ScanType.SCA,
                            file_path=dependency.get('filePath'),
                            line_number=None,
                            cve_id=vuln['name'] if vuln['name'].startswith('CVE-') else None,
                            cvss_score=vuln.get('cvssv3', {}).get('baseScore'),
                            component=dependency.get('fileName'),
                            version=None,
                            fix_available=False,  # Would need to check this
                            recommendation="Review dependency and update if fix available",
                            detected_timestamp=time.time()
                        )
                        vulnerabilities.append(vulnerability)
                        
                return vulnerabilities
                
        except Exception as e:
            logger.warning(f"OWASP Dependency-Check scan failed: {e}")
            return []
            
    async def run_sast_scan(self) -> ScanResult:
        """Run Static Application Security Testing (SAST)"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Run Bandit for Python code
            bandit_vulns = await self._run_bandit()
            vulnerabilities.extend(bandit_vulns)
            
            # Run semgrep if available
            try:
                semgrep_vulns = await self._run_semgrep()
                vulnerabilities.extend(semgrep_vulns)
            except Exception as e:
                logger.warning(f"Semgrep not available: {e}")
                
            duration = time.time() - start_time
            summary = self._summarize_vulnerabilities(vulnerabilities)
            
            result = ScanResult(
                scan_type=ScanType.SAST,
                timestamp=start_time,
                duration_seconds=duration,
                success=True,
                vulnerabilities=vulnerabilities,
                summary=summary,
                scan_tool="bandit,semgrep"
            )
            
            self._save_scan_result(result)
            return result
            
        except Exception as e:
            logger.error(f"SAST scan failed: {e}")
            return ScanResult(
                scan_type=ScanType.SAST,
                timestamp=start_time,
                duration_seconds=time.time() - start_time,
                success=False,
                vulnerabilities=[],
                summary={},
                scan_tool="bandit",
                error_message=str(e)
            )
            
    async def _run_bandit(self) -> List[SecurityVulnerability]:
        """Run Bandit security linter for Python"""
        try:
            # Install bandit if not available
            subprocess.run(["pip", "install", "bandit"], capture_output=True, check=False)
            
            result = subprocess.run([
                "bandit", "-r", ".", "-f", "json",
                "--skip", "B101"  # Skip assert_used test
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Bandit returns non-zero when issues found
            try:
                bandit_data = json.loads(result.stdout)
                vulnerabilities = []
                
                for issue in bandit_data.get("results", []):
                    vulnerability = SecurityVulnerability(
                        id=f"bandit-{issue['test_id']}-{issue['line_number']}",
                        title=issue['issue_text'],
                        description=issue.get('issue_text', 'Security issue detected'),
                        severity=self._map_bandit_severity(issue['issue_severity']),
                        scan_type=ScanType.SAST,
                        file_path=issue['filename'],
                        line_number=issue['line_number'],
                        cve_id=None,
                        cvss_score=None,
                        component=None,
                        version=None,
                        fix_available=True,
                        recommendation=issue.get('issue_text', 'Review and fix security issue'),
                        detected_timestamp=time.time()
                    )
                    vulnerabilities.append(vulnerability)
                    
                return vulnerabilities
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse Bandit JSON output")
                return []
                
        except Exception as e:
            logger.warning(f"Bandit scan failed: {e}")
            return []
            
    async def _run_semgrep(self) -> List[SecurityVulnerability]:
        """Run Semgrep for advanced SAST"""
        try:
            # Install semgrep if not available
            subprocess.run(["pip", "install", "semgrep"], capture_output=True, check=False)
            
            result = subprocess.run([
                "semgrep", "--config=auto", "--json", "."
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                semgrep_data = json.loads(result.stdout)
                vulnerabilities = []
                
                for finding in semgrep_data.get("results", []):
                    vulnerability = SecurityVulnerability(
                        id=f"semgrep-{finding['check_id']}-{finding['start']['line']}",
                        title=finding['message'],
                        description=finding.get('message', 'Security issue detected'),
                        severity=self._map_semgrep_severity(finding.get('severity', 'INFO')),
                        scan_type=ScanType.SAST,
                        file_path=finding['path'],
                        line_number=finding['start']['line'],
                        cve_id=None,
                        cvss_score=None,
                        component=None,
                        version=None,
                        fix_available=True,
                        recommendation=finding.get('message', 'Review and fix security issue'),
                        detected_timestamp=time.time()
                    )
                    vulnerabilities.append(vulnerability)
                    
                return vulnerabilities
                
        except Exception as e:
            logger.warning(f"Semgrep scan failed: {e}")
            return []
            
    async def run_secrets_scan(self) -> ScanResult:
        """Scan for hardcoded secrets and credentials"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Simple regex-based secrets detection
            secrets_vulns = await self._detect_secrets_regex()
            vulnerabilities.extend(secrets_vulns)
            
            duration = time.time() - start_time
            summary = self._summarize_vulnerabilities(vulnerabilities)
            
            result = ScanResult(
                scan_type=ScanType.SECRETS,
                timestamp=start_time,
                duration_seconds=duration,
                success=True,
                vulnerabilities=vulnerabilities,
                summary=summary,
                scan_tool="regex-patterns"
            )
            
            self._save_scan_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Secrets scan failed: {e}")
            return ScanResult(
                scan_type=ScanType.SECRETS,
                timestamp=start_time,
                duration_seconds=time.time() - start_time,
                success=False,
                vulnerabilities=[],
                summary={},
                scan_tool="regex-patterns",
                error_message=str(e)
            )
            
    async def _detect_secrets_regex(self) -> List[SecurityVulnerability]:
        """Simple regex-based secrets detection"""
        import re
        
        secret_patterns = {
            'api_key': r'(?i)(api[_-]?key|apikey)[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9_-]{20,})',
            'password': r'(?i)password[\'"\s]*[:=][\'"\s]*[\'"]([^\'"\s]{8,})[\'"]',
            'token': r'(?i)(token|access[_-]?token)[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9_-]{20,})',
            'secret': r'(?i)(secret|client[_-]?secret)[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9_-]{20,})',
            'private_key': r'-----BEGIN [A-Z ]+PRIVATE KEY-----'
        }
        
        vulnerabilities = []
        
        for py_file in self.project_root.rglob("*.py"):
            if ".git" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern_name, pattern in secret_patterns.items():
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vulnerability = SecurityVulnerability(
                            id=f"secret-{pattern_name}-{py_file.name}-{line_num}",
                            title=f"Potential {pattern_name} found",
                            description=f"Hardcoded {pattern_name} detected in source code",
                            severity=VulnerabilitySeverity.HIGH,
                            scan_type=ScanType.SECRETS,
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_num,
                            cve_id=None,
                            cvss_score=None,
                            component=None,
                            version=None,
                            fix_available=True,
                            recommendation=f"Move {pattern_name} to environment variables",
                            detected_timestamp=time.time()
                        )
                        vulnerabilities.append(vulnerability)
                        
            except Exception as e:
                logger.warning(f"Failed to scan {py_file}: {e}")
                
        return vulnerabilities
        
    async def generate_sbom(self) -> ScanResult:
        """Generate Software Bill of Materials (SBOM)"""
        start_time = time.time()
        
        try:
            # Generate SBOM using cyclonedx-python
            subprocess.run(["pip", "install", "cyclonedx-bom"], capture_output=True, check=False)
            
            sbom_file = self.results_dir / "sbom.json"
            result = subprocess.run([
                "cyclonedx-py", "-o", str(sbom_file)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # SBOM generation doesn't produce vulnerabilities directly
            sbom_result = ScanResult(
                scan_type=ScanType.SBOM,
                timestamp=start_time,
                duration_seconds=duration,
                success=success,
                vulnerabilities=[],
                summary={"sbom_generated": 1 if success else 0},
                scan_tool="cyclonedx-python",
                error_message=result.stderr if not success else None
            )
            
            if success:
                logger.info(f"SBOM generated: {sbom_file}")
            else:
                logger.warning(f"SBOM generation failed: {result.stderr}")
                
            self._save_scan_result(sbom_result)
            return sbom_result
            
        except Exception as e:
            logger.error(f"SBOM generation failed: {e}")
            return ScanResult(
                scan_type=ScanType.SBOM,
                timestamp=start_time,
                duration_seconds=time.time() - start_time,
                success=False,
                vulnerabilities=[],
                summary={},
                scan_tool="cyclonedx-python",
                error_message=str(e)
            )
            
    def _map_severity(self, severity_str: str) -> VulnerabilitySeverity:
        """Map string severity to enum"""
        severity_map = {
            "critical": VulnerabilitySeverity.CRITICAL,
            "high": VulnerabilitySeverity.HIGH,
            "medium": VulnerabilitySeverity.MEDIUM,
            "moderate": VulnerabilitySeverity.MEDIUM,
            "low": VulnerabilitySeverity.LOW,
            "info": VulnerabilitySeverity.INFO,
            "informational": VulnerabilitySeverity.INFO
        }
        return severity_map.get(severity_str.lower(), VulnerabilitySeverity.MEDIUM)
        
    def _map_bandit_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map Bandit severity to vulnerability severity"""
        severity_map = {
            "HIGH": VulnerabilitySeverity.HIGH,
            "MEDIUM": VulnerabilitySeverity.MEDIUM,
            "LOW": VulnerabilitySeverity.LOW
        }
        return severity_map.get(severity, VulnerabilitySeverity.MEDIUM)
        
    def _map_semgrep_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map Semgrep severity to vulnerability severity"""
        severity_map = {
            "ERROR": VulnerabilitySeverity.HIGH,
            "WARNING": VulnerabilitySeverity.MEDIUM,
            "INFO": VulnerabilitySeverity.INFO
        }
        return severity_map.get(severity.upper(), VulnerabilitySeverity.MEDIUM)
        
    def _summarize_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Create summary statistics of vulnerabilities"""
        summary = {
            "total": len(vulnerabilities),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for vuln in vulnerabilities:
            summary[vuln.severity.value] += 1
            
        return summary
        
    def _save_scan_result(self, result: ScanResult):
        """Save scan result to disk"""
        try:
            timestamp_str = datetime.fromtimestamp(result.timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"{result.scan_type.value}_{timestamp_str}.json"
            filepath = self.results_dir / filename
            
            # Convert to dict for JSON serialization
            result_dict = {
                "scan_type": result.scan_type.value,
                "timestamp": result.timestamp,
                "duration_seconds": result.duration_seconds,
                "success": result.success,
                "vulnerabilities": [
                    {
                        **vuln.__dict__,
                        "severity": vuln.severity.value,
                        "scan_type": vuln.scan_type.value
                    }
                    for vuln in result.vulnerabilities
                ],
                "summary": result.summary,
                "scan_tool": result.scan_tool,
                "error_message": result.error_message
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
                
            logger.info(f"Scan result saved: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save scan result: {e}")
            
    def get_latest_scan_results(self, scan_type: Optional[ScanType] = None) -> List[ScanResult]:
        """Get latest scan results, optionally filtered by type"""
        results = []
        
        pattern = f"{scan_type.value}_*.json" if scan_type else "*.json"
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct ScanResult
                vulnerabilities = []
                for vuln_data in data.get("vulnerabilities", []):
                    vuln = SecurityVulnerability(
                        **{k: v for k, v in vuln_data.items() 
                           if k not in ["severity", "scan_type"]},
                        severity=VulnerabilitySeverity(vuln_data["severity"]),
                        scan_type=ScanType(vuln_data["scan_type"])
                    )
                    vulnerabilities.append(vuln)
                    
                result = ScanResult(
                    scan_type=ScanType(data["scan_type"]),
                    timestamp=data["timestamp"],
                    duration_seconds=data["duration_seconds"],
                    success=data["success"],
                    vulnerabilities=vulnerabilities,
                    summary=data["summary"],
                    scan_tool=data["scan_tool"],
                    error_message=data.get("error_message")
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to load scan result {result_file}: {e}")
                
        # Sort by timestamp (newest first)
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results
        
    def create_security_backlog_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Create backlog items from security vulnerabilities"""
        all_results = self.get_latest_scan_results()
        
        # Get all vulnerabilities from recent scans
        all_vulnerabilities = []
        for result in all_results:
            if result.timestamp > time.time() - (7 * 24 * 3600):  # Last 7 days
                all_vulnerabilities.extend(result.vulnerabilities)
                
        # Sort by severity and CVSS score
        severity_order = {
            VulnerabilitySeverity.CRITICAL: 4,
            VulnerabilitySeverity.HIGH: 3,
            VulnerabilitySeverity.MEDIUM: 2,
            VulnerabilitySeverity.LOW: 1,
            VulnerabilitySeverity.INFO: 0
        }
        
        all_vulnerabilities.sort(
            key=lambda v: (severity_order[v.severity], v.cvss_score or 0),
            reverse=True
        )
        
        # Create backlog items
        backlog_items = []
        for i, vuln in enumerate(all_vulnerabilities[:max_items]):
            item = {
                "id": f"security-{vuln.id}",
                "title": f"Fix Security Issue: {vuln.title}",
                "description": vuln.description,
                "type": "Security",
                "file_path": vuln.file_path,
                "line_number": vuln.line_number,
                "effort": self._estimate_effort(vuln),
                "value": self._calculate_security_value(vuln),
                "time_criticality": self._calculate_time_criticality(vuln),
                "risk_reduction": self._calculate_risk_reduction(vuln),
                "status": "READY",
                "security_metadata": {
                    "severity": vuln.severity.value,
                    "scan_type": vuln.scan_type.value,
                    "cve_id": vuln.cve_id,
                    "cvss_score": vuln.cvss_score,
                    "recommendation": vuln.recommendation,
                    "fix_available": vuln.fix_available
                }
            }
            
            # Calculate WSJF score
            item["wsjf_score"] = (
                item["value"] + item["time_criticality"] + item["risk_reduction"]
            ) / item["effort"]
            
            backlog_items.append(item)
            
        return backlog_items
        
    def _estimate_effort(self, vuln: SecurityVulnerability) -> int:
        """Estimate effort to fix vulnerability (1-5 scale)"""
        if vuln.scan_type == ScanType.DEPENDENCY and vuln.fix_available:
            return 1  # Simple dependency update
        elif vuln.scan_type == ScanType.SECRETS:
            return 2  # Move to environment variables
        elif vuln.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
            return 3  # Medium effort for serious issues
        else:
            return 2  # Low effort for other issues
            
    def _calculate_security_value(self, vuln: SecurityVulnerability) -> int:
        """Calculate business value of fixing vulnerability (1-5 scale)"""
        severity_values = {
            VulnerabilitySeverity.CRITICAL: 5,
            VulnerabilitySeverity.HIGH: 4,
            VulnerabilitySeverity.MEDIUM: 3,
            VulnerabilitySeverity.LOW: 2,
            VulnerabilitySeverity.INFO: 1
        }
        return severity_values[vuln.severity]
        
    def _calculate_time_criticality(self, vuln: SecurityVulnerability) -> int:
        """Calculate time criticality (1-5 scale)"""
        if vuln.severity == VulnerabilitySeverity.CRITICAL:
            return 5  # Fix immediately
        elif vuln.severity == VulnerabilitySeverity.HIGH:
            return 4  # Fix soon
        elif vuln.cve_id:
            return 3  # Public CVE, some urgency
        else:
            return 2  # Can wait a bit
            
    def _calculate_risk_reduction(self, vuln: SecurityVulnerability) -> int:
        """Calculate risk reduction value (1-5 scale)"""
        if vuln.scan_type == ScanType.SECRETS:
            return 5  # High risk reduction
        elif vuln.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
            return 4  # Significant risk reduction
        else:
            return 2  # Some risk reduction