#!/usr/bin/env python3
"""
Claude Manager Service - Production Deployment Guide and Validator

Comprehensive production deployment preparation including infrastructure validation,
environment setup, monitoring configuration, and deployment automation.
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class DeploymentCheck:
    """Individual deployment readiness check"""
    name: str
    status: str  # 'ready', 'warning', 'fail'
    message: str
    details: Dict[str, Any]
    critical: bool = False

@dataclass
class DeploymentPlan:
    """Complete deployment plan and status"""
    environment: str
    overall_status: str
    checks: List[DeploymentCheck]
    recommendations: List[str]
    deployment_steps: List[str]
    estimated_duration: str
    prerequisites: List[str]

class ProductionDeploymentValidator:
    """Production deployment readiness validator and guide"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = self._setup_logger()
        
        # Deployment environments
        self.environments = {
            'staging': {
                'name': 'Staging',
                'requirements': ['docker', 'basic_monitoring'],
                'optional': ['k8s']
            },
            'production': {
                'name': 'Production', 
                'requirements': ['docker', 'k8s', 'monitoring', 'security', 'backup'],
                'optional': ['auto_scaling', 'cdn']
            }
        }
    
    def _setup_logger(self):
        """Setup deployment logger"""
        logger = logging.getLogger("deployment-validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def validate_deployment_readiness(self, environment: str = 'production') -> DeploymentPlan:
        """Comprehensive deployment readiness validation"""
        self.logger.info(f"Validating deployment readiness for {environment}")
        
        checks = []
        
        # Infrastructure checks
        checks.extend(await self._check_infrastructure())
        
        # Configuration checks
        checks.extend(await self._check_configuration(environment))
        
        # Security checks
        checks.extend(await self._check_security_deployment())
        
        # Monitoring checks
        checks.extend(await self._check_monitoring())
        
        # Backup and recovery checks
        checks.extend(await self._check_backup_recovery())
        
        # Performance checks
        checks.extend(await self._check_performance_readiness())
        
        # Calculate overall status
        failed_checks = [c for c in checks if c.status == 'fail']
        critical_failures = [c for c in failed_checks if c.critical]
        warning_checks = [c for c in checks if c.status == 'warning']
        
        if critical_failures:
            overall_status = 'not_ready'
        elif failed_checks:
            overall_status = 'issues'
        elif warning_checks:
            overall_status = 'ready_with_warnings'
        else:
            overall_status = 'ready'
        
        # Generate recommendations and deployment steps
        recommendations = self._generate_deployment_recommendations(checks, environment)
        deployment_steps = self._generate_deployment_steps(environment)
        prerequisites = self._generate_prerequisites(environment)
        
        plan = DeploymentPlan(
            environment=environment,
            overall_status=overall_status,
            checks=checks,
            recommendations=recommendations,
            deployment_steps=deployment_steps,
            estimated_duration=self._estimate_deployment_duration(deployment_steps),
            prerequisites=prerequisites
        )
        
        self.logger.info(f"Deployment validation completed: {overall_status}")
        
        return plan
    
    async def _check_infrastructure(self) -> List[DeploymentCheck]:
        """Check infrastructure readiness"""
        checks = []
        
        # Docker check
        docker_check = await self._check_docker()
        checks.append(docker_check)
        
        # Kubernetes check
        k8s_check = await self._check_kubernetes()
        checks.append(k8s_check)
        
        # Container registry check
        registry_check = await self._check_container_registry()
        checks.append(registry_check)
        
        return checks
    
    async def _check_docker(self) -> DeploymentCheck:
        """Check Docker setup and configuration"""
        try:
            # Check if Dockerfile exists
            dockerfile = self.project_root / 'Dockerfile'
            if not dockerfile.exists():
                return DeploymentCheck(
                    name="Docker Configuration",
                    status="fail", 
                    message="Dockerfile not found",
                    details={'dockerfile_path': str(dockerfile)},
                    critical=True
                )
            
            # Validate Dockerfile content
            with open(dockerfile) as f:
                dockerfile_content = f.read()
            
            required_directives = ['FROM', 'COPY', 'RUN', 'CMD']
            missing_directives = []
            
            for directive in required_directives:
                if directive not in dockerfile_content:
                    missing_directives.append(directive)
            
            # Check for multi-stage builds (best practice)
            multi_stage = 'AS ' in dockerfile_content or 'as ' in dockerfile_content
            
            # Check for health check
            has_healthcheck = 'HEALTHCHECK' in dockerfile_content
            
            details = {
                'dockerfile_exists': True,
                'dockerfile_size': len(dockerfile_content),
                'missing_directives': missing_directives,
                'multi_stage_build': multi_stage,
                'has_healthcheck': has_healthcheck
            }
            
            if missing_directives:
                status = "fail"
                message = f"Dockerfile missing directives: {', '.join(missing_directives)}"
            elif not has_healthcheck:
                status = "warning"
                message = "Dockerfile lacks health check (recommended for production)"
            else:
                status = "ready"
                message = "Docker configuration is production-ready"
                
            return DeploymentCheck(
                name="Docker Configuration",
                status=status,
                message=message,
                details=details,
                critical=(status == "fail")
            )
            
        except Exception as e:
            return DeploymentCheck(
                name="Docker Configuration",
                status="fail",
                message=f"Docker validation failed: {str(e)}",
                details={'error': str(e)},
                critical=True
            )
    
    async def _check_kubernetes(self) -> DeploymentCheck:
        """Check Kubernetes deployment configuration"""
        k8s_dir = self.project_root / 'k8s'
        
        if not k8s_dir.exists():
            return DeploymentCheck(
                name="Kubernetes Configuration",
                status="warning",
                message="No Kubernetes configuration found",
                details={'k8s_directory': str(k8s_dir)},
                critical=False
            )
        
        # Check for essential K8s manifests
        required_manifests = {
            'deployment.yaml': 'Deployment configuration',
            'service.yaml': 'Service configuration', 
            'configmap.yaml': 'Configuration management',
            'namespace.yaml': 'Namespace definition'
        }
        
        found_manifests = {}
        missing_manifests = {}
        
        for manifest, description in required_manifests.items():
            manifest_path = k8s_dir / manifest
            if manifest_path.exists():
                found_manifests[manifest] = description
            else:
                missing_manifests[manifest] = description
        
        # Check for production-specific configurations
        production_configs = ['production-deployment.yaml', 'ingress.yaml', 'hpa.yaml']
        production_found = [cfg for cfg in production_configs if (k8s_dir / cfg).exists()]
        
        details = {
            'k8s_directory_exists': True,
            'found_manifests': found_manifests,
            'missing_manifests': missing_manifests,
            'production_configs': production_found
        }
        
        if len(missing_manifests) > 2:
            status = "fail"
            message = f"Missing critical K8s manifests: {len(missing_manifests)}"
        elif missing_manifests:
            status = "warning" 
            message = f"Some K8s manifests missing: {len(missing_manifests)}"
        else:
            status = "ready"
            message = "Kubernetes configuration is complete"
        
        return DeploymentCheck(
            name="Kubernetes Configuration",
            status=status,
            message=message,
            details=details,
            critical=(status == "fail")
        )
    
    async def _check_container_registry(self) -> DeploymentCheck:
        """Check container registry configuration"""
        # Check for registry configuration in various files
        registry_configs = [
            self.project_root / 'docker-compose.yml',
            self.project_root / 'docker-compose.prod.yml',
            self.project_root / '.github' / 'workflows',
            self.project_root / 'k8s'
        ]
        
        registry_found = False
        registry_details = {}
        
        for config_path in registry_configs:
            if config_path.exists():
                if config_path.is_file():
                    try:
                        with open(config_path) as f:
                            content = f.read()
                        
                        # Look for registry references
                        registry_patterns = [
                            'registry', 'docker.io', 'ghcr.io', 'gcr.io', 
                            'ecr', 'harbor', 'quay.io'
                        ]
                        
                        for pattern in registry_patterns:
                            if pattern in content.lower():
                                registry_found = True
                                registry_details[str(config_path)] = f"Contains {pattern} reference"
                                break
                                
                    except Exception:
                        continue
                else:
                    # Directory - check files inside
                    for file_path in config_path.glob('*.yaml'):
                        try:
                            with open(file_path) as f:
                                content = f.read()
                            
                            if 'image:' in content and ('/' in content or ':' in content):
                                registry_found = True
                                registry_details[str(file_path)] = "Contains image registry reference"
                                
                        except Exception:
                            continue
        
        if registry_found:
            status = "ready"
            message = "Container registry configuration found"
        else:
            status = "warning"
            message = "No container registry configuration detected"
        
        return DeploymentCheck(
            name="Container Registry",
            status=status,
            message=message,
            details=registry_details,
            critical=False
        )
    
    async def _check_configuration(self, environment: str) -> List[DeploymentCheck]:
        """Check environment-specific configuration"""
        checks = []
        
        # Environment-specific config files
        env_configs = [
            f'config/{environment}.json',
            f'config.{environment}.json',
            f'.env.{environment}',
            f'{environment}.env'
        ]
        
        config_found = False
        config_details = {}
        
        for config_file in env_configs:
            config_path = self.project_root / config_file
            if config_path.exists():
                config_found = True
                config_details[config_file] = 'Found'
                
                # Validate configuration structure
                if config_file.endswith('.json'):
                    try:
                        with open(config_path) as f:
                            config_data = json.load(f)
                        
                        # Check for production-specific settings
                        if environment == 'production':
                            prod_checks = {
                                'debug_disabled': not config_data.get('debug', True),
                                'log_level_appropriate': config_data.get('log_level', '').upper() in ['INFO', 'WARNING', 'ERROR'],
                                'has_monitoring_config': 'monitoring' in config_data or 'metrics' in config_data
                            }
                            config_details['production_readiness'] = prod_checks
                            
                    except json.JSONDecodeError:
                        config_details[config_file] = 'Invalid JSON'
                    except Exception as e:
                        config_details[config_file] = f'Error: {str(e)}'
        
        if config_found:
            status = "ready"
            message = f"Environment configuration found for {environment}"
        else:
            status = "fail" if environment == 'production' else "warning"
            message = f"No environment-specific configuration for {environment}"
        
        checks.append(DeploymentCheck(
            name=f"{environment.title()} Configuration",
            status=status,
            message=message,
            details=config_details,
            critical=(environment == 'production' and status == 'fail')
        ))
        
        return checks
    
    async def _check_security_deployment(self) -> List[DeploymentCheck]:
        """Check security readiness for deployment"""
        checks = []
        
        # Secrets management
        secrets_check = await self._check_secrets_management()
        checks.append(secrets_check)
        
        # TLS/SSL configuration
        tls_check = await self._check_tls_configuration()
        checks.append(tls_check)
        
        # Security policies
        security_check = await self._check_security_policies()
        checks.append(security_check)
        
        return checks
    
    async def _check_secrets_management(self) -> DeploymentCheck:
        """Check secrets management setup"""
        secrets_indicators = [
            'secrets.yaml',
            'sealed-secrets.yaml',
            '.env.example',
            'vault',
            'ENVIRONMENT_VARIABLES.md'
        ]
        
        found_secrets_mgmt = []
        
        for indicator in secrets_indicators:
            if (self.project_root / indicator).exists():
                found_secrets_mgmt.append(indicator)
        
        # Check for external secrets documentation
        env_vars_doc = self.project_root / 'ENVIRONMENT_VARIABLES.md'
        has_env_docs = env_vars_doc.exists()
        
        if found_secrets_mgmt:
            status = "ready"
            message = "Secrets management configuration found"
        elif has_env_docs:
            status = "warning"
            message = "Environment variables documented but no secrets management"
        else:
            status = "fail"
            message = "No secrets management configuration found"
        
        return DeploymentCheck(
            name="Secrets Management",
            status=status,
            message=message,
            details={
                'found_configs': found_secrets_mgmt,
                'env_vars_documented': has_env_docs
            },
            critical=(status == 'fail')
        )
    
    async def _check_tls_configuration(self) -> DeploymentCheck:
        """Check TLS/SSL configuration"""
        tls_indicators = [
            'tls.yaml',
            'ingress.yaml',
            'certificate.yaml',
            'ssl'
        ]
        
        tls_found = []
        
        # Check in k8s directory
        k8s_dir = self.project_root / 'k8s'
        if k8s_dir.exists():
            for yaml_file in k8s_dir.glob('*.yaml'):
                try:
                    with open(yaml_file) as f:
                        content = f.read()
                    
                    if any(term in content.lower() for term in ['tls:', 'ssl:', 'https:', 'cert-manager']):
                        tls_found.append(str(yaml_file.name))
                        
                except Exception:
                    continue
        
        # Check docker-compose files
        for compose_file in ['docker-compose.yml', 'docker-compose.prod.yml']:
            compose_path = self.project_root / compose_file
            if compose_path.exists():
                try:
                    with open(compose_path) as f:
                        content = f.read()
                    
                    if 'https:' in content.lower() or 'ssl' in content.lower():
                        tls_found.append(compose_file)
                        
                except Exception:
                    continue
        
        if tls_found:
            status = "ready"
            message = "TLS configuration found"
        else:
            status = "warning"
            message = "No TLS configuration found (recommended for production)"
        
        return DeploymentCheck(
            name="TLS Configuration",
            status=status,
            message=message,
            details={'tls_configs_found': tls_found},
            critical=False
        )
    
    async def _check_security_policies(self) -> DeploymentCheck:
        """Check security policies configuration"""
        security_files = [
            'SECURITY.md',
            'security-policy.yaml',
            'network-policy.yaml',
            'pod-security-policy.yaml'
        ]
        
        found_policies = []
        
        for policy_file in security_files:
            if (self.project_root / policy_file).exists():
                found_policies.append(policy_file)
        
        # Check k8s directory for security policies
        k8s_dir = self.project_root / 'k8s'
        if k8s_dir.exists():
            for yaml_file in k8s_dir.glob('*policy*.yaml'):
                found_policies.append(f"k8s/{yaml_file.name}")
        
        if len(found_policies) >= 2:
            status = "ready"
            message = "Security policies configured"
        elif found_policies:
            status = "warning"
            message = "Some security policies found"
        else:
            status = "warning"
            message = "No security policies found"
        
        return DeploymentCheck(
            name="Security Policies",
            status=status,
            message=message,
            details={'found_policies': found_policies},
            critical=False
        )
    
    async def _check_monitoring(self) -> List[DeploymentCheck]:
        """Check monitoring and observability setup"""
        checks = []
        
        # Application monitoring
        monitoring_configs = [
            'monitoring/prometheus.yml',
            'monitoring/grafana-dashboards/',
            'docker-compose.monitoring.yml'
        ]
        
        found_monitoring = []
        
        for config in monitoring_configs:
            if (self.project_root / config).exists():
                found_monitoring.append(config)
        
        # Check for health endpoints in code
        health_endpoints = False
        python_files = list(self.project_root.glob('**/*.py'))
        
        for py_file in python_files[:10]:  # Check first 10 files
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if any(term in content.lower() for term in ['/health', '/status', 'health_check', 'healthz']):
                    health_endpoints = True
                    break
                    
            except Exception:
                continue
        
        if found_monitoring and health_endpoints:
            status = "ready"
            message = "Monitoring and health checks configured"
        elif found_monitoring or health_endpoints:
            status = "warning"
            message = "Partial monitoring configuration"
        else:
            status = "fail"
            message = "No monitoring configuration found"
        
        checks.append(DeploymentCheck(
            name="Application Monitoring",
            status=status,
            message=message,
            details={
                'monitoring_configs': found_monitoring,
                'health_endpoints': health_endpoints
            },
            critical=(status == 'fail')
        ))
        
        return checks
    
    async def _check_backup_recovery(self) -> List[DeploymentCheck]:
        """Check backup and disaster recovery setup"""
        checks = []
        
        backup_indicators = [
            'backup/',
            'scripts/backup.sh',
            'disaster-recovery.md',
            'DISASTER_RECOVERY.md'
        ]
        
        found_backup_configs = []
        
        for indicator in backup_indicators:
            if (self.project_root / indicator).exists():
                found_backup_configs.append(indicator)
        
        # Check for database backup configurations
        backup_scripts = list(self.project_root.glob('**/backup*.sh'))
        backup_scripts.extend(list(self.project_root.glob('**/backup*.py')))
        
        if backup_scripts:
            found_backup_configs.extend([str(script.name) for script in backup_scripts])
        
        if found_backup_configs:
            status = "ready"
            message = "Backup and recovery configuration found"
        else:
            status = "warning"
            message = "No backup configuration found"
        
        checks.append(DeploymentCheck(
            name="Backup & Recovery",
            status=status,
            message=message,
            details={'backup_configs': found_backup_configs},
            critical=False
        ))
        
        return checks
    
    async def _check_performance_readiness(self) -> List[DeploymentCheck]:
        """Check performance optimization for production"""
        checks = []
        
        # Check for performance configurations
        perf_indicators = [
            'performance_monitor',
            'cache',
            'concurrent',
            'pool',
            'optimization'
        ]
        
        performance_features = 0
        python_files = list(self.project_root.glob('**/*.py'))
        
        for py_file in python_files[:20]:  # Check first 20 files
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for indicator in perf_indicators:
                    if indicator in content.lower():
                        performance_features += 1
                        break  # Count once per file
                        
            except Exception:
                continue
        
        # Check for load testing
        load_test_files = list(self.project_root.glob('**/load*.py'))
        load_test_files.extend(list(self.project_root.glob('**/benchmark*.py')))
        
        if performance_features >= 10 and load_test_files:
            status = "ready"
            message = "Performance optimization and testing ready"
        elif performance_features >= 5:
            status = "warning"
            message = "Some performance optimizations found"
        else:
            status = "warning"
            message = "Limited performance optimization"
        
        checks.append(DeploymentCheck(
            name="Performance Readiness",
            status=status,
            message=message,
            details={
                'performance_features': performance_features,
                'load_test_files': len(load_test_files)
            },
            critical=False
        ))
        
        return checks
    
    def _generate_deployment_recommendations(self, checks: List[DeploymentCheck], 
                                          environment: str) -> List[str]:
        """Generate deployment recommendations based on checks"""
        recommendations = []
        
        failed_checks = [c for c in checks if c.status == 'fail']
        warning_checks = [c for c in checks if c.status == 'warning']
        
        if failed_checks:
            recommendations.append(
                f"🚨 CRITICAL: Fix {len(failed_checks)} failed checks before deployment"
            )
            
            for check in failed_checks:
                if check.critical:
                    recommendations.append(f"   • {check.name}: {check.message}")
        
        if warning_checks:
            recommendations.append(
                f"⚠️ WARNINGS: Address {len(warning_checks)} warnings for optimal deployment"
            )
        
        # Environment-specific recommendations
        if environment == 'production':
            recommendations.extend([
                "🔐 Enable security monitoring and alerting",
                "📊 Set up comprehensive logging and metrics collection", 
                "🔄 Configure automated backups and test restore procedures",
                "⚡ Perform load testing before full deployment",
                "🚨 Set up incident response procedures"
            ])
        
        # Success recommendations
        if not failed_checks and len(warning_checks) <= 2:
            recommendations.append(
                "✅ Deployment readiness looks good! Consider a staged rollout strategy"
            )
        
        return recommendations
    
    def _generate_deployment_steps(self, environment: str) -> List[str]:
        """Generate deployment steps for the environment"""
        base_steps = [
            "1. Verify all deployment checks pass",
            "2. Build and test container images",
            "3. Push images to container registry",
            "4. Update configuration files for target environment"
        ]
        
        if environment == 'staging':
            steps = base_steps + [
                "5. Deploy to staging environment",
                "6. Run smoke tests and health checks",
                "7. Validate functionality and performance",
                "8. Monitor for 24 hours"
            ]
        else:  # production
            steps = base_steps + [
                "5. Deploy to staging and validate",
                "6. Create production deployment plan",
                "7. Schedule maintenance window",
                "8. Deploy with blue-green or rolling update strategy",
                "9. Run comprehensive health checks",
                "10. Monitor metrics and logs closely",
                "11. Validate all systems operational",
                "12. Update monitoring dashboards and alerts"
            ]
        
        return steps
    
    def _generate_prerequisites(self, environment: str) -> List[str]:
        """Generate deployment prerequisites"""
        base_prereqs = [
            "Container registry access configured",
            "Deployment pipeline permissions set up",
            "Environment configuration files prepared"
        ]
        
        if environment == 'production':
            prereqs = base_prereqs + [
                "Production cluster access verified",
                "Monitoring and alerting systems ready",
                "Backup and recovery procedures tested",
                "Security scanning completed",
                "Load testing results validated",
                "Incident response team notified",
                "Rollback procedures documented and tested"
            ]
        else:
            prereqs = base_prereqs + [
                "Staging environment prepared",
                "Basic monitoring configured"
            ]
        
        return prereqs
    
    def _estimate_deployment_duration(self, steps: List[str]) -> str:
        """Estimate deployment duration based on steps"""
        base_time = len(steps) * 15  # 15 minutes per step baseline
        
        # Add extra time for production complexity
        if 'production' in ' '.join(steps).lower():
            base_time += 60  # Extra hour for production
        
        hours = base_time // 60
        minutes = base_time % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Main entry point for deployment validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Claude Manager Service - Production Deployment Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Environments:
  • staging: Basic deployment with essential monitoring
  • production: Full production deployment with all safeguards

Validation Areas:
  • Infrastructure: Docker, Kubernetes, container registry
  • Configuration: Environment-specific settings
  • Security: Secrets management, TLS, policies
  • Monitoring: Application monitoring and health checks  
  • Backup: Disaster recovery and backup procedures
  • Performance: Optimization and load testing

Examples:
  python3 deployment_guide.py --env production    # Full production validation
  python3 deployment_guide.py --env staging       # Staging validation
  python3 deployment_guide.py --output plan.json # Save deployment plan
        """)
    
    parser.add_argument('--env', '--environment',
                       choices=['staging', 'production'],
                       default='production',
                       help='Target deployment environment')
    
    parser.add_argument('--output', '-o',
                       help='Output file for deployment plan (JSON format)')
    
    parser.add_argument('--detailed', '-d',
                       action='store_true',
                       help='Show detailed check information')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ProductionDeploymentValidator()
    
    try:
        print(f"🚀 Validating deployment readiness for {args.env}...")
        
        # Run deployment validation
        plan = await validator.validate_deployment_readiness(args.env)
        
        # Display results
        status_emoji = {
            'ready': '✅',
            'ready_with_warnings': '⚠️',
            'issues': '❌', 
            'not_ready': '🚨'
        }.get(plan.overall_status, '❓')
        
        print(f"\n{status_emoji} Deployment Status: {plan.overall_status.replace('_', ' ').title()}")
        print(f"🎯 Environment: {plan.environment.title()}")
        print(f"⏱️ Estimated Duration: {plan.estimated_duration}")
        
        print(f"\n📋 Deployment Checks:")
        for check in plan.checks:
            check_emoji = {'ready': '✅', 'warning': '⚠️', 'fail': '❌'}.get(check.status, '❓')
            critical_indicator = " [CRITICAL]" if check.critical else ""
            print(f"  {check_emoji} {check.name}: {check.message}{critical_indicator}")
            
            if args.detailed and check.details:
                for key, value in check.details.items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"    {key}: {value[:3]}... ({len(value)} total)")
                    else:
                        print(f"    {key}: {value}")
        
        print(f"\n📋 Prerequisites:")
        for i, prereq in enumerate(plan.prerequisites, 1):
            print(f"  {i}. {prereq}")
        
        print(f"\n🔧 Deployment Steps:")
        for step in plan.deployment_steps:
            print(f"  {step}")
        
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(plan.recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        # Save plan if requested
        if args.output:
            plan_dict = asdict(plan)
            with open(args.output, 'w') as f:
                json.dump(plan_dict, f, indent=2, default=str)
            print(f"\n📄 Deployment plan saved to {args.output}")
        
        # Exit with appropriate code
        if plan.overall_status == 'not_ready':
            print(f"\n🚨 Deployment not ready - critical issues must be resolved")
            sys.exit(1)
        elif plan.overall_status == 'issues':
            print(f"\n❌ Deployment has issues - address before proceeding")
            sys.exit(1)
        elif plan.overall_status == 'ready_with_warnings':
            print(f"\n⚠️ Deployment ready with warnings - proceed with caution")
        else:
            print(f"\n✅ Deployment ready - all checks passed!")
            
    except Exception as e:
        print(f"❌ Deployment validation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏸️ Deployment validation cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)