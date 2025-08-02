#!/usr/bin/env python3
"""
Monitoring and observability validation script.
Validates monitoring infrastructure, alerts, and dashboards.
"""

import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any


def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def validate_prometheus_config() -> bool:
    """Validate Prometheus configuration."""
    print("🔍 Validating Prometheus configuration...")
    
    prometheus_config = Path("monitoring/prometheus.yml")
    if not prometheus_config.exists():
        print("❌ Prometheus configuration not found")
        return False
    
    print("✅ Prometheus configuration file exists")
    
    try:
        with open(prometheus_config, "r") as f:
            config = yaml.safe_load(f)
        
        # Check essential sections
        essential_sections = ["global", "scrape_configs"]
        for section in essential_sections:
            if section in config:
                print(f"✅ {section} section found")
            else:
                print(f"❌ {section} section missing")
                return False
        
        # Check scrape configs
        scrape_configs = config.get("scrape_configs", [])
        print(f"✅ {len(scrape_configs)} scrape configurations found")
        
        # Check for essential targets
        essential_targets = ["claude-manager", "prometheus"]
        configured_jobs = [job.get("job_name") for job in scrape_configs]
        
        for target in essential_targets:
            if target in configured_jobs:
                print(f"✅ Essential target '{target}' configured")
            else:
                print(f"⚠️  Essential target '{target}' not found")
        
        # Check global configuration
        global_config = config.get("global", {})
        if "scrape_interval" in global_config:
            interval = global_config["scrape_interval"]
            print(f"✅ Global scrape interval: {interval}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing Prometheus config: {e}")
        return False


def validate_alert_rules() -> bool:
    """Validate Prometheus alert rules."""
    print("\n🚨 Validating alert rules...")
    
    alerts_dir = Path("monitoring/rules")
    if not alerts_dir.exists():
        print("❌ Alert rules directory not found")
        return False
    
    print("✅ Alert rules directory exists")
    
    alert_files = list(alerts_dir.glob("*.yml"))
    if not alert_files:
        print("❌ No alert rule files found")
        return False
    
    print(f"✅ {len(alert_files)} alert rule files found")
    
    total_alerts = 0
    for alert_file in alert_files:
        try:
            with open(alert_file, "r") as f:
                alerts_config = yaml.safe_load(f)
            
            groups = alerts_config.get("groups", [])
            for group in groups:
                group_name = group.get("name", "unnamed")
                rules = group.get("rules", [])
                alert_count = len([r for r in rules if "alert" in r])
                total_alerts += alert_count
                print(f"✅ Group '{group_name}': {alert_count} alerts")
                
                # Validate each alert
                for rule in rules:
                    if "alert" in rule:
                        alert_name = rule.get("alert")
                        if not rule.get("expr"):
                            print(f"❌ Alert '{alert_name}' missing expression")
                            return False
                        if not rule.get("annotations"):
                            print(f"⚠️  Alert '{alert_name}' missing annotations")
        
        except Exception as e:
            print(f"❌ Error parsing {alert_file}: {e}")
            return False
    
    print(f"✅ Total alerts configured: {total_alerts}")
    return True


def validate_grafana_dashboards() -> bool:
    """Validate Grafana dashboard configurations."""
    print("\n📊 Validating Grafana dashboards...")
    
    dashboards_dir = Path("monitoring/grafana-dashboards")
    if not dashboards_dir.exists():
        print("❌ Grafana dashboards directory not found")
        return False
    
    print("✅ Grafana dashboards directory exists")
    
    dashboard_files = list(dashboards_dir.glob("*.json"))
    if not dashboard_files:
        print("❌ No dashboard files found")
        return False
    
    print(f"✅ {len(dashboard_files)} dashboard files found")
    
    for dashboard_file in dashboard_files:
        try:
            with open(dashboard_file, "r") as f:
                dashboard_config = json.load(f)
            
            dashboard_title = dashboard_config.get("title", "Unknown")
            panels = dashboard_config.get("panels", [])
            print(f"✅ Dashboard '{dashboard_title}': {len(panels)} panels")
            
            # Check for essential dashboard elements
            if "templating" in dashboard_config:
                templates = dashboard_config["templating"].get("list", [])
                print(f"   ✅ {len(templates)} template variables")
            
            if "annotations" in dashboard_config:
                annotations = dashboard_config["annotations"].get("list", [])
                print(f"   ✅ {len(annotations)} annotations")
        
        except Exception as e:
            print(f"❌ Error parsing {dashboard_file}: {e}")
            return False
    
    return True


def validate_health_check_endpoints() -> bool:
    """Validate health check endpoint configuration."""
    print("\n🎯 Validating health check endpoints...")
    
    health_check_file = Path("src/health_check.py")
    if not health_check_file.exists():
        print("❌ Health check module not found")
        return False
    
    print("✅ Health check module exists")
    
    try:
        with open(health_check_file, "r") as f:
            content = f.read()
        
        # Check for essential health check functions
        essential_functions = ["check", "get_status", "health"]
        for func in essential_functions:
            if f"def {func}" in content:
                print(f"✅ Health check function '{func}' found")
            else:
                print(f"⚠️  Health check function '{func}' not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading health check module: {e}")
        return False


def validate_opentelemetry_config() -> bool:
    """Validate OpenTelemetry configuration."""
    print("\n🔍 Validating OpenTelemetry configuration...")
    
    otel_config = Path("observability/opentelemetry-config.yaml")
    if not otel_config.exists():
        print("❌ OpenTelemetry configuration not found")
        return False
    
    print("✅ OpenTelemetry configuration file exists")
    
    try:
        with open(otel_config, "r") as f:
            config = yaml.safe_load(f)
        
        # Check essential sections
        essential_sections = ["receivers", "processors", "exporters", "service"]
        for section in essential_sections:
            if section in config:
                print(f"✅ {section} section found")
            else:
                print(f"❌ {section} section missing")
                return False
        
        # Check service pipelines
        service_config = config.get("service", {})
        pipelines = service_config.get("pipelines", {})
        
        essential_pipelines = ["traces", "metrics", "logs"]
        for pipeline in essential_pipelines:
            if pipeline in pipelines:
                print(f"✅ {pipeline} pipeline configured")
            else:
                print(f"⚠️  {pipeline} pipeline not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing OpenTelemetry config: {e}")
        return False


def validate_docker_compose_monitoring() -> bool:
    """Validate monitoring services in docker-compose."""
    print("\n🐳 Validating Docker Compose monitoring services...")
    
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    try:
        with open(compose_file, "r") as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get("services", {})
        
        # Check for monitoring services
        monitoring_services = ["prometheus", "grafana"]
        for service in monitoring_services:
            if service in services:
                print(f"✅ Monitoring service '{service}' configured")
                
                # Check service configuration
                service_config = services[service]
                if "ports" in service_config:
                    ports = service_config["ports"]
                    print(f"   ✅ Ports exposed: {ports}")
                
                if "volumes" in service_config:
                    volumes = service_config["volumes"]
                    print(f"   ✅ {len(volumes)} volumes mounted")
            else:
                print(f"⚠️  Monitoring service '{service}' not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing docker-compose.yml: {e}")
        return False


def validate_performance_monitoring() -> bool:
    """Validate performance monitoring setup."""
    print("\n📊 Validating performance monitoring...")
    
    perf_monitor_file = Path("src/performance_monitor.py")
    if not perf_monitor_file.exists():
        print("❌ Performance monitor module not found")
        return False
    
    print("✅ Performance monitor module exists")
    
    try:
        with open(perf_monitor_file, "r") as f:
            content = f.read()
        
        # Check for essential monitoring functionality
        essential_features = [
            "class PerformanceMonitor",
            "def record_metric",
            "def get_metrics",
            "def start_timing",
            "def end_timing"
        ]
        
        for feature in essential_features:
            if feature in content:
                print(f"✅ Feature '{feature}' found")
            else:
                print(f"⚠️  Feature '{feature}' not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading performance monitor: {e}")
        return False


def check_monitoring_dependencies() -> bool:
    """Check monitoring dependencies in requirements."""
    print("\n📦 Checking monitoring dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open(requirements_file, "r") as f:
            requirements = f.read()
        
        # Check for monitoring-related packages
        monitoring_packages = [
            "prometheus",
            "opentelemetry",
            "flask",  # For health endpoints
        ]
        
        found_packages = []
        for package in monitoring_packages:
            if package.lower() in requirements.lower():
                print(f"✅ Monitoring package '{package}' found")
                found_packages.append(package)
            else:
                print(f"⚠️  Monitoring package '{package}' not found")
        
        if len(found_packages) >= 1:
            print("✅ Basic monitoring dependencies available")
            return True
        else:
            print("❌ No monitoring dependencies found")
            return False
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False


def main():
    """Main validation function."""
    print("📈 Claude Code Manager - Monitoring & Observability Validation")
    print("=" * 70)
    
    checks = [
        ("Prometheus Configuration", validate_prometheus_config),
        ("Alert Rules", validate_alert_rules),
        ("Grafana Dashboards", validate_grafana_dashboards),
        ("Health Check Endpoints", validate_health_check_endpoints),
        ("OpenTelemetry Configuration", validate_opentelemetry_config),
        ("Docker Compose Monitoring", validate_docker_compose_monitoring),
        ("Performance Monitoring", validate_performance_monitoring),
        ("Monitoring Dependencies", check_monitoring_dependencies),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL MONITORING VALIDATION CHECKS PASSED!")
        print("✅ Monitoring and observability infrastructure is properly configured")
        return 0
    elif passed >= total * 0.8:
        print("🟡 MOSTLY CONFIGURED - Minor issues found")
        print("⚠️  Review the failed checks for optimization opportunities")
        return 0
    else:
        print(f"❌ {total - passed} critical checks failed")
        print("❌ Please review and fix the failed checks above")
        return 1


if __name__ == "__main__":
    sys.exit(main())