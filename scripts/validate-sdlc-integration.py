#!/usr/bin/env python3
"""
SDLC Integration validation script.
Validates the complete SDLC implementation across all checkpoints.
"""

import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any


def validate_checkpoint_1_foundation() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 1: Project Foundation & Documentation."""
    issues = []
    
    required_files = [
        "ARCHITECTURE.md",
        "README.md",
        "LICENSE",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "PROJECT_CHARTER.md",
    ]
    
    required_dirs = [
        "docs/adr",
        "docs/guides",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing foundation file: {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            issues.append(f"Missing foundation directory: {dir_path}")
    
    # Check ADR structure
    adr_dir = Path("docs/adr")
    if adr_dir.exists():
        adr_files = list(adr_dir.glob("ADR-*.md"))
        if len(adr_files) < 1:
            issues.append("No Architecture Decision Records found")
    
    return len(issues) == 0, issues


def validate_checkpoint_2_development() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 2: Development Environment & Tooling."""
    issues = []
    
    required_files = [
        ".devcontainer/devcontainer.json",
        ".env.example",
        ".editorconfig",
        ".pre-commit-config.yaml",
        ".pylintrc",
        "mypy.ini",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing development file: {file_path}")
    
    # Check VSCode settings (might be in .gitignore)
    if Path(".vscode").exists() and not Path(".vscode/settings.json").exists():
        issues.append("VSCode directory exists but settings.json is missing")
    
    # Validate devcontainer configuration
    devcontainer_file = Path(".devcontainer/devcontainer.json")
    if devcontainer_file.exists():
        try:
            with open(devcontainer_file) as f:
                devcontainer_config = json.load(f)
            
            if "name" not in devcontainer_config:
                issues.append("Devcontainer missing name")
            if "features" not in devcontainer_config:
                issues.append("Devcontainer missing features")
        except Exception as e:
            issues.append(f"Invalid devcontainer.json: {e}")
    
    return len(issues) == 0, issues


def validate_checkpoint_3_testing() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 3: Testing Infrastructure."""
    issues = []
    
    required_files = [
        "pytest.ini",
        "tests/conftest.py",
        "requirements-dev.txt",
    ]
    
    required_dirs = [
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "tests/fixtures",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing testing file: {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            issues.append(f"Missing testing directory: {dir_path}")
    
    # Check test files exist
    test_dirs = ["tests/unit", "tests/integration", "tests/e2e"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            test_files = list(Path(test_dir).glob("test_*.py"))
            if len(test_files) == 0:
                issues.append(f"No test files found in {test_dir}")
    
    # Validate pytest configuration
    pytest_file = Path("pytest.ini")
    if pytest_file.exists():
        try:
            with open(pytest_file) as f:
                content = f.read()
            
            if "testpaths" not in content:
                issues.append("pytest.ini missing testpaths configuration")
            if "--cov" not in content:
                issues.append("pytest.ini missing coverage configuration")
        except Exception as e:
            issues.append(f"Error reading pytest.ini: {e}")
    
    return len(issues) == 0, issues


def validate_checkpoint_4_build() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 4: Build & Containerization."""
    issues = []
    
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "Makefile",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing build file: {file_path}")
    
    # Validate Dockerfile
    dockerfile = Path("Dockerfile")
    if dockerfile.exists():
        try:
            with open(dockerfile) as f:
                content = f.read()
            
            if "FROM" not in content:
                issues.append("Dockerfile missing FROM instruction")
            if "USER" not in content:
                issues.append("Dockerfile missing USER instruction (security risk)")
            if "HEALTHCHECK" not in content:
                issues.append("Dockerfile missing HEALTHCHECK instruction")
            
            # Check for multi-stage build
            from_count = content.count("FROM")
            if from_count < 2:
                issues.append("Dockerfile not using multi-stage build")
        except Exception as e:
            issues.append(f"Error reading Dockerfile: {e}")
    
    # Validate docker-compose.yml
    compose_file = Path("docker-compose.yml")
    if compose_file.exists():
        try:
            with open(compose_file) as f:
                compose_config = yaml.safe_load(f)
            
            if "services" not in compose_config:
                issues.append("docker-compose.yml missing services")
            if "networks" not in compose_config:
                issues.append("docker-compose.yml missing networks configuration")
            if "volumes" not in compose_config:
                issues.append("docker-compose.yml missing volumes configuration")
        except Exception as e:
            issues.append(f"Error reading docker-compose.yml: {e}")
    
    return len(issues) == 0, issues


def validate_checkpoint_5_monitoring() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 5: Monitoring & Observability Setup."""
    issues = []
    
    required_files = [
        "monitoring/prometheus.yml",
        "monitoring/alertmanager.yml",
        "monitoring/rules/alerts.yml",
        "observability/opentelemetry-config.yaml",
    ]
    
    required_dirs = [
        "monitoring",
        "observability",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing monitoring file: {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            issues.append(f"Missing monitoring directory: {dir_path}")
    
    # Validate Prometheus configuration
    prometheus_file = Path("monitoring/prometheus.yml")
    if prometheus_file.exists():
        try:
            with open(prometheus_file) as f:
                prometheus_config = yaml.safe_load(f)
            
            if "global" not in prometheus_config:
                issues.append("Prometheus config missing global section")
            if "scrape_configs" not in prometheus_config:
                issues.append("Prometheus config missing scrape_configs")
            
            scrape_configs = prometheus_config.get("scrape_configs", [])
            if len(scrape_configs) == 0:
                issues.append("Prometheus config has no scrape configurations")
        except Exception as e:
            issues.append(f"Error reading prometheus.yml: {e}")
    
    return len(issues) == 0, issues


def validate_checkpoint_6_workflows() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 6: Workflow Documentation & Templates."""
    issues = []
    
    required_files = [
        "docs/github-workflows-templates/README.md",
        "docs/github-workflows-templates/ci.yml",
        "docs/github-workflows-templates/security.yml",
        "docs/GITHUB_WORKFLOWS_SETUP.md",
    ]
    
    required_dirs = [
        "docs/github-workflows-templates",
        "docs/workflows",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing workflow file: {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            issues.append(f"Missing workflow directory: {dir_path}")
    
    # Check workflow templates
    templates_dir = Path("docs/github-workflows-templates")
    if templates_dir.exists():
        workflow_files = list(templates_dir.glob("*.yml"))
        if len(workflow_files) < 3:
            issues.append("Insufficient workflow templates (expected at least 3)")
        
        # Validate workflow syntax
        for workflow_file in workflow_files:
            try:
                with open(workflow_file) as f:
                    yaml.safe_load(f)
            except Exception as e:
                issues.append(f"Invalid YAML in {workflow_file}: {e}")
    
    return len(issues) == 0, issues


def validate_checkpoint_7_metrics() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 7: Metrics & Automation Setup."""
    issues = []
    
    required_files = [
        "src/dora_metrics.py",
        "src/performance_monitor.py",
        "scripts/collect-metrics.py",
        "scripts/setup-automation.py",
    ]
    
    required_dirs = [
        "performance_data",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing metrics file: {file_path}")
        elif not os.access(file_path, os.X_OK) and file_path.startswith("scripts/"):
            issues.append(f"Script not executable: {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            issues.append(f"Missing metrics directory: {dir_path}")
    
    return len(issues) == 0, issues


def validate_checkpoint_8_integration() -> Tuple[bool, List[str]]:
    """Validate Checkpoint 8: Integration & Final Configuration."""
    issues = []
    
    # Check validation scripts
    validation_scripts = [
        "scripts/validate-testing-setup.py",
        "scripts/validate-build.py",
        "scripts/validate-monitoring.py",
        "scripts/validate-workflows.py",
        "scripts/validate-sdlc-integration.py",
    ]
    
    for script in validation_scripts:
        if not Path(script).exists():
            issues.append(f"Missing validation script: {script}")
        elif not os.access(script, os.X_OK):
            issues.append(f"Validation script not executable: {script}")
    
    # Check setup scripts
    setup_scripts = [
        "scripts/setup-workflows.sh",
        "scripts/setup-automation.py",
    ]
    
    for script in setup_scripts:
        if not Path(script).exists():
            issues.append(f"Missing setup script: {script}")
        elif not os.access(script, os.X_OK):
            issues.append(f"Setup script not executable: {script}")
    
    return len(issues) == 0, issues


def validate_repository_health() -> Tuple[bool, List[str]]:
    """Validate overall repository health."""
    issues = []
    
    # Check Git repository
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        if result.returncode != 0:
            issues.append("Not a valid Git repository")
    except FileNotFoundError:
        issues.append("Git not available")
    
    # Check Python files syntax
    python_files = list(Path("src").glob("**/*.py")) + list(Path("scripts").glob("*.py"))
    for py_file in python_files:
        try:
            with open(py_file) as f:
                compile(f.read(), py_file, 'exec')
        except SyntaxError as e:
            issues.append(f"Python syntax error in {py_file}: {e}")
        except Exception:
            # Skip files that can't be read
            pass
    
    # Check YAML files syntax
    yaml_files = list(Path(".").glob("**/*.yml")) + list(Path(".").glob("**/*.yaml"))
    for yaml_file in yaml_files:
        if ".git" in str(yaml_file) or "node_modules" in str(yaml_file):
            continue
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            issues.append(f"YAML syntax error in {yaml_file}: {e}")
        except Exception:
            # Skip files that can't be read
            pass
    
    # Check JSON files syntax
    json_files = list(Path(".").glob("**/*.json"))
    for json_file in json_files:
        if ".git" in str(json_file) or "node_modules" in str(json_file):
            continue
        try:
            with open(json_file) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"JSON syntax error in {json_file}: {e}")
        except Exception:
            # Skip files that can't be read
            pass
    
    return len(issues) == 0, issues


def generate_sdlc_report(results: Dict[str, Tuple[bool, List[str]]]) -> str:
    """Generate a comprehensive SDLC implementation report."""
    report = []
    report.append("📊 SDLC IMPLEMENTATION REPORT")
    report.append("=" * 50)
    
    total_checkpoints = len([k for k in results.keys() if k.startswith("Checkpoint")])
    passed_checkpoints = len([k for k, (success, _) in results.items() if k.startswith("Checkpoint") and success])
    
    report.append(f"\n🏁 Overall Progress: {passed_checkpoints}/{total_checkpoints} checkpoints passed")
    
    completion_percentage = (passed_checkpoints / total_checkpoints) * 100
    if completion_percentage == 100:
        report.append("🎉 SDLC IMPLEMENTATION COMPLETE!")
    elif completion_percentage >= 80:
        report.append("🟡 SDLC IMPLEMENTATION MOSTLY COMPLETE")
    elif completion_percentage >= 60:
        report.append("🟠 SDLC IMPLEMENTATION PARTIALLY COMPLETE")
    else:
        report.append("🔴 SDLC IMPLEMENTATION NEEDS WORK")
    
    report.append(f"\nCompletion: {completion_percentage:.1f}%")
    
    # Detailed results
    report.append("\n" + "-" * 50)
    report.append("CHECKPOINT DETAILS")
    report.append("-" * 50)
    
    checkpoint_names = {
        "Checkpoint 1": "Project Foundation & Documentation",
        "Checkpoint 2": "Development Environment & Tooling",
        "Checkpoint 3": "Testing Infrastructure",
        "Checkpoint 4": "Build & Containerization",
        "Checkpoint 5": "Monitoring & Observability Setup",
        "Checkpoint 6": "Workflow Documentation & Templates",
        "Checkpoint 7": "Metrics & Automation Setup",
        "Checkpoint 8": "Integration & Final Configuration",
    }
    
    for checkpoint, description in checkpoint_names.items():
        if checkpoint in results:
            success, issues = results[checkpoint]
            status = "✅ PASS" if success else "❌ FAIL"
            report.append(f"\n{status} {checkpoint}: {description}")
            
            if issues:
                report.append("   Issues found:")
                for issue in issues[:5]:  # Show first 5 issues
                    report.append(f"   - {issue}")
                if len(issues) > 5:
                    report.append(f"   ... and {len(issues) - 5} more issues")
    
    # Repository health
    if "Repository Health" in results:
        success, issues = results["Repository Health"]
        status = "✅ HEALTHY" if success else "⚠️  ISSUES"
        report.append(f"\n{status} Repository Health")
        if issues:
            report.append("   Issues found:")
            for issue in issues[:3]:
                report.append(f"   - {issue}")
    
    # Recommendations
    report.append("\n" + "-" * 50)
    report.append("RECOMMENDATIONS")
    report.append("-" * 50)
    
    if completion_percentage == 100:
        report.append("✅ SDLC implementation is complete and ready for production use")
        report.append("✅ All checkpoints have been successfully implemented")
        report.append("✅ Repository follows best practices for modern software development")
    else:
        failed_checkpoints = [k for k, (success, _) in results.items() if k.startswith("Checkpoint") and not success]
        if failed_checkpoints:
            report.append("Priority items to address:")
            for checkpoint in failed_checkpoints:
                if checkpoint in checkpoint_names:
                    report.append(f"- Complete {checkpoint}: {checkpoint_names[checkpoint]}")
    
    report.append("\n📚 For detailed setup instructions, see:")
    report.append("- docs/GITHUB_WORKFLOWS_SETUP.md")
    report.append("- docs/workflows/DEPLOYMENT_GUIDE.md")
    report.append("- docs/runbooks/monitoring-runbook.md")
    report.append("- docs/automation/AUTOMATION_GUIDE.md")
    
    return "\n".join(report)


def main():
    """Main validation function."""
    print("🔍 Claude Code Manager - SDLC Integration Validation")
    print("=" * 60)
    
    # Run all checkpoint validations
    validations = [
        ("Checkpoint 1", validate_checkpoint_1_foundation),
        ("Checkpoint 2", validate_checkpoint_2_development),
        ("Checkpoint 3", validate_checkpoint_3_testing),
        ("Checkpoint 4", validate_checkpoint_4_build),
        ("Checkpoint 5", validate_checkpoint_5_monitoring),
        ("Checkpoint 6", validate_checkpoint_6_workflows),
        ("Checkpoint 7", validate_checkpoint_7_metrics),
        ("Checkpoint 8", validate_checkpoint_8_integration),
        ("Repository Health", validate_repository_health),
    ]
    
    results = {}
    
    for checkpoint_name, validation_func in validations:
        print(f"\n🔍 Validating {checkpoint_name}...")
        try:
            success, issues = validation_func()
            results[checkpoint_name] = (success, issues)
            
            if success:
                print(f"   ✅ {checkpoint_name} validation passed")
            else:
                print(f"   ❌ {checkpoint_name} validation failed ({len(issues)} issues)")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"      - {issue}")
                if len(issues) > 3:
                    print(f"      ... and {len(issues) - 3} more issues")
        except Exception as e:
            print(f"   ❌ Error validating {checkpoint_name}: {e}")
            results[checkpoint_name] = (False, [str(e)])
    
    # Generate and display report
    print("\n" + "=" * 60)
    report = generate_sdlc_report(results)
    print(report)
    
    # Determine exit code
    total_checkpoints = len([k for k in results.keys() if k.startswith("Checkpoint")])
    passed_checkpoints = len([k for k, (success, _) in results.items() if k.startswith("Checkpoint") and success])
    
    if passed_checkpoints == total_checkpoints:
        return 0  # All checkpoints passed
    elif passed_checkpoints >= total_checkpoints * 0.8:
        return 0  # Most checkpoints passed
    else:
        return 1  # Too many failures


if __name__ == "__main__":
    sys.exit(main())