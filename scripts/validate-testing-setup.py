#!/usr/bin/env python3
"""
Testing infrastructure validation script.
Validates that all testing components are properly configured.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print result."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False


def check_directory_structure() -> bool:
    """Check test directory structure."""
    print("\n📁 Checking test directory structure...")
    
    test_dirs = [
        "tests",
        "tests/unit",
        "tests/integration", 
        "tests/e2e",
        "tests/performance",
        "tests/security",
        "tests/fixtures",
        "tests/utils"
    ]
    
    all_exist = True
    for test_dir in test_dirs:
        if Path(test_dir).is_dir():
            test_count = len(list(Path(test_dir).glob("test_*.py")))
            print(f"✅ {test_dir}/ ({test_count} test files)")
        else:
            print(f"❌ {test_dir}/ - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_configuration_files() -> bool:
    """Check testing configuration files."""
    print("\n⚙️  Checking configuration files...")
    
    config_files = [
        ("pytest.ini", "Pytest configuration"),
        ("pyproject.toml", "Project configuration"),
        ("requirements-dev.txt", "Development requirements"),
        (".pre-commit-config.yaml", "Pre-commit hooks"),
        ("mypy.ini", "MyPy type checking"),
        (".pylintrc", "Pylint configuration"),
    ]
    
    all_exist = True
    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def check_test_utilities() -> bool:
    """Check test utilities and fixtures."""
    print("\n🔧 Checking test utilities...")
    
    utilities = [
        ("tests/conftest.py", "Global test configuration"),
        ("tests/fixtures/__init__.py", "Test fixtures package"),
        ("tests/utils/test_helpers.py", "Test helper utilities"),
        ("tests/fixtures/github_responses.py", "GitHub API mock responses"),
        ("tests/fixtures/sample_data.py", "Sample test data"),
    ]
    
    all_exist = True
    for filepath, description in utilities:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def check_pytest_configuration() -> bool:
    """Check pytest configuration details."""
    print("\n🧪 Checking pytest configuration...")
    
    try:
        with open("pytest.ini", "r") as f:
            content = f.read()
            
        checks = [
            ("testpaths = tests" in content, "Test paths configured"),
            ("--cov=src" in content, "Coverage configured"),
            ("--cov-report=html" in content, "HTML coverage reports"),
            ("asyncio_mode = auto" in content, "Asyncio mode configured"),
            ("markers =" in content, "Test markers defined"),
        ]
        
        all_configured = True
        for check, description in checks:
            if check:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} - NOT CONFIGURED")
                all_configured = False
        
        return all_configured
        
    except FileNotFoundError:
        print("❌ pytest.ini not found")
        return False


def check_development_requirements() -> bool:
    """Check development requirements."""
    print("\n📦 Checking development requirements...")
    
    try:
        with open("requirements-dev.txt", "r") as f:
            content = f.read()
        
        required_packages = [
            "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock",
            "black", "isort", "flake8", "mypy", "pylint",
            "bandit", "safety", "pre-commit"
        ]
        
        all_present = True
        for package in required_packages:
            if package in content:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - NOT FOUND")
                all_present = False
        
        return all_present
        
    except FileNotFoundError:
        print("❌ requirements-dev.txt not found")
        return False


def main():
    """Main validation function."""
    print("🧪 Claude Code Manager - Testing Infrastructure Validation")
    print("=" * 60)
    
    checks = [
        check_directory_structure,
        check_configuration_files,
        check_test_utilities,
        check_pytest_configuration,
        check_development_requirements,
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("✅ Testing infrastructure is properly configured!")
        return 0
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total})")
        print("❌ Please review the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())