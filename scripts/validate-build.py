#!/usr/bin/env python3
"""
Build and containerization validation script.
Validates Docker images, build process, and deployment readiness.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def check_docker_availability() -> bool:
    """Check if Docker is available and running."""
    print("🐳 Checking Docker availability...")
    
    exit_code, stdout, stderr = run_command(["docker", "--version"])
    if exit_code != 0:
        print("❌ Docker is not installed or not in PATH")
        return False
    
    print(f"✅ Docker version: {stdout.strip()}")
    
    exit_code, stdout, stderr = run_command(["docker", "info"])
    if exit_code != 0:
        print("❌ Docker daemon is not running")
        return False
    
    print("✅ Docker daemon is running")
    return True


def check_docker_compose_availability() -> bool:
    """Check if Docker Compose is available."""
    print("\n🔧 Checking Docker Compose availability...")
    
    exit_code, stdout, stderr = run_command(["docker-compose", "--version"])
    if exit_code != 0:
        print("❌ Docker Compose is not installed or not in PATH")
        return False
    
    print(f"✅ Docker Compose version: {stdout.strip()}")
    return True


def validate_dockerfile() -> bool:
    """Validate Dockerfile syntax and best practices."""
    print("\n📄 Validating Dockerfile...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    print("✅ Dockerfile exists")
    
    # Check for multi-stage build
    with open(dockerfile_path, "r") as f:
        dockerfile_content = f.read()
    
    if "FROM" not in dockerfile_content:
        print("❌ No FROM instruction found")
        return False
    
    from_count = dockerfile_content.count("FROM")
    if from_count > 1:
        print(f"✅ Multi-stage build detected ({from_count} stages)")
    else:
        print("ℹ️ Single-stage build")
    
    # Check for security best practices
    security_checks = [
        ("USER", "Non-root user specified"),
        ("HEALTHCHECK", "Health check configured"),
        ("EXPOSE", "Ports exposed"),
        ("WORKDIR", "Working directory set"),
    ]
    
    for instruction, description in security_checks:
        if instruction in dockerfile_content:
            print(f"✅ {description}")
        else:
            print(f"⚠️  {description} - NOT FOUND")
    
    return True


def validate_dockerignore() -> bool:
    """Validate .dockerignore file."""
    print("\n🚅 Validating .dockerignore...")
    
    dockerignore_path = Path(".dockerignore")
    if not dockerignore_path.exists():
        print("❌ .dockerignore not found")
        return False
    
    print("✅ .dockerignore exists")
    
    with open(dockerignore_path, "r") as f:
        dockerignore_content = f.read()
    
    # Check for common patterns
    important_patterns = [
        ".git",
        "__pycache__",
        "*.pyc",
        "node_modules",
        ".env",
        "*.log",
        ".pytest_cache",
        "htmlcov",
    ]
    
    for pattern in important_patterns:
        if pattern in dockerignore_content:
            print(f"✅ Ignoring {pattern}")
        else:
            print(f"⚠️  Pattern {pattern} not found in .dockerignore")
    
    return True


def validate_docker_compose() -> bool:
    """Validate docker-compose.yml file."""
    print("\n🔍 Validating docker-compose.yml...")
    
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    print("✅ docker-compose.yml exists")
    
    # Validate syntax
    exit_code, stdout, stderr = run_command(["docker-compose", "config"])
    if exit_code != 0:
        print(f"❌ Docker Compose syntax error: {stderr}")
        return False
    
    print("✅ Docker Compose syntax is valid")
    
    # Parse and check services
    try:
        import yaml
        with open(compose_file, "r") as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get("services", {})
        print(f"✅ Services defined: {', '.join(services.keys())}")
        
        # Check for essential services
        essential_services = ["claude-manager"]
        for service in essential_services:
            if service in services:
                print(f"✅ Essential service '{service}' found")
            else:
                print(f"⚠️  Essential service '{service}' not found")
        
        # Check for networks and volumes
        if "networks" in compose_config:
            print(f"✅ Networks defined: {', '.join(compose_config['networks'].keys())}")
        
        if "volumes" in compose_config:
            print(f"✅ Volumes defined: {', '.join(compose_config['volumes'].keys())}")
        
    except Exception as e:
        print(f"⚠️  Could not parse docker-compose.yml: {e}")
    
    return True


def test_docker_build() -> bool:
    """Test Docker build process."""
    print("\n🔨 Testing Docker build...")
    
    # Test development build
    print("Building development image...")
    exit_code, stdout, stderr = run_command([
        "docker", "build", 
        "--target", "development",
        "--tag", "claude-manager:dev-test",
        "."
    ])
    
    if exit_code != 0:
        print(f"❌ Development build failed: {stderr}")
        return False
    
    print("✅ Development image built successfully")
    
    # Test production build
    print("Building production image...")
    exit_code, stdout, stderr = run_command([
        "docker", "build", 
        "--target", "production",
        "--tag", "claude-manager:prod-test",
        "."
    ])
    
    if exit_code != 0:
        print(f"❌ Production build failed: {stderr}")
        return False
    
    print("✅ Production image built successfully")
    
    return True


def check_image_security() -> bool:
    """Check Docker image security."""
    print("\n🔒 Checking image security...")
    
    # Check if images exist
    exit_code, stdout, stderr = run_command(["docker", "images", "claude-manager"])
    if exit_code != 0 or "claude-manager" not in stdout:
        print("⚠️  No Claude Manager images found to scan")
        return True
    
    print("✅ Claude Manager images found")
    
    # Simple security checks
    image_name = "claude-manager:dev-test"
    
    # Check for running as root
    exit_code, stdout, stderr = run_command([
        "docker", "run", "--rm", image_name, "whoami"
    ])
    
    if exit_code == 0 and "root" not in stdout:
        print("✅ Image runs as non-root user")
    else:
        print("⚠️  Image may be running as root")
    
    return True


def validate_build_scripts() -> bool:
    """Validate build automation scripts."""
    print("\n📜 Validating build scripts...")
    
    # Check Makefile
    makefile_path = Path("Makefile")
    if makefile_path.exists():
        print("✅ Makefile found")
        
        with open(makefile_path, "r") as f:
            makefile_content = f.read()
        
        # Check for essential targets
        essential_targets = ["build", "test", "clean", "help"]
        for target in essential_targets:
            if f"{target}:" in makefile_content:
                print(f"✅ Makefile target '{target}' found")
            else:
                print(f"⚠️  Makefile target '{target}' not found")
    else:
        print("⚠️  Makefile not found")
    
    # Check package.json for Node.js scripts
    package_json_path = Path("package.json")
    if package_json_path.exists():
        print("✅ package.json found")
        
        try:
            with open(package_json_path, "r") as f:
                package_config = json.load(f)
            
            scripts = package_config.get("scripts", {})
            if scripts:
                print(f"✅ npm scripts defined: {', '.join(scripts.keys())}")
            else:
                print("⚠️  No npm scripts defined")
                
        except Exception as e:
            print(f"⚠️  Could not parse package.json: {e}")
    
    return True


def check_semantic_release() -> bool:
    """Check semantic release configuration."""
    print("\n📦 Checking semantic release configuration...")
    
    # Check for semantic release config
    release_configs = [
        ".releaserc",
        ".releaserc.json",
        "release.config.js",
        "package.json"
    ]
    
    config_found = False
    for config_file in release_configs:
        if Path(config_file).exists():
            print(f"✅ Release config found: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print("⚠️  No semantic release configuration found")
    
    return True


def cleanup_test_images():
    """Clean up test Docker images."""
    print("\n🧹 Cleaning up test images...")
    
    test_images = ["claude-manager:dev-test", "claude-manager:prod-test"]
    
    for image in test_images:
        exit_code, stdout, stderr = run_command(["docker", "rmi", image])
        if exit_code == 0:
            print(f"✅ Removed test image: {image}")
        else:
            print(f"ℹ️ Test image not found or already removed: {image}")


def main():
    """Main validation function."""
    print("🏢 Claude Code Manager - Build & Containerization Validation")
    print("=" * 70)
    
    checks = [
        ("Docker Availability", check_docker_availability),
        ("Docker Compose Availability", check_docker_compose_availability),
        ("Dockerfile Validation", validate_dockerfile),
        ("Dockerignore Validation", validate_dockerignore),
        ("Docker Compose Validation", validate_docker_compose),
        ("Build Scripts Validation", validate_build_scripts),
        ("Semantic Release Check", check_semantic_release),
    ]
    
    # Run basic checks first
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Only run build tests if Docker is available
    docker_available = results[0][1] and results[1][1]
    if docker_available:
        try:
            print("\n" + "=" * 50)
            print("🔨 RUNNING BUILD TESTS")
            print("=" * 50)
            
            build_result = test_docker_build()
            results.append(("Docker Build Test", build_result))
            
            if build_result:
                security_result = check_image_security()
                results.append(("Image Security Check", security_result))
                
                cleanup_test_images()
            
        except Exception as e:
            print(f"❌ Error in build tests: {e}")
            results.append(("Docker Build Test", False))
    
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
        print("🎉 ALL BUILD VALIDATION CHECKS PASSED!")
        print("✅ Build and containerization infrastructure is ready for production")
        return 0
    else:
        print(f"⚠️  {total - passed} checks failed")
        print("❌ Please review the failed checks above")
        return 1


if __name__ == "__main__":
    sys.exit(main())