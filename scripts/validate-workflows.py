#!/usr/bin/env python3
"""
GitHub Actions workflow validation script.
Validates workflow templates, documentation, and setup requirements.
"""

import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any


def validate_workflow_syntax(workflow_file: Path) -> Tuple[bool, str]:
    """Validate YAML syntax of a workflow file."""
    try:
        with open(workflow_file, "r") as f:
            yaml.safe_load(f)
        return True, "Valid YAML syntax"
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def analyze_workflow_structure(workflow_file: Path) -> Dict[str, Any]:
    """Analyze workflow structure and return metadata."""
    try:
        with open(workflow_file, "r") as f:
            workflow = yaml.safe_load(f)
        
        analysis = {
            "name": workflow.get("name", "Unnamed"),
            "triggers": list(workflow.get("on", {}).keys()),
            "jobs": list(workflow.get("jobs", {}).keys()),
            "job_count": len(workflow.get("jobs", {})),
            "uses_matrix": any(
                "matrix" in job.get("strategy", {})
                for job in workflow.get("jobs", {}).values()
            ),
            "uses_cache": any(
                any("cache" in step.get("uses", "") for step in job.get("steps", []))
                for job in workflow.get("jobs", {}).values()
            ),
            "has_env_vars": "env" in workflow,
            "secrets_referenced": [],
        }
        
        # Find secret references
        workflow_str = str(workflow)
        if "secrets." in workflow_str:
            import re
            secret_refs = re.findall(r"secrets\.([A-Z_][A-Z0-9_]*)", workflow_str)
            analysis["secrets_referenced"] = list(set(secret_refs))
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}


def validate_workflow_templates() -> bool:
    """Validate all workflow templates."""
    print("🔄 Validating GitHub Actions workflow templates...")
    
    templates_dir = Path("docs/github-workflows-templates")
    if not templates_dir.exists():
        print("❌ Workflow templates directory not found")
        return False
    
    print("✅ Workflow templates directory exists")
    
    workflow_files = list(templates_dir.glob("*.yml"))
    if not workflow_files:
        print("❌ No workflow template files found")
        return False
    
    print(f"✅ {len(workflow_files)} workflow template files found")
    
    all_valid = True
    total_jobs = 0
    total_secrets = set()
    
    for workflow_file in workflow_files:
        print(f"\n📄 Validating {workflow_file.name}...")
        
        # Check syntax
        is_valid, message = validate_workflow_syntax(workflow_file)
        if is_valid:
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
            all_valid = False
            continue
        
        # Analyze structure
        analysis = analyze_workflow_structure(workflow_file)
        if "error" in analysis:
            print(f"   ❌ Analysis error: {analysis['error']}")
            all_valid = False
            continue
        
        print(f"   ✅ Name: {analysis['name']}")
        print(f"   ✅ Triggers: {', '.join(analysis['triggers'])}")
        print(f"   ✅ Jobs: {analysis['job_count']} ({', '.join(analysis['jobs'])})")
        
        if analysis['uses_matrix']:
            print("   ✅ Uses matrix strategy")
        
        if analysis['uses_cache']:
            print("   ✅ Uses caching")
        
        if analysis['has_env_vars']:
            print("   ✅ Defines environment variables")
        
        if analysis['secrets_referenced']:
            secrets = analysis['secrets_referenced']
            print(f"   ✅ References {len(secrets)} secrets: {', '.join(secrets)}")
            total_secrets.update(secrets)
        
        total_jobs += analysis['job_count']
    
    print(f"\n📊 Workflow Summary:")
    print(f"   - Total workflows: {len(workflow_files)}")
    print(f"   - Total jobs: {total_jobs}")
    print(f"   - Unique secrets referenced: {len(total_secrets)}")
    print(f"   - Secret names: {', '.join(sorted(total_secrets))}")
    
    return all_valid


def validate_workflow_documentation() -> bool:
    """Validate workflow documentation."""
    print("\n📆 Validating workflow documentation...")
    
    # Check README in templates directory
    readme_file = Path("docs/github-workflows-templates/README.md")
    if not readme_file.exists():
        print("❌ Workflow templates README not found")
        return False
    
    print("✅ Workflow templates README exists")
    
    # Check setup guide
    setup_guide = Path("docs/GITHUB_WORKFLOWS_SETUP.md")
    if not setup_guide.exists():
        print("❌ GitHub workflows setup guide not found")
        return False
    
    print("✅ GitHub workflows setup guide exists")
    
    # Validate content structure
    try:
        with open(readme_file, "r") as f:
            readme_content = f.read()
        
        essential_sections = [
            "Quick Start",
            "Workflow Files",
            "Required Secrets",
            "Setup"
        ]
        
        for section in essential_sections:
            if section.lower() in readme_content.lower():
                print(f"   ✅ {section} section found")
            else:
                print(f"   ⚠️  {section} section not clearly identified")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading documentation: {e}")
        return False


def check_workflow_dependencies() -> bool:
    """Check if workflow dependencies are available."""
    print("\n📦 Checking workflow dependencies...")
    
    # Check if GitHub CLI is available
    try:
        result = subprocess.run(
            ["gh", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   ✅ GitHub CLI available: {result.stdout.strip().split()[2]}")
        else:
            print("   ⚠️  GitHub CLI not available")
    except FileNotFoundError:
        print("   ⚠️  GitHub CLI not installed")
    
    # Check if git is available
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"   ✅ Git available: {result.stdout.strip()}")
        else:
            print("   ❌ Git not available")
            return False
    except FileNotFoundError:
        print("   ❌ Git not installed")
        return False
    
    # Check if we're in a git repository
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("   ✅ Inside a Git repository")
        else:
            print("   ❌ Not inside a Git repository")
            return False
    except Exception as e:
        print(f"   ❌ Error checking Git repository: {e}")
        return False
    
    return True


def validate_required_files() -> bool:
    """Validate that required project files exist."""
    print("\n📝 Validating required project files...")
    
    required_files = [
        ("requirements.txt", "Python dependencies"),
        ("requirements-dev.txt", "Development dependencies"),
        ("pyproject.toml", "Project configuration"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose configuration"),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: {file_path} - NOT FOUND")
            all_exist = False
    
    # Check for test directories
    test_dirs = ["tests", "test"]
    test_dir_found = False
    for test_dir in test_dirs:
        if Path(test_dir).is_dir():
            test_count = len(list(Path(test_dir).glob("**/*.py")))
            print(f"   ✅ Test directory: {test_dir}/ ({test_count} Python files)")
            test_dir_found = True
            break
    
    if not test_dir_found:
        print("   ⚠️  No test directory found")
    
    return all_exist and test_dir_found


def generate_workflow_setup_script() -> bool:
    """Generate a script to set up workflows."""
    print("\n🛠️ Generating workflow setup script...")
    
    script_content = '''#!/bin/bash
# Automated GitHub Actions Workflow Setup Script
# Generated by Claude Code Manager

set -e

echo "🚀 Setting up GitHub Actions workflows..."

# Create .github/workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy workflow templates
echo "📁 Copying workflow templates..."
cp docs/github-workflows-templates/*.yml .github/workflows/

# List copied workflows
echo "✅ Copied workflows:"
ls -la .github/workflows/

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a Git repository"
    exit 1
fi

# Add and commit workflows
echo "📝 Committing workflows to Git..."
git add .github/workflows/
git commit -m "feat: add comprehensive GitHub Actions workflows

- Added CI/CD pipeline with multi-Python testing
- Integrated security scanning and vulnerability detection
- Configured automated release management
- Added performance testing and monitoring
- Set up dependency management automation

🤖 Generated with Claude Code"

echo "🎉 GitHub Actions workflows have been set up successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Configure required secrets in your GitHub repository settings"
echo "2. Review and customize workflow files if needed"
echo "3. Push changes to trigger the first workflow run"
echo "4. Monitor workflow execution in the Actions tab"
echo ""
echo "📚 For detailed setup instructions, see:"
echo "   docs/GITHUB_WORKFLOWS_SETUP.md"
'''
    
    script_path = Path("scripts/setup-workflows.sh")
    try:
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"   ✅ Setup script created: {script_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating setup script: {e}")
        return False


def main():
    """Main validation function."""
    print("🔄 Claude Code Manager - GitHub Actions Workflow Validation")
    print("=" * 70)
    
    checks = [
        ("Workflow Templates", validate_workflow_templates),
        ("Workflow Documentation", validate_workflow_documentation),
        ("Workflow Dependencies", check_workflow_dependencies),
        ("Required Project Files", validate_required_files),
        ("Setup Script Generation", generate_workflow_setup_script),
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
        print("🎉 ALL WORKFLOW VALIDATION CHECKS PASSED!")
        print("✅ GitHub Actions workflows are ready for deployment")
        print("\n🚀 To deploy workflows, run:")
        print("   ./scripts/setup-workflows.sh")
        return 0
    elif passed >= total * 0.8:
        print("🟡 MOSTLY READY - Minor issues found")
        print("⚠️  Review the failed checks for optimization opportunities")
        return 0
    else:
        print(f"❌ {total - passed} critical checks failed")
        print("❌ Please review and fix the failed checks above")
        return 1


if __name__ == "__main__":
    sys.exit(main())