#!/usr/bin/env python3
"""
Comprehensive metrics collection script for Claude Code Manager.
Collects DORA metrics, performance data, and operational metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional


def run_command(cmd: List[str], capture_output: bool = True) -> Dict[str, Any]:
    """Run a command and return structured output."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, timeout=30
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out",
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


def collect_git_metrics() -> Dict[str, Any]:
    """Collect Git repository metrics."""
    print("📋 Collecting Git metrics...")
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "repository": {},
        "commits": {},
        "branches": {},
        "contributors": {},
    }
    
    # Basic repository info
    repo_info = run_command(["git", "remote", "get-url", "origin"])
    if repo_info["success"]:
        metrics["repository"]["remote_url"] = repo_info["stdout"]
    
    current_branch = run_command(["git", "branch", "--show-current"])
    if current_branch["success"]:
        metrics["repository"]["current_branch"] = current_branch["stdout"]
    
    # Commit statistics (last 30 days)
    since_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    commit_count = run_command([
        "git", "rev-list", "--count", "--since", since_date, "HEAD"
    ])
    if commit_count["success"]:
        metrics["commits"]["count_30_days"] = int(commit_count["stdout"] or 0)
    
    # Recent commits for lead time calculation
    recent_commits = run_command([
        "git", "log", "--since", since_date, "--pretty=format:%H|%ad|%an|%s", 
        "--date=iso"
    ])
    if recent_commits["success"] and recent_commits["stdout"]:
        commits_data = []
        for line in recent_commits["stdout"].split('\n'):
            if '|' in line:
                sha, date, author, message = line.split('|', 3)
                commits_data.append({
                    "sha": sha,
                    "date": date,
                    "author": author,
                    "message": message
                })
        metrics["commits"]["recent_commits"] = commits_data[:10]  # Last 10
    
    # Branch information
    branch_count = run_command(["git", "branch", "-r", "--list"])
    if branch_count["success"]:
        branches = [b.strip() for b in branch_count["stdout"].split('\n') if b.strip()]
        metrics["branches"]["remote_count"] = len(branches)
        metrics["branches"]["list"] = branches[:20]  # First 20
    
    # Contributor statistics
    contributors = run_command([
        "git", "shortlog", "-sn", "--since", since_date
    ])
    if contributors["success"] and contributors["stdout"]:
        contrib_data = []
        total_commits = 0
        for line in contributors["stdout"].split('\n'):
            if line.strip():
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    count = int(parts[0])
                    name = parts[1]
                    contrib_data.append({"name": name, "commits": count})
                    total_commits += count
        
        metrics["contributors"]["count"] = len(contrib_data)
        metrics["contributors"]["total_commits"] = total_commits
        metrics["contributors"]["top_contributors"] = contrib_data[:10]
    
    return metrics


def collect_dora_metrics() -> Dict[str, Any]:
    """Collect DORA (DevOps Research and Assessment) metrics."""
    print("📊 Collecting DORA metrics...")
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "deployment_frequency": {},
        "lead_time_for_changes": {},
        "change_failure_rate": {},
        "time_to_restore_service": {},
    }
    
    # Try to calculate deployment frequency from git tags
    tags_result = run_command(["git", "tag", "--sort=-version:refname"])
    if tags_result["success"] and tags_result["stdout"]:
        tags = tags_result["stdout"].split('\n')[:10]  # Last 10 tags
        
        deployment_dates = []
        for tag in tags:
            if tag.strip():
                tag_date = run_command(["git", "log", "-1", "--format=%ai", tag.strip()])
                if tag_date["success"]:
                    deployment_dates.append({
                        "tag": tag.strip(),
                        "date": tag_date["stdout"]
                    })
        
        metrics["deployment_frequency"]["recent_deployments"] = deployment_dates
        
        # Calculate average deployment frequency
        if len(deployment_dates) >= 2:
            date_diffs = []
            for i in range(len(deployment_dates) - 1):
                try:
                    date1 = datetime.fromisoformat(deployment_dates[i]["date"].replace(' ', 'T'))
                    date2 = datetime.fromisoformat(deployment_dates[i+1]["date"].replace(' ', 'T'))
                    diff_days = abs((date1 - date2).days)
                    date_diffs.append(diff_days)
                except:
                    continue
            
            if date_diffs:
                avg_days = sum(date_diffs) / len(date_diffs)
                metrics["deployment_frequency"]["average_days_between_deployments"] = avg_days
    
    # Lead time estimation (from commit to merge/tag)
    # This is a simplified calculation
    recent_merges = run_command([
        "git", "log", "--merges", "--since=30.days.ago", 
        "--pretty=format:%H|%ai|%s"
    ])
    
    if recent_merges["success"] and recent_merges["stdout"]:
        merge_data = []
        for line in recent_merges["stdout"].split('\n')[:10]:
            if '|' in line:
                sha, date, message = line.split('|', 2)
                merge_data.append({
                    "sha": sha,
                    "date": date,
                    "message": message
                })
        
        metrics["lead_time_for_changes"]["recent_merges"] = merge_data
        metrics["lead_time_for_changes"]["merge_count_30_days"] = len(merge_data)
    
    return metrics


def collect_system_metrics() -> Dict[str, Any]:
    """Collect system performance metrics."""
    print("🖥️ Collecting system metrics...")
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu": {},
        "memory": {},
        "disk": {},
        "network": {},
    }
    
    # CPU information
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            cpu_count = cpuinfo.count("processor")
            metrics["cpu"]["core_count"] = cpu_count
    except:
        pass
    
    # Load average (if available)
    try:
        if os.path.exists("/proc/loadavg"):
            with open("/proc/loadavg", "r") as f:
                loadavg = f.read().split()
            metrics["cpu"]["load_1min"] = float(loadavg[0])
            metrics["cpu"]["load_5min"] = float(loadavg[1])
            metrics["cpu"]["load_15min"] = float(loadavg[2])
    except:
        pass
    
    # Memory information
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            
            for line in meminfo.split('\n'):
                if line.startswith("MemTotal:"):
                    metrics["memory"]["total_kb"] = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    metrics["memory"]["available_kb"] = int(line.split()[1])
                elif line.startswith("MemFree:"):
                    metrics["memory"]["free_kb"] = int(line.split()[1])
    except:
        pass
    
    # Disk usage for current directory
    try:
        statvfs = os.statvfs('.')
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        free_bytes = statvfs.f_frsize * statvfs.f_available
        used_bytes = total_bytes - free_bytes
        
        metrics["disk"]["total_bytes"] = total_bytes
        metrics["disk"]["used_bytes"] = used_bytes
        metrics["disk"]["free_bytes"] = free_bytes
        metrics["disk"]["usage_percent"] = (used_bytes / total_bytes) * 100
    except:
        pass
    
    return metrics


def collect_docker_metrics() -> Dict[str, Any]:
    """Collect Docker container metrics if available."""
    print("🐳 Collecting Docker metrics...")
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "available": False,
        "containers": {},
        "images": {},
        "volumes": {},
    }
    
    # Check if Docker is available
    docker_version = run_command(["docker", "--version"])
    if not docker_version["success"]:
        return metrics
    
    metrics["available"] = True
    metrics["docker_version"] = docker_version["stdout"]
    
    # Container statistics
    containers = run_command(["docker", "ps", "-a", "--format", "json"])
    if containers["success"] and containers["stdout"]:
        container_list = []
        for line in containers["stdout"].split('\n'):
            if line.strip():
                try:
                    container_data = json.loads(line)
                    container_list.append(container_data)
                except:
                    continue
        
        metrics["containers"]["total_count"] = len(container_list)
        metrics["containers"]["running_count"] = len([
            c for c in container_list if c.get("State") == "running"
        ])
        metrics["containers"]["list"] = container_list[:10]  # First 10
    
    # Image statistics
    images = run_command(["docker", "images", "--format", "json"])
    if images["success"] and images["stdout"]:
        image_list = []
        for line in images["stdout"].split('\n'):
            if line.strip():
                try:
                    image_data = json.loads(line)
                    image_list.append(image_data)
                except:
                    continue
        
        metrics["images"]["count"] = len(image_list)
        metrics["images"]["list"] = image_list[:10]  # First 10
    
    # Volume statistics
    volumes = run_command(["docker", "volume", "ls", "--format", "json"])
    if volumes["success"] and volumes["stdout"]:
        volume_list = []
        for line in volumes["stdout"].split('\n'):
            if line.strip():
                try:
                    volume_data = json.loads(line)
                    volume_list.append(volume_data)
                except:
                    continue
        
        metrics["volumes"]["count"] = len(volume_list)
    
    return metrics


def collect_application_metrics() -> Dict[str, Any]:
    """Collect application-specific metrics."""
    print("📊 Collecting application metrics...")
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "python": {},
        "dependencies": {},
        "test_coverage": {},
        "code_quality": {},
    }
    
    # Python version and environment
    python_version = run_command(["python3", "--version"])
    if python_version["success"]:
        metrics["python"]["version"] = python_version["stdout"]
    
    # Package information
    if os.path.exists("requirements.txt"):
        try:
            with open("requirements.txt", "r") as f:
                requirements = f.read().strip().split('\n')
            metrics["dependencies"]["production_count"] = len([r for r in requirements if r.strip() and not r.startswith('#')])
        except:
            pass
    
    if os.path.exists("requirements-dev.txt"):
        try:
            with open("requirements-dev.txt", "r") as f:
                dev_requirements = f.read().strip().split('\n')
            metrics["dependencies"]["development_count"] = len([r for r in dev_requirements if r.strip() and not r.startswith('#')])
        except:
            pass
    
    # Test coverage (if coverage file exists)
    coverage_files = [".coverage", "coverage.xml", "htmlcov/index.html"]
    for coverage_file in coverage_files:
        if os.path.exists(coverage_file):
            metrics["test_coverage"]["coverage_file_exists"] = True
            break
    else:
        metrics["test_coverage"]["coverage_file_exists"] = False
    
    # Code quality metrics
    src_files = list(Path("src").glob("**/*.py")) if Path("src").exists() else []
    test_files = list(Path("tests").glob("**/*.py")) if Path("tests").exists() else []
    
    metrics["code_quality"]["source_files_count"] = len(src_files)
    metrics["code_quality"]["test_files_count"] = len(test_files)
    
    if src_files:
        total_lines = 0
        for file_path in src_files:
            try:
                with open(file_path, "r") as f:
                    total_lines += len(f.readlines())
            except:
                continue
        metrics["code_quality"]["total_source_lines"] = total_lines
    
    return metrics


def save_metrics(metrics_data: Dict[str, Any], output_file: str):
    """Save metrics data to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "w") as f:
            json.dump(metrics_data, f, indent=2, sort_keys=True)
        print(f"✅ Metrics saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")


def generate_metrics_summary(metrics_data: Dict[str, Any]) -> str:
    """Generate a human-readable summary of metrics."""
    summary = []
    summary.append("📊 METRICS COLLECTION SUMMARY")
    summary.append("=" * 40)
    
    # Git metrics
    if "git" in metrics_data:
        git_metrics = metrics_data["git"]
        commits = git_metrics.get("commits", {})
        contributors = git_metrics.get("contributors", {})
        
        summary.append(f"📋 Git Repository:")
        summary.append(f"  - Commits (30 days): {commits.get('count_30_days', 'N/A')}")
        summary.append(f"  - Contributors: {contributors.get('count', 'N/A')}")
        summary.append(f"  - Total commits by contributors: {contributors.get('total_commits', 'N/A')}")
    
    # DORA metrics
    if "dora" in metrics_data:
        dora_metrics = metrics_data["dora"]
        deployment_freq = dora_metrics.get("deployment_frequency", {})
        lead_time = dora_metrics.get("lead_time_for_changes", {})
        
        summary.append(f"\n📊 DORA Metrics:")
        summary.append(f"  - Recent deployments: {len(deployment_freq.get('recent_deployments', []))}")
        avg_days = deployment_freq.get("average_days_between_deployments")
        if avg_days:
            summary.append(f"  - Avg deployment frequency: {avg_days:.1f} days")
        summary.append(f"  - Recent merges (30 days): {lead_time.get('merge_count_30_days', 'N/A')}")
    
    # System metrics
    if "system" in metrics_data:
        system_metrics = metrics_data["system"]
        cpu = system_metrics.get("cpu", {})
        memory = system_metrics.get("memory", {})
        disk = system_metrics.get("disk", {})
        
        summary.append(f"\n🖥️ System Metrics:")
        if "core_count" in cpu:
            summary.append(f"  - CPU cores: {cpu['core_count']}")
        if "load_1min" in cpu:
            summary.append(f"  - Load average (1min): {cpu['load_1min']}")
        if "total_kb" in memory:
            total_mb = memory['total_kb'] // 1024
            summary.append(f"  - Total memory: {total_mb} MB")
        if "usage_percent" in disk:
            summary.append(f"  - Disk usage: {disk['usage_percent']:.1f}%")
    
    # Docker metrics
    if "docker" in metrics_data:
        docker_metrics = metrics_data["docker"]
        if docker_metrics.get("available"):
            containers = docker_metrics.get("containers", {})
            images = docker_metrics.get("images", {})
            summary.append(f"\n🐳 Docker Metrics:")
            summary.append(f"  - Containers: {containers.get('total_count', 0)} total, {containers.get('running_count', 0)} running")
            summary.append(f"  - Images: {images.get('count', 0)}")
        else:
            summary.append(f"\n🐳 Docker: Not available")
    
    # Application metrics
    if "application" in metrics_data:
        app_metrics = metrics_data["application"]
        python = app_metrics.get("python", {})
        deps = app_metrics.get("dependencies", {})
        quality = app_metrics.get("code_quality", {})
        
        summary.append(f"\n📊 Application Metrics:")
        if "version" in python:
            summary.append(f"  - Python: {python['version']}")
        summary.append(f"  - Production dependencies: {deps.get('production_count', 'N/A')}")
        summary.append(f"  - Development dependencies: {deps.get('development_count', 'N/A')}")
        summary.append(f"  - Source files: {quality.get('source_files_count', 'N/A')}")
        summary.append(f"  - Test files: {quality.get('test_files_count', 'N/A')}")
    
    return "\n".join(summary)


def main():
    """Main metrics collection function."""
    print("📈 Claude Code Manager - Comprehensive Metrics Collection")
    print("=" * 60)
    
    all_metrics = {
        "collection_timestamp": datetime.utcnow().isoformat(),
        "collection_duration_seconds": 0,
    }
    
    start_time = time.time()
    
    # Collect different types of metrics
    collectors = [
        ("git", collect_git_metrics),
        ("dora", collect_dora_metrics),
        ("system", collect_system_metrics),
        ("docker", collect_docker_metrics),
        ("application", collect_application_metrics),
    ]
    
    for metric_type, collector_func in collectors:
        try:
            print(f"\n{'-' * 30}")
            metrics = collector_func()
            all_metrics[metric_type] = metrics
        except Exception as e:
            print(f"❌ Error collecting {metric_type} metrics: {e}")
            all_metrics[metric_type] = {"error": str(e)}
    
    end_time = time.time()
    all_metrics["collection_duration_seconds"] = round(end_time - start_time, 2)
    
    # Save metrics to file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = f"performance_data/metrics_{timestamp}.json"
    save_metrics(all_metrics, output_file)
    
    # Generate and display summary
    print(f"\n{'-' * 60}")
    summary = generate_metrics_summary(all_metrics)
    print(summary)
    
    print(f"\n🕰️ Collection completed in {all_metrics['collection_duration_seconds']} seconds")
    print(f"💾 Full metrics saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())