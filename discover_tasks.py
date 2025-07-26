#!/usr/bin/env python3
"""
Simple Task Discovery Script for Autonomous Backlog Execution
Discovers actionable tasks from the current codebase without external dependencies.
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class DiscoveredTask:
    """A task discovered from codebase analysis"""
    id: str
    title: str
    description: str
    type: str
    file_path: str
    line_number: int
    context: str
    effort: int
    value: int
    time_criticality: int
    risk_reduction: int
    wsjf_score: float
    status: str = "NEW"


def scan_for_todos_fixmes(root_path: str = ".") -> List[DiscoveredTask]:
    """Scan codebase for TODO and FIXME comments"""
    tasks = []
    todo_pattern = re.compile(r'#?\s*(TODO|FIXME):?\s*(.+)', re.IGNORECASE)
    
    # File extensions to scan
    code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php'}
    
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', 'env'}]
        
        for file in files:
            if any(file.endswith(ext) for ext in code_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        match = todo_pattern.search(line)
                        if match:
                            todo_type = match.group(1).upper()
                            todo_text = match.group(2).strip()
                            
                            # Get context (3 lines before and after)
                            start_line = max(0, line_num - 4)
                            end_line = min(len(lines), line_num + 3)
                            context_lines = lines[start_line:end_line]
                            context = ''.join(f"{start_line + i + 1:3}: {l}" for i, l in enumerate(context_lines))
                            
                            # Basic scoring (can be enhanced)
                            effort = estimate_effort(todo_text, file_path)
                            value = estimate_value(todo_text, file_path)
                            criticality = estimate_criticality(todo_text)
                            risk_reduction = estimate_risk_reduction(todo_text)
                            
                            wsjf = calculate_wsjf(value, criticality, risk_reduction, effort)
                            
                            task = DiscoveredTask(
                                id=f"todo_{hash(f'{file_path}:{line_num}:{todo_text}') % 10000}",
                                title=f"Address {todo_type} in {os.path.basename(file_path)}:{line_num}",
                                description=f"{todo_type}: {todo_text}",
                                type="Refactor" if todo_type == "TODO" else "Bug",
                                file_path=file_path,
                                line_number=line_num,
                                context=context,
                                effort=effort,
                                value=value,
                                time_criticality=criticality,
                                risk_reduction=risk_reduction,
                                wsjf_score=wsjf,
                                status="READY"
                            )
                            tasks.append(task)
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    return tasks


def estimate_effort(todo_text: str, file_path: str) -> int:
    """Estimate effort on 1-13 scale based on TODO content and context"""
    todo_lower = todo_text.lower()
    
    # High effort indicators
    if any(word in todo_lower for word in ['refactor', 'rewrite', 'redesign', 'architecture']):
        return 8
    elif any(word in todo_lower for word in ['implement', 'add feature', 'new module']):
        return 5
    elif any(word in todo_lower for word in ['fix', 'bug', 'error', 'issue']):
        return 3
    elif any(word in todo_lower for word in ['update', 'improve', 'optimize']):
        return 2
    else:
        return 1


def estimate_value(todo_text: str, file_path: str) -> int:
    """Estimate business value on 1-13 scale"""
    todo_lower = todo_text.lower()
    
    # Critical files get higher value
    if any(critical in file_path for critical in ['api', 'security', 'auth', 'core']):
        base_value = 8
    elif any(important in file_path for important in ['service', 'model', 'controller']):
        base_value = 5
    else:
        base_value = 3
        
    # Security-related todos are critical
    if any(word in todo_lower for word in ['security', 'vulnerability', 'auth', 'permission']):
        return min(13, base_value + 5)
    elif any(word in todo_lower for word in ['performance', 'memory', 'leak']):
        return min(13, base_value + 3)
    elif any(word in todo_lower for word in ['error', 'exception', 'crash']):
        return min(13, base_value + 2)
    else:
        return base_value


def estimate_criticality(todo_text: str) -> int:
    """Estimate time criticality on 1-13 scale"""
    todo_lower = todo_text.lower()
    
    if any(word in todo_lower for word in ['urgent', 'asap', 'critical', 'blocker']):
        return 13
    elif any(word in todo_lower for word in ['soon', 'important', 'needed']):
        return 8
    elif any(word in todo_lower for word in ['security', 'vulnerability', 'auth']):
        return 13
    elif any(word in todo_lower for word in ['bug', 'error', 'fix']):
        return 5
    else:
        return 2


def estimate_risk_reduction(todo_text: str) -> int:
    """Estimate risk reduction on 1-13 scale"""
    todo_lower = todo_text.lower()
    
    if any(word in todo_lower for word in ['security', 'vulnerability', 'injection', 'xss']):
        return 13
    elif any(word in todo_lower for word in ['validation', 'sanitization', 'escape']):
        return 8
    elif any(word in todo_lower for word in ['error handling', 'exception', 'crash']):
        return 5
    elif any(word in todo_lower for word in ['test', 'testing', 'coverage']):
        return 3
    else:
        return 1


def calculate_wsjf(value: int, criticality: int, risk_reduction: int, effort: int) -> float:
    """Calculate WSJF score"""
    if effort == 0:
        effort = 1
    return (value + criticality + risk_reduction) / effort


def discover_failing_tests() -> List[DiscoveredTask]:
    """Discover failing tests by running pytest with minimal output"""
    tasks = []
    try:
        import subprocess
        result = subprocess.run(['python3', '-m', 'pytest', '--tb=no', '-q'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            # Parse failed tests from output
            lines = result.stdout.split('\n')
            for line in lines:
                if '::' in line and 'FAILED' in line:
                    test_path = line.split()[0] if line.split() else ""
                    task = DiscoveredTask(
                        id=f"test_fail_{hash(test_path) % 10000}",
                        title=f"Fix failing test: {test_path}",
                        description=f"Test failure detected in {test_path}",
                        type="Bug",
                        file_path=test_path.split('::')[0] if '::' in test_path else test_path,
                        line_number=0,
                        context=line,
                        effort=3,
                        value=8,
                        time_criticality=8,
                        risk_reduction=5,
                        wsjf_score=7.0,
                        status="READY"
                    )
                    tasks.append(task)
    except Exception as e:
        print(f"Could not run tests: {e}")
    
    return tasks


def discover_type_errors() -> List[DiscoveredTask]:
    """Discover type errors by running mypy"""
    tasks = []
    try:
        import subprocess
        result = subprocess.run(['python3', '-m', 'mypy', 'src/', '--ignore-missing-imports'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if ':' in line and 'error:' in line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1] if parts[1].isdigit() else "0"
                        error_msg = ':'.join(parts[2:]).strip()
                        
                        task = DiscoveredTask(
                            id=f"type_error_{hash(line) % 10000}",
                            title=f"Fix type error in {os.path.basename(file_path)}:{line_num}",
                            description=f"Type error: {error_msg}",
                            type="Bug",
                            file_path=file_path,
                            line_number=int(line_num) if line_num.isdigit() else 0,
                            context=line,
                            effort=2,
                            value=5,
                            time_criticality=3,
                            risk_reduction=3,
                            wsjf_score=5.5,
                            status="READY"
                        )
                        tasks.append(task)
    except Exception as e:
        print(f"Could not run mypy: {e}")
    
    return tasks


def main():
    """Main discovery function"""
    print("🔍 Discovering tasks from codebase...")
    
    all_tasks = []
    
    # Discover TODO/FIXME comments
    print("📝 Scanning for TODO/FIXME comments...")
    todo_tasks = scan_for_todos_fixmes()
    all_tasks.extend(todo_tasks)
    print(f"Found {len(todo_tasks)} TODO/FIXME items")
    
    # Discover failing tests
    print("🧪 Checking for failing tests...")
    test_tasks = discover_failing_tests()
    all_tasks.extend(test_tasks)
    print(f"Found {len(test_tasks)} failing tests")
    
    # Discover type errors
    print("🔍 Checking for type errors...")
    type_tasks = discover_type_errors()
    all_tasks.extend(type_tasks)
    print(f"Found {len(type_tasks)} type errors")
    
    # Sort by WSJF score (highest first)
    all_tasks.sort(key=lambda t: t.wsjf_score, reverse=True)
    
    # Generate report
    timestamp = datetime.now().isoformat()
    report = {
        "timestamp": timestamp,
        "total_tasks": len(all_tasks),
        "tasks_by_type": {
            "Bug": len([t for t in all_tasks if t.type == "Bug"]),
            "Refactor": len([t for t in all_tasks if t.type == "Refactor"]),
        },
        "top_priority_tasks": [
            {
                "id": task.id,
                "title": task.title,
                "type": task.type,
                "wsjf_score": task.wsjf_score,
                "file_path": task.file_path,
                "description": task.description
            }
            for task in all_tasks[:10]
        ],
        "all_tasks": [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "type": task.type,
                "file_path": task.file_path,
                "line_number": task.line_number,
                "effort": task.effort,
                "value": task.value,
                "time_criticality": task.time_criticality,
                "risk_reduction": task.risk_reduction,
                "wsjf_score": task.wsjf_score,
                "status": task.status
            }
            for task in all_tasks
        ]
    }
    
    # Save report
    os.makedirs("docs/status", exist_ok=True)
    report_file = f"docs/status/discovered_backlog_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Discovery Complete!")
    print(f"Total tasks found: {len(all_tasks)}")
    print(f"Report saved to: {report_file}")
    
    if all_tasks:
        print(f"\n🏆 Top 5 Priority Items (by WSJF):")
        for i, task in enumerate(all_tasks[:5], 1):
            print(f"{i}. {task.title} (WSJF: {task.wsjf_score:.1f})")
            print(f"   Type: {task.type}, File: {task.file_path}")
            print(f"   Description: {task.description}")
            print()
    
    return all_tasks


if __name__ == "__main__":
    main()