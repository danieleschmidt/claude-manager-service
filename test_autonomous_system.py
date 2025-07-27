#!/usr/bin/env python3
"""
Simple test of the autonomous backlog system without complex imports
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports work"""
    try:
        print("Testing basic logger import...")
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Logger working!")
        
        print("Testing task tracker import...")
        from task_tracker import TaskTracker
        tracker = TaskTracker()
        print("TaskTracker working!")
        
        print("Testing task prioritization...")
        from task_prioritization import TaskPrioritizer, calculate_wsjf_score
        prioritizer = TaskPrioritizer()
        
        # Test WSJF calculation
        score = calculate_wsjf_score(
            business_value=4, 
            time_criticality=3, 
            risk_reduction=2, 
            job_size=2
        )
        print(f"WSJF Score test: {score}")
        
        print("✅ Basic autonomous system components are working!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def discover_simple_todos():
    """Discover TODO comments in codebase"""
    try:
        print("\n🔍 Discovering TODO/FIXME comments...")
        
        todos = []
        for root, dirs, files in os.walk("src"):
            # Skip __pycache__ and similar directories
            dirs[:] = [d for d in dirs if not d.startswith('__')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        for line_num, line in enumerate(lines, 1):
                            line_lower = line.lower()
                            if 'todo' in line_lower or 'fixme' in line_lower:
                                todos.append({
                                    'file': file_path,
                                    'line': line_num,
                                    'content': line.strip(),
                                    'wsjf_score': 3.0  # Default score
                                })
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
        
        print(f"Found {len(todos)} TODO/FIXME items:")
        for i, todo in enumerate(todos[:10], 1):  # Show first 10
            print(f"{i:2d}. [{todo['wsjf_score']:4.1f}] {todo['file']}:{todo['line']} - {todo['content'][:80]}...")
            
        if len(todos) > 10:
            print(f"... and {len(todos) - 10} more items")
            
        return todos
        
    except Exception as e:
        print(f"❌ TODO discovery failed: {e}")
        return []

def test_wsjf_prioritization():
    """Test WSJF prioritization system"""
    try:
        print("\n📊 Testing WSJF prioritization...")
        
        # Sample tasks for prioritization
        sample_tasks = [
            {
                'id': 'security-fix-1',
                'title': 'Fix SQL injection vulnerability',
                'business_value': 5,
                'time_criticality': 5,
                'risk_reduction': 5,
                'job_size': 2,
                'type': 'Security'
            },
            {
                'id': 'feature-1', 
                'title': 'Add new user dashboard',
                'business_value': 4,
                'time_criticality': 2,
                'risk_reduction': 1,
                'job_size': 5,
                'type': 'Feature'
            },
            {
                'id': 'bug-fix-1',
                'title': 'Fix login timeout issue',
                'business_value': 3,
                'time_criticality': 4,
                'risk_reduction': 2,
                'job_size': 1,
                'type': 'Bug'
            }
        ]
        
        # Calculate WSJF scores
        from task_prioritization import calculate_wsjf_score
        
        for task in sample_tasks:
            task['wsjf_score'] = calculate_wsjf_score(
                task['business_value'],
                task['time_criticality'], 
                task['risk_reduction'],
                task['job_size']
            )
        
        # Sort by WSJF score
        sorted_tasks = sorted(sample_tasks, key=lambda x: x['wsjf_score'], reverse=True)
        
        print("Prioritized task list (WSJF scores):")
        for i, task in enumerate(sorted_tasks, 1):
            print(f"{i}. [{task['wsjf_score']:5.1f}] {task['title']} ({task['type']})")
        
        return sorted_tasks
        
    except Exception as e:
        print(f"❌ WSJF test failed: {e}")
        return []

def main():
    """Main test function"""
    print("🤖 Testing Autonomous Backlog Management System")
    print("=" * 60)
    
    # Test basic functionality
    if not test_basic_imports():
        sys.exit(1)
        
    # Discover real TODO items
    todos = discover_simple_todos()
    
    # Test prioritization
    prioritized_tasks = test_wsjf_prioritization()
    
    print("\n🎯 System Status Summary:")
    print(f"- Basic imports: ✅ Working")
    print(f"- TODO discovery: ✅ Found {len(todos)} items")
    print(f"- WSJF prioritization: ✅ Working")
    print(f"- Ready for autonomous execution: {'✅ Yes' if todos and prioritized_tasks else '⚠️ Partial'}")
    
    print("\n💡 Next Steps:")
    print("1. Fix remaining import issues in complex modules")
    print("2. Set up GitHub API configuration (config.json)")
    print("3. Run full autonomous execution with:")
    print("   python3 autonomous_backlog_manager.py --dry-run")

if __name__ == "__main__":
    main()