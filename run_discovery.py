#!/usr/bin/env python3
"""
Simple discovery script to demonstrate the autonomous backlog system
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from logger import get_logger
from task_prioritization import TaskPrioritizer, calculate_wsjf_score

logger = get_logger(__name__)


async def discover_and_prioritize_backlog():
    """Discover and prioritize all backlog items"""
    
    print("🤖 AUTONOMOUS BACKLOG MANAGEMENT SYSTEM")
    print("=" * 60)
    print("🔍 DISCOVERING BACKLOG ITEMS...")
    
    # Initialize components
    prioritizer = TaskPrioritizer()
    all_tasks = []
    
    # 1. Discover TODO/FIXME comments
    print("\n📝 Scanning for TODO/FIXME comments...")
    todos = discover_todo_comments()
    all_tasks.extend(todos)
    print(f"Found {len(todos)} TODO/FIXME items")
    
    # 2. Discover from backlog markdown
    print("\n📋 Scanning BACKLOG.md...")
    backlog_tasks = discover_backlog_md_tasks()
    all_tasks.extend(backlog_tasks)
    print(f"Found {len(backlog_tasks)} tasks from BACKLOG.md")
    
    # 3. Score all tasks using WSJF
    print("\n📊 SCORING TASKS WITH WSJF...")
    for task in all_tasks:
        if 'wsjf_score' not in task:
            task['wsjf_score'] = calculate_wsjf_score(
                business_value=task.get('business_value', 3),
                time_criticality=task.get('time_criticality', 2), 
                risk_reduction=task.get('risk_reduction', 2),
                job_size=task.get('job_size', 3)
            )
    
    # 4. Sort by priority
    prioritized_tasks = sorted(all_tasks, key=lambda x: x['wsjf_score'], reverse=True)
    
    # 5. Display results
    print(f"\n🎯 PRIORITIZED BACKLOG ({len(prioritized_tasks)} total items)")
    print("=" * 60)
    
    # Show top 20 items
    for i, task in enumerate(prioritized_tasks[:20], 1):
        status_emoji = get_status_emoji(task)
        print(f"{i:2d}. {status_emoji} [{task['wsjf_score']:5.1f}] {task['title']}")
        print(f"    📁 {task.get('file_path', 'N/A')} | Type: {task.get('type', 'Unknown')}")
        
    if len(prioritized_tasks) > 20:
        print(f"\n... and {len(prioritized_tasks) - 20} more items")
    
    # 6. Save discovery results
    save_discovery_results(prioritized_tasks)
    
    # 7. Analysis summary
    print(f"\n📈 BACKLOG ANALYSIS")
    print("=" * 60)
    print_backlog_statistics(prioritized_tasks)
    
    return prioritized_tasks


def discover_todo_comments():
    """Scan codebase for TODO/FIXME comments"""
    import os
    
    todos = []
    for root, dirs, files in os.walk("src"):
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
                            # Classify TODO type and priority
                            todo_type = classify_todo_type(line)
                            priority_scores = estimate_todo_priority(line, file_path)
                            
                            todos.append({
                                'id': f"todo-{hash(f'{file_path}:{line_num}')}", 
                                'title': line.strip().replace('#', '').replace('//', '').strip(),
                                'description': f"TODO comment in {file_path}:{line_num}",
                                'type': todo_type,
                                'file_path': file_path,
                                'line_number': line_num,
                                'status': 'READY',
                                **priority_scores  # business_value, time_criticality, etc.
                            })
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
                    
    return todos


def discover_backlog_md_tasks():
    """Extract tasks from BACKLOG.md"""
    backlog_tasks = []
    
    try:
        with open("BACKLOG.md", 'r') as f:
            content = f.read()
            
        # Simple parser for markdown sections
        lines = content.split('\n')
        current_task = None
        
        for line in lines:
            # Look for task headers with WSJF scores
            if 'WSJF Score:' in line and '###' in line:
                if current_task:
                    backlog_tasks.append(current_task)
                    
                # Extract WSJF score from line
                try:
                    wsjf_text = line.split('WSJF Score:')[1].split('|')[0].strip()
                    wsjf_score = float(wsjf_text)
                except:
                    wsjf_score = 3.0
                    
                # Extract title
                title = line.split('###')[1].split('**WSJF')[0].strip()
                if title.startswith('✅'):
                    continue  # Skip completed tasks
                    
                current_task = {
                    'id': f"backlog-{hash(title)}",
                    'title': title,
                    'wsjf_score': wsjf_score,
                    'type': 'Backlog Item',
                    'status': 'READY',
                    'source': 'BACKLOG.md'
                }
                
        if current_task:
            backlog_tasks.append(current_task)
            
    except Exception as e:
        logger.warning(f"Could not read BACKLOG.md: {e}")
        
    return backlog_tasks


def classify_todo_type(line):
    """Classify TODO comment type"""
    line_lower = line.lower()
    
    if any(word in line_lower for word in ['security', 'auth', 'password', 'token', 'inject']):
        return 'Security'
    elif any(word in line_lower for word in ['bug', 'fix', 'error', 'issue']):
        return 'Bug'
    elif any(word in line_lower for word in ['performance', 'optimize', 'slow', 'memory']):
        return 'Performance'
    elif any(word in line_lower for word in ['test', 'coverage', 'mock', 'assert']):
        return 'Testing'
    elif any(word in line_lower for word in ['refactor', 'clean', 'improve', 'simplify']):
        return 'Refactoring'
    else:
        return 'Feature'


def estimate_todo_priority(line, file_path):
    """Estimate WSJF components for TODO"""
    line_lower = line.lower()
    
    # Business value based on file importance and content
    business_value = 3  # Default
    if 'security' in file_path or any(word in line_lower for word in ['critical', 'important', 'urgent']):
        business_value = 5
    elif 'test' in file_path:
        business_value = 2
        
    # Time criticality based on keywords
    time_criticality = 2  # Default
    if any(word in line_lower for word in ['urgent', 'asap', 'now', 'immediately']):
        time_criticality = 5
    elif any(word in line_lower for word in ['soon', 'quick', 'simple']):
        time_criticality = 3
        
    # Risk reduction
    risk_reduction = 2  # Default
    if any(word in line_lower for word in ['security', 'vulnerability', 'bug', 'error']):
        risk_reduction = 4
        
    # Job size estimate  
    job_size = 2  # Default (small)
    if any(word in line_lower for word in ['refactor', 'rewrite', 'major', 'complex']):
        job_size = 4
    elif any(word in line_lower for word in ['simple', 'quick', 'small']):
        job_size = 1
        
    return {
        'business_value': business_value,
        'time_criticality': time_criticality, 
        'risk_reduction': risk_reduction,
        'job_size': job_size
    }


def get_status_emoji(task):
    """Get emoji for task status/type"""
    task_type = task.get('type', '').lower()
    if 'security' in task_type:
        return '🔒'
    elif 'bug' in task_type:
        return '🐛'
    elif 'performance' in task_type:
        return '⚡'
    elif 'test' in task_type:
        return '🧪'
    elif 'refactor' in task_type:
        return '♻️'
    else:
        return '✨'


def save_discovery_results(tasks):
    """Save discovery results to file"""
    results = {
        'timestamp': str(Path().cwd()),
        'total_tasks': len(tasks),
        'tasks': tasks
    }
    
    output_file = Path("docs/status") / f"discovered_backlog_{Path().cwd().name}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\n💾 Results saved to: {output_file}")


def print_backlog_statistics(tasks):
    """Print backlog analysis statistics"""
    
    # Type distribution
    type_counts = {}
    for task in tasks:
        task_type = task.get('type', 'Unknown')
        type_counts[task_type] = type_counts.get(task_type, 0) + 1
    
    print("📊 Task Type Distribution:")
    for task_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(tasks)) * 100
        print(f"   {task_type:15} {count:3d} ({percentage:5.1f}%)")
    
    # Priority distribution  
    high_priority = [t for t in tasks if t['wsjf_score'] >= 7.0]
    medium_priority = [t for t in tasks if 4.0 <= t['wsjf_score'] < 7.0]
    low_priority = [t for t in tasks if t['wsjf_score'] < 4.0]
    
    print(f"\n🎯 Priority Distribution:")
    print(f"   High Priority (≥7.0):   {len(high_priority):3d} ({len(high_priority)/len(tasks)*100:5.1f}%)")
    print(f"   Medium Priority (4-7):   {len(medium_priority):3d} ({len(medium_priority)/len(tasks)*100:5.1f}%)")
    print(f"   Low Priority (<4.0):     {len(low_priority):3d} ({len(low_priority)/len(tasks)*100:5.1f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if len(high_priority) > 0:
        print(f"   • Start with {min(5, len(high_priority))} high-priority items")
    if len([t for t in tasks if t.get('type') == 'Security']) > 0:
        print(f"   • Address security items immediately")
    if len(tasks) > 50:
        print(f"   • Consider breaking large tasks into smaller pieces")
    print(f"   • Use autonomous execution: python3 autonomous_backlog_manager.py --dry-run")


async def main():
    """Main entry point"""
    try:
        prioritized_backlog = await discover_and_prioritize_backlog()
        
        print(f"\n🎉 DISCOVERY COMPLETE!")
        print(f"Found {len(prioritized_backlog)} actionable items ready for autonomous execution.")
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())