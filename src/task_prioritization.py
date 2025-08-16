"""
Task Prioritization System for Claude Manager Service

This module provides intelligent prioritization of discovered tasks based on:
- Business impact assessment
- Complexity analysis  
- Urgency evaluation
- Task type classification

Features:
- WSJF-inspired scoring algorithm
- Context-aware priority calculation
- Integration with existing task discovery
- Configurable priority weights
"""
import re
import math
from typing import Dict, List, Any, Optional
from src.logger import get_logger

logger = get_logger(__name__)

# Priority weights for different factors
PRIORITY_WEIGHTS = {
    'business_impact': 0.35,
    'urgency': 0.30,
    'complexity': 0.20,
    'task_type': 0.15
}

# Task type priority multipliers
TASK_TYPE_PRIORITIES = {
    'security': 10.0,
    'bug': 8.0,
    'performance': 6.0,
    'testing': 5.0,
    'refactoring': 4.0,
    'feature': 4.0,
    'documentation': 2.0,
    'cleanup': 2.0,
    'unknown': 3.0
}

# File path importance weights (higher = more critical to business)
FILE_IMPORTANCE = {
    # Core business logic
    'auth': 10, 'security': 10, 'payment': 10, 'billing': 10,
    'api': 9, 'github_api': 9, 'orchestrator': 9,
    'database': 8, 'models': 8, 'data': 8,
    'config': 7, 'settings': 7,
    'utils': 5, 'helpers': 5,
    'tests': 4, 'test': 4,
    'docs': 2, 'documentation': 2, 'readme': 2,
    'scripts': 3, 'tools': 3, 'dev': 3
}

# Complexity indicators and their weights
COMPLEXITY_KEYWORDS = {
    # High complexity (4-6 points) - Reduced to prevent over-scoring
    'sql': 5, 'injection': 6, 'vulnerability': 6, 'security': 5,
    'authentication': 4, 'authorization': 4, 'encryption': 4,
    'database': 4, 'migration': 4, 'transaction': 4,
    'async': 4, 'threading': 5, 'concurrency': 5,
    'algorithm': 4, 'optimization': 3,
    
    # Medium complexity (2-3 points)  
    'api': 3, 'integration': 3, 'external': 2,
    'retry': 2, 'timeout': 2, 'error handling': 3,
    'validation': 2, 'parsing': 2, 'regex': 3,
    'refactor': 2, 'cleanup': 1,
    
    # Low complexity (0.5-1 points)
    'logging': 1, 'formatting': 1, 'indentation': 0.5,
    'documentation': 1, 'comment': 0.5, 'readme': 1,
    'typo': 0.5, 'spelling': 0.5
}

# Urgency keywords and their weights
URGENCY_KEYWORDS = {
    # Critical urgency (8-10 points)
    'critical': 10, 'urgent': 9, 'blocking': 10, 'broken': 9,
    'failure': 8, 'crash': 9, 'error': 7, 'exception': 7,
    'bug': 8, 'issue': 6, 'problem': 6,
    'security': 9, 'vulnerability': 10,
    
    # Medium urgency (4-7 points)
    'slow': 6, 'performance': 4, 'optimization': 3,  # Reduced for optimization tasks
    'improvement': 4, 'enhance': 4, 'update': 4,
    'missing': 5, 'incomplete': 5,
    
    # Low urgency (1-3 points)
    'documentation': 3, 'comment': 2, 'cleanup': 3,
    'refactor': 3, 'todo': 2, 'consider': 2,
    'maybe': 1, 'future': 1, 'later': 1
}


def classify_task_type(content: str, file_path: str) -> str:
    """
    Classify the type of task based on content and file path
    
    Args:
        content (str): Task content/description
        file_path (str): File path where task was found
        
    Returns:
        str: Task type classification
    """
    content_lower = content.lower()
    file_lower = file_path.lower()
    
    # Security tasks
    security_indicators = ['security', 'vulnerability', 'injection', 'xss', 'csrf', 
                          'auth', 'password', 'token', 'encryption', 'ssl']
    if any(keyword in content_lower for keyword in security_indicators):
        return 'security'
    
    # Performance tasks
    performance_indicators = ['performance', 'slow', 'optimization', 'optimize', 
                             'cache', 'memory', 'speed', 'efficiency']
    if any(keyword in content_lower for keyword in performance_indicators):
        return 'performance'
    
    # Bug fixes
    bug_indicators = ['bug', 'fix', 'broken', 'error', 'exception', 'crash', 
                     'issue', 'problem', 'incorrect', 'wrong']
    if any(keyword in content_lower for keyword in bug_indicators) or content.startswith('FIXME'):
        return 'bug'
    
    # Testing tasks
    test_indicators = ['test', 'testing', 'coverage', 'mock', 'unittest', 'integration']
    if (any(keyword in content_lower for keyword in test_indicators) or 
        'test' in file_lower):
        return 'testing'
    
    # Documentation tasks
    doc_indicators = ['documentation', 'document', 'readme', 'docs', 'comment', 'docstring']
    if (any(keyword in content_lower for keyword in doc_indicators) or
        file_lower.endswith('.md') or 'readme' in file_lower):
        return 'documentation'
    
    # Refactoring tasks
    refactor_indicators = ['refactor', 'cleanup', 'reorganize', 'simplify', 'improve']
    if any(keyword in content_lower for keyword in refactor_indicators):
        return 'refactoring'
    
    # Feature development
    feature_indicators = ['add', 'implement', 'create', 'build', 'develop', 'feature']
    if any(keyword in content_lower for keyword in feature_indicators):
        return 'feature'
    
    # Cleanup tasks
    cleanup_indicators = ['cleanup', 'remove', 'delete', 'unused', 'deprecated', 'dead code']
    if any(keyword in content_lower for keyword in cleanup_indicators):
        return 'cleanup'
    
    return 'unknown'


def analyze_task_complexity(content: str, file_path: str) -> float:
    """
    Analyze the complexity of a task based on content and context
    
    Args:
        content (str): Task description
        file_path (str): File path where task was found
        
    Returns:
        float: Complexity score (1-10, higher = more complex)
    """
    content_lower = content.lower()
    complexity_score = 0.0
    
    # Base complexity from keywords - cap to prevent stacking
    keyword_score = 0.0
    keywords_found = []
    for keyword, weight in COMPLEXITY_KEYWORDS.items():
        if keyword in content_lower:
            keyword_score += weight
            keywords_found.append(keyword)
            logger.debug(f"Found complexity keyword '{keyword}' (+{weight})")
    
    # Cap keyword contribution to prevent excessive stacking
    complexity_score += min(keyword_score, 5.5)  # Max 5.5 points from keywords - balanced
    
    # File path complexity modifiers - reduced to prevent over-penalization
    file_lower = file_path.lower()
    for path_indicator, importance in FILE_IMPORTANCE.items():
        if path_indicator in file_lower:
            # Core files add to complexity because changes are riskier
            if importance >= 8:
                complexity_score += 1.0  # Further reduced
            elif importance >= 6:
                complexity_score += 0.4  # Further reduced
            break
    
    # Content length indicator (longer descriptions often indicate complexity)
    if len(content) > 100:
        complexity_score += 1
    if len(content) > 200:
        complexity_score += 1
        
    # Multiple sentences indicate more complex task
    sentence_count = len([s for s in content.split('.') if s.strip()])
    if sentence_count > 2:
        complexity_score += min(sentence_count - 2, 3)
    
    # Technical terms increase complexity
    technical_terms = ['algorithm', 'database', 'api', 'async', 'concurrent', 
                      'distributed', 'microservice', 'protocol', 'framework']
    technical_count = sum(1 for term in technical_terms if term in content_lower)
    complexity_score += technical_count * 0.3  # Reduced from 0.5
    
    # Security boost for high-complexity security tasks
    security_terms = ['sql', 'injection', 'vulnerability', 'security', 'authentication']
    security_count = sum(1 for term in security_terms if term in content_lower)
    if security_count >= 3:  # Multiple security indicators
        complexity_score += 1.0  # Security complexity boost
    
    # Normalize to 1-10 scale
    return min(max(complexity_score, 1.0), 10.0)


def determine_business_impact(content: str, file_path: str) -> float:
    """
    Determine the business impact of a task
    
    Args:
        content (str): Task description
        file_path (str): File path where task was found
        
    Returns:
        float: Business impact score (1-10, higher = more impact)
    """
    content_lower = content.lower()
    
    # Base impact from file importance
    file_lower = file_path.lower()
    base_impact = 5.0  # Default medium impact
    
    for path_indicator, importance in FILE_IMPORTANCE.items():
        if path_indicator in file_lower:
            base_impact = importance
            break
    
    # Impact modifiers based on content
    impact_modifiers = 0.0
    
    # High impact keywords
    high_impact_terms = ['outage', 'downtime', 'failure', 'crash', 'critical',
                        'production', 'customer', 'user', 'revenue', 'business']
    for term in high_impact_terms:
        if term in content_lower:
            impact_modifiers += 2.0
    
    # Medium impact keywords  
    medium_impact_terms = ['performance', 'slow', 'optimization', 'experience',
                          'reliability', 'availability', 'scalability']
    for term in medium_impact_terms:
        if term in content_lower:
            impact_modifiers += 0.8  # Reduced from 1.0
    
    # Security issues always have high business impact
    security_terms = ['security', 'vulnerability', 'injection', 'breach', 'exploit']
    if any(term in content_lower for term in security_terms):
        impact_modifiers += 3.0
    
    # Integration and API issues affect multiple systems
    integration_terms = ['api', 'integration', 'external', 'service', 'endpoint']
    if any(term in content_lower for term in integration_terms):
        impact_modifiers += 1.5
    
    total_impact = base_impact + impact_modifiers
    return min(max(total_impact, 1.0), 10.0)


def assess_urgency(content: str) -> float:
    """
    Assess the urgency of a task based on keywords and context
    
    Args:
        content (str): Task description
        
    Returns:
        float: Urgency score (1-10, higher = more urgent)
    """
    content_lower = content.lower()
    urgency_score = 0.0
    
    # Base urgency from keywords - cap to prevent stacking
    keyword_urgency = 0.0
    for keyword, weight in URGENCY_KEYWORDS.items():
        if keyword in content_lower:
            keyword_urgency += weight
            logger.debug(f"Found urgency keyword '{keyword}' (+{weight})")
    
    # Cap keyword contribution to handle overlapping keywords  
    urgency_score += min(keyword_urgency, 4.0)  # Max 4.0 points from keywords
    
    # FIXME indicates higher urgency than TODO (4.0 vs 1.0)
    # FIXME suggests something is broken and needs immediate attention
    # TODO suggests planned future work with lower time criticality
    if content.startswith('FIXME'):
        urgency_score += 4.0  # Increased to compensate for lower keyword cap
    elif content.startswith('TODO'):
        urgency_score += 1.0
    
    # Exclamation marks indicate urgency
    exclamation_count = content.count('!')
    urgency_score += min(exclamation_count * 0.5, 2.0)
    
    # ALL CAPS words indicate urgency
    caps_words = len([word for word in content.split() if word.isupper() and len(word) > 2])
    urgency_score += min(caps_words * 0.5, 2.0)
    
    # Time-sensitive keywords
    time_sensitive = ['asap', 'immediately', 'now', 'urgent', 'deadline', 'before']
    for term in time_sensitive:
        if term in content_lower:
            urgency_score += 2.0
    
    # Default minimum urgency
    if urgency_score == 0:
        urgency_score = 2.0
    
    # Normalize to 1-10 scale
    return min(max(urgency_score, 1.0), 10.0)


def calculate_task_priority(task_data: Dict[str, Any]) -> float:
    """
    Calculate overall priority score for a task using WSJF-inspired methodology
    
    Args:
        task_data (dict): Task information including type, content, file_path, etc.
        
    Returns:
        float: Priority score (1-10, higher = higher priority)
    """
    task_type = task_data.get('type', 'unknown')
    content = task_data.get('content', '')
    file_path = task_data.get('file_path', '')
    
    # Get individual scores
    business_impact = determine_business_impact(content, file_path)
    urgency = assess_urgency(content)
    complexity = analyze_task_complexity(content, file_path)
    
    # Task type multiplier
    type_multiplier = TASK_TYPE_PRIORITIES.get(task_type, TASK_TYPE_PRIORITIES['unknown'])
    
    # For security and critical tasks, use a base priority that ensures high scores
    if task_type == 'security':
        base_priority = 8.5  # Start with high base for security
        # Adjust based on actual values
        value_multiplier = (business_impact + urgency) / 20.0  # Scale 0-1
        complexity_penalty = min(complexity / 15.0, 0.5)  # Cap complexity penalty
        priority_score = base_priority + value_multiplier - complexity_penalty
        
    elif task_type == 'bug':
        base_priority = 7.0  # High base for bugs
        value_multiplier = (business_impact + urgency) / 20.0
        complexity_penalty = min(complexity / 12.0, 0.8)
        priority_score = base_priority + value_multiplier - complexity_penalty
        
    else:
        # For other task types, use adjusted weighted formula to meet expected ranges
        value_score = (
            business_impact * PRIORITY_WEIGHTS['business_impact'] +
            urgency * PRIORITY_WEIGHTS['urgency'] +
            type_multiplier * PRIORITY_WEIGHTS['task_type']
        ) * 1.4  # Scale up to ensure proper ranges
        
        # Normalize complexity as job size (higher complexity = larger job = lower priority)
        complexity_factor = complexity * PRIORITY_WEIGHTS['complexity']
        
        # Final priority calculation with adjusted divisor
        priority_score = value_score / (1 + complexity_factor / 12)
    
    logger.debug(f"Priority calculation for {task_type}: "
                f"business_impact={business_impact:.2f}, urgency={urgency:.2f}, "
                f"complexity={complexity:.2f}, type_multiplier={type_multiplier:.2f}, "
                f"final_score={priority_score:.2f}")
    
    return min(max(priority_score, 1.0), 10.0)


def prioritize_discovered_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prioritize a list of discovered tasks
    
    Args:
        tasks (list): List of task dictionaries
        
    Returns:
        list: Tasks sorted by priority (highest first) with priority information added
    """
    logger.info(f"Prioritizing {len(tasks)} discovered tasks")
    
    prioritized_tasks = []
    
    for task in tasks:
        # Classify task type if not already set
        if 'type' not in task:
            task['type'] = classify_task_type(
                task.get('content', ''), 
                task.get('file_path', '')
            )
        
        # Calculate priority score
        priority_score = calculate_task_priority(task)
        
        # Generate priority explanation
        task_type = task['type']
        priority_reason = _generate_priority_reason(task, priority_score)
        
        # Add priority information to task
        enhanced_task = task.copy()
        enhanced_task.update({
            'priority_score': priority_score,
            'priority_reason': priority_reason,
            'task_type': task_type
        })
        
        prioritized_tasks.append(enhanced_task)
        
        logger.debug(f"Task '{task.get('id', 'unknown')}' assigned priority {priority_score:.2f} ({task_type})")
    
    # Sort by priority score (highest first)
    prioritized_tasks.sort(key=lambda x: x['priority_score'], reverse=True)
    
    logger.info(f"Task prioritization complete. Highest priority: {prioritized_tasks[0]['priority_score']:.2f}, "
               f"Lowest priority: {prioritized_tasks[-1]['priority_score']:.2f}")
    
    return prioritized_tasks


def enhance_task_with_priority(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance a single task with priority information
    
    Args:
        task (dict): Task dictionary
        
    Returns:
        dict: Enhanced task with priority information
    """
    return prioritize_discovered_tasks([task])[0]


def calculate_wsjf_score(business_value: float, time_criticality: float, risk_reduction: float, job_size: float) -> float:
    """
    Calculate WSJF (Weighted Shortest Job First) score
    
    Args:
        business_value: Business value score (1-10)
        time_criticality: Time criticality score (1-10)
        risk_reduction: Risk reduction score (1-10)
        job_size: Job size/effort score (1-10)
        
    Returns:
        WSJF score
    """
    if job_size <= 0:
        job_size = 1.0  # Prevent division by zero
    
    cost_of_delay = business_value + time_criticality + risk_reduction
    return cost_of_delay / job_size


class TaskPrioritizer:
    """
    Task prioritization system using WSJF methodology
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.TaskPrioritizer")
    
    def prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize a list of tasks using WSJF scoring
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            Prioritized tasks with WSJF scores
        """
        return prioritize_discovered_tasks(tasks)
    
    def calculate_priority(self, task: Dict[str, Any]) -> float:
        """
        Calculate priority score for a single task
        
        Args:
            task: Task dictionary
            
        Returns:
            Priority score
        """
        return calculate_task_priority(task)
    
    def get_task_classification(self, content: str, file_path: str) -> str:
        """
        Get task type classification
        
        Args:
            content: Task content
            file_path: File path
            
        Returns:
            Task type
        """
        return classify_task_type(content, file_path)


def _generate_priority_reason(task: Dict[str, Any], priority_score: float) -> str:
    """
    Generate a human-readable explanation for the priority score
    
    Args:
        task (dict): Task information
        priority_score (float): Calculated priority score
        
    Returns:
        str: Human-readable priority explanation
    """
    task_type = task.get('type', 'unknown')
    content = task.get('content', '')
    file_path = task.get('file_path', '')
    
    reasons = []
    
    # Task type reasoning
    if task_type == 'security':
        reasons.append("Security issue (highest priority)")
    elif task_type == 'bug':
        reasons.append("Bug fix (high priority)")
    elif task_type == 'performance':
        reasons.append("Performance improvement")
    elif task_type == 'testing':
        reasons.append("Testing enhancement")
    else:
        reasons.append(f"{task_type.title()} task")
    
    # File importance
    file_lower = file_path.lower()
    for path_indicator, importance in FILE_IMPORTANCE.items():
        if path_indicator in file_lower and importance >= 8:
            reasons.append(f"in critical {path_indicator} module")
            break
        elif path_indicator in file_lower and importance >= 6:
            reasons.append(f"in core {path_indicator} module")
            break
    
    # Urgency indicators
    content_lower = content.lower()
    if any(urgent in content_lower for urgent in ['critical', 'urgent', 'blocking']):
        reasons.append("with urgent keywords")
    elif any(security in content_lower for security in ['security', 'vulnerability']):
        reasons.append("with security implications")
    
    # Priority level description
    if priority_score >= 8.0:
        priority_level = "Very High"
    elif priority_score >= 6.0:
        priority_level = "High"
    elif priority_score >= 4.0:
        priority_level = "Medium"
    else:
        priority_level = "Low"
    
    reason_text = f"{priority_level} priority: {' '.join(reasons)}"
    return reason_text


# Example usage and testing
if __name__ == "__main__":
    # Test the prioritization system
    test_tasks = [
        {
            'id': 'task1',
            'content': 'TODO: Fix SQL injection vulnerability in user authentication',
            'file_path': 'src/auth.py',
            'line_number': 42
        },
        {
            'id': 'task2',
            'content': 'TODO: Add unit tests for utility functions',
            'file_path': 'src/utils.py',
            'line_number': 15
        },
        {
            'id': 'task3',
            'content': 'TODO: Update README documentation',
            'file_path': 'README.md',
            'line_number': 1
        }
    ]
    
    prioritized = prioritize_discovered_tasks(test_tasks)
    
    print("Prioritized Tasks:")
    for task in prioritized:
        print(f"  {task['id']}: {task['priority_score']:.2f} - {task['priority_reason']}")