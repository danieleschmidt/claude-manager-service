"""
Tests for task prioritization system
"""
import pytest
from unittest.mock import Mock, patch
import sys

sys.path.append('/root/repo/src')


class TestTaskPrioritization:
    """Test cases for task prioritization functionality"""

    def test_calculate_task_priority_basic(self):
        """Test basic priority calculation for different task types"""
        from task_prioritization import calculate_task_priority
        
        # High priority: security issue in critical file
        security_task = {
            'type': 'security',
            'file_path': 'src/auth.py',
            'complexity_indicators': ['password', 'token', 'auth'],
            'urgency_keywords': ['vulnerability', 'security', 'critical'],
            'line_count': 15
        }
        priority = calculate_task_priority(security_task)
        assert priority >= 8.0, "Security tasks should have high priority"
        
        # Medium priority: performance improvement
        performance_task = {
            'type': 'performance',
            'file_path': 'src/utils.py',
            'complexity_indicators': ['loop', 'optimization'],
            'urgency_keywords': ['slow', 'performance'],
            'line_count': 30
        }
        priority = calculate_task_priority(performance_task)
        assert 4.0 <= priority < 8.0, "Performance tasks should have medium priority"
        
        # Low priority: documentation TODO
        doc_task = {
            'type': 'documentation',
            'file_path': 'README.md',
            'complexity_indicators': [],
            'urgency_keywords': ['documentation', 'readme'],
            'line_count': 5
        }
        priority = calculate_task_priority(doc_task)
        assert priority < 4.0, "Documentation tasks should have lower priority"

    def test_analyze_task_complexity(self):
        """Test task complexity analysis"""
        from task_prioritization import analyze_task_complexity
        
        # High complexity: database and authentication related
        complex_content = "TODO: Fix SQL injection vulnerability in user authentication system"
        complexity = analyze_task_complexity(complex_content, 'src/auth.py')
        assert complexity >= 7, "Security + DB + Auth should be high complexity"
        
        # Medium complexity: API integration
        medium_content = "TODO: Add retry logic for external API calls"
        complexity = analyze_task_complexity(medium_content, 'src/api_client.py')
        assert 4 <= complexity < 7, "API changes should be medium complexity"
        
        # Low complexity: simple formatting
        simple_content = "TODO: Fix indentation in this function"
        complexity = analyze_task_complexity(simple_content, 'src/utils.py')
        assert complexity < 4, "Formatting should be low complexity"

    def test_determine_business_impact(self):
        """Test business impact assessment"""
        from task_prioritization import determine_business_impact
        
        # High impact: core functionality
        core_file = 'src/github_api.py'
        core_content = 'TODO: Fix authentication failure causing service outage'
        impact = determine_business_impact(core_content, core_file)
        assert impact >= 8, "Core service issues should have high business impact"
        
        # Medium impact: user experience
        ui_file = 'src/prompt_builder.py'
        ui_content = 'TODO: Improve error messages for better user experience'
        impact = determine_business_impact(ui_content, ui_file)
        assert 4 <= impact < 8, "UX improvements should have medium impact"
        
        # Low impact: internal tooling
        tool_file = 'scripts/helper.py'
        tool_content = 'TODO: Add debug logging to development script'
        impact = determine_business_impact(tool_content, tool_file)
        assert impact < 4, "Internal tooling should have low impact"

    def test_assess_urgency(self):
        """Test urgency assessment based on keywords and context"""
        from task_prioritization import assess_urgency
        
        # High urgency: blocking issues
        blocking_content = "FIXME: Critical bug blocking deployment"
        urgency = assess_urgency(blocking_content)
        assert urgency >= 8, "Blocking issues should be high urgency"
        
        # Medium urgency: user-affecting bugs
        user_content = "TODO: Fix bug causing incorrect data display"
        urgency = assess_urgency(user_content)
        assert 4 <= urgency < 8, "User-affecting bugs should be medium urgency"
        
        # Low urgency: optimization tasks
        optimization_content = "TODO: Optimize this algorithm for better performance"
        urgency = assess_urgency(optimization_content)
        assert urgency < 6, "Optimizations should be lower urgency"

    def test_prioritize_discovered_tasks(self):
        """Test prioritization of a list of discovered tasks"""
        from task_prioritization import prioritize_discovered_tasks
        
        tasks = [
            {
                'id': 'task1',
                'content': 'TODO: Fix memory leak in authentication module',
                'file_path': 'src/auth.py',
                'line_number': 42,
                'type': 'bug'
            },
            {
                'id': 'task2', 
                'content': 'TODO: Add unit tests for utility functions',
                'file_path': 'src/utils.py',
                'line_number': 15,
                'type': 'testing'
            },
            {
                'id': 'task3',
                'content': 'FIXME: SQL injection vulnerability in user query',
                'file_path': 'src/database.py', 
                'line_number': 89,
                'type': 'security'
            }
        ]
        
        prioritized = prioritize_discovered_tasks(tasks)
        
        # Should return tasks sorted by priority (highest first)
        assert len(prioritized) == 3
        assert all('priority_score' in task for task in prioritized)
        assert all('priority_reason' in task for task in prioritized)
        
        # Security task should be highest priority
        security_task = next(task for task in prioritized if task['type'] == 'security')
        assert security_task['priority_score'] >= 8.0
        
        # Priority scores should be in descending order
        scores = [task['priority_score'] for task in prioritized]
        assert scores == sorted(scores, reverse=True)

    def test_task_type_classification(self):
        """Test automatic classification of task types"""
        from task_prioritization import classify_task_type
        
        # Security issues
        security_content = "TODO: Fix XSS vulnerability in user input"
        assert classify_task_type(security_content, 'src/web.py') == 'security'
        
        # Performance issues
        perf_content = "TODO: Optimize slow database query"
        assert classify_task_type(perf_content, 'src/models.py') == 'performance'
        
        # Bug fixes
        bug_content = "FIXME: Function returns wrong value in edge case"
        assert classify_task_type(bug_content, 'src/logic.py') == 'bug'
        
        # Testing tasks
        test_content = "TODO: Add integration tests for API endpoints"
        assert classify_task_type(test_content, 'tests/api.py') == 'testing'
        
        # Documentation
        doc_content = "TODO: Update API documentation"
        assert classify_task_type(doc_content, 'docs/api.md') == 'documentation'

    def test_priority_score_ranges(self):
        """Test that priority scores fall within expected ranges"""
        from task_prioritization import calculate_task_priority
        
        # Test multiple scenarios to ensure scores are reasonable
        test_cases = [
            {
                'type': 'security',
                'file_path': 'src/auth.py',
                'complexity_indicators': ['sql', 'injection', 'vulnerability'],
                'urgency_keywords': ['critical', 'security'],
                'line_count': 20,
                'expected_range': (8.0, 10.0)
            },
            {
                'type': 'documentation',
                'file_path': 'README.md',
                'complexity_indicators': [],
                'urgency_keywords': ['update'],
                'line_count': 3,
                'expected_range': (1.0, 3.0)
            },
            {
                'type': 'refactoring',
                'file_path': 'src/utils.py',
                'complexity_indicators': ['refactor', 'cleanup'],
                'urgency_keywords': ['improve'],
                'line_count': 10,
                'expected_range': (3.0, 6.0)
            }
        ]
        
        for case in test_cases:
            score = calculate_task_priority(case)
            min_score, max_score = case['expected_range']
            assert min_score <= score <= max_score, f"Score {score} not in expected range {case['expected_range']} for {case['type']}"

    def test_integration_with_task_analyzer(self):
        """Test integration with existing task analyzer"""
        from task_prioritization import enhance_task_with_priority
        
        # Mock task from task analyzer
        task = {
            'title': 'Fix TODO in authentication module',
            'content': 'TODO: Add rate limiting to prevent brute force attacks',
            'file_path': 'src/auth.py',
            'line_number': 156,
            'repository': 'myorg/myapp'
        }
        
        enhanced_task = enhance_task_with_priority(task)
        
        # Should add priority information
        assert 'priority_score' in enhanced_task
        assert 'priority_reason' in enhanced_task
        assert 'task_type' in enhanced_task
        assert enhanced_task['priority_score'] > 0
        assert len(enhanced_task['priority_reason']) > 0