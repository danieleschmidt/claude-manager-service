"""
Unit tests for the enhanced prompt template system with Jinja2 support.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.prompt_builder import (
    PromptTemplateEngine, 
    build_prompt, 
    get_template_for_labels,
    validate_template,
    get_template_engine
)


class TestPromptTemplateEngine:
    """Test suite for PromptTemplateEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = PromptTemplateEngine(template_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        assert self.engine.template_dir == self.temp_dir
        assert self.engine.env is not None
        
    def test_engine_initialization_missing_dir(self):
        """Test engine initialization with missing directory."""
        non_existent_dir = "/path/that/does/not/exist"
        engine = PromptTemplateEngine(template_dir=non_existent_dir)
        assert engine.template_dir == non_existent_dir
        assert engine.env is None
    
    def test_custom_filters(self):
        """Test that custom Jinja2 filters work correctly."""
        # Test truncate_lines filter
        long_text = "\n".join([f"Line {i}" for i in range(25)])
        truncated = self.engine._truncate_lines_filter(long_text, 5)
        lines = truncated.split('\n')
        assert len(lines) == 6  # 5 lines + truncation message
        assert "truncated 20 more lines" in truncated
        
        # Test format_list filter
        long_list = [f"item{i}" for i in range(15)]
        formatted = self.engine._format_list_filter(long_list, ", ", 5)
        assert "and 10 more" in formatted
        assert formatted.count(",") == 4  # 5 items means 4 commas
        
        # Test safe_filename filter
        unsafe_name = "My Unsafe/File:Name*With?Special<>Chars"
        safe_name = self.engine._safe_filename_filter(unsafe_name)
        assert "/" not in safe_name
        assert ":" not in safe_name
        assert "*" not in safe_name
        assert safe_name.islower()
    
    def test_enhance_context(self):
        """Test context enhancement with defaults and utility functions."""
        minimal_context = {"issue_title": "Test Issue"}
        enhanced = self.engine._enhance_context(minimal_context)
        
        # Check defaults are added
        assert enhanced["issue_body"] == "No description provided"
        assert enhanced["repository"] == "Unknown repository"
        
        # Check utility functions
        assert callable(enhanced["has_label"])
        assert enhanced["is_bug"] == False  # No bug labels
        assert enhanced["is_feature"] == False  # No feature labels
        
        # Test with labels
        context_with_labels = {"labels": "bug, critical, fix"}
        enhanced_with_labels = self.engine._enhance_context(context_with_labels)
        assert enhanced_with_labels["is_bug"] == True
    
    def test_build_jinja2_prompt_success(self):
        """Test successful Jinja2 prompt building."""
        # Create a test template
        template_content = """Title: {{ issue_title }}
{% if is_bug %}
This is a bug fix task.
{% else %}
This is a general task.
{% endif %}
Description: {{ issue_body }}"""
        
        template_path = os.path.join(self.temp_dir, "test.j2")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        context = {
            "issue_title": "Fix login bug",
            "issue_body": "Users cannot log in",
            "labels": "bug, critical"
        }
        
        prompt = self.engine.build_prompt("test.j2", context)
        assert "Title: Fix login bug" in prompt
        assert "This is a bug fix task." in prompt
        assert "Description: Users cannot log in" in prompt
    
    def test_build_fallback_prompt(self):
        """Test fallback prompt building when Jinja2 fails."""
        # Create a simple template with placeholders
        template_content = "Title: {issue_title}\nDescription: {issue_body}"
        template_path = os.path.join(self.temp_dir, "simple.txt")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        context = {
            "issue_title": "Test Issue",
            "issue_body": "Test Description"
        }
        
        prompt = self.engine._build_fallback_prompt("simple.txt", context)
        assert "Title: Test Issue" in prompt
        assert "Description: Test Description" in prompt
    
    def test_emergency_fallback_prompt(self):
        """Test emergency fallback when all else fails."""
        context = {
            "issue_title": "Emergency Test",
            "issue_body": "Emergency Description"
        }
        
        prompt = self.engine._build_emergency_fallback_prompt(context)
        assert "Title: Emergency Test" in prompt
        assert "Emergency Description" in prompt
    
    def test_validate_template_success(self):
        """Test successful template validation."""
        template_content = """{# @title: Test Template #}
{# @description: A test template #}
Title: {{ issue_title }}
{% if is_bug %}
Bug fix instructions
{% endif %}"""
        
        template_path = os.path.join(self.temp_dir, "validate_test.j2")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        result = self.engine.validate_template(template_path)
        assert result["valid"] == True
        assert len(result["errors"]) == 0
        assert "issue_title" in result["variables"]
        assert "is_bug" in result["conditional_blocks"][0]
        assert result["metadata"]["title"] == "Test Template"
    
    def test_validate_template_syntax_error(self):
        """Test template validation with syntax errors."""
        template_content = "{% if unclosed_block %}\nContent without endif"
        template_path = os.path.join(self.temp_dir, "syntax_error.j2")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        result = self.engine.validate_template(template_path)
        assert result["valid"] == False
        assert len(result["errors"]) > 0
        assert "syntax error" in result["errors"][0].lower()
    
    def test_validate_template_missing_file(self):
        """Test template validation with missing file."""
        result = self.engine.validate_template("/nonexistent/template.j2")
        assert result["valid"] == False
        assert "not found" in result["errors"][0].lower()


class TestTemplateLabelSelection:
    """Test suite for template selection based on labels."""
    
    def test_feature_template_selection(self):
        """Test that feature labels select the feature template."""
        labels = ["feature", "enhancement"]
        template = get_template_for_labels(labels)
        assert template == "feature_implementation.j2"
        
        labels = ["new", "api"]
        template = get_template_for_labels(labels)
        assert template == "feature_implementation.j2"
    
    def test_refactor_template_selection(self):
        """Test that refactor labels select the refactor template."""
        labels = ["refactor", "cleanup"]
        template = get_template_for_labels(labels)
        assert template == "refactor_code.j2"
        
        labels = ["todo", "tech-debt"]
        template = get_template_for_labels(labels)
        assert template == "refactor_code.j2"
    
    def test_bug_template_selection(self):
        """Test that bug labels select the bug fix template."""
        labels = ["bug", "error"]
        template = get_template_for_labels(labels)
        assert template == "fix_issue.j2"
        
        labels = ["fix", "issue"]
        template = get_template_for_labels(labels)
        assert template == "fix_issue.j2"
    
    def test_default_template_selection(self):
        """Test default template selection."""
        # No labels
        template = get_template_for_labels([])
        assert template == "fix_issue.j2"
        
        # Unrecognized labels
        template = get_template_for_labels(["random", "unknown"])
        assert template == "fix_issue.j2"
    
    def test_priority_order(self):
        """Test that template selection follows priority order."""
        # Feature should take priority over bug
        labels = ["feature", "bug"]
        template = get_template_for_labels(labels)
        assert template == "feature_implementation.j2"
        
        # Refactor should take priority over bug
        labels = ["refactor", "bug"]
        template = get_template_for_labels(labels)
        assert template == "refactor_code.j2"


class TestPublicAPI:
    """Test suite for public API functions."""
    
    def test_build_prompt_function(self):
        """Test the public build_prompt function."""
        with patch('src.prompt_builder.get_template_engine') as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.build_prompt.return_value = "Test prompt"
            mock_get_engine.return_value = mock_engine
            
            result = build_prompt("test.j2", {"key": "value"})
            assert result == "Test prompt"
            mock_engine.build_prompt.assert_called_once_with("test.j2", {"key": "value"})
    
    def test_validate_template_function(self):
        """Test the public validate_template function."""
        with patch('src.prompt_builder.get_template_engine') as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.validate_template.return_value = {"valid": True}
            mock_get_engine.return_value = mock_engine
            
            result = validate_template("test.j2")
            assert result == {"valid": True}
            mock_engine.validate_template.assert_called_once_with("test.j2")
    
    def test_get_template_engine_singleton(self):
        """Test that get_template_engine returns a singleton."""
        engine1 = get_template_engine()
        engine2 = get_template_engine()
        assert engine1 is engine2


class TestTemplateIntegration:
    """Integration tests for the complete template system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_template_processing(self):
        """Test complete template processing workflow."""
        # Create a feature template
        template_content = """{# @title: Feature Template #}
Task: {{ issue_title }}
{% if is_feature %}
This is a feature implementation task.
{% endif %}
{% if is_urgent %}
URGENT: High priority task!
{% endif %}
Labels: {{ labels | format_list }}"""
        
        template_path = os.path.join(self.temp_dir, "feature_implementation.j2")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        # Create engine with our temp directory
        engine = PromptTemplateEngine(template_dir=self.temp_dir)
        
        # Test context
        context = {
            "issue_title": "Add user authentication",
            "labels": ["feature", "urgent", "security"]
        }
        
        # Build prompt
        prompt = engine.build_prompt("feature_implementation.j2", context)
        
        # Verify content
        assert "Task: Add user authentication" in prompt
        assert "This is a feature implementation task." in prompt
        assert "URGENT: High priority task!" in prompt
        assert "feature, urgent, security" in prompt
    
    def test_template_with_missing_variables(self):
        """Test template processing with missing variables."""
        template_content = "Title: {{ issue_title }}\nOptional: {{ missing_var | default('Not provided') }}"
        template_path = os.path.join(self.temp_dir, "test_missing.j2")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        engine = PromptTemplateEngine(template_dir=self.temp_dir)
        context = {"issue_title": "Test"}
        
        prompt = engine.build_prompt("test_missing.j2", context)
        assert "Title: Test" in prompt
        assert "Optional: Not provided" in prompt
    
    def test_complex_conditional_logic(self):
        """Test complex conditional logic in templates."""
        template_content = """{% if is_bug and is_urgent %}
CRITICAL BUG FIX REQUIRED
{% elif is_bug %}
Bug fix needed
{% elif is_feature %}
New feature implementation
{% else %}
General task
{% endif %}"""
        
        template_path = os.path.join(self.temp_dir, "conditional.j2")
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        engine = PromptTemplateEngine(template_dir=self.temp_dir)
        
        # Test critical bug
        context1 = {"labels": "bug, urgent"}
        prompt1 = engine.build_prompt("conditional.j2", context1)
        assert "CRITICAL BUG FIX REQUIRED" in prompt1
        
        # Test regular bug
        context2 = {"labels": "bug"}
        prompt2 = engine.build_prompt("conditional.j2", context2)
        assert "Bug fix needed" in prompt2
        
        # Test feature
        context3 = {"labels": "feature"}
        prompt3 = engine.build_prompt("conditional.j2", context3)
        assert "New feature implementation" in prompt3
        
        # Test general task
        context4 = {"labels": "documentation"}
        prompt4 = engine.build_prompt("conditional.j2", context4)
        assert "General task" in prompt4


if __name__ == "__main__":
    pytest.main([__file__])