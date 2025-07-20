"""
Unit tests for prompt_builder.py module
"""
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open

import sys
sys.path.append('/root/repo/src')
from prompt_builder import build_prompt, get_template_for_labels


class TestPromptBuilder:
    """Test cases for prompt builder functions"""

    def test_build_prompt_with_existing_template(self):
        """Test prompt building with existing template file"""
        template_content = "Title: {issue_title}\nRepo: {repository}\nDescription: {issue_body}"
        context = {
            "issue_title": "Test Issue",
            "repository": "test/repo",
            "issue_body": "This is a test issue"
        }
        
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("os.path.exists", return_value=True):
                result = build_prompt("test_template.txt", context)
        
        expected = "Title: Test Issue\nRepo: test/repo\nDescription: This is a test issue"
        assert result == expected

    def test_build_prompt_with_missing_template(self):
        """Test prompt building with missing template file"""
        context = {
            "issue_title": "Test Issue",
            "issue_body": "This is a test issue"
        }
        
        with patch("os.path.exists", return_value=False):
            result = build_prompt("nonexistent.txt", context)
        
        expected = "Please work on the following task:\n\nTitle: Test Issue\n\nDescription:\nThis is a test issue"
        assert result == expected

    def test_build_prompt_with_partial_placeholders(self):
        """Test prompt building with some missing placeholders"""
        template_content = "Title: {issue_title}\nRepo: {repository}\nMissing: {missing_key}"
        context = {
            "issue_title": "Test Issue",
            "repository": "test/repo"
        }
        
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("os.path.exists", return_value=True):
                result = build_prompt("test_template.txt", context)
        
        # Missing placeholders should remain unchanged
        expected = "Title: Test Issue\nRepo: test/repo\nMissing: {missing_key}"
        assert result == expected

    def test_build_prompt_with_empty_context(self):
        """Test prompt building with empty context"""
        template_content = "Title: {issue_title}\nBody: {issue_body}"
        context = {}
        
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("os.path.exists", return_value=True):
                result = build_prompt("test_template.txt", context)
        
        # Placeholders should remain unchanged
        expected = "Title: {issue_title}\nBody: {issue_body}"
        assert result == expected

    def test_build_prompt_file_read_error(self):
        """Test prompt building when file read fails"""
        context = {
            "issue_title": "Test Issue",
            "issue_body": "This is a test issue"
        }
        
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", side_effect=IOError("Permission denied")):
                result = build_prompt("error_template.txt", context)
        
        # Should fallback to default prompt
        expected = "Please work on the following task:\n\nTitle: Test Issue\n\nDescription:\nThis is a test issue"
        assert result == expected

    def test_build_prompt_with_missing_title_and_body(self):
        """Test prompt building with missing title and body in context"""
        context = {"repository": "test/repo"}
        
        with patch("os.path.exists", return_value=False):
            result = build_prompt("nonexistent.txt", context)
        
        expected = "Please work on the following task:\n\nTitle: No title\n\nDescription:\nNo description"
        assert result == expected

    def test_get_template_for_labels_refactor(self):
        """Test template selection for refactor labels"""
        labels = ["refactor", "cleanup"]
        result = get_template_for_labels(labels)
        assert result == "prompts/refactor_code.txt"

    def test_get_template_for_labels_todo(self):
        """Test template selection for todo labels"""
        labels = ["todo", "task"]
        result = get_template_for_labels(labels)
        assert result == "prompts/refactor_code.txt"

    def test_get_template_for_labels_bug(self):
        """Test template selection for bug labels"""
        labels = ["bug", "urgent"]
        result = get_template_for_labels(labels)
        assert result == "prompts/fix_issue.txt"

    def test_get_template_for_labels_fix(self):
        """Test template selection for fix labels"""
        labels = ["fix", "hotfix"]
        result = get_template_for_labels(labels)
        assert result == "prompts/fix_issue.txt"

    def test_get_template_for_labels_issue(self):
        """Test template selection for issue labels"""
        labels = ["issue", "problem"]
        result = get_template_for_labels(labels)
        assert result == "prompts/fix_issue.txt"

    def test_get_template_for_labels_default(self):
        """Test template selection for unknown labels defaults to fix_issue"""
        labels = ["enhancement", "feature"]
        result = get_template_for_labels(labels)
        assert result == "prompts/fix_issue.txt"

    def test_get_template_for_labels_empty(self):
        """Test template selection for empty labels"""
        labels = []
        result = get_template_for_labels(labels)
        assert result == "prompts/fix_issue.txt"

    def test_get_template_for_labels_case_insensitive(self):
        """Test template selection is case insensitive"""
        labels = ["REFACTOR", "TODO"]
        result = get_template_for_labels(labels)
        assert result == "prompts/refactor_code.txt"

    def test_get_template_for_labels_mixed_case(self):
        """Test template selection with mixed case labels"""
        labels = ["Bug", "FIX"]
        result = get_template_for_labels(labels)
        assert result == "prompts/fix_issue.txt"

    def test_build_prompt_with_special_characters(self):
        """Test prompt building with special characters in context"""
        template_content = "Title: {issue_title}\nSpecial: {special_chars}"
        context = {
            "issue_title": "Test with émojis 🐛",
            "special_chars": "Line1\nLine2\tTabbed"
        }
        
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch("os.path.exists", return_value=True):
                result = build_prompt("test_template.txt", context)
        
        expected = "Title: Test with émojis 🐛\nSpecial: Line1\nLine2\tTabbed"
        assert result == expected