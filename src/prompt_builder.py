import os
import re
from typing import Dict, Any, List, Optional, Set
from jinja2 import Environment, FileSystemLoader, Template, TemplateError, select_autoescape, meta
from jinja2.exceptions import TemplateSyntaxError, TemplateNotFound, UndefinedError
try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

logger = get_logger(__name__)

class PromptTemplateEngine:
    """Enhanced prompt template engine with Jinja2 support, validation, and conditional sections."""
    
    def __init__(self, template_dir: str = "prompts"):
        """
        Initialize the template engine.
        
        Args:
            template_dir (str): Directory containing template files
        """
        self.template_dir = template_dir
        self.env: Optional[Environment] = None
        self._setup_jinja2_environment()
        logger.info(f"PromptTemplateEngine initialized with template directory: {template_dir}")
    
    def _setup_jinja2_environment(self) -> None:
        """Set up Jinja2 environment with custom filters and security settings."""
        try:
            if os.path.exists(self.template_dir):
                self.env = Environment(
                    loader=FileSystemLoader(self.template_dir),
                    autoescape=select_autoescape(['html', 'xml']),
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                
                # Add custom filters
                if self.env:
                    self.env.filters['truncate_lines'] = self._truncate_lines_filter
                    self.env.filters['format_list'] = self._format_list_filter
                    self.env.filters['safe_filename'] = self._safe_filename_filter
                
                logger.debug("Jinja2 environment configured successfully")
            else:
                logger.warning(f"Template directory {self.template_dir} does not exist")
                self.env = None
        except Exception as e:
            logger.error(f"Failed to setup Jinja2 environment: {e}")
            self.env = None
    
    def _truncate_lines_filter(self, text: str, max_lines: int = 20) -> str:
        """Jinja2 filter to truncate text to specified number of lines."""
        if not text:
            return ""
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
        return '\n'.join(lines[:max_lines]) + f"\n... (truncated {len(lines) - max_lines} more lines)"
    
    def _format_list_filter(self, items: List[Any], separator: str = ", ", max_items: int = 10) -> str:
        """Jinja2 filter to format lists with optional truncation."""
        if not items:
            return "None"
        str_items = [str(item) for item in items]
        if len(str_items) <= max_items:
            return separator.join(str_items)
        return separator.join(str_items[:max_items]) + f" (and {len(str_items) - max_items} more)"
    
    def _safe_filename_filter(self, text: str) -> str:
        """Jinja2 filter to create safe filenames from text."""
        if not text:
            return "unknown"
        # Remove or replace unsafe characters
        safe = re.sub(r'[^\w\s-]', '', text)
        safe = re.sub(r'[-\s]+', '-', safe)
        return safe.strip('-').lower()[:50]
    
    def validate_template(self, template_path: str) -> Dict[str, Any]:
        """
        Validate a template file for syntax errors and extract metadata.
        
        Args:
            template_path (str): Path to template file
            
        Returns:
            Dict containing validation results and metadata
        """
        validation_result: Dict[str, Any] = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "variables": set(),
            "conditional_blocks": [],
            "metadata": {}
        }
        
        try:
            if not os.path.exists(template_path):
                validation_result["errors"].append(f"Template file not found: {template_path}")
                return validation_result
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Parse metadata from template comments
            metadata_pattern = r'{#\s*@(\w+):\s*(.+?)\s*#}'
            metadata_matches = re.findall(metadata_pattern, template_content)
            for key, value in metadata_matches:
                if isinstance(validation_result["metadata"], dict):
                    validation_result["metadata"][key] = value.strip()
            
            if self.env:
                try:
                    # Parse template with Jinja2
                    template = self.env.from_string(template_content)
                    
                    # Extract variables used in template
                    ast = self.env.parse(template_content)
                    validation_result["variables"] = set(meta.find_undeclared_variables(ast))
                    
                    # Find conditional blocks
                    conditional_pattern = r'{%\s*if\s+(.+?)\s*%}'
                    conditionals = re.findall(conditional_pattern, template_content)
                    validation_result["conditional_blocks"] = conditionals
                    
                    validation_result["valid"] = True
                    logger.debug(f"Template {template_path} validated successfully")
                    
                except TemplateSyntaxError as e:
                    validation_result["errors"].append(f"Syntax error in template: {e}")
                except Exception as e:
                    if isinstance(validation_result["errors"], list):
                        validation_result["errors"].append(f"Template parsing error: {e}")
            else:
                # Fallback validation for simple placeholder templates
                placeholders = set(re.findall(r'{(\w+)}', template_content))
                validation_result["variables"] = placeholders
                validation_result["valid"] = True
                logger.debug(f"Template {template_path} validated with fallback method")
            
        except Exception as e:
            if isinstance(validation_result["errors"], list):
                validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result
    
    def build_prompt(self, template_file: str, context: Dict[str, Any]) -> str:
        """
        Build a prompt using Jinja2 templating with validation and conditional sections.
        
        Args:
            template_file (str): Path to template file (relative to template_dir)
            context (dict): Dictionary containing template variables
            
        Returns:
            str: Generated prompt
        """
        logger.debug(f"Building prompt from template: {template_file}")
        logger.debug(f"Context keys: {list(context.keys())}")
        
        # Add default context values
        enhanced_context = self._enhance_context(context)
        
        try:
            if self.env:
                return self._build_jinja2_prompt(template_file, enhanced_context)
            else:
                return self._build_fallback_prompt(template_file, enhanced_context)
                
        except Exception as e:
            logger.error(f"Error building prompt from {template_file}: {e}")
            return self._build_emergency_fallback_prompt(enhanced_context)
    
    def _enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add default values and utility functions to context."""
        enhanced = context.copy()
        
        # Add default values for common variables
        defaults = {
            'issue_title': 'No title provided',
            'issue_body': 'No description provided',
            'issue_number': 'N/A',
            'repository': 'Unknown repository',
            'labels': 'No labels',
            'issue_url': '#',
            'assignees': 'None assigned',
            'milestone': 'No milestone',
            'created_at': 'Unknown',
            'updated_at': 'Unknown'
        }
        
        for key, default_value in defaults.items():
            if key not in enhanced or enhanced[key] is None:
                enhanced[key] = default_value
        
        # Add utility functions
        enhanced['has_label'] = lambda label: label.lower() in str(enhanced.get('labels', '')).lower()
        enhanced['is_bug'] = enhanced['has_label']('bug') or enhanced['has_label']('error')
        enhanced['is_feature'] = enhanced['has_label']('feature') or enhanced['has_label']('enhancement')
        enhanced['is_urgent'] = enhanced['has_label']('urgent') or enhanced['has_label']('critical')
        
        return enhanced
    
    def _build_jinja2_prompt(self, template_file: str, context: Dict[str, Any]) -> str:
        """Build prompt using Jinja2 template engine."""
        try:
            if self.env is None:
                raise TemplateError("Jinja2 environment not initialized")
            template = self.env.get_template(template_file)
            prompt = template.render(**context)
            logger.info(f"Jinja2 prompt built successfully from {template_file}")
            logger.debug(f"Generated prompt length: {len(prompt)} characters")
            return prompt
            
        except TemplateNotFound:
            logger.warning(f"Jinja2 template not found: {template_file}, falling back to simple method")
            return self._build_fallback_prompt(template_file, context)
        except UndefinedError as e:
            logger.warning(f"Undefined variable in template {template_file}: {e}")
            return self._build_fallback_prompt(template_file, context)
        except TemplateError as e:
            logger.error(f"Template error in {template_file}: {e}")
            return self._build_fallback_prompt(template_file, context)
    
    def _build_fallback_prompt(self, template_file: str, context: Dict[str, Any]) -> str:
        """Build prompt using simple string replacement (fallback method)."""
        template_path = os.path.join(self.template_dir, template_file)
        
        if not os.path.exists(template_path):
            logger.warning(f"Template file {template_path} not found. Using emergency fallback.")
            return self._build_emergency_fallback_prompt(context)
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Simple placeholder replacement
            prompt = template
            replacements_made = 0
            
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
                    replacements_made += 1
            
            logger.info(f"Fallback prompt built: {replacements_made} placeholders replaced")
            return prompt
            
        except Exception as e:
            logger.error(f"Fallback prompt building failed: {e}")
            return self._build_emergency_fallback_prompt(context)
    
    def _build_emergency_fallback_prompt(self, context: Dict[str, Any]) -> str:
        """Build a basic prompt when all other methods fail."""
        logger.debug("Building emergency fallback prompt")
        title = context.get('issue_title', 'No title')
        body = context.get('issue_body', 'No description')
        return f"Please work on the following task:\n\nTitle: {title}\n\nDescription:\n{body}"

# Global template engine instance
_template_engine: Optional[PromptTemplateEngine] = None

def get_template_engine() -> PromptTemplateEngine:
    """Get or create the global template engine instance."""
    global _template_engine
    if _template_engine is None:
        _template_engine = PromptTemplateEngine()
    return _template_engine

def build_prompt(template_file: str, context: Dict[str, Any]) -> str:
    """
    Builds a detailed prompt from a template file and a context dictionary.
    
    Args:
        template_file (str): Path to the template file
        context (dict): Dictionary containing placeholder values
        
    Returns:
        str: Generated prompt with placeholders replaced
    """
    engine = get_template_engine()
    return engine.build_prompt(template_file, context)

def get_template_for_labels(labels: List[str]) -> str:
    """
    Returns the appropriate template file based on issue labels.
    
    Args:
        labels (list): List of issue label strings
        
    Returns:
        str: Path to the appropriate template file
    """
    logger.debug(f"Selecting template for labels: {labels}")
    
    if not labels:
        logger.debug("No labels provided, using default template")
        return 'fix_issue.j2'
    
    label_names = [label.lower() for label in labels]
    logger.debug(f"Normalized label names: {label_names}")
    
    # Priority order: feature > refactor > bug/fix
    if any(word in label_names for word in ['feature', 'enhancement', 'new']):
        template = 'feature_implementation.j2'
        logger.debug(f"Selected feature template based on labels: {[l for l in label_names if l in ['feature', 'enhancement', 'new']]}")
        return template
    elif any(word in label_names for word in ['refactor', 'todo', 'cleanup', 'tech-debt']):
        template = 'refactor_code.j2'
        logger.debug(f"Selected refactor template based on labels: {[l for l in label_names if l in ['refactor', 'todo', 'cleanup', 'tech-debt']]}")
        return template
    elif any(word in label_names for word in ['bug', 'fix', 'issue', 'error']):
        template = 'fix_issue.j2'
        logger.debug(f"Selected fix_issue template based on labels: {[l for l in label_names if l in ['bug', 'fix', 'issue', 'error']]}")
        return template
    else:
        logger.debug("No matching label patterns, using default fix_issue template")
        return 'fix_issue.j2'  # Default template

def validate_template(template_path: str) -> Dict[str, Any]:
    """
    Validate a template file for syntax errors and extract metadata.
    
    Args:
        template_path (str): Path to template file
        
    Returns:
        Dict containing validation results and metadata
    """
    engine = get_template_engine()
    return engine.validate_template(template_path)