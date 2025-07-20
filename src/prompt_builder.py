import os
from logger import get_logger

logger = get_logger(__name__)

def build_prompt(template_file, context):
    """
    Builds a detailed prompt from a template file and a context dictionary.
    
    Args:
        template_file (str): Path to the template file
        context (dict): Dictionary containing placeholder values
        
    Returns:
        str: Generated prompt with placeholders replaced
    """
    logger.debug(f"Building prompt from template: {template_file}")
    logger.debug(f"Context keys: {list(context.keys())}")
    
    if not os.path.exists(template_file):
        logger.warning(f"Template file {template_file} not found. Using default prompt.")
        default_prompt = f"Please work on the following task:\n\nTitle: {context.get('issue_title', 'No title')}\n\nDescription:\n{context.get('issue_body', 'No description')}"
        logger.debug(f"Generated default prompt length: {len(default_prompt)} characters")
        return default_prompt
    
    try:
        logger.debug(f"Reading template file: {template_file}")
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        logger.debug(f"Template loaded, length: {len(template)} characters")
        
        # Replace placeholders in the template with values from context
        prompt = template
        replacements_made = 0
        
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
                replacements_made += 1
                logger.debug(f"Replaced placeholder '{placeholder}' with value of length {len(str(value))}")
        
        logger.info(f"Prompt built successfully: {replacements_made} placeholders replaced")
        logger.debug(f"Final prompt length: {len(prompt)} characters")
        return prompt
        
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_file}")
        return _build_fallback_prompt(context)
    except PermissionError:
        logger.error(f"Permission denied reading template file: {template_file}")
        return _build_fallback_prompt(context)
    except Exception as e:
        logger.error(f"Unexpected error building prompt from {template_file}: {e}")
        return _build_fallback_prompt(context)

def _build_fallback_prompt(context):
    """Build a basic fallback prompt when template processing fails"""
    logger.debug("Building fallback prompt")
    fallback = f"Please work on the following task:\n\nTitle: {context.get('issue_title', 'No title')}\n\nDescription:\n{context.get('issue_body', 'No description')}"
    logger.debug(f"Fallback prompt length: {len(fallback)} characters")
    return fallback

def get_template_for_labels(labels):
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
        return 'prompts/fix_issue.txt'
    
    label_names = [label.lower() for label in labels]
    logger.debug(f"Normalized label names: {label_names}")
    
    if any(word in label_names for word in ['refactor', 'todo', 'cleanup']):
        template = 'prompts/refactor_code.txt'
        logger.debug(f"Selected refactor template based on labels: {[l for l in label_names if l in ['refactor', 'todo', 'cleanup']]}")
        return template
    elif any(word in label_names for word in ['bug', 'fix', 'issue']):
        template = 'prompts/fix_issue.txt'
        logger.debug(f"Selected fix_issue template based on labels: {[l for l in label_names if l in ['bug', 'fix', 'issue']]}")
        return template
    else:
        logger.debug("No matching label patterns, using default fix_issue template")
        return 'prompts/fix_issue.txt'  # Default template