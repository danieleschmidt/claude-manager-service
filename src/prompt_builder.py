import os

def build_prompt(template_file, context):
    """
    Builds a detailed prompt from a template file and a context dictionary.
    """
    print(f"Building prompt from template: {template_file}")
    
    if not os.path.exists(template_file):
        print(f"Warning: Template file {template_file} not found. Using default prompt.")
        return f"Please work on the following task:\n\nTitle: {context.get('issue_title', 'No title')}\n\nDescription:\n{context.get('issue_body', 'No description')}"
    
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Replace placeholders in the template with values from context
        prompt = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
        
        return prompt
        
    except Exception as e:
        print(f"Error building prompt: {e}")
        # Fallback to basic prompt
        return f"Please work on the following task:\n\nTitle: {context.get('issue_title', 'No title')}\n\nDescription:\n{context.get('issue_body', 'No description')}"

def get_template_for_labels(labels):
    """
    Returns the appropriate template file based on issue labels.
    """
    label_names = [label.lower() for label in labels]
    
    if any(word in label_names for word in ['refactor', 'todo', 'cleanup']):
        return 'prompts/refactor_code.txt'
    elif any(word in label_names for word in ['bug', 'fix', 'issue']):
        return 'prompts/fix_issue.txt'
    else:
        return 'prompts/fix_issue.txt'  # Default template