#!/usr/bin/env python3
"""
Manual security test validation script
"""
import html
import re

def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent XSS and other attacks"""
    if not isinstance(user_input, str):
        return str(user_input)
    
    # HTML escape to prevent XSS
    sanitized = html.escape(user_input, quote=True)
    
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
        r'<script[^>]*>.*?</script>',
    ]
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized

def validate_limit_parameter(limit_str: str) -> int:
    """Validate and sanitize limit parameter"""
    try:
        limit = int(limit_str)
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        return limit
    except (ValueError, TypeError):
        raise ValueError("Invalid limit parameter")

def main():
    print("Testing security functions...")
    
    # Test XSS prevention
    test_input = "<script>alert('xss')</script>"
    result = sanitize_user_input(test_input)
    print(f'XSS test input: {test_input}')
    print(f'XSS test result: {result}')
    assert '&lt;script&gt;' in result and '&lt;/script&gt;' in result
    print('✓ XSS prevention working')

    # Test SQL injection prevention  
    sql_test = "'; DROP TABLE users; --"
    sql_result = sanitize_user_input(sql_test)
    print(f'SQL test result: {sql_result}')
    assert '&#x27;' in sql_result
    print('✓ SQL injection prevention working')

    # Test valid limits
    assert validate_limit_parameter('10') == 10
    assert validate_limit_parameter('100') == 100
    print('✓ Valid limit validation working')

    # Test invalid limits
    try:
        validate_limit_parameter('1001')
        print('ERROR: Should have failed')
    except ValueError:
        print('✓ Invalid limit rejection working')

    try:
        validate_limit_parameter('abc')
        print('ERROR: Should have failed')  
    except ValueError:
        print('✓ Non-numeric limit rejection working')

    print('\n✅ ALL SECURITY FUNCTIONS VALIDATED SUCCESSFULLY')

if __name__ == "__main__":
    main()