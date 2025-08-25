#!/usr/bin/env python3
"""
MOCK AUTHENTICATION SYSTEM - Generation 2
Mock authentication for testing without requiring real GitHub tokens
"""

import os
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MockAuthUser:
    """Mock authenticated user"""
    login: str
    id: int
    avatar_url: str
    email: Optional[str] = None
    name: Optional[str] = None
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class MockGitHubAPI:
    """Mock GitHub API for testing without real authentication"""
    
    def __init__(self, mock_token: Optional[str] = None):
        self.mock_token = mock_token or "ghp_mock_token_for_testing_only"
        self.mock_user = MockAuthUser(
            login="test-user",
            id=12345,
            avatar_url="https://github.com/images/error/test-user_happy.gif",
            email="test@example.com",
            name="Test User"
        )
        self.rate_limit = {
            "core": {
                "limit": 5000,
                "remaining": 4999,
                "reset": int(time.time() + 3600),
                "used": 1
            }
        }
        
    def get_user(self) -> MockAuthUser:
        """Get authenticated user (mock)"""
        logger.info("Mock API: Getting authenticated user")
        return self.mock_user
    
    def get_repo(self, repo_name: str) -> Dict[str, Any]:
        """Get repository information (mock)"""
        logger.info(f"Mock API: Getting repository {repo_name}")
        
        # Simulate different repository scenarios
        if "nonexistent" in repo_name.lower():
            return None
        
        return {
            "id": 67890,
            "full_name": repo_name,
            "name": repo_name.split("/")[-1] if "/" in repo_name else repo_name,
            "private": False,
            "html_url": f"https://github.com/{repo_name}",
            "description": f"Mock repository {repo_name}",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "language": "Python",
            "stargazers_count": 42,
            "forks_count": 7,
            "open_issues_count": 3
        }
    
    def get_issues(self, repo_name: str, state: str = "open") -> list:
        """Get repository issues (mock)"""
        logger.info(f"Mock API: Getting {state} issues for {repo_name}")
        
        if state == "open":
            return [
                {
                    "id": 1,
                    "number": 1,
                    "title": "Mock Issue 1",
                    "body": "This is a mock issue for testing",
                    "state": "open",
                    "labels": [{"name": "bug"}, {"name": "help wanted"}],
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-02T00:00:00Z",
                    "html_url": f"https://github.com/{repo_name}/issues/1",
                    "assignees": [],
                    "user": {
                        "login": "test-user",
                        "id": 12345
                    }
                },
                {
                    "id": 2,
                    "number": 2,
                    "title": "Mock Issue 2",
                    "body": "Another mock issue for testing",
                    "state": "open",
                    "labels": [{"name": "enhancement"}],
                    "created_at": "2023-01-03T00:00:00Z",
                    "updated_at": "2023-01-03T00:00:00Z",
                    "html_url": f"https://github.com/{repo_name}/issues/2",
                    "assignees": [],
                    "user": {
                        "login": "test-user",
                        "id": 12345
                    }
                }
            ]
        else:
            return []
    
    def search_code(self, query: str) -> list:
        """Search code in repositories (mock)"""
        logger.info(f"Mock API: Searching code with query: {query}")
        
        # Simulate search results based on query
        if "TODO" in query.upper() or "FIXME" in query.upper():
            return [
                {
                    "name": "main.py",
                    "path": "src/main.py",
                    "sha": "abc123",
                    "url": "https://api.github.com/repositories/123/contents/src/main.py",
                    "html_url": "https://github.com/test/repo/blob/main/src/main.py",
                    "repository": {
                        "full_name": "test/repo"
                    }
                },
                {
                    "name": "utils.py",
                    "path": "src/utils.py", 
                    "sha": "def456",
                    "url": "https://api.github.com/repositories/123/contents/src/utils.py",
                    "html_url": "https://github.com/test/repo/blob/main/src/utils.py",
                    "repository": {
                        "full_name": "test/repo"
                    }
                }
            ]
        else:
            return []
    
    def get_contents(self, repo_name: str, path: str) -> Dict[str, Any]:
        """Get file contents (mock)"""
        logger.info(f"Mock API: Getting contents of {path} from {repo_name}")
        
        # Mock file contents based on file type
        if path.endswith('.py'):
            content = """#!/usr/bin/env python3
# TODO: Implement error handling
# FIXME: Performance optimization needed

def main():
    # TODO: Add logging
    print("Hello, World!")
    # FIXME: Handle edge cases
    pass

if __name__ == "__main__":
    main()
"""
        elif path.endswith('.md'):
            content = """# Mock File

This is a mock README file for testing.

## TODO
- Add more documentation
- FIXME: Update installation instructions

## Features
- Mock feature 1
- Mock feature 2
"""
        else:
            content = "# Mock file content for testing\n# TODO: Add real content\n"
        
        import base64
        encoded_content = base64.b64encode(content.encode()).decode()
        
        return {
            "name": path.split("/")[-1],
            "path": path,
            "sha": "mock_sha_" + path.replace("/", "_"),
            "size": len(content),
            "url": f"https://api.github.com/repositories/123/contents/{path}",
            "html_url": f"https://github.com/{repo_name}/blob/main/{path}",
            "download_url": f"https://raw.githubusercontent.com/{repo_name}/main/{path}",
            "type": "file",
            "content": encoded_content,
            "encoding": "base64",
            "decoded_content": content.encode()
        }
    
    def create_issue(self, repo_name: str, title: str, body: str, labels: list = None) -> Dict[str, Any]:
        """Create issue (mock)"""
        logger.info(f"Mock API: Creating issue '{title}' in {repo_name}")
        
        issue_number = hash(f"{repo_name}-{title}") % 1000 + 1
        
        return {
            "id": issue_number * 100,
            "number": issue_number,
            "title": title,
            "body": body,
            "state": "open",
            "labels": [{"name": label} for label in (labels or [])],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "html_url": f"https://github.com/{repo_name}/issues/{issue_number}",
            "user": asdict(self.mock_user)
        }
    
    def create_comment(self, repo_name: str, issue_number: int, body: str) -> Dict[str, Any]:
        """Create issue comment (mock)"""
        logger.info(f"Mock API: Creating comment on issue #{issue_number} in {repo_name}")
        
        comment_id = hash(f"{repo_name}-{issue_number}-{body}") % 10000 + 1
        
        return {
            "id": comment_id,
            "body": body,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "html_url": f"https://github.com/{repo_name}/issues/{issue_number}#issuecomment-{comment_id}",
            "user": asdict(self.mock_user)
        }
    
    def get_rate_limit(self) -> Dict[str, Any]:
        """Get rate limit status (mock)"""
        logger.info("Mock API: Getting rate limit status")
        return {
            "rate": self.rate_limit["core"],
            "resources": {
                "core": self.rate_limit["core"],
                "search": {
                    "limit": 30,
                    "remaining": 29,
                    "reset": int(time.time() + 3600),
                    "used": 1
                }
            }
        }


class MockAuthenticationManager:
    """Mock authentication manager for testing"""
    
    def __init__(self):
        self.authenticated = False
        self.github_api = None
        self.mock_token = None
    
    def authenticate_with_token(self, token: str) -> bool:
        """Authenticate with token (mock)"""
        logger.info("Mock Auth: Authenticating with token")
        
        # Accept any token that looks like a GitHub token or is a test token
        if (token.startswith(('ghp_', 'gho_', 'ghs_', 'ghr_')) or 
            token == "test_token" or 
            "mock" in token.lower()):
            
            self.mock_token = token
            self.github_api = MockGitHubAPI(token)
            self.authenticated = True
            logger.info("Mock Auth: Authentication successful")
            return True
        else:
            logger.warning("Mock Auth: Authentication failed - invalid token format")
            return False
    
    def authenticate_without_token(self) -> bool:
        """Authenticate without token for testing (mock)"""
        logger.info("Mock Auth: Authenticating without token (test mode)")
        
        self.mock_token = "mock_test_token"
        self.github_api = MockGitHubAPI(self.mock_token)
        self.authenticated = True
        
        return True
    
    def is_authenticated(self) -> bool:
        """Check if authenticated"""
        return self.authenticated
    
    def get_github_api(self) -> Optional[MockGitHubAPI]:
        """Get GitHub API client"""
        if self.authenticated:
            return self.github_api
        return None
    
    def logout(self):
        """Logout (clear mock authentication)"""
        logger.info("Mock Auth: Logging out")
        self.authenticated = False
        self.github_api = None
        self.mock_token = None


def create_mock_authentication_system() -> MockAuthenticationManager:
    """Create mock authentication system for testing"""
    return MockAuthenticationManager()


# Integration with existing GitHub API
def patch_github_api_for_testing():
    """Monkey patch existing GitHub API to use mock for testing"""
    try:
        from src import github_api
        
        # Store original class
        if not hasattr(github_api, '_original_github_api'):
            github_api._original_github_api = github_api.GitHubAPI
        
        # Create mock replacement
        class MockGitHubAPIWrapper:
            def __init__(self):
                self.mock_auth = create_mock_authentication_system()
                # Try to authenticate without token first
                if not self.mock_auth.authenticate_without_token():
                    logger.warning("Mock authentication failed")
                
                self.client = self.mock_auth.get_github_api()
                self.token = self.mock_auth.mock_token
                self.logger = logger
                
                # Mock successful user access
                if self.client:
                    logger.info(f"Mock GitHub API initialized for user: {self.client.mock_user.login}")
            
            def get_repo(self, repo_name: str):
                """Get repository (mock)"""
                if self.client:
                    return self.client.get_repo(repo_name)
                return None
            
            def create_issue(self, repo_name: str, title: str, body: str, labels: list = None):
                """Create issue (mock)"""
                if self.client:
                    result = self.client.create_issue(repo_name, title, body, labels)
                    logger.info(f"Mock issue created: #{result['number']}")
                    return result
                return None
            
            def get_issue(self, repo_name: str, issue_number: int):
                """Get issue (mock)"""
                if self.client:
                    # Return a mock issue
                    return {
                        "number": issue_number,
                        "title": f"Mock Issue #{issue_number}",
                        "body": "This is a mock issue for testing",
                        "state": "open",
                        "html_url": f"https://github.com/{repo_name}/issues/{issue_number}"
                    }
                return None
            
            def add_comment_to_issue(self, repo_name: str, issue_number: int, comment_body: str):
                """Add comment to issue (mock)"""
                if self.client:
                    result = self.client.create_comment(repo_name, issue_number, comment_body)
                    logger.info(f"Mock comment added to issue #{issue_number}")
                    return result
                return None
        
        # Replace GitHubAPI class
        github_api.GitHubAPI = MockGitHubAPIWrapper
        logger.info("GitHub API patched with mock implementation")
        return True
        
    except ImportError:
        logger.warning("Could not patch GitHub API - module not found")
        return False


def restore_github_api():
    """Restore original GitHub API implementation"""
    try:
        from src import github_api
        
        if hasattr(github_api, '_original_github_api'):
            github_api.GitHubAPI = github_api._original_github_api
            logger.info("Original GitHub API restored")
        
    except ImportError:
        logger.warning("Could not restore GitHub API - module not found")


# Demo
if __name__ == "__main__":
    logger.info("Testing mock authentication system")
    
    # Test mock authentication
    auth = create_mock_authentication_system()
    
    # Test authentication without token
    success = auth.authenticate_without_token()
    logger.info(f"Authentication without token: {success}")
    
    if success:
        api = auth.get_github_api()
        user = api.get_user()
        logger.info(f"Authenticated user: {user.login}")
        
        # Test API calls
        repo = api.get_repo("test/repo")
        logger.info(f"Repository: {repo['full_name']}")
        
        issues = api.get_issues("test/repo")
        logger.info(f"Found {len(issues)} issues")
    
    # Test GitHub API patching
    if patch_github_api_for_testing():
        logger.info("GitHub API successfully patched for testing")
    
    logger.info("Mock authentication system test completed")