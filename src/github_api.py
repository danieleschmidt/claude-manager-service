import os
from typing import Optional, List
from github import Github, GithubException, Repository, Issue
from logger import get_logger, log_performance
from performance_monitor import monitor_api_call
from security import get_secure_config, validate_repo_name, sanitize_issue_content
from error_handler import with_error_recovery, safe_github_operation
from enhanced_error_handler import github_api_operation, get_rate_limiter, NetworkError, RateLimitError
from enhanced_security import sanitize_repo_name, sanitize_issue_content_enhanced
from enhanced_validation import validate_api_parameters

class GitHubAPI:
    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        
        # Use secure config to get token
        self.token = get_secure_config().get_github_token()
        
        self.logger.info("Initializing GitHub API client with secure token")
        self.client = Github(self.token)
        
        # Test API connection with specific error handling
        try:
            user = self.client.get_user()
            self.logger.info(f"GitHub API client initialized successfully for user: {user.login}")
        except GithubException as e:
            if e.status == 401:
                self.logger.error("GitHub API authentication failed - invalid token")
                raise NetworkError("Authentication failed", "github_api_init", e)
            elif e.status == 403:
                self.logger.error("GitHub API rate limit or permissions issue")
                raise RateLimitError("API access denied", "github_api_init")
            else:
                self.logger.error(f"GitHub API error during initialization: {e.status} - {e.data}")
                raise NetworkError(f"GitHub API error: {e.status}", "github_api_init", e)
        except ConnectionError as e:
            self.logger.error(f"Network connection failed during GitHub API initialization: {e}")
            raise NetworkError("Connection failed", "github_api_init", e)
        except TimeoutError as e:
            self.logger.error(f"Timeout during GitHub API initialization: {e}")
            raise NetworkError("Request timeout", "github_api_init", e)

    @github_api_operation("get_repository")
    @monitor_api_call("github_get_repository")
    @log_performance
    def get_repo(self, repo_name: str) -> Optional[Repository.Repository]:
        """Get repository object from GitHub API with enhanced validation and error handling"""
        # Enhanced repository name validation
        try:
            validated_repo_name = sanitize_repo_name(repo_name)
        except ValueError as e:
            self.logger.error(f"Repository name validation failed: {e}")
            raise NetworkError(f"Invalid repository name: {str(e)}", "get_repository", e)
        
        # Validate API parameters
        try:
            validate_api_parameters({"repo_name": validated_repo_name}, "get_repository")
        except Exception as e:
            self.logger.error(f"API parameter validation failed: {e}")
            raise NetworkError(f"Parameter validation failed: {str(e)}", "get_repository", e)
        
        self.logger.debug(f"Fetching repository: {validated_repo_name}")
        
        # Enhanced GitHub operation with specific error handling
        try:
            return self.client.get_repo(validated_repo_name)
        except GithubException as e:
            if e.status == 404:
                self.logger.warning(f"Repository not found: {validated_repo_name}")
                return None
            elif e.status == 403:
                self.logger.error(f"Access denied to repository: {validated_repo_name}")
                raise RateLimitError("Repository access denied", "get_repository")
            else:
                self.logger.error(f"GitHub API error getting repository {validated_repo_name}: {e.status}")
                raise NetworkError(f"GitHub API error: {e.status}", "get_repository", e)

    @github_api_operation("create_issue")
    @monitor_api_call("github_create_issue")
    @log_performance
    def create_issue(self, repo_name: str, title: str, body: Optional[str], labels: List[str]) -> None:
        """Create a new issue in the specified repository with enhanced validation and error handling"""
        # Enhanced input validation and sanitization
        try:
            validated_repo_name = sanitize_repo_name(repo_name)
            sanitized_title = sanitize_issue_content_enhanced(title)
            sanitized_body = sanitize_issue_content_enhanced(body) if body else ""
            
            # Validate API parameters
            validate_api_parameters({
                "repo_name": validated_repo_name,
                "title": sanitized_title,
                "body": sanitized_body,
                "labels": labels
            }, "create_issue")
            
        except Exception as e:
            self.logger.error(f"Input validation failed for create_issue: {e}")
            raise NetworkError(f"Input validation failed: {str(e)}", "create_issue", e)
        
        self.logger.info(f"Creating issue '{sanitized_title[:50]}...' in repository {validated_repo_name}")
        self.logger.debug(f"Issue labels: {labels}")
        
        repo = self.get_repo(validated_repo_name)
        if not repo:
            self.logger.warning(f"Cannot create issue - repository {validated_repo_name} not accessible")
            raise NetworkError(f"Repository not accessible: {validated_repo_name}", "create_issue", 
                             Exception("Repository access failed"))
        
        try:
            # Check for existing issues with similar titles
            self.logger.debug("Checking for duplicate issues")
            existing_issues = repo.get_issues(state='open')
            
            for issue in existing_issues:
                if issue.title.lower() == sanitized_title.lower():
                    self.logger.warning(f"Duplicate issue found: '{sanitized_title}' already exists (#{issue.number})")
                    return
            
            # Create the issue if no duplicate found
            self.logger.debug("No duplicate found, creating new issue")
            new_issue = repo.create_issue(title=sanitized_title, body=sanitized_body, labels=labels)
            self.logger.info(f"Issue created successfully: #{new_issue.number} - '{sanitized_title}'")
            
        except GithubException as e:
            if e.status == 403:
                self.logger.error(f"Permission denied creating issue in {validated_repo_name}")
                raise RateLimitError("Issue creation permission denied", "create_issue")
            elif e.status == 422:
                self.logger.error(f"Invalid issue data: {e.data}")
                raise NetworkError(f"Invalid issue data: {e.data}", "create_issue", e)
            else:
                self.logger.error(f"GitHub API error creating issue: {e.status} - {e.data}")
                raise NetworkError(f"Issue creation failed: {e.status}", "create_issue", e)

    def get_issue(self, repo_name: str, issue_number: int) -> Optional[Issue.Issue]:
        """Get a specific issue from the repository"""
        self.logger.debug(f"Fetching issue #{issue_number} from {repo_name}")
        
        repo = self.get_repo(repo_name)
        if not repo:
            self.logger.warning(f"Cannot get issue - repository {repo_name} not accessible")
            return None
        
        try:
            issue = repo.get_issue(number=issue_number)
            self.logger.debug(f"Successfully retrieved issue #{issue_number}")
            return issue
        except GithubException as e:
            self.logger.error(f"Failed to get issue #{issue_number}: {e.status} - {e.data}")
            return None

    @monitor_api_call("github_add_comment")
    @log_performance
    def add_comment_to_issue(self, repo_name: str, issue_number: int, comment_body: str) -> None:
        """Add a comment to the specified issue"""
        self.logger.info(f"Adding comment to issue #{issue_number} in {repo_name}")
        self.logger.debug(f"Comment length: {len(comment_body)} characters")
        
        issue = self.get_issue(repo_name, issue_number)
        if not issue:
            self.logger.warning(f"Cannot add comment - issue #{issue_number} not found")
            return
        
        try:
            issue.create_comment(comment_body)
            self.logger.info(f"Comment posted successfully to issue #{issue_number}")
        except GithubException as e:
            self.logger.error(f"Failed to post comment to issue #{issue_number}: {e.status} - {e.data}")
        except Exception as e:
            self.logger.error(f"Unexpected error posting comment to issue #{issue_number}: {e}")