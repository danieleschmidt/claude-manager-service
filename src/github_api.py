import os
from github import Github, GithubException
from logger import get_logger, log_performance
from security import get_secure_config, validate_repo_name, sanitize_issue_content
from error_handler import with_error_recovery, safe_github_operation

class GitHubAPI:
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Use secure config to get token
        self.token = get_secure_config().get_github_token()
        
        self.logger.info("Initializing GitHub API client with secure token")
        self.client = Github(self.token)
        
        # Test API connection
        try:
            user = self.client.get_user()
            self.logger.info(f"GitHub API client initialized successfully for user: {user.login}")
        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub API client: {e}")
            raise

    @log_performance
    @with_error_recovery("get_repository", max_attempts=3, delay=2.0)
    def get_repo(self, repo_name):
        """Get repository object from GitHub API with security validation"""
        # Validate repository name for security
        if not validate_repo_name(repo_name):
            self.logger.error(f"Invalid repository name format: {repo_name}")
            return None
        
        self.logger.debug(f"Fetching repository: {repo_name}")
        return safe_github_operation(
            self.client.get_repo, 
            f"get repository {repo_name}",
            repo_name
        )

    @log_performance
    @with_error_recovery("create_issue", max_attempts=3, delay=1.0)
    def create_issue(self, repo_name, title, body, labels):
        """Create a new issue in the specified repository with duplicate checking and content sanitization"""
        # Sanitize inputs for security
        title = sanitize_issue_content(title)
        body = sanitize_issue_content(body) if body else ""
        
        self.logger.info(f"Creating issue '{title[:50]}...' in repository {repo_name}")
        self.logger.debug(f"Issue labels: {labels}")
        
        repo = self.get_repo(repo_name)
        if not repo:
            self.logger.warning(f"Cannot create issue - repository {repo_name} not accessible")
            return
        
        try:
            # Check for existing issues with similar titles
            self.logger.debug("Checking for duplicate issues")
            existing_issues = repo.get_issues(state='open')
            
            for issue in existing_issues:
                if issue.title.lower() == title.lower():
                    self.logger.warning(f"Duplicate issue found: '{title}' already exists (#{issue.number})")
                    return
            
            # Create the issue if no duplicate found
            self.logger.debug("No duplicate found, creating new issue")
            new_issue = repo.create_issue(title=title, body=body, labels=labels)
            self.logger.info(f"Issue created successfully: #{new_issue.number} - '{title}'")
            
        except GithubException as e:
            self.logger.error(f"Failed to create issue '{title}': {e.status} - {e.data}")
        except Exception as e:
            self.logger.error(f"Unexpected error creating issue '{title}': {e}")

    def get_issue(self, repo_name, issue_number):
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

    @log_performance
    def add_comment_to_issue(self, repo_name, issue_number, comment_body):
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