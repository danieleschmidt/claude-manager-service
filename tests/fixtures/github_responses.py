"""
Test fixtures for GitHub API responses.
"""

from typing import Dict, Any, List


class GitHubFixtures:
    """Collection of GitHub API response fixtures for testing."""

    @staticmethod
    def repository_response() -> Dict[str, Any]:
        """Mock GitHub repository response."""
        return {
            "id": 123456789,
            "name": "test-repo",
            "full_name": "test-user/test-repo",
            "description": "A test repository for unit testing",
            "private": False,
            "fork": False,
            "html_url": "https://github.com/test-user/test-repo",
            "clone_url": "https://github.com/test-user/test-repo.git",
            "ssh_url": "git@github.com:test-user/test-repo.git",
            "default_branch": "main",
            "language": "Python",
            "stargazers_count": 42,
            "watchers_count": 15,
            "forks_count": 8,
            "open_issues_count": 5,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "pushed_at": "2025-01-01T12:00:00Z",
            "owner": {
                "login": "test-user",
                "id": 987654321,
                "type": "User",
                "html_url": "https://github.com/test-user"
            }
        }

    @staticmethod
    def issue_response(
        number: int = 1,
        title: str = "Test Issue",
        state: str = "open",
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """Mock GitHub issue response."""
        if labels is None:
            labels = ["bug", "priority-medium"]
        
        return {
            "id": 123456789 + number,
            "number": number,
            "title": title,
            "body": f"This is the body of issue #{number}",
            "state": state,
            "locked": False,
            "html_url": f"https://github.com/test-user/test-repo/issues/{number}",
            "created_at": "2024-12-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "closed_at": None if state == "open" else "2025-01-01T12:00:00Z",
            "user": {
                "login": "test-user",
                "id": 987654321,
                "type": "User",
                "html_url": "https://github.com/test-user"
            },
            "labels": [
                {
                    "id": i,
                    "name": label,
                    "color": "d73a49",
                    "description": f"Label for {label}"
                }
                for i, label in enumerate(labels, 1)
            ],
            "assignees": [],
            "milestone": None,
            "comments": 0,
            "pull_request": None
        }

    @staticmethod
    def pull_request_response(
        number: int = 1,
        title: str = "Test PR",
        state: str = "open"
    ) -> Dict[str, Any]:
        """Mock GitHub pull request response."""
        return {
            "id": 123456789 + number,
            "number": number,
            "title": title,
            "body": f"This is the body of PR #{number}",
            "state": state,
            "locked": False,
            "html_url": f"https://github.com/test-user/test-repo/pull/{number}",
            "created_at": "2024-12-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "closed_at": None if state == "open" else "2025-01-01T12:00:00Z",
            "merged_at": None if state != "closed" else "2025-01-01T12:00:00Z",
            "head": {
                "ref": "feature-branch",
                "sha": "abc123def456",
                "repo": {
                    "name": "test-repo",
                    "full_name": "test-user/test-repo"
                }
            },
            "base": {
                "ref": "main",
                "sha": "def456abc123",
                "repo": {
                    "name": "test-repo",
                    "full_name": "test-user/test-repo"
                }
            },
            "user": {
                "login": "test-user",
                "id": 987654321,
                "type": "User",
                "html_url": "https://github.com/test-user"
            },
            "draft": False,
            "mergeable": True,
            "rebaseable": True,
            "comments": 0,
            "review_comments": 0,
            "commits": 1,
            "additions": 10,
            "deletions": 5,
            "changed_files": 2
        }

    @staticmethod
    def file_content_response(
        path: str = "src/example.py",
        content: str = None
    ) -> Dict[str, Any]:
        """Mock GitHub file content response."""
        if content is None:
            content = '''
# Example Python file with TODO comments
def example_function():
    # TODO: Implement this function
    pass

def another_function():
    # FIXME: This function has a bug
    return "Hello, World!"
'''
        
        import base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        return {
            "name": path.split('/')[-1],
            "path": path,
            "sha": "abc123def456ghi789",
            "size": len(content),
            "url": f"https://api.github.com/repos/test-user/test-repo/contents/{path}",
            "html_url": f"https://github.com/test-user/test-repo/blob/main/{path}",
            "git_url": f"https://api.github.com/repos/test-user/test-repo/git/blobs/abc123def456ghi789",
            "download_url": f"https://raw.githubusercontent.com/test-user/test-repo/main/{path}",
            "type": "file",
            "content": encoded_content,
            "encoding": "base64"
        }

    @staticmethod
    def search_code_response(query: str = "TODO") -> Dict[str, Any]:
        """Mock GitHub code search response."""
        return {
            "total_count": 2,
            "incomplete_results": False,
            "items": [
                {
                    "name": "example.py",
                    "path": "src/example.py",
                    "sha": "abc123def456",
                    "url": "https://api.github.com/repos/test-user/test-repo/contents/src/example.py",
                    "git_url": "https://api.github.com/repos/test-user/test-repo/git/blobs/abc123def456",
                    "html_url": "https://github.com/test-user/test-repo/blob/main/src/example.py",
                    "repository": GitHubFixtures.repository_response(),
                    "score": 1.0
                },
                {
                    "name": "utils.py",
                    "path": "src/utils.py",
                    "sha": "def456abc123",
                    "url": "https://api.github.com/repos/test-user/test-repo/contents/src/utils.py",
                    "git_url": "https://api.github.com/repos/test-user/test-repo/git/blobs/def456abc123",
                    "html_url": "https://github.com/test-user/test-repo/blob/main/src/utils.py",
                    "repository": GitHubFixtures.repository_response(),
                    "score": 0.8
                }
            ]
        }

    @staticmethod
    def commit_response(sha: str = "abc123def456") -> Dict[str, Any]:
        """Mock GitHub commit response."""
        return {
            "sha": sha,
            "commit": {
                "author": {
                    "name": "Test User",
                    "email": "test@example.com",
                    "date": "2025-01-01T00:00:00Z"
                },
                "committer": {
                    "name": "Test User",
                    "email": "test@example.com",
                    "date": "2025-01-01T00:00:00Z"
                },
                "message": "feat: add new feature\n\nThis commit adds a new feature to the application.",
                "tree": {
                    "sha": "tree123abc456",
                    "url": "https://api.github.com/repos/test-user/test-repo/git/trees/tree123abc456"
                },
                "url": "https://api.github.com/repos/test-user/test-repo/git/commits/abc123def456",
                "comment_count": 0
            },
            "url": "https://api.github.com/repos/test-user/test-repo/commits/abc123def456",
            "html_url": "https://github.com/test-user/test-repo/commit/abc123def456",
            "author": {
                "login": "test-user",
                "id": 987654321,
                "type": "User",
                "html_url": "https://github.com/test-user"
            },
            "committer": {
                "login": "test-user",
                "id": 987654321,
                "type": "User",
                "html_url": "https://github.com/test-user"
            },
            "parents": [
                {
                    "sha": "parent123abc456",
                    "url": "https://api.github.com/repos/test-user/test-repo/commits/parent123abc456",
                    "html_url": "https://github.com/test-user/test-repo/commit/parent123abc456"
                }
            ],
            "stats": {
                "total": 15,
                "additions": 10,
                "deletions": 5
            },
            "files": [
                {
                    "sha": "file123abc456",
                    "filename": "src/example.py",
                    "status": "modified",
                    "additions": 5,
                    "deletions": 2,
                    "changes": 7,
                    "blob_url": "https://github.com/test-user/test-repo/blob/abc123def456/src/example.py",
                    "raw_url": "https://github.com/test-user/test-repo/raw/abc123def456/src/example.py",
                    "contents_url": "https://api.github.com/repos/test-user/test-repo/contents/src/example.py?ref=abc123def456",
                    "patch": "@@ -1,5 +1,8 @@\n # Example file\n+# Added new comment\n def example():\n+    # New code\n     pass"
                }
            ]
        }

    @staticmethod
    def workflow_run_response(
        run_id: int = 123456789,
        status: str = "completed",
        conclusion: str = "success"
    ) -> Dict[str, Any]:
        """Mock GitHub workflow run response."""
        return {
            "id": run_id,
            "name": "CI",
            "node_id": "MDEyOldvcmtmbG93UnVuMTIzNDU2Nzg5",
            "head_branch": "main",
            "head_sha": "abc123def456",
            "run_number": 42,
            "event": "push",
            "status": status,
            "conclusion": conclusion,
            "workflow_id": 987654321,
            "url": f"https://api.github.com/repos/test-user/test-repo/actions/runs/{run_id}",
            "html_url": f"https://github.com/test-user/test-repo/actions/runs/{run_id}",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:05:00Z",
            "run_started_at": "2025-01-01T00:00:00Z",
            "jobs_url": f"https://api.github.com/repos/test-user/test-repo/actions/runs/{run_id}/jobs",
            "logs_url": f"https://api.github.com/repos/test-user/test-repo/actions/runs/{run_id}/logs",
            "check_suite_url": f"https://api.github.com/repos/test-user/test-repo/check-suites/{run_id}",
            "artifacts_url": f"https://api.github.com/repos/test-user/test-repo/actions/runs/{run_id}/artifacts",
            "cancel_url": f"https://api.github.com/repos/test-user/test-repo/actions/runs/{run_id}/cancel",
            "rerun_url": f"https://api.github.com/repos/test-user/test-repo/actions/runs/{run_id}/rerun",
            "head_commit": GitHubFixtures.commit_response(),
            "repository": GitHubFixtures.repository_response(),
            "head_repository": GitHubFixtures.repository_response()
        }

    @staticmethod
    def error_response(status_code: int = 404, message: str = "Not Found") -> Dict[str, Any]:
        """Mock GitHub API error response."""
        return {
            "message": message,
            "documentation_url": "https://docs.github.com/rest",
            "status": status_code
        }