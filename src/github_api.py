import os
from github import Github, GithubException

class GitHubAPI:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable not set.")
        self.client = Github(self.token)

    def get_repo(self, repo_name):
        try:
            return self.client.get_repo(repo_name)
        except GithubException as e:
            print(f"Error getting repo {repo_name}: {e}")
            return None

    def create_issue(self, repo_name, title, body, labels):
        print(f"Creating issue '{title}' in repo {repo_name}")
        repo = self.get_repo(repo_name)
        if repo:
            try:
                # Check for existing issues with similar titles
                existing_issues = repo.get_issues(state='open')
                for issue in existing_issues:
                    if issue.title.lower() == title.lower():
                        print(f"Issue with title '{title}' already exists (#{issue.number})")
                        return
                
                # Create the issue if no duplicate found
                new_issue = repo.create_issue(title=title, body=body, labels=labels)
                print(f"Issue created successfully (#{new_issue.number})")
            except GithubException as e:
                print(f"Error creating issue: {e}")

    def get_issue(self, repo_name, issue_number):
        repo = self.get_repo(repo_name)
        if repo:
            try:
                return repo.get_issue(number=issue_number)
            except GithubException as e:
                print(f"Error getting issue #{issue_number}: {e}")
        return None

    def add_comment_to_issue(self, repo_name, issue_number, comment_body):
        print(f"Adding comment to issue #{issue_number} in {repo_name}")
        issue = self.get_issue(repo_name, issue_number)
        if issue:
            try:
                issue.create_comment(comment_body)
                print("Comment posted successfully.")
            except GithubException as e:
                print(f"Error posting comment: {e}")