import json
import datetime
from github_api import GitHubAPI

def find_todo_comments(github_api, repo, manager_repo_name):
    print(f"Scanning {repo.full_name} for TODO comments...")
    try:
        # Search for TODO and FIXME comments in the repository
        search_queries = ['TODO:', 'FIXME:', 'TODO', 'FIXME']
        
        for query in search_queries:
            try:
                # Use GitHub's code search API
                search_results = github_api.client.search_code(
                    query=f"{query} repo:{repo.full_name}"
                )
                
                for result in search_results[:5]:  # Limit to first 5 results per query
                    file_path = result.path
                    
                    # Get the file content to extract context
                    try:
                        file_content = repo.get_contents(file_path)
                        content_lines = file_content.decoded_content.decode('utf-8').split('\n')
                        
                        # Find the line with the TODO/FIXME
                        for line_num, line in enumerate(content_lines, 1):
                            if query.lower() in line.lower():
                                # Extract some context around the TODO
                                start_line = max(0, line_num - 3)
                                end_line = min(len(content_lines), line_num + 2)
                                context = '\n'.join(content_lines[start_line:end_line])
                                
                                title = f"Address {query} in {file_path}:{line_num}"
                                body = f"""A `{query}` comment was found that may require action.

**Repository:** {repo.full_name}
**File:** `{file_path}`
**Line:** {line_num}

**Context:**
```
{context}
```

**Direct Link:** {result.html_url}
"""
                                
                                github_api.create_issue(
                                    manager_repo_name, 
                                    title, 
                                    body, 
                                    ["task-proposal", "refactor", "todo"]
                                )
                                break  # Only create one issue per file
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error searching for {query}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in find_todo_comments: {e}")

def analyze_open_issues(github_api, repo, manager_repo_name):
    print(f"Analyzing open issues in {repo.full_name}...")
    try:
        # Get open issues from the repository
        open_issues = repo.get_issues(state='open')
        
        # Current time for comparison
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for issue in open_issues:
            # Skip if it's a pull request
            if issue.pull_request:
                continue
                
            # Check if issue has relevant labels
            issue_labels = [label.name.lower() for label in issue.labels]
            relevant_labels = ['bug', 'help wanted', 'good first issue', 'enhancement']
            
            has_relevant_label = any(label in issue_labels for label in relevant_labels)
            
            if has_relevant_label:
                # Check if issue is stale (no activity in last 30 days)
                days_since_update = (now - issue.updated_at).days
                
                if days_since_update > 30:
                    title = f"Review stale issue: '{issue.title}'"
                    body = f"""The issue #{issue.number} in `{repo.full_name}` has been inactive for {days_since_update} days and may need attention.

**Original Issue:** {issue.title}
**Labels:** {', '.join([label.name for label in issue.labels])}
**Created:** {issue.created_at.strftime('%Y-%m-%d')}
**Last Updated:** {issue.updated_at.strftime('%Y-%m-%d')}
**Assignees:** {', '.join([assignee.login for assignee in issue.assignees]) if issue.assignees else 'None'}

**Description:**
{issue.body[:500]}{'...' if len(issue.body) > 500 else ''}

**Link:** {issue.html_url}

This issue may be a good candidate for AI-assisted resolution.
"""
                    
                    github_api.create_issue(
                        manager_repo_name, 
                        title, 
                        body, 
                        ["task-proposal", "stale-issue", "review"]
                    )
                    
    except Exception as e:
        print(f"Error in analyze_open_issues: {e}")

if __name__ == "__main__":
    try:
        with open('config.json') as f:
            config = json.load(f)

        api = GitHubAPI()
        manager_repo_name = config['github']['managerRepo']
        
        for repo_name in config['github']['reposToScan']:
            print(f"\n--- Analyzing repository: {repo_name} ---")
            repo = api.get_repo(repo_name)
            if not repo:
                print(f"Skipping {repo_name} - could not access repository")
                continue

            if config['analyzer']['scanForTodos']:
                find_todo_comments(api, repo, manager_repo_name)

            if config['analyzer']['scanOpenIssues']:
                analyze_open_issues(api, repo, manager_repo_name)

        print("\n--- Task analysis cycle complete ---")
        
    except Exception as e:
        print(f"Fatal error in task analyzer: {e}")
        raise