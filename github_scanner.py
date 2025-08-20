#!/usr/bin/env python3
"""
GitHub Repository Scanner - Simple Implementation
Scans GitHub repositories for issues and TODOs
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("PyGithub not available. GitHub scanning disabled.")

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from simple_main import SimpleConfig

class GitHubScanner:
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.github_token = os.getenv("GITHUB_TOKEN")
        
        if not GITHUB_AVAILABLE:
            self.client = None
            return
            
        if self.github_token:
            self.client = Github(self.github_token)
        else:
            self.client = None
            rprint("[yellow]Warning: No GITHUB_TOKEN found. Public repo access only.[/yellow]")
    
    def scan_repositories(self) -> List[Dict[str, Any]]:
        """Scan configured repositories for issues and TODOs"""
        if not self.client:
            rprint("[red]Error: GitHub client not available[/red]")
            return []
        
        all_tasks = []
        repos_to_scan = self.config.get("github.reposToScan", [])
        
        if not repos_to_scan:
            rprint("[yellow]No repositories configured for scanning[/yellow]")
            return []
        
        for repo_name in repos_to_scan:
            rprint(f"[blue]Scanning repository: {repo_name}[/blue]")
            try:
                repo_tasks = self._scan_single_repository(repo_name)
                all_tasks.extend(repo_tasks)
            except Exception as e:
                rprint(f"[red]Error scanning {repo_name}: {e}[/red]")
                continue
        
        return all_tasks
    
    def _scan_single_repository(self, repo_name: str) -> List[Dict[str, Any]]:
        """Scan a single repository"""
        tasks = []
        
        try:
            repo = self.client.get_repo(repo_name)
            
            # Scan open issues
            if self.config.get("analyzer.scanOpenIssues", True):
                tasks.extend(self._scan_issues(repo, repo_name))
            
            # Scan repository files for TODOs
            if self.config.get("analyzer.scanForTodos", True):
                tasks.extend(self._scan_repo_files(repo, repo_name))
            
        except GithubException as e:
            rprint(f"[red]GitHub API error for {repo_name}: {e}[/red]")
        except Exception as e:
            rprint(f"[red]Unexpected error scanning {repo_name}: {e}[/red]")
        
        return tasks
    
    def _scan_issues(self, repo, repo_name: str) -> List[Dict[str, Any]]:
        """Scan repository issues"""
        tasks = []
        
        try:
            # Get open issues (not pull requests)
            issues = repo.get_issues(state='open')
            current_time = datetime.now(timezone.utc)
            
            for issue in issues:
                if issue.pull_request:  # Skip pull requests
                    continue
                
                # Check if issue is stale
                days_since_update = (current_time - issue.updated_at).days
                
                # Focus on issues with specific labels or stale issues
                issue_labels = [label.name.lower() for label in issue.labels]
                priority_labels = ['bug', 'help wanted', 'good first issue', 'enhancement']
                
                has_priority_label = any(label in issue_labels for label in priority_labels)
                is_stale = days_since_update > 30
                
                if has_priority_label or is_stale:
                    priority = self._calculate_issue_priority(issue_labels, days_since_update)
                    
                    task = {
                        "id": f"github_issue_{repo_name}_{issue.number}",
                        "title": f"GitHub Issue: {issue.title}",
                        "description": f"Repository: {repo_name}\nIssue #{issue.number}\n\n{issue.body[:500]}{'...' if len(issue.body or '') > 500 else ''}",
                        "type": "github_issue",
                        "priority": priority,
                        "repo_name": repo_name,
                        "issue_number": issue.number,
                        "issue_url": issue.html_url,
                        "labels": [label.name for label in issue.labels],
                        "days_since_update": days_since_update,
                        "created_at": issue.created_at.isoformat(),
                        "updated_at": issue.updated_at.isoformat()
                    }
                    tasks.append(task)
                    
        except Exception as e:
            rprint(f"[yellow]Warning: Error scanning issues: {e}[/yellow]")
        
        return tasks
    
    def _scan_repo_files(self, repo, repo_name: str) -> List[Dict[str, Any]]:
        """Scan repository files for TODOs using GitHub API search"""
        tasks = []
        
        try:
            # Use GitHub search to find TODO comments
            search_terms = ['TODO', 'FIXME', 'HACK', 'XXX']
            
            for term in search_terms:
                try:
                    # Search for the term in the repository
                    search_results = self.client.search_code(
                        query=f"{term} repo:{repo_name}",
                        sort='indexed',
                        order='desc'
                    )
                    
                    # Limit results to avoid rate limiting
                    count = 0
                    for result in search_results:
                        if count >= 10:  # Limit to 10 results per search term
                            break
                        
                        try:
                            # Get file content to extract context
                            file_content = repo.get_contents(result.path)
                            content_lines = file_content.decoded_content.decode('utf-8', errors='ignore').split('\n')
                            
                            # Find the line with the search term
                            for line_num, line in enumerate(content_lines, 1):
                                if term.lower() in line.lower():
                                    task = {
                                        "id": f"github_todo_{repo_name}_{result.path}_{line_num}",
                                        "title": f"Address {term} in {repo_name}:{result.path}",
                                        "description": f"Repository: {repo_name}\nFile: {result.path}\nLine {line_num}: {line.strip()}",
                                        "type": "github_todo",
                                        "priority": self._calculate_todo_priority(term),
                                        "repo_name": repo_name,
                                        "file_path": result.path,
                                        "line_number": line_num,
                                        "file_url": result.html_url,
                                        "search_term": term,
                                        "created_at": datetime.now().isoformat()
                                    }
                                    tasks.append(task)
                                    count += 1
                                    break
                                    
                        except Exception as e:
                            continue  # Skip problematic files
                            
                except Exception as e:
                    rprint(f"[yellow]Warning: Search error for '{term}': {e}[/yellow]")
                    continue
                    
        except Exception as e:
            rprint(f"[yellow]Warning: Error scanning repository files: {e}[/yellow]")
        
        return tasks
    
    def _calculate_issue_priority(self, labels: List[str], days_since_update: int) -> int:
        """Calculate priority for GitHub issues"""
        priority = 5  # Default priority
        
        # Increase priority for important labels
        if 'bug' in labels:
            priority += 3
        if 'critical' in labels or 'urgent' in labels:
            priority += 4
        if 'help wanted' in labels:
            priority += 2
        if 'good first issue' in labels:
            priority += 1
        
        # Increase priority for stale issues
        if days_since_update > 90:
            priority += 2
        elif days_since_update > 60:
            priority += 1
        
        return min(priority, 10)  # Cap at 10
    
    def _calculate_todo_priority(self, term: str) -> int:
        """Calculate priority for TODO items"""
        priority_map = {
            'FIXME': 9,
            'HACK': 8,
            'XXX': 7,
            'TODO': 6
        }
        return priority_map.get(term, 5)
    
    def create_github_issue(self, repo_name: str, title: str, body: str, labels: List[str] = None) -> bool:
        """Create a GitHub issue"""
        if not self.client:
            rprint("[red]Error: GitHub client not available[/red]")
            return False
        
        try:
            repo = self.client.get_repo(repo_name)
            
            # Check for existing issue with same title
            existing_issues = repo.get_issues(state='open')
            for issue in existing_issues:
                if issue.title.lower() == title.lower():
                    rprint(f"[yellow]Issue already exists: #{issue.number}[/yellow]")
                    return False
            
            # Create the issue
            issue = repo.create_issue(
                title=title,
                body=body,
                labels=labels or []
            )
            
            rprint(f"[green]✓ Created issue #{issue.number}: {title}[/green]")
            return True
            
        except GithubException as e:
            rprint(f"[red]Error creating issue: {e}[/red]")
            return False
        except Exception as e:
            rprint(f"[red]Unexpected error: {e}[/red]")
            return False

# CLI interface
app = typer.Typer(
    name="github-scanner",
    help="GitHub Repository Scanner",
    add_completion=False
)
console = Console()

@app.callback()
def main(
    config_file: str = typer.Option(
        "config.json",
        "--config",
        "-c",
        help="Configuration file path",
    ),
):
    """GitHub Repository Scanner"""
    global config, scanner
    config = SimpleConfig(config_file)
    scanner = GitHubScanner(config)

@app.command()
def scan(
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results",
    ),
):
    """Scan GitHub repositories"""
    if not GITHUB_AVAILABLE:
        rprint("[red]PyGithub not installed. Run: pip install PyGithub[/red]")
        return
    
    rprint("[bold blue]🔍 Scanning GitHub repositories...[/bold blue]")
    
    tasks = scanner.scan_repositories()
    
    if tasks:
        # Display results
        table = Table(title=f"Found {len(tasks)} GitHub tasks")
        table.add_column("Type", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Repository", style="green")
        table.add_column("Title", style="white")
        
        for task in sorted(tasks, key=lambda x: x.get("priority", 0), reverse=True):
            table.add_row(
                task.get("type", "unknown"),
                str(task.get("priority", 0)),
                task.get("repo_name", "N/A"),
                task["title"][:50] + "..." if len(task["title"]) > 50 else task["title"]
            )
        
        console.print(table)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(tasks, f, indent=2, default=str)
            rprint(f"[green]✓[/green] Results saved to {output}")
    else:
        rprint("[green]✓[/green] No GitHub tasks found!")

@app.command()
def create_issue(
    repo: str = typer.Argument(..., help="Repository name (owner/repo)"),
    title: str = typer.Argument(..., help="Issue title"),
    body: str = typer.Option("", "--body", "-b", help="Issue body"),
    labels: Optional[str] = typer.Option(None, "--labels", "-l", help="Comma-separated labels"),
):
    """Create a GitHub issue"""
    if not GITHUB_AVAILABLE:
        rprint("[red]PyGithub not installed. Run: pip install PyGithub[/red]")
        return
    
    label_list = [l.strip() for l in labels.split(",")] if labels else []
    
    success = scanner.create_github_issue(repo, title, body, label_list)
    if not success:
        raise typer.Exit(1)

@app.command()
def test_auth():
    """Test GitHub authentication"""
    if not GITHUB_AVAILABLE:
        rprint("[red]PyGithub not installed. Run: pip install PyGithub[/red]")
        return
    
    if not scanner.client:
        rprint("[red]No GitHub token found. Set GITHUB_TOKEN environment variable.[/red]")
        return
    
    try:
        user = scanner.client.get_user()
        rprint(f"[green]✓ Authenticated as: {user.login}[/green]")
        rprint(f"Rate limit: {scanner.client.get_rate_limit().core.remaining}/{scanner.client.get_rate_limit().core.limit}")
    except Exception as e:
        rprint(f"[red]Authentication failed: {e}[/red]")

if __name__ == "__main__":
    app()