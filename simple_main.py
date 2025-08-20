#!/usr/bin/env python3
"""
Claude Manager Service - Simple Generation 1 Implementation
MAKE IT WORK - Basic functional version
"""

import json
import os
import sys
from pathlib import Path
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Simple configuration manager
class SimpleConfig:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                rprint(f"[yellow]Warning: Error loading config: {e}[/yellow]")
        
        # Default configuration
        default_config = {
            "github": {
                "username": "your-username",
                "managerRepo": "your-username/claude-manager-service",
                "reposToScan": []
            },
            "analyzer": {
                "scanForTodos": True,
                "scanOpenIssues": True
            },
            "executor": {
                "terragonUsername": "@terragon-labs"
            }
        }
        
        # Save default config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            rprint(f"[green]Created default config at {self.config_path}[/green]")
        except Exception as e:
            rprint(f"[yellow]Warning: Could not save default config: {e}[/yellow]")
        
        return default_config
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# Simple task analyzer
class SimpleTaskAnalyzer:
    def __init__(self, config: SimpleConfig):
        self.config = config
        
    def scan_repository(self, repo_path: str = ".") -> List[Dict[str, Any]]:
        """Simple repository scanning for TODOs and basic issues"""
        tasks = []
        
        # Scan for TODO comments
        if self.config.get("analyzer.scanForTodos", True):
            tasks.extend(self._scan_todos(repo_path))
        
        # Check for basic project health
        tasks.extend(self._check_project_health(repo_path))
        
        return tasks
    
    def _scan_todos(self, repo_path: str) -> List[Dict[str, Any]]:
        """Scan for TODO/FIXME comments"""
        tasks = []
        search_patterns = ["TODO", "FIXME", "HACK", "XXX"]
        
        try:
            for pattern in ["**/*.py", "**/*.js", "**/*.md", "**/*.txt"]:
                for file_path in Path(repo_path).glob(pattern):
                    if file_path.is_file() and not any(x in str(file_path) for x in ['.git', '__pycache__', 'node_modules', 'venv']):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                    for search_pattern in search_patterns:
                                        if search_pattern.lower() in line.lower():
                                            tasks.append({
                                                "id": f"todo_{len(tasks)}",
                                                "title": f"Address {search_pattern} in {file_path.name}",
                                                "description": f"Line {line_num}: {line.strip()}",
                                                "file_path": str(file_path),
                                                "line_number": line_num,
                                                "type": "code_improvement",
                                                "priority": self._calculate_priority(search_pattern),
                                                "created_at": datetime.now().isoformat()
                                            })
                                            break
                        except Exception as e:
                            continue
        except Exception as e:
            rprint(f"[yellow]Warning: Error scanning TODOs: {e}[/yellow]")
            
        return tasks
    
    def _check_project_health(self, repo_path: str) -> List[Dict[str, Any]]:
        """Check basic project health indicators"""
        tasks = []
        
        # Check for missing essential files
        essential_files = {
            "README.md": "Add project README documentation",
            "requirements.txt": "Add Python requirements file",
            ".gitignore": "Add gitignore file",
            "LICENSE": "Add license file"
        }
        
        for file_name, description in essential_files.items():
            file_path = Path(repo_path) / file_name
            if not file_path.exists():
                tasks.append({
                    "id": f"missing_{file_name}",
                    "title": f"Missing {file_name}",
                    "description": description,
                    "type": "project_setup",
                    "priority": 7,
                    "created_at": datetime.now().isoformat()
                })
        
        return tasks
    
    def _calculate_priority(self, pattern: str) -> int:
        """Calculate task priority based on pattern"""
        priority_map = {
            "FIXME": 9,
            "TODO": 6,
            "HACK": 8,
            "XXX": 7
        }
        return priority_map.get(pattern, 5)

# Simple health checker
class SimpleHealthCheck:
    def __init__(self, config: SimpleConfig):
        self.config = config
    
    def check_health(self) -> Dict[str, Any]:
        """Perform basic health checks"""
        checks = {}
        
        # Check configuration
        checks["configuration"] = {
            "status": "OK" if self.config.config else "ERROR",
            "message": "Configuration loaded successfully" if self.config.config else "No configuration found"
        }
        
        # Check GitHub token
        github_token = os.getenv("GITHUB_TOKEN")
        checks["github_token"] = {
            "status": "OK" if github_token else "WARNING",
            "message": "GitHub token found" if github_token else "GITHUB_TOKEN environment variable not set"
        }
        
        # Check Python environment
        checks["python_environment"] = {
            "status": "OK",
            "message": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        
        # Check working directory
        checks["working_directory"] = {
            "status": "OK",
            "message": f"Current directory: {os.getcwd()}"
        }
        
        return checks

# Initialize Typer app
app = typer.Typer(
    name="claude-manager-simple",
    help="Claude Manager Service - Simple Generation 1 Implementation",
    add_completion=False,
    rich_markup_mode="rich",
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Claude Manager Service - Simple Generation 1"""
    global config
    config = SimpleConfig(config_file)
    if verbose:
        rprint(f"[green]ℹ[/green] Loaded configuration from {config_file}")

@app.command()
def scan(
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to scan",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results",
    ),
):
    """Scan for tasks and issues"""
    rprint("[bold blue]🔍 Scanning for tasks...[/bold blue]")
    
    analyzer = SimpleTaskAnalyzer(config)
    tasks = analyzer.scan_repository(path)
    
    if tasks:
        # Display results
        table = Table(title=f"Found {len(tasks)} tasks")
        table.add_column("Type", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Title", style="green")
        table.add_column("File", style="dim")
        
        for task in sorted(tasks, key=lambda x: x.get("priority", 0), reverse=True):
            table.add_row(
                task.get("type", "unknown"),
                str(task.get("priority", 0)),
                task["title"],
                task.get("file_path", "N/A")
            )
        
        console.print(table)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(tasks, f, indent=2)
            rprint(f"[green]✓[/green] Results saved to {output}")
    else:
        rprint("[green]✓[/green] No issues found!")

@app.command()
def health():
    """Perform system health check"""
    rprint("[bold blue]🏥 Performing health check...[/bold blue]")
    
    health_checker = SimpleHealthCheck(config)
    checks = health_checker.check_health()
    
    table = Table(title="System Health Check")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Message", style="dim")
    
    all_ok = True
    for check_name, result in checks.items():
        status = result["status"]
        if status != "OK":
            all_ok = False
        
        status_style = "green" if status == "OK" else ("yellow" if status == "WARNING" else "red")
        table.add_row(
            check_name.replace("_", " ").title(),
            f"[{status_style}]{status}[/{status_style}]",
            result["message"]
        )
    
    console.print(table)
    
    if all_ok:
        rprint("\n[green]✓ All checks passed![/green]")
    else:
        rprint("\n[yellow]⚠ Some checks have warnings or errors[/yellow]")

@app.command()
def status():
    """Show system status"""
    rprint("[bold blue]📊 System Status[/bold blue]")
    
    status_info = {
        "version": "1.0.0 (Generation 1)",
        "mode": "Simple",
        "config_file": config.config_path,
        "repos_configured": len(config.get("github.reposToScan", [])),
        "scan_todos": config.get("analyzer.scanForTodos", False),
        "scan_issues": config.get("analyzer.scanOpenIssues", False),
    }
    
    table = Table(title="Current Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in status_info.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)

@app.command()
def config_show():
    """Show current configuration"""
    rprint("[bold blue]⚙️  Configuration[/bold blue]")
    
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_config = flatten_dict(config.config)
    
    table = Table(title="Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in flat_config.items():
        table.add_row(key, str(value))
    
    console.print(table)

@app.command()
def init():
    """Initialize a new configuration"""
    rprint("[bold blue]🚀 Initializing Claude Manager...[/bold blue]")
    
    # Interactive setup
    github_username = typer.prompt("GitHub username")
    manager_repo = typer.prompt("Manager repository (e.g., username/claude-manager-service)")
    repos_to_scan = typer.prompt("Repositories to scan (comma-separated)", default="")
    
    new_config = {
        "github": {
            "username": github_username,
            "managerRepo": manager_repo,
            "reposToScan": [repo.strip() for repo in repos_to_scan.split(",") if repo.strip()]
        },
        "analyzer": {
            "scanForTodos": True,
            "scanOpenIssues": True
        },
        "executor": {
            "terragonUsername": "@terragon-labs"
        }
    }
    
    try:
        with open(config.config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        rprint(f"[green]✓[/green] Configuration saved to {config.config_path}")
        
        # Reload config
        config.config = new_config
        rprint("[green]✓[/green] Configuration reloaded")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error saving configuration: {e}")

if __name__ == "__main__":
    app()