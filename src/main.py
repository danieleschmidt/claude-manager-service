#!/usr/bin/env python3
"""
Claude Manager Service - Main CLI Entry Point

This is the primary entry point for the Claude Manager Service, providing a unified
command-line interface for all system operations including task analysis, orchestration,
quantum-enhanced planning, and system management.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Import core system components
try:
    from .core_system import CoreSystem
    from .services.configuration_service import ConfigurationService
    from .services.database_service import DatabaseService
    from .autonomous_status_reporter import AutonomousStatusReporter
    from .continuous_backlog_executor import ContinuousBacklogExecutor
    from .health_check import HealthCheck
    from .logger import setup_logger
    # Quantum CLI import (with fallback)
    try:
        from .quantum_cli import main as quantum_main
    except ImportError:
        print("Warning: Quantum CLI not available")
        quantum_main = None
except ImportError as e:
    print(f"Error importing core components: {e}")
    print("Some functionality may not be available.")
    # Create minimal stubs
    CoreSystem = None
    ConfigurationService = None
    quantum_main = None

# Initialize Typer app and Rich console
app = typer.Typer(
    name="claude-manager",
    help="Claude Manager Service - Autonomous SDLC Management System",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Global configuration
config_service: Optional[ConfigurationService] = None
config_file_path: str = "config.json"
logger = None


@app.callback()
def main(
    config_file: str = typer.Option(
        "config.json",
        "--config",
        "-c",
        help="Configuration file path",
        show_default=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d", 
        help="Enable debug mode",
    ),
):
    """Claude Manager Service - Autonomous SDLC Management System"""
    global config_service, logger
    
    # Setup logging
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    if 'setup_logger' in globals():
        logger = setup_logger("claude-manager", log_level)
    else:
        # Fallback to basic logging
        import logging
        logging.basicConfig(level=getattr(logging, log_level))
        logger = logging.getLogger("claude-manager")
    
    # Store configuration path for async initialization
    global config_file_path
    config_file_path = config_file
    if verbose:
        rprint(f"[green]ℹ[/green] Configuration will be loaded from {config_file}")


@app.command()
def start(
    mode: str = typer.Option(
        "interactive",
        "--mode",
        "-m",
        help="Operation mode: interactive, daemon, one-shot",
        show_default=True,
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Web dashboard port (daemon mode only)",
        show_default=True,
    ),
):
    """Start the Claude Manager Service"""
    rprint("[bold blue]🚀 Starting Claude Manager Service[/bold blue]")
    
    if mode == "interactive":
        asyncio.run(_start_interactive_mode())
    elif mode == "daemon":
        asyncio.run(_start_daemon_mode(port))
    elif mode == "one-shot":
        asyncio.run(_start_one_shot_mode())
    else:
        rprint(f"[red]✗[/red] Unknown mode: {mode}")
        sys.exit(1)


@app.command()
def scan(
    repos: Optional[List[str]] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Specific repositories to scan (default: all configured)",
    ),
    todos: bool = typer.Option(
        True,
        "--todos/--no-todos",
        help="Scan for TODO comments",
        show_default=True,
    ),
    issues: bool = typer.Option(
        True,
        "--issues/--no-issues", 
        help="Scan for open issues",
        show_default=True,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for scan results (JSON format)",
    ),
):
    """Scan repositories for tasks and issues"""
    rprint("[bold blue]🔍 Scanning repositories...[/bold blue]")
    asyncio.run(_scan_repositories(repos, todos, issues, output))


@app.command()
def execute(
    task_id: Optional[str] = typer.Option(
        None,
        "--task-id",
        "-t",
        help="Specific task ID to execute",
    ),
    issue_number: Optional[int] = typer.Option(
        None,
        "--issue",
        "-i",
        help="GitHub issue number to execute",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Repository name (required with --issue)",
    ),
    executor: str = typer.Option(
        "auto",
        "--executor",
        "-e",
        help="Executor type: auto, terragon, claude-flow",
        show_default=True,
    ),
):
    """Execute a specific task or GitHub issue"""
    if issue_number and not repo:
        rprint("[red]✗[/red] Repository name required when specifying issue number")
        sys.exit(1)
        
    rprint("[bold blue]⚡ Executing task...[/bold blue]")
    asyncio.run(_execute_task(task_id, issue_number, repo, executor))


@app.command()
def status(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed system status",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
        show_default=True,
    ),
):
    """Show system status and health"""
    asyncio.run(_show_status(detailed, format))


@app.command()
def quantum(
    args: List[str] = typer.Argument(
        None,
        help="Arguments to pass to quantum CLI",
    ),
):
    """Launch quantum-enhanced task planner CLI"""
    rprint("[bold purple]🔮 Launching Quantum CLI...[/bold purple]")
    
    # Temporarily modify sys.argv to pass arguments to quantum CLI
    original_argv = sys.argv.copy()
    sys.argv = ["quantum"] + (args or [])
    
    try:
        quantum_main()
    finally:
        sys.argv = original_argv


@app.command()
def health():
    """Perform system health check"""
    rprint("[bold blue]🏥 Performing health check...[/bold blue]")
    asyncio.run(_health_check())


@app.command()
def config(
    key: Optional[str] = typer.Option(
        None,
        "--key",
        "-k",
        help="Configuration key to display",
    ),
    set_value: Optional[str] = typer.Option(
        None,
        "--set",
        "-s",
        help="Set configuration value (use with --key)",
    ),
    list_all: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all configuration keys",
    ),
):
    """Manage system configuration"""
    asyncio.run(_manage_config(key, set_value, list_all))


# Async implementation functions

async def _start_interactive_mode():
    """Start interactive mode with continuous execution"""
    try:
        # Initialize configuration service asynchronously
        global config_service
        config_service = ConfigurationService(config_file_path)
        await config_service.initialize()
        
        core_system = await CoreSystem.create(config_service)
        executor = ContinuousBacklogExecutor(
            config_service=config_service,
            core_system=core_system,
            logger=logger
        )
        
        rprint("[green]✓[/green] Interactive mode started. Press Ctrl+C to stop.")
        await executor.run_continuous()
        
    except KeyboardInterrupt:
        rprint("[yellow]⏸[/yellow] Stopping interactive mode...")
    except Exception as e:
        rprint(f"[red]✗[/red] Error in interactive mode: {e}")
        if logger:
            logger.exception("Interactive mode error")
        sys.exit(1)


async def _start_daemon_mode(port: int):
    """Start daemon mode with web dashboard"""
    try:
        # Initialize configuration service asynchronously
        global config_service
        config_service = ConfigurationService(config_file_path)
        await config_service.initialize()
        
        # Import web app using proper path structure
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from web.app import app as web_app
        
        rprint(f"[green]✓[/green] Starting web dashboard on port {port}")
        # Note: Flask app.run() is blocking, so we need to handle this differently
        # For now, we'll run it in a thread to maintain async compatibility
        import threading
        def run_flask():
            web_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # Keep the async event loop alive
        while True:
            await asyncio.sleep(1)
        
    except ImportError as e:
        rprint(f"[red]✗[/red] Web dashboard not available: {e}")
        rprint("[yellow]ℹ[/yellow] To enable web dashboard, ensure Flask dependencies are installed")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]✗[/red] Error starting daemon mode: {e}")
        if logger:
            logger.exception("Daemon mode error")
        sys.exit(1)


async def _start_one_shot_mode():
    """Start one-shot execution mode"""
    try:
        # Initialize configuration service asynchronously
        global config_service
        config_service = ConfigurationService(config_file_path)
        await config_service.initialize()
        
        core_system = await CoreSystem.create(config_service)
        
        # Run single execution cycle
        from .task_analyzer import main as analyzer_main
        await analyzer_main()
        
        rprint("[green]✓[/green] One-shot execution completed")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error in one-shot mode: {e}")
        if logger:
            logger.exception("One-shot mode error")
        sys.exit(1)


async def _scan_repositories(repos: Optional[List[str]], todos: bool, issues: bool, output: Optional[str]):
    """Scan repositories for tasks"""
    try:
        # Initialize configuration service if not already done
        global config_service
        if config_service is None:
            config_service = ConfigurationService(config_file_path)
            await config_service.initialize()
        
        core_system = await CoreSystem.create(config_service)
        
        # Configure scan options
        scan_config = {
            "scanForTodos": todos,
            "scanOpenIssues": issues,
        }
        
        if repos:
            await config_service.set_config("github.reposToScan", repos)
        
        # Perform scan
        results = await core_system.scan_repositories()
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            rprint(f"[green]✓[/green] Scan results saved to {output}")
        else:
            rprint(f"[green]✓[/green] Scan completed. Found {len(results)} items.")
            
    except Exception as e:
        rprint(f"[red]✗[/red] Scan failed: {e}")
        if logger:
            logger.exception("Repository scan error")
        sys.exit(1)


async def _execute_task(task_id: Optional[str], issue_number: Optional[int], repo: Optional[str], executor: str):
    """Execute a specific task"""
    try:
        # Initialize configuration service if not already done
        global config_service
        if config_service is None:
            config_service = ConfigurationService(config_file_path)
            await config_service.initialize()
        
        core_system = await CoreSystem.create(config_service)
        
        if task_id:
            result = await core_system.execute_task_by_id(task_id, executor)
        elif issue_number and repo:
            result = await core_system.execute_github_issue(repo, issue_number, executor)
        else:
            rprint("[red]✗[/red] Must specify either --task-id or --issue with --repo")
            sys.exit(1)
            
        if result:
            rprint("[green]✓[/green] Task executed successfully")
        else:
            rprint("[yellow]⚠[/yellow] Task execution completed with warnings")
            
    except Exception as e:
        rprint(f"[red]✗[/red] Task execution failed: {e}")
        if logger:
            logger.exception("Task execution error")
        sys.exit(1)


async def _show_status(detailed: bool, format_type: str):
    """Show system status"""
    try:
        # Initialize configuration service if not already done
        global config_service
        if config_service is None:
            config_service = ConfigurationService(config_file_path)
            await config_service.initialize()
        
        # Initialize status reporter
        reporter = AutonomousStatusReporter(config_service)
        status = await reporter.generate_comprehensive_status()
        
        if format_type == "json":
            print(json.dumps(status, indent=2, default=str))
            return
            
        # Display as table
        table = Table(title="Claude Manager Service Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        for component, info in status.items():
            if isinstance(info, dict):
                status_str = info.get("status", "Unknown")
                details = info.get("details", "")
            else:
                status_str = str(info)
                details = ""
                
            table.add_row(component, status_str, details)
        
        console.print(table)
        
        if detailed:
            console.print("\n[bold]Detailed Information:[/bold]")
            console.print(json.dumps(status, indent=2, default=str))
            
    except Exception as e:
        rprint(f"[red]✗[/red] Failed to get system status: {e}")
        if logger:
            logger.exception("Status check error")
        sys.exit(1)


async def _health_check():
    """Perform comprehensive health check"""
    try:
        # Initialize configuration service if not already done
        global config_service
        if config_service is None:
            config_service = ConfigurationService(config_file_path)
            await config_service.initialize()
        
        health = HealthCheck()
        results = await health.comprehensive_health_check()
        
        table = Table(title="System Health Check")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Message", style="dim")
        
        all_healthy = True
        for check_name, result in results.items():
            status = "✓ PASS" if result["healthy"] else "✗ FAIL"
            if not result["healthy"]:
                all_healthy = False
                
            table.add_row(
                check_name.replace("_", " ").title(),
                status,
                result["message"]
            )
        
        console.print(table)
        
        if all_healthy:
            rprint("\n[green]✓ All health checks passed![/green]")
        else:
            rprint("\n[red]✗ Some health checks failed. Please review the issues above.[/red]")
            sys.exit(1)
            
    except Exception as e:
        rprint(f"[red]✗[/red] Health check failed: {e}")
        if logger:
            logger.exception("Health check error")
        sys.exit(1)


async def _manage_config(key: Optional[str], set_value: Optional[str], list_all: bool):
    """Manage configuration"""
    try:
        # Initialize configuration service if not already done
        global config_service
        if config_service is None:
            config_service = ConfigurationService(config_file_path)
            await config_service.initialize()
        
        if list_all:
            table = Table(title="Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="dim")
            
            def flatten_config(config, prefix=""):
                items = []
                for k, v in config.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.extend(flatten_config(v, full_key))
                    else:
                        items.append((full_key, str(v)))
                return items
            
            current_config = await config_service.get_config()
            for config_key, value in flatten_config(current_config):
                table.add_row(config_key, value)
                
            console.print(table)
            
        elif key and set_value:
            # Set configuration value
            await config_service.set_config(key, set_value, persist=True)
            rprint(f"[green]✓[/green] Configuration updated: {key} = {set_value}")
            
        elif key:
            # Get configuration value
            try:
                value = await config_service.get_config(key)
                rprint(f"{key}: {value}")
            except Exception:
                rprint(f"[red]✗[/red] Configuration key not found: {key}")
                sys.exit(1)
        else:
            rprint("[yellow]⚠[/yellow] Use --list to show all config, --key to get/set values")
            
    except Exception as e:
        rprint(f"[red]✗[/red] Configuration management failed: {e}")
        if logger:
            logger.exception("Configuration management error")
        sys.exit(1)


# Entry point for setuptools
def cli():
    """Entry point for setuptools console script"""
    app()


if __name__ == "__main__":
    app()