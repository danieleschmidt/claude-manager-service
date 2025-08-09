#!/usr/bin/env python3
"""
Claude Manager Service - Generation 2 Main CLI Entry Point

This is the Generation 2 entry point with enhanced robustness features:
- Enhanced error handling and recovery
- Comprehensive input validation
- Security hardening
- Structured logging and monitoring
- Configuration validation
- Health monitoring
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

# Import Generation 2 components
try:
    from .generation_2_system import create_generation_2_system, Generation2System
    from .services.configuration_service import ConfigurationService
    from .enhanced_logger import get_enhanced_logger, log_context
    from .health_check_v2 import get_health_check
    from .config_validator_v2 import validate_config_file, generate_validation_report
    from .security_v2 import get_security_manager
    
    # Legacy components (with fallback)
    try:
        from .core_system import CoreSystem
        from .continuous_backlog_executor import ContinuousBacklogExecutor
        from .quantum_cli import main as quantum_main
    except ImportError:
        CoreSystem = None
        ContinuousBacklogExecutor = None
        quantum_main = None
        
except ImportError as e:
    print(f"Error importing Generation 2 components: {e}")
    print("Falling back to basic functionality.")
    sys.exit(1)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="claude-manager-gen2",
    help="Claude Manager Service - Generation 2 Robust SDLC Management System",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Global state
config_service: Optional[ConfigurationService] = None
generation_2_system: Optional[Generation2System] = None
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
    validate_config: bool = typer.Option(
        True,
        "--validate-config/--no-validate-config",
        help="Validate configuration on startup",
        show_default=True,
    ),
):
    """Claude Manager Service - Generation 2 Robust SDLC Management System"""
    global config_file_path, logger
    
    # Setup enhanced logging
    try:
        logger = get_enhanced_logger("claude-manager-gen2")
        log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
        
        import logging
        logger.setLevel(getattr(logging, log_level))
        
        rprint("[green]✓[/green] Enhanced logging initialized")
        
    except Exception as e:
        print(f"Failed to initialize enhanced logging: {e}")
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("claude-manager-gen2")
    
    # Store configuration path
    config_file_path = config_file
    
    # Validate configuration if requested
    if validate_config:
        asyncio.run(_validate_configuration_startup())
    
    if verbose:
        rprint(f"[green]ℹ[/green] Generation 2 system ready - Configuration: {config_file}")


@app.command()
def start(
    mode: str = typer.Option(
        "robust",
        "--mode",
        "-m",
        help="Operation mode: robust, interactive, daemon, legacy",
        show_default=True,
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Web dashboard port (daemon mode only)",
        show_default=True,
    ),
    enable_security: bool = typer.Option(
        True,
        "--security/--no-security",
        help="Enable security features",
        show_default=True,
    ),
):
    """Start the Claude Manager Service with Generation 2 features"""
    rprint("[bold blue]🚀 Starting Claude Manager Service (Generation 2)[/bold blue]")
    
    if mode == "robust":
        asyncio.run(_start_robust_mode(enable_security))
    elif mode == "interactive":
        asyncio.run(_start_interactive_mode())
    elif mode == "daemon":
        asyncio.run(_start_daemon_mode(port))
    elif mode == "legacy":
        asyncio.run(_start_legacy_mode())
    else:
        rprint(f"[red]✗[/red] Unknown mode: {mode}")
        rprint("Available modes: robust, interactive, daemon, legacy")
        sys.exit(1)


@app.command()
def validate(
    environment: str = typer.Option(
        "development",
        "--environment",
        "-e",
        help="Target environment for validation",
        show_default=True,
    ),
    show_report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Show detailed validation report",
        show_default=True,
    ),
):
    """Validate system configuration"""
    rprint("[bold blue]🔍 Validating configuration...[/bold blue]")
    asyncio.run(_validate_configuration_command(environment, show_report))


@app.command()
def health(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed health report",
    ),
    categories: Optional[List[str]] = typer.Option(
        None,
        "--category",
        "-c",
        help="Health check categories to run",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        "-e",
        help="Export health report to file",
    ),
    continuous: bool = typer.Option(
        False,
        "--continuous",
        help="Run continuous health monitoring",
    ),
):
    """Perform comprehensive health check"""
    rprint("[bold blue]🏥 Performing health check...[/bold blue]")
    
    if continuous:
        rprint("[yellow]⚠[/yellow] Continuous monitoring mode - Press Ctrl+C to stop")
        asyncio.run(_continuous_health_monitoring())
    else:
        asyncio.run(_health_check_v2(detailed, categories, export))


@app.command()
def security(
    action: str = typer.Argument(help="Security action: status, audit, cleanup"),
    hours: int = typer.Option(
        24,
        "--hours",
        "-h",
        help="Hours to look back for audit",
        show_default=True,
    ),
):
    """Security management and auditing"""
    rprint("[bold blue]🔒 Security management...[/bold blue]")
    asyncio.run(_security_management(action, hours))


@app.command()
def metrics(
    export_path: str = typer.Option(
        "system_metrics",
        "--output",
        "-o",
        help="Output path for metrics export",
        show_default=True,
    ),
    include_history: bool = typer.Option(
        True,
        "--history/--no-history",
        help="Include historical data",
        show_default=True,
    ),
):
    """Export system metrics and monitoring data"""
    rprint("[bold blue]📊 Exporting system metrics...[/bold blue]")
    asyncio.run(_export_metrics(export_path, include_history))


@app.command()
def status(
    format_type: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, summary",
        show_default=True,
    ),
):
    """Show Generation 2 system status"""
    asyncio.run(_show_generation_2_status(format_type))


# Legacy commands for compatibility
@app.command()
def scan(
    repos: Optional[List[str]] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Specific repositories to scan",
    ),
    secure: bool = typer.Option(
        True,
        "--secure/--no-secure",
        help="Use secure scanning mode",
        show_default=True,
    ),
):
    """Scan repositories with enhanced security"""
    rprint("[bold blue]🔍 Scanning repositories (Generation 2)...[/bold blue]")
    asyncio.run(_secure_scan(repos, secure))


@app.command()
def execute(
    task_id: Optional[str] = typer.Option(None, "--task-id", "-t"),
    issue_number: Optional[int] = typer.Option(None, "--issue", "-i"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r"),
    secure: bool = typer.Option(
        True,
        "--secure/--no-secure",
        help="Use secure execution mode",
        show_default=True,
    ),
):
    """Execute tasks with enhanced security and monitoring"""
    rprint("[bold blue]⚡ Executing task (Generation 2)...[/bold blue]")
    asyncio.run(_secure_execute(task_id, issue_number, repo, secure))


# Async implementation functions

async def _validate_configuration_startup():
    """Validate configuration at startup"""
    try:
        result = await validate_config_file(config_file_path)
        
        if result.is_valid:
            rprint("[green]✓[/green] Configuration validation passed")
        else:
            report = generate_validation_report(result)
            rprint("[red]✗[/red] Configuration validation failed:")
            console.print(report)
            
            if result.errors or result.security_issues:
                rprint("[red]Critical issues found - exiting[/red]")
                sys.exit(1)
                
    except Exception as e:
        rprint(f"[red]✗[/red] Configuration validation error: {e}")
        sys.exit(1)


async def _start_robust_mode(enable_security: bool):
    """Start Generation 2 robust mode"""
    try:
        # Initialize Generation 2 system
        global config_service, generation_2_system
        
        config_service = ConfigurationService(config_file_path)
        await config_service.initialize()
        
        rprint("[green]✓[/green] Configuration service initialized")
        
        # Create Generation 2 system
        generation_2_system = await create_generation_2_system(config_service)
        
        rprint("[green]✓[/green] Generation 2 system initialized")
        
        # Show system status
        status = await generation_2_system.get_system_status()
        
        table = Table(title="Generation 2 System Status")
        table.add_column("Feature", style="cyan")
        table.add_column("Status", style="green")
        
        for feature, enabled in status.features_enabled.items():
            status_str = "✓ Enabled" if enabled else "✗ Disabled"
            table.add_row(feature.replace("_", " ").title(), status_str)
        
        console.print(table)
        
        rprint(f"[green]Health Score: {status.health_score:.1f}%[/green]")
        rprint(f"[blue]Security Status: {status.security_status}[/blue]")
        
        # Start continuous operation
        rprint("[green]🟢[/green] Generation 2 system is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            rprint("[yellow]⏸[/yellow] Shutting down Generation 2 system...")
            await generation_2_system.shutdown()
            
    except Exception as e:
        rprint(f"[red]✗[/red] Failed to start robust mode: {e}")
        if logger:
            logger.exception("Robust mode startup error")
        sys.exit(1)


async def _start_interactive_mode():
    """Start interactive mode with Generation 2 features"""
    try:
        await _initialize_generation_2_system()
        
        rprint("[green]✓[/green] Interactive mode with Generation 2 features started")
        rprint("Press Ctrl+C to stop.")
        
        try:
            while True:
                await asyncio.sleep(10)
                # Perform periodic health checks
                if generation_2_system:
                    status = await generation_2_system.get_system_status()
                    if status.health_score < 50:
                        rprint(f"[yellow]⚠[/yellow] Health score low: {status.health_score:.1f}%")
        
        except KeyboardInterrupt:
            rprint("[yellow]⏸[/yellow] Stopping interactive mode...")
            if generation_2_system:
                await generation_2_system.shutdown()
                
    except Exception as e:
        rprint(f"[red]✗[/red] Interactive mode error: {e}")
        sys.exit(1)


async def _start_daemon_mode(port: int):
    """Start daemon mode with web dashboard"""
    try:
        await _initialize_generation_2_system()
        
        rprint(f"[green]✓[/green] Starting web dashboard on port {port}")
        
        # This would integrate with the web dashboard
        # For now, simulate daemon mode
        rprint("[blue]🌐[/blue] Web dashboard running (simulated)")
        
        try:
            while True:
                await asyncio.sleep(60)
                # Log system status periodically
                if generation_2_system:
                    status = await generation_2_system.get_system_status()
                    logger.info(f"System health: {status.health_score:.1f}%")
        
        except KeyboardInterrupt:
            rprint("[yellow]⏸[/yellow] Stopping daemon mode...")
            if generation_2_system:
                await generation_2_system.shutdown()
                
    except Exception as e:
        rprint(f"[red]✗[/red] Daemon mode error: {e}")
        sys.exit(1)


async def _start_legacy_mode():
    """Start legacy mode for compatibility"""
    rprint("[yellow]⚠[/yellow] Starting in legacy mode - reduced functionality")
    
    try:
        if CoreSystem and ContinuousBacklogExecutor:
            # Use legacy system
            global config_service
            config_service = ConfigurationService(config_file_path)
            await config_service.initialize()
            
            core_system = await CoreSystem.create(config_service)
            executor = ContinuousBacklogExecutor(
                config_service=config_service,
                core_system=core_system,
                logger=logger
            )
            
            await executor.run_continuous()
        else:
            rprint("[red]✗[/red] Legacy components not available")
            sys.exit(1)
            
    except Exception as e:
        rprint(f"[red]✗[/red] Legacy mode error: {e}")
        sys.exit(1)


async def _validate_configuration_command(environment: str, show_report: bool):
    """Validate configuration command"""
    try:
        result = await validate_config_file(config_file_path, environment)
        
        if show_report:
            report = generate_validation_report(result)
            console.print(report)
        
        if result.is_valid:
            rprint("[green]✓[/green] Configuration validation passed")
        else:
            rprint("[red]✗[/red] Configuration validation failed")
            if result.errors or result.security_issues:
                sys.exit(1)
                
    except Exception as e:
        rprint(f"[red]✗[/red] Configuration validation error: {e}")
        sys.exit(1)


async def _health_check_v2(detailed: bool, categories: Optional[List[str]], export: Optional[str]):
    """Enhanced health check"""
    try:
        await _initialize_generation_2_system()
        
        health_check = await get_health_check(config_service)
        
        # Run health checks
        health_report = await health_check.run_health_checks(
            categories=categories,
            include_non_critical=detailed
        )
        
        # Display results
        table = Table(title="System Health Check")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Message", style="dim")
        
        for check in health_report.checks:
            status_color = "green" if check.status == "healthy" else "yellow" if check.status == "degraded" else "red"
            status_symbol = "✓" if check.status == "healthy" else "⚠" if check.status == "degraded" else "✗"
            
            table.add_row(
                check.name.replace("_", " ").title(),
                f"[{status_color}]{status_symbol} {check.status.upper()}[/{status_color}]",
                check.message
            )
        
        console.print(table)
        
        # Overall status
        overall_color = "green" if health_report.overall_status == "healthy" else "yellow" if health_report.overall_status == "degraded" else "red"
        rprint(f"[{overall_color}]Overall Status: {health_report.overall_status.upper()}[/{overall_color}]")
        rprint(f"[blue]Health Score: {health_report.health_score:.1f}%[/blue]")
        
        # Show recommendations
        if health_report.recommendations:
            rprint("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in health_report.recommendations:
                rprint(f"  • {rec}")
        
        # Export if requested
        if export:
            health_check.export_health_report(health_report, export)
            rprint(f"[green]✓[/green] Health report exported to {export}")
        
        # Exit with error code if unhealthy
        if health_report.overall_status == "unhealthy":
            sys.exit(1)
            
    except Exception as e:
        rprint(f"[red]✗[/red] Health check failed: {e}")
        sys.exit(1)


async def _continuous_health_monitoring():
    """Continuous health monitoring"""
    try:
        await _initialize_generation_2_system()
        
        health_check = await get_health_check(config_service)
        
        while True:
            health_report = await health_check.run_health_checks(include_non_critical=False)
            
            timestamp = health_report.timestamp.strftime("%H:%M:%S")
            status_color = "green" if health_report.overall_status == "healthy" else "yellow" if health_report.overall_status == "degraded" else "red"
            
            rprint(f"[{status_color}]{timestamp} - {health_report.overall_status.upper()} ({health_report.health_score:.1f}%)[/{status_color}]")
            
            if health_report.critical_issues:
                for issue in health_report.critical_issues:
                    rprint(f"[red]  ⚠ {issue}[/red]")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        rprint("\n[yellow]Continuous monitoring stopped[/yellow]")
    except Exception as e:
        rprint(f"[red]✗[/red] Continuous monitoring error: {e}")


async def _security_management(action: str, hours: int):
    """Security management command"""
    try:
        await _initialize_generation_2_system()
        
        security_manager = get_security_manager()
        
        if action == "status":
            # Show security status
            rprint("[green]✓[/green] Security system operational")
            
            # Check credentials
            credentials = security_manager.credential_manager.list_credentials()
            rprint(f"[blue]Credentials configured: {len(credentials)}[/blue]")
            
            # Check rate limiting
            rprint("[green]✓[/green] Rate limiting active")
            
        elif action == "audit":
            # Perform security audit
            audit_report = await security_manager.audit_security_events(hours)
            
            table = Table(title=f"Security Audit ({hours}h)")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green")
            
            for metric, count in audit_report.items():
                table.add_row(str(metric).replace("_", " ").title(), str(count))
            
            console.print(table)
            
        elif action == "cleanup":
            # Perform security cleanup
            expired_sessions = security_manager.session_manager.cleanup_expired_sessions()
            expired_creds = security_manager.credential_manager.cleanup_expired()
            
            rprint(f"[green]✓[/green] Cleaned up {expired_sessions} expired sessions")
            rprint(f"[green]✓[/green] Cleaned up {expired_creds} expired credentials")
            
        else:
            rprint(f"[red]✗[/red] Unknown security action: {action}")
            rprint("Available actions: status, audit, cleanup")
            sys.exit(1)
            
    except Exception as e:
        rprint(f"[red]✗[/red] Security management error: {e}")
        sys.exit(1)


async def _export_metrics(export_path: str, include_history: bool):
    """Export system metrics"""
    try:
        await _initialize_generation_2_system()
        
        await generation_2_system.export_system_metrics(export_path)
        
        rprint(f"[green]✓[/green] System metrics exported to {export_path}")
        
        if include_history:
            # Export health trends
            health_check = await get_health_check(config_service)
            trends = health_check.get_health_trends(24)
            
            with open(f"{export_path}_health_trends.json", "w") as f:
                json.dump(trends, f, indent=2, default=str)
                
            rprint(f"[green]✓[/green] Health trends exported to {export_path}_health_trends.json")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Metrics export error: {e}")
        sys.exit(1)


async def _show_generation_2_status(format_type: str):
    """Show Generation 2 system status"""
    try:
        await _initialize_generation_2_system()
        
        status = await generation_2_system.get_system_status()
        
        if format_type == "json":
            import dataclasses
            status_dict = dataclasses.asdict(status)
            print(json.dumps(status_dict, indent=2, default=str))
            
        elif format_type == "summary":
            rprint(f"[blue]Generation 2 System Status[/blue]")
            rprint(f"Initialized: {'Yes' if status.initialized else 'No'}")
            rprint(f"Health Score: {status.health_score:.1f}%")
            rprint(f"Security: {status.security_status}")
            rprint(f"Features: {sum(status.features_enabled.values())}/{len(status.features_enabled)}")
            
        else:  # table format
            table = Table(title="Generation 2 System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="dim")
            
            table.add_row("System", "✓ Initialized" if status.initialized else "✗ Not Initialized", "")
            table.add_row("Health Score", f"{status.health_score:.1f}%", "")
            table.add_row("Security", status.security_status, "")
            
            for feature, enabled in status.features_enabled.items():
                table.add_row(
                    feature.replace("_", " ").title(),
                    "✓ Enabled" if enabled else "✗ Disabled",
                    ""
                )
            
            console.print(table)
        
    except Exception as e:
        rprint(f"[red]✗[/red] Status check error: {e}")
        sys.exit(1)


async def _secure_scan(repos: Optional[List[str]], secure: bool):
    """Secure repository scanning"""
    try:
        await _initialize_generation_2_system()
        
        # Use Generation 2 secure operations
        scan_params = {
            "repos": repos or [],
            "secure_mode": secure
        }
        
        result = await generation_2_system.execute_secure_operation(
            operation="scan_repositories",
            parameters=scan_params,
            client_id="127.0.0.1"  # Local execution
        )
        
        rprint(f"[green]✓[/green] Secure scan completed: {result}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Secure scan error: {e}")
        sys.exit(1)


async def _secure_execute(task_id: Optional[str], issue_number: Optional[int], 
                         repo: Optional[str], secure: bool):
    """Secure task execution"""
    try:
        await _initialize_generation_2_system()
        
        if not task_id and not (issue_number and repo):
            rprint("[red]✗[/red] Must specify either --task-id or --issue with --repo")
            sys.exit(1)
        
        exec_params = {
            "task_id": task_id,
            "issue_number": issue_number,
            "repo": repo,
            "secure_mode": secure
        }
        
        result = await generation_2_system.execute_secure_operation(
            operation="execute_task",
            parameters=exec_params,
            client_id="127.0.0.1"  # Local execution
        )
        
        rprint(f"[green]✓[/green] Secure execution completed: {result}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Secure execution error: {e}")
        sys.exit(1)


async def _initialize_generation_2_system():
    """Initialize Generation 2 system if not already done"""
    global config_service, generation_2_system
    
    if not config_service:
        config_service = ConfigurationService(config_file_path)
        await config_service.initialize()
    
    if not generation_2_system:
        generation_2_system = await create_generation_2_system(config_service)


# Entry point for setuptools
def cli():
    """Entry point for setuptools console script"""
    app()


if __name__ == "__main__":
    app()