"""
Async Orchestrator for Claude Manager Service

This module provides async orchestration capabilities for the entire
task analysis and management workflow with significant performance
improvements through concurrent processing.

Features:
- Async/await orchestration of all operations
- Concurrent repository processing
- Non-blocking I/O for all external operations
- Performance monitoring and error handling
- Backward compatibility with synchronous components
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .async_task_analyzer import AsyncTaskAnalyzer
from .logger import get_logger
from .performance_monitor import monitor_performance, get_monitor
from .config_validator import get_validated_config
from .error_handler import NetworkError, with_enhanced_error_handling


logger = get_logger(__name__)


class AsyncOrchestrator:
    """
    Async orchestrator for the complete task management workflow
    
    This class coordinates all async operations including repository analysis,
    issue creation, and performance monitoring with concurrent execution.
    """
    
    def __init__(self, config_path: str = 'config.json', max_concurrent_repos: int = 5):
        """
        Initialize async orchestrator
        
        Args:
            config_path: Path to configuration file
            max_concurrent_repos: Maximum concurrent repository operations
        """
        self.logger = get_logger(__name__)
        self.config = get_validated_config(config_path)
        self.max_concurrent_repos = max_concurrent_repos
        
        # Initialize async task analyzer
        self.task_analyzer = AsyncTaskAnalyzer(config_path, max_concurrent_repos)
        
        # Performance monitoring
        self.monitor = get_monitor()
        
        self.logger.info(f"Async orchestrator initialized with max_concurrent={max_concurrent_repos}")
    
    @with_enhanced_error_handling("async_orchestration", use_rate_limiter=True, use_circuit_breaker=True)
    async def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete async task analysis workflow
        
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        self.logger.info("Starting async orchestration workflow")
        
        workflow_results = {
            'workflow_start_time': datetime.now().isoformat(),
            'task_analysis': {},
            'performance_metrics': {},
            'errors': [],
            'total_execution_time': 0,
            'success': False
        }
        
        try:
            # Run async task analysis
            self.logger.info("Executing async task analysis...")
            with monitor_performance("async_task_analysis"):
                analysis_results = await self.task_analyzer.run_analysis_async()
            
            workflow_results['task_analysis'] = analysis_results
            
            # Collect performance metrics
            workflow_results['performance_metrics'] = await self._collect_performance_metrics_async()
            
            # Mark as successful
            workflow_results['success'] = True
            workflow_results['total_execution_time'] = time.time() - start_time
            
            self.logger.info(
                f"Async orchestration completed successfully in {workflow_results['total_execution_time']:.2f}s"
            )
            
            return workflow_results
            
        except Exception as e:
            error_msg = f"Async orchestration failed: {str(e)}"
            self.logger.error(error_msg)
            workflow_results['errors'].append(error_msg)
            workflow_results['total_execution_time'] = time.time() - start_time
            
            raise NetworkError("Async orchestration failed", "run_full_analysis", e)
    
    async def _collect_performance_metrics_async(self) -> Dict[str, Any]:
        """
        Collect performance metrics asynchronously
        
        Returns:
            Performance metrics dictionary
        """
        try:
            # Run metrics collection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def get_sync_metrics():
                return {
                    'api_call_stats': self.monitor.get_api_call_stats(),
                    'performance_history': self.monitor.get_performance_history(),
                    'memory_usage': self.monitor.get_memory_usage(),
                    'error_statistics': self.monitor.get_error_statistics()
                }
            
            metrics = await loop.run_in_executor(None, get_sync_metrics)
            
            self.logger.debug("Performance metrics collected asynchronously")
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect performance metrics: {e}")
            return {'error': str(e)}
    
    async def run_scheduled_analysis(self, interval_seconds: int = 3600) -> None:
        """
        Run scheduled async analysis at regular intervals
        
        Args:
            interval_seconds: Interval between analyses in seconds
        """
        self.logger.info(f"Starting scheduled async analysis (interval: {interval_seconds}s)")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                self.logger.info(f"Starting scheduled analysis iteration {iteration}")
                
                # Run full analysis
                results = await self.run_full_analysis()
                
                # Log results summary
                task_results = results.get('task_analysis', {})
                self.logger.info(
                    f"Scheduled analysis {iteration} completed: "
                    f"{task_results.get('repositories_scanned', 0)} repos, "
                    f"{task_results.get('todos_found', 0)} TODOs, "
                    f"execution time: {results.get('total_execution_time', 0):.2f}s"
                )
                
                # Wait for next iteration
                self.logger.info(f"Waiting {interval_seconds}s until next analysis...")
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("Scheduled analysis interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Scheduled analysis iteration {iteration} failed: {e}")
                # Continue with next iteration after shorter delay
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def run_single_repository_analysis(self, repo_name: str) -> Dict[str, Any]:
        """
        Run async analysis for a single repository
        
        Args:
            repo_name: Repository name to analyze
            
        Returns:
            Analysis results for the repository
        """
        self.logger.info(f"Running async analysis for single repository: {repo_name}")
        
        # Temporarily override configuration for single repo
        original_repos = self.task_analyzer.repos_to_scan
        self.task_analyzer.repos_to_scan = [repo_name]
        
        try:
            results = await self.task_analyzer.run_analysis_async()
            self.logger.info(f"Single repository analysis completed for {repo_name}")
            return results
            
        finally:
            # Restore original configuration
            self.task_analyzer.repos_to_scan = original_repos
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform async health check of all components
        
        Returns:
            Health check results
        """
        self.logger.info("Running async health check")
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'checks_completed': 0,
            'total_checks': 3
        }
        
        try:
            # Check GitHub API connectivity
            health_results['components']['github_api'] = await self._check_github_api_health()
            health_results['checks_completed'] += 1
            
            # Check configuration validity
            health_results['components']['configuration'] = await self._check_configuration_health()
            health_results['checks_completed'] += 1
            
            # Check performance monitoring
            health_results['components']['monitoring'] = await self._check_monitoring_health()
            health_results['checks_completed'] += 1
            
            # Determine overall status
            component_statuses = [comp.get('status', 'unhealthy') 
                                for comp in health_results['components'].values()]
            
            if all(status == 'healthy' for status in component_statuses):
                health_results['overall_status'] = 'healthy'
            elif any(status == 'healthy' for status in component_statuses):
                health_results['overall_status'] = 'degraded'
            else:
                health_results['overall_status'] = 'unhealthy'
            
            self.logger.info(f"Health check completed: {health_results['overall_status']}")
            return health_results
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_results['overall_status'] = 'unhealthy'
            health_results['error'] = str(e)
            return health_results
    
    async def _check_github_api_health(self) -> Dict[str, Any]:
        """Check GitHub API health asynchronously"""
        try:
            from async_github_api import AsyncGitHubAPI
            
            async with AsyncGitHubAPI() as api:
                # Try to get the first configured repository
                if self.task_analyzer.repos_to_scan:
                    test_repo = await api.get_repo(self.task_analyzer.repos_to_scan[0])
                    if test_repo:
                        return {'status': 'healthy', 'message': 'GitHub API accessible'}
            
            return {'status': 'degraded', 'message': 'GitHub API accessible but no test repositories found'}
            
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'GitHub API error: {str(e)}'}
    
    async def _check_configuration_health(self) -> Dict[str, Any]:
        """Check configuration health asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            
            def validate_config():
                # Basic configuration validation
                required_keys = ['github']
                for key in required_keys:
                    if key not in self.config:
                        raise ValueError(f"Missing required config key: {key}")
                return True
            
            await loop.run_in_executor(None, validate_config)
            return {'status': 'healthy', 'message': 'Configuration valid'}
            
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Configuration error: {str(e)}'}
    
    async def _check_monitoring_health(self) -> Dict[str, Any]:
        """Check monitoring system health asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            
            def check_monitor():
                if self.monitor and hasattr(self.monitor, 'get_api_call_stats'):
                    stats = self.monitor.get_api_call_stats()
                    return isinstance(stats, dict)
                return False
            
            is_healthy = await loop.run_in_executor(None, check_monitor)
            
            if is_healthy:
                return {'status': 'healthy', 'message': 'Monitoring system operational'}
            else:
                return {'status': 'degraded', 'message': 'Monitoring system not fully operational'}
                
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Monitoring error: {str(e)}'}


# Convenience functions
async def run_async_orchestration(config_path: str = 'config.json', 
                                max_concurrent: int = 5) -> Dict[str, Any]:
    """
    Convenience function to run async orchestration
    
    Args:
        config_path: Path to configuration file
        max_concurrent: Maximum concurrent operations
        
    Returns:
        Orchestration results
    """
    orchestrator = AsyncOrchestrator(config_path, max_concurrent)
    return await orchestrator.run_full_analysis()


async def run_health_check(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Convenience function to run health check
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Health check results
    """
    orchestrator = AsyncOrchestrator(config_path)
    return await orchestrator.health_check_async()


# Example usage
async def example_async_orchestration():
    """Example of running async orchestration"""
    try:
        # Run health check first
        health = await run_health_check()
        logger.info(f"Health check: {health['overall_status']}")
        
        if health['overall_status'] in ['healthy', 'degraded']:
            # Run full orchestration
            results = await run_async_orchestration()
            logger.info(f"Orchestration completed: {results['success']}")
            return results
        else:
            logger.error("System unhealthy, skipping orchestration")
            return None
            
    except Exception as e:
        logger.error(f"Async orchestration example failed: {e}")
        raise


if __name__ == "__main__":
    # Test async orchestration
    asyncio.run(example_async_orchestration())