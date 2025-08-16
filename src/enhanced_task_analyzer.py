"""
Enhanced Task Analyzer with Concurrent Repository Scanning

This module provides concurrent repository scanning capabilities that significantly
improve performance over the sequential approach used in the original task_analyzer.py.

Features:
- Concurrent repository scanning with configurable concurrency limits  
- Backwards compatibility with existing task analysis functions
- Enhanced performance monitoring and metrics
- Robust error handling and recovery
- Integration with existing configuration and validation systems
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.logger import get_logger
from src.github_api import GitHubAPI
from src.concurrent_repository_scanner import ConcurrentRepositoryScanner
from src.config_validator import get_validated_config
from src.task_tracker import get_task_tracker
from src.performance_monitor import monitor_performance, get_monitor
from src.error_handler import get_error_tracker


logger = get_logger(__name__)


class EnhancedTaskAnalyzer:
    """
    Enhanced task analyzer with concurrent repository scanning capabilities
    
    This class provides the same functionality as the original task_analyzer.py
    but with significant performance improvements through concurrent processing.
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize the enhanced task analyzer
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config = get_validated_config(config_path)
        
        # Initialize GitHub API
        self.github_api = GitHubAPI()
        
        # Get configuration values
        self.manager_repo_name = self.config['github']['managerRepo']
        self.repos_to_scan = self.config['github']['reposToScan']
        self.analyzer_config = self.config.get('analyzer', {})
        
        # Concurrent scanning configuration
        self.max_concurrent = self.analyzer_config.get('maxConcurrentScans', 5)
        self.scan_timeout = self.analyzer_config.get('scanTimeoutSeconds', 300)
        
        # Initialize task tracker
        self.task_tracker = get_task_tracker()
        
        self.logger.info(
            f"Enhanced task analyzer initialized: "
            f"{len(self.repos_to_scan)} repos, max_concurrent={self.max_concurrent}"
        )
    
    async def cleanup_old_tasks(self) -> int:
        """
        Clean up old task entries
        
        Returns:
            Number of tasks cleaned up
        """
        cleanup_days = self.analyzer_config.get('cleanupTasksOlderThanDays', 90)
        
        self.logger.info(f"Cleaning up tasks older than {cleanup_days} days")
        
        cleaned_count = self.task_tracker.cleanup_old_tasks(days=cleanup_days)
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old task entries")
        else:
            self.logger.debug("No old task entries found for cleanup")
        
        return cleaned_count
    
    @monitor_performance(track_memory=True, custom_name="enhanced_task_analysis")
    async def analyze_repositories_concurrently(self) -> Dict[str, Any]:
        """
        Analyze all configured repositories concurrently
        
        Returns:
            Dictionary with analysis results and performance metrics
        """
        self.logger.info(f"Starting concurrent analysis of {len(self.repos_to_scan)} repositories")
        
        start_time = time.time()
        
        # Perform cleanup first
        cleanup_count = await self.cleanup_old_tasks()
        
        # Initialize concurrent scanner
        scanner = ConcurrentRepositoryScanner(
            max_concurrent=self.max_concurrent,
            timeout=self.scan_timeout
        )
        
        try:
            # Scan repositories concurrently
            scan_results = await scanner.scan_repositories(
                self.github_api,
                self.repos_to_scan,
                self.manager_repo_name,
                scan_todos=self.analyzer_config.get('scanForTodos', True),
                scan_issues=self.analyzer_config.get('scanOpenIssues', True)
            )
            
            # Collect results
            total_duration = time.time() - start_time
            successful_scans = sum(1 for result in scan_results if result['success'])
            failed_scans = len(scan_results) - successful_scans
            
            # Get performance statistics
            scanner_stats = scanner.get_performance_stats()
            
            analysis_results = {
                'analysis_completed_at': datetime.now().isoformat(),
                'total_repositories': len(self.repos_to_scan),
                'successful_scans': successful_scans,
                'failed_scans': failed_scans,
                'total_duration_seconds': total_duration,
                'cleanup_count': cleanup_count,
                'scanner_stats': scanner_stats,
                'scan_results': scan_results,
                'performance_improvement': self._calculate_performance_improvement(scanner_stats)
            }
            
            self.logger.info(
                f"Concurrent analysis completed: {successful_scans}/{len(scan_results)} successful "
                f"in {total_duration:.2f}s (concurrency: {scanner_stats.get('concurrency_utilized', 0):.2f}x)"
            )
            
            return analysis_results
            
        finally:
            scanner.cleanup()
    
    def _calculate_performance_improvement(self, scanner_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance improvement metrics compared to sequential scanning"""
        
        # Estimate sequential time based on individual scan durations
        total_scan_time = scanner_stats.get('total_scan_time', 0)
        actual_wall_time = scanner_stats.get('total_wall_time', total_scan_time)
        
        if actual_wall_time > 0 and total_scan_time > 0:
            speed_improvement = total_scan_time / actual_wall_time
            time_saved = total_scan_time - actual_wall_time
            efficiency = scanner_stats.get('concurrency_utilized', 1.0)
        else:
            speed_improvement = 1.0
            time_saved = 0
            efficiency = 1.0
        
        return {
            'speed_improvement_factor': speed_improvement,
            'time_saved_seconds': time_saved,
            'concurrency_efficiency': efficiency,
            'estimated_sequential_time': total_scan_time,
            'actual_concurrent_time': actual_wall_time
        }
    
    async def analyze_single_repository(self, repo_name: str) -> Dict[str, Any]:
        """
        Analyze a single repository (for testing or selective analysis)
        
        Args:
            repo_name: Repository name to analyze
            
        Returns:
            Analysis results for the repository
        """
        self.logger.info(f"Analyzing single repository: {repo_name}")
        
        scanner = ConcurrentRepositoryScanner(max_concurrent=1, timeout=self.scan_timeout)
        
        try:
            result = await scanner.scan_repository(
                self.github_api,
                repo_name,
                self.manager_repo_name,
                scan_todos=self.analyzer_config.get('scanForTodos', True),
                scan_issues=self.analyzer_config.get('scanOpenIssues', True)
            )
            
            return result
            
        finally:
            scanner.cleanup()
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get current analysis statistics and performance metrics"""
        
        # Get performance monitor stats
        perf_monitor = get_monitor()
        perf_report = perf_monitor.get_performance_report(24)  # Last 24 hours
        
        # Get error tracking stats
        error_tracker = get_error_tracker()
        error_stats = error_tracker.get_error_statistics()
        
        return {
            'configuration': {
                'repositories_configured': len(self.repos_to_scan),
                'max_concurrent_scans': self.max_concurrent,
                'scan_timeout_seconds': self.scan_timeout,
                'scan_todos_enabled': self.analyzer_config.get('scanForTodos', True),
                'scan_issues_enabled': self.analyzer_config.get('scanOpenIssues', True)
            },
            'performance_metrics': perf_report.get('overall_stats', {}),
            'error_statistics': error_stats,
            'repositories': self.repos_to_scan
        }


# Backwards compatibility functions
async def run_concurrent_analysis(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Run concurrent repository analysis (main entry point)
    
    This function provides backwards compatibility while using the new
    concurrent scanning approach.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Analysis results and performance metrics
    """
    analyzer = EnhancedTaskAnalyzer(config_path)
    return await analyzer.analyze_repositories_concurrently()


def run_sequential_analysis_for_comparison(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Run the original sequential analysis for performance comparison
    
    This function runs the original sequential logic to provide
    baseline performance metrics for comparison.
    """
    # Import original functions
    from task_analyzer import find_todo_comments, analyze_open_issues
    
    logger.info("Running sequential analysis for comparison")
    
    start_time = time.time()
    
    try:
        config = get_validated_config(config_path)
        api = GitHubAPI()
        manager_repo_name = config['github']['managerRepo']
        repos_to_scan = config['github']['reposToScan']
        
        successful_scans = 0
        failed_scans = 0
        
        for repo_name in repos_to_scan:
            logger.info(f"Sequential analysis: {repo_name}")
            try:
                repo = api.get_repo(repo_name)
                if not repo:
                    failed_scans += 1
                    continue
                
                if config['analyzer']['scanForTodos']:
                    find_todo_comments(api, repo, manager_repo_name)
                
                if config['analyzer']['scanOpenIssues']:
                    analyze_open_issues(api, repo, manager_repo_name)
                
                successful_scans += 1
                
            except Exception as e:
                logger.error(f"Sequential scan failed for {repo_name}: {e}")
                failed_scans += 1
        
        total_duration = time.time() - start_time
        
        return {
            'analysis_type': 'sequential',
            'total_repositories': len(repos_to_scan),
            'successful_scans': successful_scans,
            'failed_scans': failed_scans,
            'total_duration_seconds': total_duration,
            'average_time_per_repo': total_duration / len(repos_to_scan) if repos_to_scan else 0
        }
        
    except Exception as e:
        logger.error(f"Sequential analysis failed: {e}")
        raise


# CLI interface for enhanced task analyzer
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Task Analyzer with Concurrent Scanning")
    parser.add_argument(
        '--config', 
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=5,
        help='Maximum concurrent repository scans'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds for each repository scan'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare concurrent vs sequential performance'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show analysis statistics only'
    )
    
    args = parser.parse_args()
    
    if args.stats:
        analyzer = EnhancedTaskAnalyzer(args.config)
        stats = analyzer.get_analysis_stats()
        print(json.dumps(stats, indent=2))
        return
    
    if args.compare:
        logger.info("Running performance comparison between concurrent and sequential analysis")
        
        # Run concurrent analysis
        logger.info("=== Running Concurrent Analysis ===")
        concurrent_results = await run_concurrent_analysis(args.config)
        
        # Run sequential analysis
        logger.info("=== Running Sequential Analysis ===")
        sequential_results = run_sequential_analysis_for_comparison(args.config)
        
        # Compare results
        improvement_factor = (
            sequential_results['total_duration_seconds'] / 
            concurrent_results['total_duration_seconds']
        )
        
        print(f"\n🚀 Performance Comparison Results:")
        print(f"  Sequential Time: {sequential_results['total_duration_seconds']:.2f}s")
        print(f"  Concurrent Time: {concurrent_results['total_duration_seconds']:.2f}s")
        print(f"  Speed Improvement: {improvement_factor:.2f}x faster")
        print(f"  Time Saved: {sequential_results['total_duration_seconds'] - concurrent_results['total_duration_seconds']:.2f}s")
        
    else:
        # Run standard concurrent analysis
        results = await run_concurrent_analysis(args.config)
        
        print(f"\n✅ Analysis Complete:")
        print(f"  Repositories: {results['successful_scans']}/{results['total_repositories']} successful")
        print(f"  Duration: {results['total_duration_seconds']:.2f}s")
        print(f"  Concurrency: {results['scanner_stats']['concurrency_utilized']:.2f}x")
        
        if results['performance_improvement']['speed_improvement_factor'] > 1:
            print(f"  Performance: {results['performance_improvement']['speed_improvement_factor']:.2f}x faster than sequential")


if __name__ == "__main__":
    asyncio.run(main())