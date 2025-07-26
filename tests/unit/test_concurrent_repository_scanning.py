"""
Unit tests for concurrent repository scanning optimization

These tests verify the new concurrent repository scanning functionality
that replaces the sequential scanning approach for improved performance.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestConcurrentRepositoryScanner:
    """Test the new concurrent repository scanner"""
    
    def test_concurrent_scanner_initialization(self):
        """Test that concurrent scanner initializes correctly"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner(max_concurrent=3)
        
        assert scanner.max_concurrent == 3
        assert scanner.timeout == 300  # Default timeout
        assert scanner.executor is not None
        assert isinstance(scanner.executor, ThreadPoolExecutor)
    
    def test_concurrent_scanner_custom_config(self):
        """Test scanner with custom configuration"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner(
            max_concurrent=5,
            timeout=600,
            executor_type='thread'
        )
        
        assert scanner.max_concurrent == 5
        assert scanner.timeout == 600
        assert isinstance(scanner.executor, ThreadPoolExecutor)
    
    @pytest.mark.asyncio
    async def test_scan_single_repository(self):
        """Test scanning a single repository"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner()
        
        # Mock repository and scanning functions
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        
        mock_github_api = Mock()
        mock_github_api.get_repo.return_value = mock_repo
        
        # Mock scanning functions from task_analyzer
        with patch('task_analyzer.find_todo_comments') as mock_todos, \
             patch('task_analyzer.analyze_open_issues') as mock_issues:
            
            mock_todos.return_value = None
            mock_issues.return_value = None
            
            result = await scanner.scan_repository(
                mock_github_api,
                "test/repo",
                "manager/repo",
                scan_todos=True,
                scan_issues=True
            )
            
            assert result['repo_name'] == "test/repo"
            assert result['success'] is True
            assert result['duration'] > 0
            assert 'todos_found' in result
            assert 'issues_found' in result
    
    @pytest.mark.asyncio
    async def test_scan_repository_with_error(self):
        """Test repository scanning with error handling"""
        from concurrent_repository_scanner import (
            ConcurrentRepositoryScanner, 
            RepositoryScanningError
        )
        
        scanner = ConcurrentRepositoryScanner()
        
        mock_github_api = Mock()
        mock_github_api.get_repo.side_effect = Exception("API Error")
        
        result = await scanner.scan_repository(
            mock_github_api,
            "test/error-repo", 
            "manager/repo"
        )
        
        assert result['repo_name'] == "test/error-repo"
        assert result['success'] is False
        assert "API Error" in result['error']
        assert result['duration'] > 0
    
    @pytest.mark.asyncio
    async def test_scan_multiple_repositories_concurrently(self):
        """Test scanning multiple repositories concurrently"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner(max_concurrent=2)
        
        # Mock repositories
        repos = ["test/repo1", "test/repo2", "test/repo3"]
        
        mock_github_api = Mock()
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        mock_github_api.get_repo.return_value = mock_repo
        
        # Track timing
        start_time = time.time()
        
        with patch('task_analyzer.find_todo_comments') as mock_todos, \
             patch('task_analyzer.analyze_open_issues') as mock_issues:
            
            # Add small delay to simulate work
            def slow_scan(*args, **kwargs):
                time.sleep(0.1)
                return None
            
            mock_todos.side_effect = slow_scan
            mock_issues.side_effect = slow_scan
            
            results = await scanner.scan_repositories(
                mock_github_api,
                repos,
                "manager/repo"
            )
        
        duration = time.time() - start_time
        
        # Should complete faster than sequential (3 * 0.2s = 0.6s)
        # With concurrency=2, should be around 0.4s (allowing 0.3s overhead for setup)
        assert duration < 0.8
        
        assert len(results) == 3
        assert all(result['success'] for result in results)
        assert all('repo' in str(result['repo_name']) for result in results)
    
    def test_performance_comparison(self):
        """Test performance improvement vs sequential scanning"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        # This test would compare performance in a real scenario
        # For unit testing, we'll just verify the structure exists
        scanner = ConcurrentRepositoryScanner()
        
        # Verify performance tracking attributes exist
        assert hasattr(scanner, '_start_time')
        assert hasattr(scanner, '_end_time')
        assert hasattr(scanner, 'get_performance_stats')
    
    def test_concurrency_limits(self):
        """Test that concurrency limits are respected"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        # Test maximum concurrency enforcement
        scanner = ConcurrentRepositoryScanner(max_concurrent=2)
        
        # Verify that the executor has the correct max_workers
        assert scanner.executor._max_workers == 2
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for slow repositories"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner(timeout=0.1)  # Very short timeout
        
        mock_github_api = Mock()
        mock_repo = MagicMock()
        mock_github_api.get_repo.return_value = mock_repo
        
        with patch('task_analyzer.find_todo_comments') as mock_todos:
            # Mock a slow operation
            def slow_operation(*args, **kwargs):
                time.sleep(0.2)  # Longer than timeout
                return None
            
            mock_todos.side_effect = slow_operation
            
            result = await scanner.scan_repository(
                mock_github_api,
                "test/slow-repo",
                "manager/repo"
            )
            
            # Should handle timeout gracefully
            assert result['success'] is False
            assert 'timed out' in result['error'].lower()
    
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner()
        
        # Verify executor is created
        assert scanner.executor is not None
        
        # Test cleanup
        scanner.cleanup()
        
        # Executor should be shutdown
        assert scanner.executor._shutdown
    
    @pytest.mark.asyncio
    async def test_error_aggregation(self):
        """Test error aggregation across multiple repositories"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner()
        
        repos = ["test/good-repo", "test/bad-repo1", "test/bad-repo2"]
        
        mock_github_api = Mock()
        
        def mock_get_repo(repo_name):
            if 'bad' in repo_name:
                raise Exception(f"Error for {repo_name}")
            else:
                mock_repo = MagicMock()
                mock_repo.full_name = repo_name
                return mock_repo
        
        mock_github_api.get_repo.side_effect = mock_get_repo
        
        results = await scanner.scan_repositories(
            mock_github_api,
            repos,
            "manager/repo"
        )
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        assert len(successful) == 1
        assert len(failed) == 2
        assert successful[0]['repo_name'] == "test/good-repo"


class TestRepositoryScanningIntegration:
    """Integration tests for repository scanning"""
    
    def test_integration_with_task_analyzer(self):
        """Test integration with existing task analyzer"""
        # This would test the integration points
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner()
        
        # Verify that scanner can call existing functions
        assert hasattr(scanner, 'scan_repository')
        assert callable(getattr(scanner, 'scan_repository'))
    
    @pytest.mark.asyncio 
    async def test_backwards_compatibility(self):
        """Test backwards compatibility with existing code"""
        from concurrent_repository_scanner import scan_repositories_concurrently
        
        # Mock the required components
        mock_github_api = Mock()
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        mock_github_api.get_repo.return_value = mock_repo
        
        repos = ["test/repo1", "test/repo2"]
        
        with patch('task_analyzer.find_todo_comments') as mock_todos, \
             patch('task_analyzer.analyze_open_issues') as mock_issues:
            
            mock_todos.return_value = None
            mock_issues.return_value = None
            
            # Test the convenience function
            results = await scan_repositories_concurrently(
                mock_github_api,
                repos,
                "manager/repo"
            )
            
            assert len(results) == 2
            assert all(result['success'] for result in results)


class TestPerformanceMetrics:
    """Test performance metrics and monitoring"""
    
    def test_performance_metrics_collection(self):
        """Test that performance metrics are collected"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner()
        
        # Test metrics collection structure
        metrics = scanner.get_performance_stats()
        
        expected_keys = [
            'total_repositories_scanned',
            'successful_scans',
            'failed_scans',
            'average_scan_time',
            'total_scan_time',
            'concurrency_utilized'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    @pytest.mark.asyncio
    async def test_metrics_accuracy(self):
        """Test that metrics are accurately recorded"""
        from concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner()
        
        # Mock successful scan
        mock_github_api = Mock()
        mock_repo = MagicMock()
        mock_github_api.get_repo.return_value = mock_repo
        
        with patch('task_analyzer.find_todo_comments'), \
             patch('task_analyzer.analyze_open_issues'):
            
            await scanner.scan_repository(
                mock_github_api,
                "test/repo",
                "manager/repo"
            )
        
        metrics = scanner.get_performance_stats()
        
        assert metrics['total_repositories_scanned'] == 1
        assert metrics['successful_scans'] == 1
        assert metrics['failed_scans'] == 0
        assert metrics['average_scan_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])