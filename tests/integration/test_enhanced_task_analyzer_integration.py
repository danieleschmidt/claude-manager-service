"""
Integration tests for Enhanced Task Analyzer with concurrent scanning
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestEnhancedTaskAnalyzerIntegration:
    """Integration tests for enhanced task analyzer"""
    
    @pytest.mark.asyncio
    async def test_enhanced_analyzer_initialization_with_mock(self):
        """Test enhanced analyzer initialization with mocked dependencies"""
        
        # Mock the security config to avoid needing real GitHub token
        with patch('enhanced_task_analyzer.get_validated_config') as mock_config, \
             patch('enhanced_task_analyzer.GitHubAPI') as mock_github_api:
            
            mock_config.return_value = {
                'github': {
                    'managerRepo': 'test/manager',
                    'reposToScan': ['test/repo1', 'test/repo2']
                },
                'analyzer': {
                    'maxConcurrentScans': 3,
                    'scanTimeoutSeconds': 120,
                    'scanForTodos': True,
                    'scanOpenIssues': True
                }
            }
            
            mock_github_api.return_value = Mock()
            
            from enhanced_task_analyzer import EnhancedTaskAnalyzer
            
            analyzer = EnhancedTaskAnalyzer()
            
            assert analyzer.max_concurrent == 3
            assert analyzer.scan_timeout == 120
            assert len(analyzer.repos_to_scan) == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_workflow(self):
        """Test the full concurrent analysis workflow"""
        
        with patch('enhanced_task_analyzer.get_validated_config') as mock_config, \
             patch('enhanced_task_analyzer.GitHubAPI') as mock_github_api, \
             patch('enhanced_task_analyzer.get_task_tracker') as mock_tracker:
            
            # Setup config
            mock_config.return_value = {
                'github': {
                    'managerRepo': 'test/manager',
                    'reposToScan': ['test/repo1', 'test/repo2']
                },
                'analyzer': {
                    'maxConcurrentScans': 2,
                    'scanTimeoutSeconds': 60
                }
            }
            
            # Setup GitHub API mock
            mock_api_instance = Mock()
            mock_github_api.return_value = mock_api_instance
            
            # Setup task tracker
            mock_tracker_instance = Mock()
            mock_tracker_instance.cleanup_old_tasks.return_value = 5
            mock_tracker.return_value = mock_tracker_instance
            
            from enhanced_task_analyzer import EnhancedTaskAnalyzer
            
            analyzer = EnhancedTaskAnalyzer()
            
            # Mock the concurrent scanner
            with patch('enhanced_task_analyzer.ConcurrentRepositoryScanner') as mock_scanner:
                mock_scanner_instance = Mock()
                mock_scanner_instance.scan_repositories = AsyncMock(return_value=[
                    {
                        'repo_name': 'test/repo1',
                        'success': True,
                        'duration': 2.5,
                        'todos_found': 3,
                        'issues_found': 1
                    },
                    {
                        'repo_name': 'test/repo2', 
                        'success': True,
                        'duration': 1.8,
                        'todos_found': 2,
                        'issues_found': 0
                    }
                ])
                mock_scanner_instance.get_performance_stats.return_value = {
                    'total_scan_time': 4.3,
                    'total_wall_time': 2.5,
                    'concurrency_utilized': 1.72,
                    'successful_scans': 2,
                    'failed_scans': 0
                }
                mock_scanner_instance.cleanup = Mock()
                mock_scanner.return_value = mock_scanner_instance
                
                # Run concurrent analysis
                results = await analyzer.analyze_repositories_concurrently()
                
                # Verify results structure
                assert 'analysis_completed_at' in results
                assert results['total_repositories'] == 2
                assert results['successful_scans'] == 2
                assert results['failed_scans'] == 0
                assert results['cleanup_count'] == 5
                assert 'performance_improvement' in results
                assert 'scanner_stats' in results
                
                # Verify performance improvement calculation
                perf_improvement = results['performance_improvement']
                assert perf_improvement['speed_improvement_factor'] > 1.0
                assert perf_improvement['concurrency_efficiency'] > 1.0
                
                # Verify cleanup was called
                mock_scanner_instance.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_single_repository_analysis(self):
        """Test analyzing a single repository"""
        
        with patch('enhanced_task_analyzer.get_validated_config') as mock_config, \
             patch('enhanced_task_analyzer.GitHubAPI') as mock_github_api:
            
            mock_config.return_value = {
                'github': {
                    'managerRepo': 'test/manager',
                    'reposToScan': ['test/repo1']
                },
                'analyzer': {}
            }
            
            mock_github_api.return_value = Mock()
            
            from enhanced_task_analyzer import EnhancedTaskAnalyzer
            
            analyzer = EnhancedTaskAnalyzer()
            
            # Mock the concurrent scanner for single repository
            with patch('enhanced_task_analyzer.ConcurrentRepositoryScanner') as mock_scanner:
                mock_scanner_instance = Mock()
                mock_scanner_instance.scan_repository = AsyncMock(return_value={
                    'repo_name': 'test/single-repo',
                    'success': True,
                    'duration': 3.2,
                    'todos_found': 5,
                    'issues_found': 2
                })
                mock_scanner_instance.cleanup = Mock()
                mock_scanner.return_value = mock_scanner_instance
                
                result = await analyzer.analyze_single_repository('test/single-repo')
                
                assert result['repo_name'] == 'test/single-repo'
                assert result['success'] is True
                assert result['duration'] == 3.2
                assert result['todos_found'] == 5
                assert result['issues_found'] == 2
                
                # Verify cleanup was called
                mock_scanner_instance.cleanup.assert_called_once()
    
    def test_analysis_statistics(self):
        """Test analysis statistics collection"""
        
        with patch('enhanced_task_analyzer.get_validated_config') as mock_config, \
             patch('enhanced_task_analyzer.GitHubAPI') as mock_github_api, \
             patch('enhanced_task_analyzer.get_monitor') as mock_monitor, \
             patch('enhanced_task_analyzer.get_error_tracker') as mock_error_tracker:
            
            mock_config.return_value = {
                'github': {
                    'managerRepo': 'test/manager',
                    'reposToScan': ['test/repo1', 'test/repo2', 'test/repo3']
                },
                'analyzer': {
                    'maxConcurrentScans': 5,
                    'scanTimeoutSeconds': 300,
                    'scanForTodos': True,
                    'scanOpenIssues': False
                }
            }
            
            mock_github_api.return_value = Mock()
            
            # Mock performance monitor
            mock_perf_monitor = Mock()
            mock_perf_monitor.get_performance_report.return_value = {
                'overall_stats': {
                    'total_operations': 150,
                    'average_duration': 2.3,
                    'error_rate': 0.05
                }
            }
            mock_monitor.return_value = mock_perf_monitor
            
            # Mock error tracker
            mock_error_instance = Mock()
            mock_error_instance.get_error_statistics.return_value = {
                'total_errors': 12,
                'error_rate': 0.08,
                'common_errors': ['NetworkError', 'TimeoutError']
            }
            mock_error_tracker.return_value = mock_error_instance
            
            from enhanced_task_analyzer import EnhancedTaskAnalyzer
            
            analyzer = EnhancedTaskAnalyzer()
            stats = analyzer.get_analysis_stats()
            
            # Verify configuration section
            config_section = stats['configuration']
            assert config_section['repositories_configured'] == 3
            assert config_section['max_concurrent_scans'] == 5
            assert config_section['scan_timeout_seconds'] == 300
            assert config_section['scan_todos_enabled'] is True
            assert config_section['scan_issues_enabled'] is False
            
            # Verify performance metrics
            assert 'performance_metrics' in stats
            assert stats['performance_metrics']['total_operations'] == 150
            
            # Verify error statistics
            assert 'error_statistics' in stats
            assert stats['error_statistics']['total_errors'] == 12
            
            # Verify repositories list
            assert 'repositories' in stats
            assert len(stats['repositories']) == 3


class TestBackwardsCompatibility:
    """Test backwards compatibility functions"""
    
    @pytest.mark.asyncio
    async def test_run_concurrent_analysis(self):
        """Test the main entry point function"""
        
        with patch('enhanced_task_analyzer.EnhancedTaskAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_repositories_concurrently = AsyncMock(return_value={
                'total_repositories': 2,
                'successful_scans': 2,
                'failed_scans': 0,
                'total_duration_seconds': 5.2
            })
            mock_analyzer_class.return_value = mock_analyzer
            
            from enhanced_task_analyzer import run_concurrent_analysis
            
            results = await run_concurrent_analysis('test-config.json')
            
            assert results['total_repositories'] == 2
            assert results['successful_scans'] == 2
            mock_analyzer_class.assert_called_once_with('test-config.json')
            mock_analyzer.analyze_repositories_concurrently.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])