"""
Load testing for API endpoints and concurrent operations.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestLoadTesting:
    """Load testing scenarios for the Claude Manager Service."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(
        self, mock_async_github_api, mock_async_database
    ):
        """Test concurrent task processing under load."""
        from src.async_orchestrator import AsyncOrchestrator
        
        orchestrator = AsyncOrchestrator(
            github_api=mock_async_github_api,
            database=mock_async_database
        )
        
        # Simulate processing 50 tasks concurrently
        tasks = []
        for i in range(50):
            task = {
                "id": f"task_{i}",
                "title": f"Test task {i}",
                "repository": "test/repo",
                "priority": "medium"
            }
            tasks.append(orchestrator.process_task(task))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify all tasks completed
        assert len(results) == 50
        assert all(not isinstance(r, Exception) for r in results)
        
        # Performance assertion - should complete within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 30.0, f"Processing took too long: {processing_time}s"

    @pytest.mark.slow
    def test_repository_scanning_load(self, mock_github_api):
        """Test repository scanning under load with multiple repositories."""
        from src.concurrent_repository_scanner import ConcurrentRepositoryScanner
        
        scanner = ConcurrentRepositoryScanner(github_api=mock_github_api)
        repositories = [f"test/repo{i}" for i in range(20)]
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(scanner.scan_repository, repo)
                for repo in repositories
            ]
            results = [future.result() for future in futures]
        end_time = time.time()
        
        # Verify all scans completed
        assert len(results) == 20
        
        # Performance assertion
        scanning_time = end_time - start_time
        assert scanning_time < 60.0, f"Scanning took too long: {scanning_time}s"

    @pytest.mark.performance
    def test_memory_usage_during_bulk_operations(self, mock_database):
        """Test memory usage remains stable during bulk operations."""
        import psutil
        import gc
        
        from src.database_task_tracker import DatabaseTaskTracker
        
        tracker = DatabaseTaskTracker(database=mock_database)
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform bulk operations
        for i in range(1000):
            task_data = {
                "title": f"Bulk task {i}",
                "description": f"Description for task {i}",
                "status": "pending",
                "priority": "low"
            }
            tracker.create_task(task_data)
            
            # Force garbage collection every 100 iterations
            if i % 100 == 0:
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_github_api_rate_limiting(self, mock_async_github_api):
        """Test GitHub API rate limiting behavior under load."""
        from src.async_github_api import AsyncGitHubAPI
        
        # Configure mock to simulate rate limiting
        mock_async_github_api.get_repo.side_effect = [
            {"name": "repo"} for _ in range(50)
        ]
        
        api = AsyncGitHubAPI(token="test_token")
        api._client = mock_async_github_api
        
        # Make rapid requests
        start_time = time.time()
        tasks = [api.get_repo("test/repo") for _ in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify all requests completed
        assert len(results) == 50
        assert all(not isinstance(r, Exception) for r in results)
        
        # Should handle rate limiting gracefully
        request_time = end_time - start_time
        assert request_time >= 1.0, "Rate limiting should slow down requests"

    @pytest.mark.slow
    def test_error_handling_under_load(self, mock_github_api):
        """Test error handling behavior under high load conditions."""
        from src.error_handler import ErrorHandler
        
        error_handler = ErrorHandler()
        
        # Simulate processing errors rapidly
        errors = []
        for i in range(100):
            try:
                if i % 10 == 0:
                    raise ValueError(f"Test error {i}")
                elif i % 15 == 0:
                    raise ConnectionError(f"Connection error {i}")
                else:
                    # Normal operation
                    pass
            except Exception as e:
                errors.append(error_handler.handle_error(e))
        
        # Verify error handling didn't crash
        assert len(errors) == 13  # 10 ValueError + 3 ConnectionError (excluding overlaps)
        
        # Verify error handler remains functional
        assert error_handler.is_healthy()


class TestStressScenarios:
    """Stress testing scenarios for edge cases."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_database_connection_stress(self, mock_async_database):
        """Test database under connection stress."""
        from src.services.database_service import DatabaseService
        
        # Simulate rapid connection/disconnection
        service = DatabaseService(database=mock_async_database)
        
        tasks = []
        for i in range(100):
            tasks.append(service.health_check())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most connections should succeed
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful >= 90, f"Only {successful}/100 connections succeeded"

    @pytest.mark.performance
    def test_large_payload_handling(self, mock_github_api):
        """Test handling of large payloads."""
        from src.task_analyzer import TaskAnalyzer
        
        analyzer = TaskAnalyzer(github_api=mock_github_api)
        
        # Create a large fake repository content
        large_content = "TODO: " + "x" * 10000  # 10KB TODO comment
        large_file = {
            "name": "large_file.py",
            "content": large_content * 100  # 1MB file
        }
        
        start_time = time.time()
        result = analyzer.analyze_file_content(large_file)
        end_time = time.time()
        
        # Should handle large files within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Large file processing took {processing_time}s"
        assert result is not None

    @pytest.mark.slow
    def test_concurrent_database_operations(self, mock_database):
        """Test concurrent database operations for race conditions."""
        from src.database_task_tracker import DatabaseTaskTracker
        
        tracker = DatabaseTaskTracker(database=mock_database)
        
        def create_and_update_task(task_id: int):
            """Create and immediately update a task."""
            task_data = {
                "id": f"stress_task_{task_id}",
                "title": f"Stress test task {task_id}",
                "status": "pending"
            }
            tracker.create_task(task_data)
            
            # Immediately update
            task_data["status"] = "in_progress"
            tracker.update_task(task_data["id"], task_data)
            
            return task_data["id"]
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(create_and_update_task, i)
                for i in range(50)
            ]
            results = [future.result() for future in futures]
        
        # All operations should complete successfully
        assert len(results) == 50
        assert len(set(results)) == 50  # All unique task IDs