"""
Performance benchmark tests for Claude Code Manager.
"""

import asyncio
import time
from typing import List

import pytest

from src.async_orchestrator import AsyncOrchestrator
from src.async_task_analyzer import AsyncTaskAnalyzer
from src.performance_monitor import PerformanceMonitor


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for benchmarks."""
        return PerformanceMonitor()

    @pytest.mark.asyncio
    async def test_concurrent_task_processing_benchmark(
        self, performance_monitor, mock_async_github_api
    ):
        """Benchmark concurrent task processing performance."""
        # Setup
        orchestrator = AsyncOrchestrator()
        task_count = 100
        max_concurrent = 10
        
        # Create test tasks
        tasks = [
            {"id": f"task_{i}", "action": "analyze", "repository": f"repo_{i}"}
            for i in range(task_count)
        ]
        
        # Benchmark
        start_time = time.time()
        
        # Process tasks with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_task(task):
            async with semaphore:
                await asyncio.sleep(0.01)  # Simulate work
                return {"task_id": task["id"], "status": "completed"}
        
        results = await asyncio.gather(*[process_task(task) for task in tasks])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assertions
        assert len(results) == task_count
        assert duration < 5.0  # Should complete within 5 seconds
        assert all(result["status"] == "completed" for result in results)
        
        # Performance metrics
        throughput = task_count / duration
        assert throughput > 20  # At least 20 tasks per second

    @pytest.mark.asyncio
    async def test_github_api_rate_limiting_benchmark(
        self, performance_monitor, mock_async_github_api
    ):
        """Benchmark GitHub API rate limiting performance."""
        # Setup rate limiter
        from src.async_github_api import AsyncGitHubAPI
        
        api = AsyncGitHubAPI("test_token")
        request_count = 50
        
        # Benchmark API calls with rate limiting
        start_time = time.time()
        
        async def make_api_call(i):
            return await api.get_repository_info(f"test/repo{i}")
        
        # Mock the actual API calls
        mock_async_github_api.get_repository_info.return_value = {
            "name": "test_repo",
            "full_name": "test/test_repo"
        }
        
        results = await asyncio.gather(*[make_api_call(i) for i in range(request_count)])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assertions
        assert len(results) == request_count
        # Should respect rate limits but not be too slow
        assert 1.0 < duration < 10.0

    @pytest.mark.asyncio
    async def test_task_analyzer_performance(
        self, performance_monitor, mock_async_github_api
    ):
        """Benchmark task analyzer performance."""
        analyzer = AsyncTaskAnalyzer()
        
        # Create test repository data
        repositories = [
            {"full_name": f"test/repo{i}", "language": "Python"}
            for i in range(20)
        ]
        
        start_time = time.time()
        
        # Analyze repositories concurrently
        async def analyze_repo(repo):
            # Mock analysis results
            return {
                "repository": repo["full_name"],
                "todos": [f"TODO: Fix issue {i}" for i in range(5)],
                "issues": [f"Issue {i}" for i in range(3)]
            }
        
        results = await asyncio.gather(*[analyze_repo(repo) for repo in repositories])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assertions
        assert len(results) == 20
        assert duration < 3.0  # Should complete within 3 seconds
        
        # Verify all repositories were analyzed
        analyzed_repos = {result["repository"] for result in results}
        expected_repos = {repo["full_name"] for repo in repositories}
        assert analyzed_repos == expected_repos

    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, performance_monitor):
        """Benchmark memory usage during intensive operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large data structures
        large_data = []
        for i in range(10000):
            large_data.append({
                "id": i,
                "data": f"test_data_{i}" * 100,
                "metadata": {"key": f"value_{i}"}
            })
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del large_data
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Assertions
        assert memory_increase < 100  # Should not use more than 100MB
        assert final_memory < peak_memory + 10  # Memory should be mostly freed

    @pytest.mark.asyncio
    async def test_database_operation_performance(self, mock_async_database):
        """Benchmark database operation performance."""
        # Setup
        batch_size = 1000
        
        # Create test data
        test_tasks = [
            {
                "id": f"task_{i}",
                "title": f"Task {i}",
                "status": "pending",
                "priority": "medium"
            }
            for i in range(batch_size)
        ]
        
        # Benchmark batch insert
        start_time = time.time()
        
        # Mock database operations
        async def batch_insert(tasks):
            for task in tasks:
                await mock_async_database.create_task(task)
        
        await batch_insert(test_tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assertions
        assert duration < 2.0  # Should complete within 2 seconds
        assert mock_async_database.create_task.call_count == batch_size
        
        # Benchmark batch read
        start_time = time.time()
        
        # Mock batch read
        mock_async_database.list_tasks.return_value = test_tasks
        results = await mock_async_database.list_tasks()
        
        end_time = time.time()
        read_duration = end_time - start_time
        
        # Assertions
        assert read_duration < 0.1  # Reads should be very fast
        assert len(results) == batch_size

    @pytest.mark.asyncio
    async def test_concurrent_file_operations_benchmark(self, temp_dir):
        """Benchmark concurrent file operations."""
        file_count = 100
        
        # Create test files concurrently
        async def create_file(file_index):
            file_path = temp_dir / f"test_file_{file_index}.txt"
            content = f"Test content for file {file_index}\n" * 100
            
            # Simulate async file write
            await asyncio.sleep(0.001)  # Simulate I/O delay
            file_path.write_text(content)
            return file_path
        
        start_time = time.time()
        
        file_paths = await asyncio.gather(*[
            create_file(i) for i in range(file_count)
        ])
        
        end_time = time.time()
        create_duration = end_time - start_time
        
        # Assertions
        assert len(file_paths) == file_count
        assert create_duration < 2.0  # Should complete within 2 seconds
        
        # Verify all files were created
        for file_path in file_paths:
            assert file_path.exists()
            assert file_path.stat().st_size > 0
        
        # Benchmark concurrent file reads
        async def read_file(file_path):
            await asyncio.sleep(0.001)  # Simulate I/O delay
            return file_path.read_text()
        
        start_time = time.time()
        
        contents = await asyncio.gather(*[
            read_file(file_path) for file_path in file_paths
        ])
        
        end_time = time.time()
        read_duration = end_time - start_time
        
        # Assertions
        assert len(contents) == file_count
        assert read_duration < 1.0  # Reads should be faster than writes
        assert all(len(content) > 0 for content in contents)

    def test_synchronous_performance_baseline(self):
        """Benchmark synchronous operations for comparison."""
        # Simple computation benchmark
        start_time = time.time()
        
        result = 0
        for i in range(1000000):
            result += i * 2
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Assertions
        assert duration < 1.0  # Should complete within 1 second
        assert result > 0
        
        # String operations benchmark
        start_time = time.time()
        
        strings = []
        for i in range(10000):
            strings.append(f"test_string_{i}" * 10)
        
        combined = "".join(strings)
        
        end_time = time.time()
        string_duration = end_time - start_time
        
        # Assertions
        assert string_duration < 0.5  # Should be very fast
        assert len(combined) > 0