"""
Test fixtures package for Claude Manager Service.

This package contains various fixtures and sample data for testing different
components of the system.
"""

from .github_responses import GitHubFixtures
from .sample_data import SampleData

__all__ = ["GitHubFixtures", "SampleData"]