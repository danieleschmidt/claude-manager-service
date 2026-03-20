"""Tests for Claude Manager Service."""
import pytest
import time

from claude_manager.session import ClaudeSession, Message
from claude_manager.rate_limiter import RateLimiter
from claude_manager.router import RequestRouter, RequestType
from claude_manager.tracker import UsageTracker


# ─────────────────────────────────────────────
# Session tests
# ─────────────────────────────────────────────

def test_session_creation():
    """A new session gets a UUID, correct defaults, and empty history."""
    session = ClaudeSession()
    assert session.session_id is not None
    assert len(session.session_id) == 36  # UUID format
    assert session.token_budget == 100_000
    assert len(session) == 0
    assert session.total_tokens() == 0


def test_session_add_message():
    """Messages are appended and token count is updated."""
    session = ClaudeSession()
    session.add_message("user", "Hello!", tokens=10)
    session.add_message("assistant", "Hi there!", tokens=15)
    assert len(session) == 2
    assert session.total_tokens() == 25
    history = session.history()
    assert history[0].role == "user"
    assert history[0].content == "Hello!"
    assert history[1].role == "assistant"


def test_session_token_budget():
    """Remaining budget decreases as tokens are used."""
    session = ClaudeSession(token_budget=1000)
    assert session.remaining_budget() == 1000
    session.add_message("user", "test", tokens=200)
    assert session.remaining_budget() == 800
    assert session.is_within_budget(500) is True


def test_session_budget_exceeded():
    """is_within_budget returns False when estimated tokens exceed budget."""
    session = ClaudeSession(token_budget=100)
    session.add_message("user", "test", tokens=90)
    assert session.is_within_budget(20) is False
    assert session.remaining_budget() == 10


def test_session_history():
    """History returns messages in insertion order."""
    session = ClaudeSession()
    session.add_message("user", "first", tokens=5)
    session.add_message("assistant", "second", tokens=8)
    session.add_message("user", "third", tokens=3)
    history = session.history()
    assert len(history) == 3
    assert [m.content for m in history] == ["first", "second", "third"]
    # Returned list is a copy
    history.pop()
    assert len(session) == 3


def test_session_clear():
    """clear() removes all messages and resets token count."""
    session = ClaudeSession()
    session.add_message("user", "hello", tokens=50)
    session.add_message("assistant", "world", tokens=60)
    assert len(session) == 2
    session.clear()
    assert len(session) == 0
    assert session.total_tokens() == 0
    assert session.remaining_budget() == session.token_budget


# ─────────────────────────────────────────────
# Rate limiter tests
# ─────────────────────────────────────────────

def test_rate_limiter_allows_first():
    """First request on a fresh key should always be allowed."""
    limiter = RateLimiter(requests_per_minute=60, burst=5)
    assert limiter.acquire("key-alpha") is True


def test_rate_limiter_throttles():
    """After exhausting the burst capacity, requests are denied."""
    limiter = RateLimiter(requests_per_minute=60, burst=3)
    key = "throttle-test-key"
    # Drain the bucket
    results = [limiter.acquire(key) for _ in range(3)]
    assert all(results), "All burst requests should succeed"
    # Next one should fail
    assert limiter.acquire(key) is False


def test_rate_limiter_different_keys():
    """Different API keys have independent buckets."""
    limiter = RateLimiter(requests_per_minute=60, burst=2)
    # Drain key A
    limiter.acquire("key-a")
    limiter.acquire("key-a")
    assert limiter.acquire("key-a") is False
    # Key B should still be fine
    assert limiter.acquire("key-b") is True


# ─────────────────────────────────────────────
# Router tests
# ─────────────────────────────────────────────

def test_router_classify_simple():
    """Short prompts without special keywords classify as SIMPLE."""
    r = RequestRouter()
    assert r.classify("What time is it?") == RequestType.SIMPLE
    assert r.classify("Hello") == RequestType.SIMPLE


def test_router_classify_complex():
    """Prompts with code-related keywords classify as COMPLEX."""
    r = RequestRouter()
    assert r.classify("Please implement a binary search function in Python") == RequestType.COMPLEX
    assert r.classify("How do I debug this code?") == RequestType.COMPLEX


def test_router_route_returns_model():
    """route() returns a non-empty model string for any prompt."""
    r = RequestRouter()
    model = r.route("What is 2+2?")
    assert isinstance(model, str)
    assert model.startswith("claude-")

    model_complex = r.route("Implement a quicksort algorithm")
    assert model_complex == "claude-3-5-sonnet-20241022"


# ─────────────────────────────────────────────
# Tracker tests
# ─────────────────────────────────────────────

def test_tracker_log_request():
    """log_request stores an entry retrievable via get_usage."""
    tracker = UsageTracker()  # in-memory
    tracker.log_request("sess-1", "claude-3-haiku-20240307", 100, 200)
    records = tracker.get_usage()
    assert len(records) == 1
    assert records[0]["session_id"] == "sess-1"
    assert records[0]["input_tokens"] == 100
    assert records[0]["output_tokens"] == 200


def test_tracker_total_cost():
    """total_cost computes correct USD based on PRICING table."""
    tracker = UsageTracker()
    # 1M input tokens @ $0.25, 1M output @ $1.25 for haiku
    tracker.log_request("sess-x", "claude-3-haiku-20240307", 1_000_000, 1_000_000)
    cost = tracker.total_cost()
    assert abs(cost - 1.50) < 0.0001  # $0.25 + $1.25

    # Filter by session
    tracker.log_request("sess-y", "claude-3-opus-20240229", 1_000_000, 0)
    assert abs(tracker.total_cost("sess-x") - 1.50) < 0.0001
    assert abs(tracker.total_cost("sess-y") - 15.0) < 0.0001


def test_tracker_summary():
    """summary() returns correct aggregate counts."""
    tracker = UsageTracker()
    tracker.log_request("s1", "claude-3-haiku-20240307", 100, 50)
    tracker.log_request("s1", "claude-3-haiku-20240307", 200, 80)
    tracker.log_request("s2", "claude-3-5-sonnet-20241022", 300, 120)
    summary = tracker.summary()
    assert summary["total_requests"] == 3
    assert summary["total_tokens"] == 100 + 50 + 200 + 80 + 300 + 120  # 850
    assert isinstance(summary["total_cost"], float)
    assert summary["total_cost"] > 0
