import time
from threading import Lock


class RateLimiter:
    """Token bucket rate limiter per API key."""

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rpm = requests_per_minute
        self.burst = burst
        self._buckets: dict[str, dict] = {}  # key -> {tokens, last_refill}
        self._lock = Lock()

    def _get_or_create_bucket(self, api_key: str) -> dict:
        if api_key not in self._buckets:
            self._buckets[api_key] = {
                "tokens": float(self.burst),
                "last_refill": time.time(),
            }
        return self._buckets[api_key]

    def _refill(self, bucket: dict):
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - bucket["last_refill"]
        # tokens per second = rpm / 60
        tokens_per_sec = self.rpm / 60.0
        new_tokens = elapsed * tokens_per_sec
        bucket["tokens"] = min(float(self.burst), bucket["tokens"] + new_tokens)
        bucket["last_refill"] = now

    def is_allowed(self, api_key: str) -> bool:
        """Check if a request is allowed without consuming a token."""
        with self._lock:
            bucket = self._get_or_create_bucket(api_key)
            self._refill(bucket)
            return bucket["tokens"] >= 1.0

    def acquire(self, api_key: str) -> bool:
        """Attempt to consume a token. Returns True if allowed, False if rate limited."""
        with self._lock:
            bucket = self._get_or_create_bucket(api_key)
            self._refill(bucket)
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False

    def reset(self, api_key: str):
        """Reset a bucket to full capacity."""
        with self._lock:
            self._buckets[api_key] = {
                "tokens": float(self.burst),
                "last_refill": time.time(),
            }

    def status(self, api_key: str) -> dict:
        """Return the current status of a bucket."""
        with self._lock:
            bucket = self._get_or_create_bucket(api_key)
            self._refill(bucket)
            return {
                "tokens": bucket["tokens"],
                "rpm": self.rpm,
                "burst": self.burst,
            }
