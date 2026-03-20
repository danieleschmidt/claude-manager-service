import sqlite3
import time
from contextlib import contextmanager


class UsageTracker:
    """SQLite-backed usage logging for Claude API calls."""

    PRICING = {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # For in-memory databases, keep a single persistent connection
        self._persistent_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON usage(session_id)")

    @contextmanager
    def _connect(self):
        if self._persistent_conn is not None:
            # Reuse the persistent in-memory connection; caller is responsible for commit
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given model and token counts."""
        pricing = UsageTracker.PRICING.get(model)
        if not pricing:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def log_request(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
    ):
        """Log a single API request."""
        if cost_usd == 0.0:
            cost_usd = self.calculate_cost(model, input_tokens, output_tokens)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage (session_id, model, input_tokens, output_tokens, cost_usd, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, model, input_tokens, output_tokens, cost_usd, time.time()),
            )

    def get_usage(self, session_id: str = None) -> list[dict]:
        """Get usage records, optionally filtered by session."""
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM usage WHERE session_id = ? ORDER BY timestamp DESC",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM usage ORDER BY timestamp DESC").fetchall()
        return [dict(row) for row in rows]

    def total_cost(self, session_id: str = None) -> float:
        """Get total cost in USD, optionally filtered by session."""
        with self._connect() as conn:
            if session_id:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM usage").fetchone()
        return float(row[0])

    def total_tokens(self, session_id: str = None) -> int:
        """Get total token count (input + output), optionally filtered by session."""
        with self._connect() as conn:
            if session_id:
                row = conn.execute(
                    "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM usage WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM usage"
                ).fetchone()
        return int(row[0])

    def summary(self) -> dict:
        """Return a summary of all usage."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_requests,
                    COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
                    COALESCE(SUM(cost_usd), 0) as total_cost
                FROM usage
                """
            ).fetchone()
        return {
            "total_requests": int(row["total_requests"]),
            "total_tokens": int(row["total_tokens"]),
            "total_cost": float(row["total_cost"]),
        }
