from dataclasses import dataclass, field
from typing import Optional
import time
import uuid


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0


class ClaudeSession:
    """Manages conversation history and token budget for a Claude session."""

    def __init__(
        self,
        session_id: str = None,
        token_budget: int = 100_000,
        model: str = "claude-3-haiku",
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.model = model
        self.token_budget = token_budget
        self._messages: list[Message] = []
        self._total_tokens_used = 0
        self.created_at = time.time()

    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add a message to the session history."""
        msg = Message(role=role, content=content, tokens=tokens)
        self._messages.append(msg)
        self._total_tokens_used += tokens

    def history(self) -> list[Message]:
        """Return the full message history."""
        return list(self._messages)

    def total_tokens(self) -> int:
        """Return total tokens used in this session."""
        return self._total_tokens_used

    def remaining_budget(self) -> int:
        """Return the number of tokens remaining in the budget."""
        return max(0, self.token_budget - self._total_tokens_used)

    def is_within_budget(self, estimated_tokens: int = 0) -> bool:
        """Check if we're within budget, optionally accounting for an estimated next call."""
        return (self._total_tokens_used + estimated_tokens) <= self.token_budget

    def clear(self):
        """Clear all messages and reset token count."""
        self._messages = []
        self._total_tokens_used = 0

    def to_dict(self) -> dict:
        """Serialize the session to a dictionary."""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "token_budget": self.token_budget,
            "total_tokens_used": self._total_tokens_used,
            "remaining_budget": self.remaining_budget(),
            "message_count": len(self._messages),
            "created_at": self.created_at,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "tokens": m.tokens,
                }
                for m in self._messages
            ],
        }

    def __len__(self) -> int:
        """Return the number of messages in this session."""
        return len(self._messages)
