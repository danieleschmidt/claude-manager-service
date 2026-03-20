from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

from .session import ClaudeSession
from .rate_limiter import RateLimiter
from .router import RequestRouter, RequestType
from .tracker import UsageTracker

app = FastAPI(title="Claude Manager Service", version="0.1.0")

# In-memory state for demo
sessions: dict[str, ClaudeSession] = {}
limiter = RateLimiter(requests_per_minute=60)
router = RequestRouter()
tracker = UsageTracker()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str
    model: Optional[str] = None  # override routing if provided


class ChatResponse(BaseModel):
    session_id: str
    model: str
    request_type: str
    tokens_used: int
    response: str  # simulated (echo for demo)
    budget_remaining: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_api_key: str = Header(default="default")):
    """Submit a chat prompt. Handles rate limiting, routing, session management, and usage tracking."""
    # Check rate limit
    if not limiter.acquire(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Get or create session
    if req.session_id and req.session_id in sessions:
        session = sessions[req.session_id]
    else:
        session = ClaudeSession(session_id=req.session_id)
        sessions[session.session_id] = session

    # Route to model
    req_type = router.classify(req.prompt)
    model = req.model or router.route(req.prompt)

    # Estimate tokens (rough: 1 token ~ 4 chars)
    estimated_input_tokens = max(1, len(req.prompt) // 4)
    estimated_output_tokens = max(1, len(req.prompt) // 8)  # simulated echo is shorter
    total_estimated = estimated_input_tokens + estimated_output_tokens

    # Check budget
    if not session.is_within_budget(total_estimated):
        raise HTTPException(status_code=402, detail="Token budget exceeded for this session")

    # Simulate response (echo for demo — no actual API call)
    simulated_response = f"[{model}] Echo: {req.prompt}"

    # Add messages to session
    session.add_message("user", req.prompt, tokens=estimated_input_tokens)
    session.add_message("assistant", simulated_response, tokens=estimated_output_tokens)

    # Log usage
    cost = UsageTracker.calculate_cost(model, estimated_input_tokens, estimated_output_tokens)
    tracker.log_request(
        session_id=session.session_id,
        model=model,
        input_tokens=estimated_input_tokens,
        output_tokens=estimated_output_tokens,
        cost_usd=cost,
    )

    return ChatResponse(
        session_id=session.session_id,
        model=model,
        request_type=req_type.value,
        tokens_used=total_estimated,
        response=simulated_response,
        budget_remaining=session.remaining_budget(),
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details about a specific session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].to_dict()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"deleted": session_id}


@app.get("/usage")
async def get_usage():
    """Get overall usage summary."""
    return tracker.summary()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
