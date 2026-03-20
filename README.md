# Claude Manager Service

A FastAPI-based service for managing Claude AI sessions with rate limiting, intelligent request routing, and usage tracking.

## Features

- **Session Management** — Conversation history with configurable token budgets per session
- **Rate Limiting** — Token-bucket rate limiter per API key (configurable RPM and burst)
- **Request Routing** — Heuristic-based routing to the right Claude model (Haiku / Sonnet / Opus)
- **Usage Tracking** — SQLite-backed cost and token logging with per-session filtering
- **REST API** — FastAPI endpoints for chat, session management, and usage stats
- **CLI** — Command-line tool for routing inspection and session info

## Quick Start

```bash
pip install fastapi pydantic uvicorn
uvicorn claude_manager.app:app --reload
```

API docs at: http://localhost:8000/docs

## REST API

| Method   | Path                    | Description                      |
|----------|-------------------------|----------------------------------|
| `POST`   | `/chat`                 | Submit a prompt (rate-limited)   |
| `GET`    | `/sessions/{id}`        | Get session details              |
| `DELETE` | `/sessions/{id}`        | Delete a session                 |
| `GET`    | `/usage`                | Usage summary                    |
| `GET`    | `/health`               | Health check                     |

### Chat request

```json
{
  "prompt": "Implement a binary search in Python",
  "session_id": "optional-existing-id",
  "model": "optional-model-override"
}
```

Set the `x-api-key` header to identify your API key for rate limiting.

## Request Routing

Prompts are automatically classified and routed:

| Type       | Keywords                                  | Model                        |
|------------|-------------------------------------------|------------------------------|
| `simple`   | Short (&lt;100 chars), no keywords           | claude-3-haiku-20240307      |
| `complex`  | code, implement, debug, function          | claude-3-5-sonnet-20241022   |
| `creative` | write, story, poem, creative              | claude-3-5-sonnet-20241022   |
| `analysis` | analyze, data, statistics, compare        | claude-3-opus-20240229       |

## CLI

```bash
claude-manager route "Implement a REST API in FastAPI"
# Request type : complex
# Routed model : claude-3-5-sonnet-20241022

claude-manager session --new --budget 50000
claude-manager usage
```

## Architecture

```
claude_manager/
  __init__.py    — Package init
  session.py     — ClaudeSession: history + token budget
  rate_limiter.py — RateLimiter: token-bucket per API key
  router.py      — RequestRouter: heuristic model selection
  tracker.py     — UsageTracker: SQLite cost/token logging
  app.py         — FastAPI application
  cli.py         — CLI entry point
tests/
  test_claude_manager.py — 15 unit tests
```

## Pricing (per 1M tokens)

| Model                      | Input    | Output   |
|----------------------------|----------|----------|
| claude-3-haiku-20240307    | $0.25    | $1.25    |
| claude-3-5-sonnet-20241022 | $3.00    | $15.00   |
| claude-3-opus-20240229     | $15.00   | $75.00   |

## Running Tests

```bash
python -m pytest tests/ -v
```
