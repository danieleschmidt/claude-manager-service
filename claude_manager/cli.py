#!/usr/bin/env python3
"""Claude Manager Service CLI."""
import argparse
import json
import sys

from .session import ClaudeSession
from .router import RequestRouter
from .tracker import UsageTracker


def main():
    parser = argparse.ArgumentParser(description="Claude Manager Service CLI")
    subparsers = parser.add_subparsers(dest="cmd")

    # route subcommand
    route_p = subparsers.add_parser("route", help="Route a prompt to a model")
    route_p.add_argument("prompt", help="Prompt text to classify and route")

    # session subcommand
    session_p = subparsers.add_parser("session", help="Manage sessions")
    session_p.add_argument("--new", action="store_true", help="Create a new session")
    session_p.add_argument("--id", help="Session ID to inspect")
    session_p.add_argument("--budget", type=int, default=100_000, help="Token budget for new session")

    # usage subcommand
    usage_p = subparsers.add_parser("usage", help="Show usage summary")

    args = parser.parse_args()

    if args.cmd == "route":
        r = RequestRouter()
        req_type = r.classify(args.prompt)
        model = r.route(args.prompt)
        print(f"Request type : {req_type.value}")
        print(f"Routed model : {model}")

    elif args.cmd == "session":
        if args.new:
            session = ClaudeSession(token_budget=args.budget)
            print(json.dumps(session.to_dict(), indent=2))
        elif args.id:
            # Without a persistent store, just create a mock session with the given ID
            session = ClaudeSession(session_id=args.id, token_budget=args.budget)
            print(json.dumps(session.to_dict(), indent=2))
        else:
            session_p.print_help()

    elif args.cmd == "usage":
        tracker = UsageTracker()
        summary = tracker.summary()
        print(json.dumps(summary, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
