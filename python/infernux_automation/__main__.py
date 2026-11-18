"""Thin CLI for the same public Host used by MCP and external harnesses."""
import argparse
import json
from pathlib import Path
import sys

from . import AutomationClient, AutomationError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    search = commands.add_parser("operations")
    search.add_argument("query", nargs="?", default="")
    schema = commands.add_parser("schema")
    schema.add_argument("operation")
    for name in ("call", "submit"):
        call = commands.add_parser(name)
        call.add_argument("operation")
        arguments = call.add_mutually_exclusive_group()
        arguments.add_argument("--arguments", default="{}", help="JSON object")
        arguments.add_argument("--arguments-file", help="UTF-8 JSON object file")
    for name in ("job", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("job_id")
    reports = commands.add_parser("reports")
    reports.add_argument("consumer")
    reports.add_argument("--cycle-id", required=True, help="The host's stable real user-turn ID")
    args = parser.parse_args(argv)
    try:
        with AutomationClient(args.project) as engine:
            if args.command == "status":
                value = engine.info()
            elif args.command == "operations":
                value = engine.operations(args.query)
            elif args.command == "schema":
                value = engine.schema(args.operation)
            elif args.command in {"call", "submit"}:
                text = Path(args.arguments_file).read_text(encoding="utf-8") if args.arguments_file else args.arguments
                arguments = json.loads(text)
                if not isinstance(arguments, dict):
                    raise ValueError("Operation arguments must be a JSON object")
                invoke = engine.call if args.command == "call" else engine.submit
                value = invoke(args.operation, arguments)
            elif args.command == "reports":
                value = engine.reports(args.consumer, cycle_id=args.cycle_id)
            else:
                value = getattr(engine, args.command)(args.job_id)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except (AutomationError, OSError, ValueError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "client.invalid_arguments"),
                                     "message": str(exc)}}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
