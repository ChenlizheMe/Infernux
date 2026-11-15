"""Claude UserPromptSubmit hook: only actual prompt events advance reports."""
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infernux_automation import AutomationClient, AutomationError


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    project = parser.parse_args().project
    event = json.load(sys.stdin)
    consumer = "claude:" + event["session_id"]
    # Do not turn retry/hook invocation counts into user-turn identities. Some
    # hosts provide an event ID; the base hook schema does not promise one.
    cycle = event.get("prompt_id") or event.get("request_id") or event.get("event_id")
    try:
        reports = AutomationClient(project or event["cwd"]).request(
            "reports", consumer=consumer, cycle_id=cycle, changes_only=cycle is None)
        context = json.dumps({"infernux_consumer": consumer, "cycle_id": cycle, "reports": reports,
                              "turn_delivery": "stable client event ID" if cycle else
                              "Unavailable without a stable user-turn ID; only change reports were read. "
                              "Use conversation_turn with the host's real turn ID for turns mode."}, ensure_ascii=False)
    except AutomationError as exc:
        context = f"Infernux is not connected: {exc}"
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
