import importlib
import json
import sys
import traceback

modules = sys.argv[1:]
report = {}
for name in modules:
    try:
        importlib.import_module(name)
        report[name] = "ok"
    except Exception as exc:
        report[name] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, indent=2, ensure_ascii=False))
failed = [k for k, v in report.items() if v != "ok"]
sys.exit(1 if failed else 0)
