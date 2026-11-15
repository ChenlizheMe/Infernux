"""Identity and artifact location of the active automation host."""

from dataclasses import dataclass
from pathlib import Path
import uuid


@dataclass(frozen=True, slots=True)
class AutomationContext:
    project_root: str
    instance_id: str
    artifact_root: str


_context = None


def configure(project_root):
    global _context
    root = Path(project_root).resolve()
    identity = uuid.uuid4().hex
    artifacts = root / "Cache" / "Automation" / "artifacts" / identity
    artifacts.mkdir(parents=True, exist_ok=True)
    _context = AutomationContext(str(root), identity, str(artifacts))
    return _context


def automation_context():
    if _context is None:
        raise RuntimeError("The engine automation host is not running")
    return _context
