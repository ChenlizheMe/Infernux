"""Fail closed when the backend-neutral RHI starts depending on Vulkan."""

from __future__ import annotations

import re
import sys
from pathlib import Path


_VULKAN_INCLUDE = re.compile(r"#\s*include\s*[<\"](?:vulkan/|vk_mem_alloc\.h)|\bVk[A-Z][A-Za-z0-9_]*")


def main(root: Path) -> int:
    rhi = root / "cpp" / "infernux" / "function" / "renderer" / "rhi"
    violations: list[str] = []
    for path in sorted((*rhi.rglob("*.h"), *rhi.rglob("*.hpp"), *rhi.rglob("*.cpp"))):
        if _VULKAN_INCLUDE.search(path.read_text(encoding="utf-8")):
            violations.append(path.relative_to(root).as_posix())
    if violations:
        print("RHI boundary violation: Vulkan types/includes in backend-neutral sources", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("RHI boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parents[1]))
