"""Platform-neutral contracts shared by build frontends and exporters."""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from Infernux.engine.build_cancellation import BuildCancelled
from Infernux.engine.build_target import BuildTargetId
from Infernux.engine.path_utils import resolve_project_path, resolved_path


_OPTION_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class BuildConfiguration(str, Enum):
    DEVELOPMENT = "development"
    RELEASE = "release"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class BuildOptionKind(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    STRING = "string"
    PATH = "path"
    ENUM = "enum"


@dataclass(frozen=True, slots=True)
class BuildOptionChoice:
    value: str
    label: str

    def __post_init__(self) -> None:
        value = str(self.value or "").strip()
        label = str(self.label or "").strip()
        if not value or not label:
            raise ValueError("Build option choices require a value and label")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class BuildOption:
    """One exporter-owned, typed Player packaging option."""

    key: str
    label: str
    kind: BuildOptionKind
    default: object
    description: str = ""
    choices: tuple[BuildOptionChoice, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    step: int = 1
    editor_visible: bool = True
    visible_when: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.key or "").strip()
        label = str(self.label or "").strip()
        kind = BuildOptionKind(self.kind)
        if not _OPTION_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                "Build option keys must use lower-case snake_case tokens: "
                f"{self.key!r}"
            )
        if not label:
            raise ValueError(f"Build option {key!r} requires a label")
        choices = tuple(self.choices)
        if kind is BuildOptionKind.ENUM:
            if not choices:
                raise ValueError(f"Enum build option {key!r} requires choices")
            values = tuple(item.value for item in choices)
            if len(set(values)) != len(values):
                raise ValueError(f"Enum build option {key!r} has duplicate choices")
        elif choices:
            raise ValueError(f"Only enum build options may declare choices: {key!r}")
        minimum = self.minimum
        maximum = self.maximum
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"Build option {key!r} has an invalid numeric range")
        if int(self.step) <= 0:
            raise ValueError(f"Build option {key!r} requires a positive step")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "visible_when", _frozen_mapping(self.visible_when))
        object.__setattr__(self, "default", self.normalize(self.default))

    def normalize(self, value: object) -> object:
        if self.kind is BuildOptionKind.BOOLEAN:
            if not isinstance(value, bool):
                raise TypeError(f"Build option {self.key} must be a boolean")
            return value
        if self.kind is BuildOptionKind.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Build option {self.key} must be an integer")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(
                    f"Build option {self.key} must be at least {self.minimum}"
                )
            if self.maximum is not None and value > self.maximum:
                raise ValueError(
                    f"Build option {self.key} must be at most {self.maximum}"
                )
            return value
        if not isinstance(value, str):
            raise TypeError(f"Build option {self.key} must be a string")
        normalized = value.strip()
        if self.kind is BuildOptionKind.ENUM:
            allowed = {item.value for item in self.choices}
            if normalized not in allowed:
                raise ValueError(
                    f"Build option {self.key} must be one of: "
                    + ", ".join(sorted(allowed))
                )
        return normalized

    def is_visible(self, values: Mapping[str, object]) -> bool:
        return self.editor_visible and all(
            values.get(key) == expected
            for key, expected in self.visible_when.items()
        )


@dataclass(frozen=True, slots=True)
class PlatformCapabilities:
    graphics_api: str
    threads: bool = True
    dynamic_loading: bool = True
    filesystem: bool = True
    network: bool = True
    audio: bool = True
    pointer_input: bool = True
    text_input: bool = True
    gamepad_input: bool = True
    python_native_modules: bool = True
    # Compiler runtimes are a platform payload decision, never an implicit
    # consequence of registering a target.  Desktop Vulkan opts in below;
    # constrained/new targets stay compiler-free until their exporter proves
    # and declares support.
    cpu_jit: bool = False
    persistent_storage: bool = True
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        graphics_api = str(self.graphics_api or "").strip().casefold()
        if graphics_api not in {"vulkan", "webgpu"}:
            raise ValueError(
                "Infernux platform targets support only Vulkan or WebGPU"
            )
        object.__setattr__(self, "graphics_api", graphics_api)
        object.__setattr__(
            self,
            "features",
            frozenset(str(item).strip() for item in self.features if str(item).strip()),
        )


@dataclass(frozen=True, slots=True)
class BuildTarget:
    id: BuildTargetId
    display_name: str
    platform: str
    architecture: str
    capabilities: PlatformCapabilities

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", BuildTargetId(self.id))
        for field_name in ("display_name", "platform", "architecture"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"BuildTarget.{field_name} is required")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class BuildProfile:
    configuration: BuildConfiguration = BuildConfiguration.DEVELOPMENT
    debug_symbols: bool = True
    compress_resources: bool = False
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "configuration", BuildConfiguration(self.configuration)
        )
        object.__setattr__(self, "options", _frozen_mapping(self.options))


class BuildCancellationToken:
    """Thread-safe cancellation signal shared by every build frontend."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise BuildCancelled("Build cancelled")


@dataclass(frozen=True, slots=True)
class BuildProgress:
    phase: str
    completed: int
    total: int
    message: str
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = str(self.phase or "").strip()
        if not phase:
            raise ValueError("BuildProgress.phase is required")
        completed = int(self.completed)
        total = int(self.total)
        if total < 0 or completed < 0 or (total and completed > total):
            raise ValueError("Build progress counters are invalid")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "detail", _frozen_mapping(self.detail))


ProgressCallback = Callable[[BuildProgress], None]


@dataclass(frozen=True, slots=True)
class BuildRequest:
    project_root: str
    target: BuildTargetId
    output_dir: str
    profile: BuildProfile = field(default_factory=BuildProfile)
    asset_catalog_entries: tuple[Mapping[str, object], ...] = ()
    cancellation: BuildCancellationToken = field(
        default_factory=BuildCancellationToken,
        compare=False,
        repr=False,
    )
    progress: ProgressCallback | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        project_root = str(self.project_root or "").strip()
        output_dir = str(self.output_dir or "").strip()
        if not project_root:
            raise ValueError("BuildRequest.project_root is required")
        if not output_dir:
            raise ValueError("BuildRequest.output_dir is required")
        project_root = resolved_path(project_root)
        output_dir = resolve_project_path(output_dir, project_root)
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "target", BuildTargetId(self.target))
        object.__setattr__(
            self,
            "asset_catalog_entries",
            tuple(_frozen_mapping(item) for item in self.asset_catalog_entries),
        )

    def report(
        self,
        phase: str,
        completed: int,
        total: int,
        message: str,
        **detail: object,
    ) -> None:
        self.cancellation.raise_if_cancelled()
        if self.progress is not None:
            self.progress(
                BuildProgress(phase, completed, total, message, detail)
            )


@dataclass(frozen=True, slots=True)
class BuildDiagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    source: str = ""
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", DiagnosticSeverity(self.severity))
        object.__setattr__(self, "code", str(self.code or "").strip())
        object.__setattr__(self, "message", str(self.message or "").strip())
        object.__setattr__(self, "source", str(self.source or "").strip())
        object.__setattr__(self, "detail", _frozen_mapping(self.detail))
        if not self.code or not self.message:
            raise ValueError("Build diagnostics require a code and message")


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    available: bool
    diagnostics: tuple[BuildDiagnostic, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "details", _frozen_mapping(self.details))
        if self.available and any(
            item.severity is DiagnosticSeverity.ERROR for item in diagnostics
        ):
            raise ValueError("An available capability report cannot contain errors")


@dataclass(frozen=True, slots=True)
class BuildStep:
    id: str
    title: str
    phase: str
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("id", "title", "phase"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"BuildStep.{field_name} is required")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "detail", _frozen_mapping(self.detail))


@dataclass(frozen=True, slots=True)
class BuildPlan:
    target: BuildTargetId
    steps: tuple[BuildStep, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", BuildTargetId(self.target))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BuildArtifact:
    path: str
    kind: str
    size: int = 0

    def __post_init__(self) -> None:
        path = str(self.path or "").strip()
        kind = str(self.kind or "").strip()
        if not path or not kind or int(self.size) < 0:
            raise ValueError("Build artifact path, kind, and size are invalid")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size", int(self.size))


@dataclass(frozen=True, slots=True)
class BuildResult:
    target: BuildTargetId
    success: bool
    artifacts: tuple[BuildArtifact, ...] = ()
    diagnostics: tuple[BuildDiagnostic, ...] = ()
    manifest: Mapping[str, object] = field(default_factory=dict)
    logs: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", BuildTargetId(self.target))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "manifest", _frozen_mapping(self.manifest))
        object.__setattr__(self, "logs", tuple(str(item) for item in self.logs))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        if self.elapsed_seconds < 0:
            raise ValueError("BuildResult.elapsed_seconds cannot be negative")


class PlatformExporter(ABC):
    """Current interface implemented by core and InxPackage exporters."""

    @property
    @abstractmethod
    def exporter_id(self) -> str:
        """Return a stable reverse-DNS or package-style exporter identity."""

    @abstractmethod
    def targets(self) -> Sequence[BuildTarget]:
        """Return all targets contributed by this exporter."""

    def build_options(self, target: BuildTarget) -> Sequence[BuildOption]:
        """Declare every build option accepted for ``target``."""

        del target
        return ()

    @abstractmethod
    def doctor(self, request: BuildRequest) -> CapabilityReport:
        """Inspect the host toolchain without mutating the project."""

    @abstractmethod
    def create_plan(self, request: BuildRequest) -> BuildPlan:
        """Create an inspectable plan before performing build mutations."""

    @abstractmethod
    def execute(self, request: BuildRequest, plan: BuildPlan) -> BuildResult:
        """Execute an accepted plan and atomically publish its artifacts."""

    def audit(self, request: BuildRequest, result: BuildResult) -> BuildResult:
        """Validate the produced package; exporters may return richer results."""

        return result

    def smoke(self, request: BuildRequest, result: BuildResult) -> BuildResult:
        """Optionally run a target-specific launch or installation smoke test."""

        return result


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


__all__ = [
    "BuildArtifact",
    "BuildCancellationToken",
    "BuildConfiguration",
    "BuildDiagnostic",
    "BuildPlan",
    "BuildOption",
    "BuildOptionChoice",
    "BuildOptionKind",
    "BuildProfile",
    "BuildProgress",
    "BuildRequest",
    "BuildResult",
    "BuildStep",
    "BuildTarget",
    "BuildTargetId",
    "CapabilityReport",
    "DiagnosticSeverity",
    "PlatformCapabilities",
    "PlatformExporter",
    "ProgressCallback",
]
