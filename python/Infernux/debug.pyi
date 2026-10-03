from __future__ import annotations

from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, List, Callable


class LogType(Enum):
    """Type of log message."""

    LOG = auto()
    WARNING = auto()
    ERROR = auto()
    ASSERT = auto()
    EXCEPTION = auto()


class LogEntry:
    """A single log entry with message, type, timestamp, and metadata."""

    message: str
    log_type: LogType
    timestamp: datetime
    stack_trace: str
    context: Any
    internal: bool
    source_file: str
    source_line: int

    def __init__(
        self,
        message: str,
        log_type: LogType,
        timestamp: datetime,
        stack_trace: str = ...,
        context: Any = ...,
        internal: bool = ...,
        source_file: str = ...,
        source_line: int = ...,
    ) -> None: ...
    def get_formatted_time(self) -> str:
        """Return the timestamp as a formatted string."""
        ...
    def get_icon(self) -> str:
        """Return the icon character for this log type."""
        ...


class DebugConsole:
    """Singleton console that collects and filters log entries."""

    _instance: Optional[DebugConsole]

    def __new__(cls) -> DebugConsole: ...
    def __init__(self) -> None: ...
    @classmethod
    def instance(cls) -> DebugConsole: ...
    def add_listener(self, callback: Callable[[LogEntry], None]) -> None:
        """Register a callback invoked on every new log entry."""
        ...
    def remove_listener(self, callback: Callable[[LogEntry], None]) -> None:
        """Unregister a log listener callback."""
        ...
    def log(self, entry: LogEntry) -> None:
        """Add a log entry to the console."""
        ...
    def get_entries(self) -> List[LogEntry]:
        """Get all log entries."""
        ...
    def get_filtered_entries(
        self,
        show_logs: bool = ...,
        show_warnings: bool = ...,
        show_errors: bool = ...,
    ) -> List[LogEntry]:
        """Get log entries filtered by type."""
        ...
    def clear(self) -> None:
        """Clear all log entries."""
        ...
    @property
    def log_count(self) -> int:
        """Number of log-level entries."""
        ...
    @property
    def warning_count(self) -> int:
        """Number of warning-level entries."""
        ...
    @property
    def error_count(self) -> int:
        """Number of error-level entries."""
        ...


class Debug:
    """Utility class for logging messages to the console."""

    @staticmethod
    def log(message: Any, context: Any = ...) -> None:
        """Log a message to the console."""
        ...
    @staticmethod
    def log_warning(message: Any, context: Any = ...) -> None:
        """Log a warning message to the console."""
        ...
    @staticmethod
    def log_error(
        message: Any,
        context: Any = ...,
        *,
        source_file: str = ...,
        source_line: int = ...,
    ) -> None:
        """Log an error message to the console."""
        ...
    @staticmethod
    def log_exception(exception: Exception, context: Any = ...) -> None:
        """Log an exception to the console."""
        ...
    @staticmethod
    def log_suppressed(
        where: str,
        exc: BaseException,
        context: Any = ...,
    ) -> None:
        """Log an exception that was deliberately swallowed by the caller.

        Routes through ``log_warning`` so suppressed events are visible
        in the editor console.
        """
        ...
    @staticmethod
    def log_assert(
        condition: bool, message: Any = ..., context: Any = ...
    ) -> None:
        """Assert a condition and log if it fails."""
        ...
    @staticmethod
    def clear_console() -> None:
        """Clear all messages in the debug console."""
        ...
    @staticmethod
    def log_internal(message: Any, context: Any = ...) -> None:
        """Log an internal engine message (hidden from user by default)."""
        ...
    @staticmethod
    def log_debug(message: Any, context: Any = ...) -> None:
        """Emit a debug-only internal marker filtered from release output."""
        ...


def log(message: Any, context: Any = ...) -> None:
    """Log a message to the console (module-level shortcut)."""
    ...
def log_warning(message: Any, context: Any = ...) -> None:
    """Log a warning to the console (module-level shortcut)."""
    ...
def log_error(
    message: Any,
    context: Any = ...,
    *,
    source_file: str = ...,
    source_line: int = ...,
) -> None:
    """Log an error to the console (module-level shortcut)."""
    ...
def log_exception(exception: Exception, context: Any = ...) -> None: ...


class _DebugProxy:
    def __getattr__(self, name: str) -> Any: ...
    def __repr__(self) -> str: ...


debug: _DebugProxy
